# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Dict, Optional, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import gc
from diffusers.loaders import PeftAdapterMixin, FromOriginalModelMixin
from src.attention import attention_register
from src.transformer.efficiency.list_clear import unwrap_single_item_list
from src.transformer.efficiency.mod import InplaceRMSNorm
from src.transformer.efficiency.ops import (
    apply_gate_inplace,
    apply_scale_shift_inplace,
    chunked_feed_forward_inplace,
)

__all__ = ["SCAILModel"]


def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """
    Modulated layer norm with chunking to reduce peak memory.
    """
    B, S, D = hidden_states.shape
    in_dtype = hidden_states.dtype

    if chunk_size is None:
        out = norm_layer(hidden_states).to(in_dtype)
        apply_scale_shift_inplace(out, scale, shift)
        return out

    if S <= chunk_size:
        out = norm_layer(hidden_states).to(in_dtype)
        apply_scale_shift_inplace(out, scale, shift)
        return out

    out = torch.empty_like(hidden_states)
    scale_per_token = scale.dim() == 3 and scale.shape[1] == S

    for i in range(0, S, chunk_size):
        end = min(i + chunk_size, S)
        hs_chunk = hidden_states[:, i:end, :]

        if scale_per_token:
            scale_chunk = scale[:, i:end, :]
            shift_chunk = shift[:, i:end, :]
        else:
            scale_chunk = scale
            shift_chunk = shift

        out[:, i:end, :].copy_(norm_layer(hs_chunk))
        apply_scale_shift_inplace(out[:, i:end, :], scale_chunk, shift_chunk)

    return out


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
) -> torch.Tensor:
    """Chunked feed-forward to reduce peak memory."""
    return chunked_feed_forward_inplace(
        ff, hidden_states, chunk_dim=chunk_dim, chunk_size=chunk_size
    )


def _chunked_norm(
    norm_layer: nn.Module, hidden_states: torch.Tensor, chunk_size: Optional[int] = 8192
) -> torch.Tensor:
    """
    LayerNorm in chunks along the sequence dimension to reduce peak memory.
    """
    if isinstance(norm_layer, nn.Identity):
        return hidden_states

    if hidden_states.ndim != 3:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    if chunk_size is None:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    B, S, D = hidden_states.shape
    if S <= chunk_size:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    out = torch.empty_like(hidden_states)
    for i in range(0, S, chunk_size):
        end = min(i + chunk_size, S)
        out[:, i:end, :] = norm_layer(hidden_states[:, i:end, :])
    return out


T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def _rope_split_freqs(freqs: torch.Tensor, head_dim: int):
    """
    Split frequency table into (t, h, w) parts matching the per-head rotary layout.

    freqs: [max_seq, head_dim/2] complex
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    c = head_dim // 2
    return freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)


def _rope_complex_multiply_inplace(
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
) -> None:
    """
    In-place complex multiply on real/imag pairs:
      (x_real + i*x_imag) *= (freqs_real + i*freqs_imag)
    """
    x_real_orig = x_real.clone()
    x_real.mul_(freqs_real).addcmul_(x_imag, freqs_imag, value=-1.0)
    x_imag.mul_(freqs_real).addcmul_(x_real_orig, freqs_imag, value=1.0)


@amp.autocast(enabled=False)
def _rope_apply_3d_chunked(
    x: torch.Tensor,
    freqs: torch.Tensor,
    *,
    f: int,
    h: int,
    w: int,
    shift_f: int,
    shift_h: int,
    shift_w: int,
    downsample_hw_by_2: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply 3D RoPE (t/h/w) to x in chunks to reduce peak memory.

    x: [B, S, N, D] where D is per-head dim (must be even).
    freqs: [max_seq, D/2] complex
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D [B,S,N,D], got shape {tuple(x.shape)}")

    b, s, n, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"Per-head dim must be even for RoPE, got {d}")

    freqs_t, freqs_h, freqs_w = _rope_split_freqs(freqs, d)

    if downsample_hw_by_2:
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"Expected even H/W for downsampled RoPE, got H={h}, W={w}"
            )
        seq_len = f * (h // 2) * (w // 2)
    else:
        seq_len = f * h * w

    if seq_len != s:
        raise AssertionError(f"seq_len mismatch: expected {seq_len}, got {s}")

    # Basic range checks (helps catch misconfigured shifts early).
    if shift_f < 0 or shift_h < 0 or shift_w < 0:
        raise ValueError(
            f"Negative RoPE shift not supported: {shift_f=}, {shift_h=}, {shift_w=}"
        )
    if shift_f + f > freqs_t.size(0):
        raise AssertionError(f"{shift_f + f} > {freqs_t.size(0)}")
    if shift_h + h > freqs_h.size(0):
        raise AssertionError(f"{shift_h + h} > {freqs_h.size(0)}")
    if shift_w + w > freqs_w.size(0):
        raise AssertionError(f"{shift_w + w} > {freqs_w.size(0)}")

    if chunk_size is None or chunk_size <= 0:
        chunk_size = s

    out = torch.empty((b, s, n, d), device=x.device, dtype=torch.float32)

    # Pre-slice the ranges we will index into (keeps index math simple).
    freqs_t = freqs_t[shift_f : shift_f + f]
    freqs_h = freqs_h[shift_h : shift_h + h]
    freqs_w = freqs_w[shift_w : shift_w + w]

    if downsample_hw_by_2:
        # Match the old avg_pool2d(kernel=2,stride=2) behavior on (H,W) after slicing.
        freqs_h = 0.5 * (freqs_h[0::2] + freqs_h[1::2])
        freqs_w = 0.5 * (freqs_w[0::2] + freqs_w[1::2])
        hw = (h // 2) * (w // 2)
        w_eff = w // 2
    else:
        hw = h * w
        w_eff = w

    for start in range(0, s, chunk_size):
        end = min(start + chunk_size, s)
        pos = torch.arange(start, end, device=x.device, dtype=torch.long)

        t_idx = pos // hw
        rem = pos - t_idx * hw
        h_idx = rem // w_eff
        w_idx = rem - h_idx * w_eff

        mult = torch.cat(
            [
                freqs_t.index_select(0, t_idx),
                freqs_h.index_select(0, h_idx),
                freqs_w.index_select(0, w_idx),
            ],
            dim=1,
        )  # [chunk, D/2] complex

        # Complex multiply in float64 for numerical stability (matches original).
        x_chunk = x[:, start:end].to(torch.float64).reshape(b, end - start, n, -1, 2)
        x_chunk = torch.view_as_complex(x_chunk)  # [B, chunk, N, D/2]
        y_chunk = x_chunk * mult.view(1, end - start, 1, -1)
        y_chunk = torch.view_as_real(y_chunk).flatten(-2)  # [B, chunk, N, D]
        out[:, start:end] = y_chunk.float()

    return out


@amp.autocast(enabled=False)
def rope_apply_ref(x, freqs, **kwargs):
    f = 1
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = 0
    shift_h = kwargs["rope_H_shift"]
    shift_w = kwargs["rope_W_shift"]
    rope_chunk_size = kwargs.get("rope_chunk_size", None)

    return _rope_apply_3d_chunked(
        x,
        freqs,
        f=f,
        h=h,
        w=w,
        shift_f=shift_f,
        shift_h=shift_h,
        shift_w=shift_w,
        downsample_hw_by_2=False,
        chunk_size=rope_chunk_size,
    )


@amp.autocast(enabled=False)
def rope_apply_video(x, freqs, **kwargs):
    f = kwargs["rope_T"]
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = 1 + int(kwargs.get("rope_T_shift", 0) or 0)  # reference frame
    shift_h = kwargs["rope_H_shift"]
    shift_w = kwargs["rope_W_shift"]
    rope_chunk_size = kwargs.get("rope_chunk_size", None)

    return _rope_apply_3d_chunked(
        x,
        freqs,
        f=f,
        h=h,
        w=w,
        shift_f=shift_f,
        shift_h=shift_h,
        shift_w=shift_w,
        downsample_hw_by_2=False,
        chunk_size=rope_chunk_size,
    )


@amp.autocast(enabled=False)
def rope_apply_pose(x, freqs, **kwargs):
    f = kwargs["rope_T"]
    h = kwargs["rope_H"]
    w = kwargs["rope_W"]
    shift_f = 1 + int(kwargs.get("rope_T_shift", 0) or 0)  # reference frame
    shift_h = kwargs["rope_H_shift"] + kwargs["global_rope_H"]
    shift_w = kwargs["rope_W_shift"] + kwargs["global_rope_W"]
    rope_chunk_size = kwargs.get("rope_chunk_size", None)

    return _rope_apply_3d_chunked(
        x,
        freqs,
        f=f,
        h=h,
        w=w,
        shift_f=shift_f,
        shift_h=shift_h,
        shift_w=shift_w,
        downsample_hw_by_2=True,
        chunk_size=rope_chunk_size,
    )


def rope_apply_scail(x, **kwargs):
    """
    x: [b, s, n, d]
    """
    ref_length = kwargs["ref_length"]
    video_length = kwargs["seq_length"]
    pose_length = kwargs["pose_length"]

    x_ref = x[:, :ref_length]
    x_video = x[:, ref_length : ref_length + video_length]
    x_pose = x[:, -pose_length:]

    return torch.cat(
        [
            rope_apply_ref(x_ref, **kwargs),
            rope_apply_video(x_video, **kwargs),
            rope_apply_pose(x_pose, **kwargs),
        ],
        dim=1,
    )


@amp.autocast(enabled=False)
def _rope_apply_3d_inplace_chunked(
    x: torch.Tensor,
    freqs_t: torch.Tensor,
    freqs_h: torch.Tensor,
    freqs_w: torch.Tensor,
    *,
    f: int,
    h: int,
    w: int,
    shift_f: int,
    shift_h: int,
    shift_w: int,
    downsample_hw_by_2: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply 3D RoPE (t/h/w) to `x` IN-PLACE in chunks to reduce peak memory.

    x: [B, S, N, D] where D is per-head dim (must be even).
    freqs_*: complex tensors on CPU or GPU.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D [B,S,N,D], got shape {tuple(x.shape)}")

    b, s, n, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"Per-head dim must be even for RoPE, got {d}")

    if downsample_hw_by_2:
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"Expected even H/W for downsampled RoPE, got H={h}, W={w}"
            )
        seq_len = f * (h // 2) * (w // 2)
    else:
        seq_len = f * h * w

    if seq_len != s:
        raise AssertionError(f"seq_len mismatch: expected {seq_len}, got {s}")

    if shift_f < 0 or shift_h < 0 or shift_w < 0:
        raise ValueError(
            f"Negative RoPE shift not supported: {shift_f=}, {shift_h=}, {shift_w=}"
        )

    if chunk_size is None or chunk_size <= 0:
        chunk_size = s

    # Pre-slice the ranges we will index into, then move the (small) slices to the
    # correct device once per segment (enables rope freqs on CPU).
    freqs_t = freqs_t[shift_f : shift_f + f]
    freqs_h = freqs_h[shift_h : shift_h + h]
    freqs_w = freqs_w[shift_w : shift_w + w]
    if freqs_t.device != x.device:
        freqs_t = freqs_t.to(x.device)
    if freqs_h.device != x.device:
        freqs_h = freqs_h.to(x.device)
    if freqs_w.device != x.device:
        freqs_w = freqs_w.to(x.device)

    if downsample_hw_by_2:
        freqs_h = 0.5 * (freqs_h[0::2] + freqs_h[1::2])
        freqs_w = 0.5 * (freqs_w[0::2] + freqs_w[1::2])
        hw = (h // 2) * (w // 2)
        w_eff = w // 2
    else:
        hw = h * w
        w_eff = w

    for start in range(0, s, chunk_size):
        end = min(start + chunk_size, s)
        pos = torch.arange(start, end, device=x.device, dtype=torch.long)

        t_idx = pos // hw
        rem = pos - t_idx * hw
        h_idx = rem // w_eff
        w_idx = rem - h_idx * w_eff

        # Build real/imag multipliers directly to avoid allocating complex intermediates.
        mt = freqs_t.index_select(0, t_idx)
        mh = freqs_h.index_select(0, h_idx)
        mw = freqs_w.index_select(0, w_idx)
        mult_real = torch.cat(
            [
                mt.real.to(dtype=torch.float32),
                mh.real.to(dtype=torch.float32),
                mw.real.to(dtype=torch.float32),
            ],
            dim=1,
        ).view(1, end - start, 1, -1)
        mult_imag = torch.cat(
            [
                mt.imag.to(dtype=torch.float32),
                mh.imag.to(dtype=torch.float32),
                mw.imag.to(dtype=torch.float32),
            ],
            dim=1,
        ).view(1, end - start, 1, -1)

        x_chunk = x[:, start:end]  # [B, chunk, N, D]
        x_pairs = x_chunk.unflatten(-1, (-1, 2))
        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]

        # Do RoPE math in fp32 for fp16/bf16 inputs (chunked to cap peak memory).
        if x.dtype in (torch.float16, torch.bfloat16):
            x_real_f = x_real.float()
            x_imag_f = x_imag.float()
            _rope_complex_multiply_inplace(x_real_f, x_imag_f, mult_real, mult_imag)
            x_real.copy_(x_real_f.to(dtype=x.dtype))
            x_imag.copy_(x_imag_f.to(dtype=x.dtype))
        else:
            _rope_complex_multiply_inplace(
                x_real,
                x_imag,
                mult_real.to(dtype=x.dtype),
                mult_imag.to(dtype=x.dtype),
            )

    return x


def rope_apply_scail_inplace(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    In-place RoPE for SCAIL's concatenated (ref | video | pose) token stream.

    x: [B, S, N, D]
    """
    ref_length = int(kwargs["ref_length"])
    video_length = int(kwargs["seq_length"])
    pose_length = int(kwargs["pose_length"])

    rope_t = int(kwargs["rope_T"])
    rope_h = int(kwargs["rope_H"])
    rope_w = int(kwargs["rope_W"])
    rope_h_shift = int(kwargs["rope_H_shift"])
    rope_w_shift = int(kwargs["rope_W_shift"])
    global_rope_h = int(kwargs["global_rope_H"])
    global_rope_w = int(kwargs["global_rope_W"])
    rope_t_shift = 1 + int(kwargs.get("rope_T_shift", 0) or 0)  # reference frame offset
    rope_chunk_size = kwargs.get("rope_chunk_size", None)

    freqs = kwargs["freqs"]
    _, _, _, d = x.shape
    freqs_t, freqs_h, freqs_w = _rope_split_freqs(freqs, d)

    # ref segment
    _rope_apply_3d_inplace_chunked(
        x[:, :ref_length],
        freqs_t,
        freqs_h,
        freqs_w,
        f=1,
        h=rope_h,
        w=rope_w,
        shift_f=0,
        shift_h=rope_h_shift,
        shift_w=rope_w_shift,
        downsample_hw_by_2=False,
        chunk_size=rope_chunk_size,
    )

    # video segment
    _rope_apply_3d_inplace_chunked(
        x[:, ref_length : ref_length + video_length],
        freqs_t,
        freqs_h,
        freqs_w,
        f=rope_t,
        h=rope_h,
        w=rope_w,
        shift_f=rope_t_shift,
        shift_h=rope_h_shift,
        shift_w=rope_w_shift,
        downsample_hw_by_2=False,
        chunk_size=rope_chunk_size,
    )

    # pose segment
    _rope_apply_3d_inplace_chunked(
        x[:, -pose_length:],
        freqs_t,
        freqs_h,
        freqs_w,
        f=rope_t,
        h=rope_h,
        w=rope_w,
        shift_f=rope_t_shift,
        shift_h=rope_h_shift + global_rope_h,
        shift_w=rope_w_shift + global_rope_w,
        downsample_hw_by_2=True,
        chunk_size=rope_chunk_size,
    )

    return x


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = (
            InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
            if qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
            if qk_norm
            else nn.Identity()
        )

    def forward(self, x, seq_lens, rope_apply_func, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # Enable list-based early ref dropping (caller may pass `[tensor]`).
        x = unwrap_single_item_list(x)
        seq_lens = unwrap_single_item_list(seq_lens)

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        device_index = x.device.index
        device_type = x.device.type

        # Projections + in-place Q/K norm (saves one buffer vs `norm(q(x))` style).
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # Clear input tensor early to reduce peak memory (q+k+v+x -> q+k+v)
        del x

        self.norm_q(q)
        self.norm_k(k)
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)

        # Apply RoPE in-place for inference. For training/grad, allow out-of-place
        # fallback to preserve autograd correctness.
        if torch.is_grad_enabled() or q.requires_grad:
            q = rope_apply_func(q)
        else:
            rope_apply_func(q)
        if torch.is_grad_enabled() or k.requires_grad:
            k = rope_apply_func(k)
        else:
            rope_apply_func(k)
        
        x = attention_register.call(
            q=q.transpose(1, 2),
            k=k.transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=seq_lens,
            window_size=self.window_size,
            key="flash"
        ).transpose(1, 2)
        del q, k, v

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        x = unwrap_single_item_list(x)
        context = unwrap_single_item_list(context)
        context_lens = unwrap_single_item_list(context_lens)

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value (with in-place Q/K norm)
        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        self.norm_q(q)
        self.norm_k(k)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)
    
        # compute attention
        x = attention_register.call(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            k_lens=context_lens,
            window_size=self.window_size,
            key="flash"
        ).transpose(1, 2)
        
        del q, k, v

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = (
            InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
            if qk_norm
            else nn.Identity()
        )

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        x = unwrap_single_item_list(x)
        context = unwrap_single_item_list(context)
        context_lens = unwrap_single_item_list(context_lens)

        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim
        device_index = x.device.index
        device_type = x.device.type

        # compute query, key, value (with in-place Q/K norm)
        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        k_img = self.k_img(context_img)
        v_img = self.v_img(context_img)

        self.norm_q(q)
        self.norm_k(k)
        self.norm_k_img(k_img)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)
        k_img = k_img.view(b, -1, n, d)
        v_img = v_img.view(b, -1, n, d)
        
        
        img_x = attention_register.call(
            q.transpose(1, 2),
            k_img.transpose(1, 2),
            v_img.transpose(1, 2),
            k_lens=None,
            window_size=self.window_size,
            key="flash"
        ).transpose(1, 2)
        del k_img, v_img

        x = attention_register.call(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            k_lens=context_lens,
            window_size=self.window_size,
            key="flash"
        ).transpose(1, 2)
        del q, k, v

        # output
        if (not torch.is_grad_enabled()) and (not x.requires_grad) and (not img_x.requires_grad):
            x.add_(img_x)
        else:
            x = x + img_x
        x = x.flatten(2)
        x = self.o(x)

        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # Chunking configuration (disabled by default)
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1
        self._mod_norm_chunk_size: Optional[int] = None
        self._norm_chunk_size: Optional[int] = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        self._ff_chunk_size = chunk_size
        self._ff_chunk_dim = dim
    
    def set_chunk_norms(
        self,
        *,
        modulated_norm_chunk_size: Optional[int] = None,
        norm_chunk_size: Optional[int] = None,
    ) -> None:
        self._mod_norm_chunk_size = modulated_norm_chunk_size
        self._norm_chunk_size = norm_chunk_size
  
    def forward(
        self,
        x,
        e,
        seq_lens,
        context,
        context_lens,
        **kwargs,  # contains rope_apply_func
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        
        x = unwrap_single_item_list(x)
        e = unwrap_single_item_list(e)
        seq_lens = unwrap_single_item_list(seq_lens)
        context = unwrap_single_item_list(context)
        context_lens = unwrap_single_item_list(context_lens)

        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        # self-attention with chunked modulated norm
        norm_x = _chunked_modulated_norm(
            self.norm1, x, e[1], e[0], chunk_size=self._mod_norm_chunk_size
        )
        # Pass as single-item lists so `unwrap_single_item_list` clears references early.
        norm_x_list = [norm_x]
        seq_lens_list = [seq_lens]
        del norm_x, seq_lens
        y = self.self_attn(norm_x_list, seq_lens_list, **kwargs)
        del norm_x_list, seq_lens_list
      
        # Inference-only in-place residual/gating to avoid allocating `x + y * gate`.
        if (not torch.is_grad_enabled()) and (not x.requires_grad) and (not y.requires_grad):
            apply_gate_inplace(y, e[2].to(dtype=y.dtype))
            x.add_(y)
        else:
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[2]
        
    
        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            norm3_x = _chunked_norm(self.norm3, x, chunk_size=self._norm_chunk_size)
            norm3_x_list = [norm3_x]
            context_list = [context]
            context_lens_list = [context_lens]
            del norm3_x, context, context_lens
            ca = self.cross_attn(norm3_x_list, context_list, context_lens_list)
            

            if (not torch.is_grad_enabled()) and (not x.requires_grad) and (not ca.requires_grad):
                x.add_(ca)
            else:
                x = x + ca
            del ca
   
            ffn_x = _chunked_modulated_norm(
                self.norm2, x, e[4], e[3], chunk_size=self._mod_norm_chunk_size
            )
            if self._ff_chunk_size is not None:
                y = _chunked_feed_forward(
                    self.ffn, ffn_x, self._ff_chunk_dim, self._ff_chunk_size
                )

            else:
                y = self.ffn(ffn_x)
            del ffn_x

            if (not torch.is_grad_enabled()) and (not x.requires_grad) and (not y.requires_grad):
                apply_gate_inplace(y, e[5].to(dtype=y.dtype))
                x.add_(y)

            else:
                with amp.autocast(dtype=torch.float32):
                    x = x + y * e[5]
            
            return x

        x = cross_attn_ffn(x, context, context_lens, e)

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        # Chunking configuration
        self._mod_norm_chunk_size: Optional[int] = None

    def set_chunk_norms(self, modulated_norm_chunk_size: Optional[int] = None) -> None:
        self._mod_norm_chunk_size = modulated_norm_chunk_size

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        x = unwrap_single_item_list(x)
        e = unwrap_single_item_list(e)

        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            norm_x = _chunked_modulated_norm(
                self.norm, x, e[1], e[0], chunk_size=self._mod_norm_chunk_size
            )
            x = self.head(norm_x)
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280)
            )

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


from einops import rearrange
from functools import partial, reduce
from operator import mul


class SCAILModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    SCAIL diffusion backbone.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    # Chunking profile presets
    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        "none": {
            "ffn_chunk_size": None,
            "ffn_chunk_dim": 1,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
            "rope_chunk_size": None,
        },
        "light": {
            "ffn_chunk_size": 2048,
            "ffn_chunk_dim": 1,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
            "rope_chunk_size": 8192,
        },
        "balanced": {
            "ffn_chunk_size": 512,
            "ffn_chunk_dim": 1,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
            "rope_chunk_size": 4096,
        },
        "aggressive": {
            "ffn_chunk_size": 256,
            "ffn_chunk_dim": 1,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
            "rope_chunk_size": 1024,
        },
    }

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        pose_rope_shift=[0, 0, 120],  # shift in (t, h, w) for pose rope embedding
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v", "vace"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.pose_rope_shift = pose_rope_shift
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.patch_embedding_pose = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(8192, d - 4 * (d // 6)),
                rope_params(8192, 2 * (d // 6)),
                rope_params(8192, 2 * (d // 6)),
            ],
            dim=1,
        )
        self.hidden_size_head = d

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

        # Default: no chunking unless explicitly enabled
        self.set_chunking_profile("none")

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        """Enable/disable chunked feed-forward on all transformer blocks."""
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        """Return available chunking profile names."""
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunking_profile(self, profile_name: str) -> None:
        """
        Apply a predefined chunking profile across the whole model.
        """
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. "
                f"Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )

        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name
        self._out_modulated_norm_chunk_size = p.get(
            "out_modulated_norm_chunk_size", None
        )
        self._rope_chunk_size = p.get("rope_chunk_size", None)

        # Configure FFN chunking (both size and chunk dimension).
        # This drives `WanAttentionBlock._ff_chunk_size/_ff_chunk_dim`.
        ffn_chunk_size = p.get("ffn_chunk_size", None)
        ffn_chunk_dim = p.get("ffn_chunk_dim", 1)
        self.set_chunk_feed_forward(ffn_chunk_size, dim=int(ffn_chunk_dim or 1))
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

        # Also configure head chunking
        self.head.set_chunk_norms(
            modulated_norm_chunk_size=p.get("out_modulated_norm_chunk_size", None)
        )

    def apply_i2v_ones_masks(self, inputs: torch.Tensor, mask_dim: int = 4):
        b, d, t, h, w = inputs.shape
        out = torch.empty(
            (b, d + mask_dim, t, h, w), device=inputs.device, dtype=inputs.dtype
        )
        out[:, :d].copy_(inputs)
        out[:, d:].fill_(1)
        return out

    def apply_i2v_zeros_masks(self, inputs: torch.Tensor, mask_dim: int = 4):
        b, d, t, h, w = inputs.shape
        out = torch.empty(
            (b, d + mask_dim, t, h, w), device=inputs.device, dtype=inputs.dtype
        )
        out[:, :d].copy_(inputs)
        out[:, d:].zero_()
        return out

    def merge_list_of_tensors_to_batch(self, inputs: list[torch.Tensor]):
        return torch.cat([u.unsqueeze(0) for u in inputs], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states_pose: torch.Tensor,
        encoder_hidden_states_reference: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        seq_len: int,
        encoder_hidden_states_clip: torch.Tensor = None,
        rope_on_cpu: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            ref_latents (list[Tensor]):
                list of reference latents, each with shape [C_in, 1, H, W]
            pose_latents (list[Tensor]):
                list of downsampled pose video latents, each with shape [C_in, F, H / 2, W / 2]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]

        Args:
            rope_on_cpu (`bool`, *optional*):
                If True, keep RoPE frequency tables on CPU and only move the small slices needed per chunk to GPU.
                This reduces peak VRAM usage at the cost of some extra CPU↔GPU transfers.
        """
        # Accept list/tuple wrappers (common in some pipelines) and clear list references early.
        hidden_states = unwrap_single_item_list(hidden_states)
        encoder_hidden_states_pose = unwrap_single_item_list(encoder_hidden_states_pose)
        encoder_hidden_states_reference = unwrap_single_item_list(
            encoder_hidden_states_reference
        )
        encoder_hidden_states = unwrap_single_item_list(encoder_hidden_states)
        encoder_hidden_states_clip = unwrap_single_item_list(encoder_hidden_states_clip)

        rope_t_shift = int(kwargs.get("rope_T_shift", 0) or 0)
        # Backward compatible: allow `rope_on_cpu` to still be passed via `**kwargs`.
        if rope_on_cpu is None:
            rope_on_cpu = bool(kwargs.get("rope_on_cpu", False))
        clip_fea = encoder_hidden_states_clip
        t = timestep

        assert clip_fea is not None
        device = hidden_states.device

        # Keep RoPE freqs on CPU when requested, otherwise cache a device copy for
        # the (training / reference) out-of-place RoPE path.
        freqs = self.freqs
        if not rope_on_cpu:
            if not hasattr(self, "_freqs_device_cache"):
                self._freqs_device_cache = {}
            cache_key = str(device)
            if cache_key not in self._freqs_device_cache:
                self._freqs_device_cache[cache_key] = self.freqs.to(device)
            freqs = self._freqs_device_cache[cache_key]

        # Avoid list<->tensor roundtrips; work directly on batched tensors.
        x_in = self.apply_i2v_zeros_masks(hidden_states)
        ref_in = self.apply_i2v_ones_masks(encoder_hidden_states_reference)
        pose_in = self.apply_i2v_ones_masks(encoder_hidden_states_pose)

        B, _, T, H, W = x_in.shape

        assert pose_in.shape[3] == H // 2
        assert pose_in.shape[4] == W // 2

        ref_length = 1 * H * W // reduce(mul, self.patch_size)
        seq_length = T * ref_length
        pose_length = T * (H // 2) * (W // 2) // reduce(mul, self.patch_size)

        # embeddings
        # Run patch embedding separately for ref/video to avoid a large concat in latent space.
        ref_emb = self.patch_embedding(ref_in)
        vid_emb = self.patch_embedding(x_in)
        pose_emb = self.patch_embedding_pose(pose_in)
        del x_in, ref_in, pose_in

        x = torch.cat(
            [
                rearrange(ref_emb, "b c t h w -> b (t h w) c"),
                rearrange(vid_emb, "b c t h w -> b (t h w) c"),
                rearrange(pose_emb, "b c t h w -> b (t h w) c"),
            ],
            dim=1,
        )
        del ref_emb, vid_emb, pose_emb

        # seq_lens is used for flash attention k_lens (constant for this model).
        seq_lens = torch.full((B,), x.shape[1], dtype=torch.long)
        # seq_lens is used for flash attention k_lens

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = encoder_hidden_states
        if context.shape[1] < self.text_len:
            pad = torch.zeros(
                (context.shape[0], self.text_len - context.shape[1], context.shape[2]),
                device=context.device,
                dtype=context.dtype,
            )
            context = torch.cat([context, pad], dim=1)
        elif context.shape[1] > self.text_len:
            context = context[:, : self.text_len]
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        rope_t = T // self.patch_size[0]
        rope_h = H // self.patch_size[1]
        rope_w = W // self.patch_size[2]

        # grid_sizes:
        # Original spatial-temporal grid dimensions before patching,
        # shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)
        grid_sizes = torch.stack(
            [torch.tensor((rope_t, rope_h, rope_w), dtype=torch.long) for _ in range(B)]
        )

        # arguments
        model_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=context_lens,
            ref_length=ref_length,
            seq_length=seq_length,
            pose_length=pose_length,
        )

        model_kwargs["rope_T"] = rope_t
        model_kwargs["rope_H"] = rope_h
        model_kwargs["rope_W"] = rope_w
        model_kwargs["hidden_size_head"] = self.hidden_size_head

        model_kwargs["global_rope_H"] = self.pose_rope_shift[1]
        model_kwargs["global_rope_W"] = self.pose_rope_shift[2]
        model_kwargs["rope_chunk_size"] = getattr(self, "_rope_chunk_size", None)
        model_kwargs["rope_T_shift"] = rope_t_shift
        model_kwargs["rope_on_cpu"] = rope_on_cpu

        # TODO: add shift based on rank of sequence parallelism
        model_kwargs["rope_H_shift"] = 0
        model_kwargs["rope_W_shift"] = 0

        def apply_rope_scail(x):
            """
            x: [b, s, n, d]
            """
            if torch.is_grad_enabled() or x.requires_grad:
                return rope_apply_scail(x, **model_kwargs)
            return rope_apply_scail_inplace(x, **model_kwargs)

        model_kwargs["rope_apply_func"] = apply_rope_scail

        for block in self.blocks:
            # Pass as a single-item list so the block can clear the wrapper early.
            x_list = [x]
            del x
            x = block(x_list, **model_kwargs)


        # head
        x_list = [x]
        e_list = [e]
        del x, e
        x = self.head(x_list, e_list)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, offset=ref_length)
        return torch.stack([u.float() for u in x])

    def unpatchify(self, x, grid_sizes, offset: int = 0):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # only keep denoised part of u
            u = u[offset : offset + math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
