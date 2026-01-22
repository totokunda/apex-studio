import torch
from torch import nn

import torch.cuda.amp as amp
import math
import torch
from typing import Dict, Optional, Tuple
from einops import rearrange
from torch import nn
from einops import rearrange
from diffusers import ModelMixin, ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.configuration_utils import register_to_config
from src.attention import attention_register
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from src.transformer.efficiency.ops import (
    apply_gate_inplace,
    apply_scale_shift_inplace,
    chunked_feed_forward_inplace,
)
from src.transformer.efficiency.mod import InplaceRMSNorm
from src.transformer.efficiency.list_clear import unwrap_single_item_list
import warnings

warnings.filterwarnings("ignore")


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


class DummyAdapterLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=13,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=1536,
        context_tokens=16,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.audio_proj_glob_1 = DummyAdapterLayer(
            nn.Linear(self.input_dim, intermediate_dim)
        )
        self.audio_proj_glob_2 = DummyAdapterLayer(
            nn.Linear(intermediate_dim, intermediate_dim)
        )
        self.audio_proj_glob_3 = DummyAdapterLayer(
            nn.Linear(intermediate_dim, context_tokens * output_dim)
        )

        self.audio_proj_glob_norm = DummyAdapterLayer(nn.LayerNorm(output_dim))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, audio_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.audio_proj_glob_1(audio_embeds))
        audio_embeds = torch.relu(self.audio_proj_glob_2(audio_embeds))

        context_tokens = self.audio_proj_glob_3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        context_tokens = self.audio_proj_glob_norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens


import types


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
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    # Backward-compatible shim (older code called `rope_apply` directly).
    return rope_apply_video(x, grid_sizes=grid_sizes, freqs=freqs, chunk_size=None)


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
def _rope_apply_3d_inplace_chunked(
    x: torch.Tensor,
    freqs_t: torch.Tensor,
    freqs_h: torch.Tensor,
    freqs_w: torch.Tensor,
    *,
    f: int,
    h: int,
    w: int,
    shift_f: int = 0,
    shift_h: int = 0,
    shift_w: int = 0,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply 3D RoPE (t/h/w) to `x` IN-PLACE in chunks to reduce peak memory.

    x: [B, S, N, D] where D is per-head dim (must be even).
    freqs_*: complex tensors on CPU or GPU.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D [B,S,N,D], got shape {tuple(x.shape)}")

    _, s, _, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"Per-head dim must be even for RoPE, got {d}")

    seq_len = f * h * w
    if seq_len != s:
        raise AssertionError(f"seq_len mismatch: expected {seq_len}, got {s}")

    if shift_f < 0 or shift_h < 0 or shift_w < 0:
        raise ValueError(
            f"Negative RoPE shift not supported: {shift_f=}, {shift_h=}, {shift_w=}"
        )

    if chunk_size is None or chunk_size <= 0:
        chunk_size = s

    # Pre-slice ranges, then move (small) slices to the correct device once per segment.
    freqs_t = freqs_t[shift_f : shift_f + f]
    freqs_h = freqs_h[shift_h : shift_h + h]
    freqs_w = freqs_w[shift_w : shift_w + w]
    if freqs_t.device != x.device:
        freqs_t = freqs_t.to(x.device)
    if freqs_h.device != x.device:
        freqs_h = freqs_h.to(x.device)
    if freqs_w.device != x.device:
        freqs_w = freqs_w.to(x.device)

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


@amp.autocast(enabled=False)
def _rope_apply_3d_chunked(
    x: torch.Tensor,
    freqs: torch.Tensor,
    *,
    f: int,
    h: int,
    w: int,
    shift_f: int = 0,
    shift_h: int = 0,
    shift_w: int = 0,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply 3D RoPE (t/h/w) to x in chunks to reduce peak memory (out-of-place).

    x: [B, S, N, D] where D is per-head dim (must be even).
    freqs: [max_seq, D/2] complex (can live on CPU).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D [B,S,N,D], got shape {tuple(x.shape)}")

    b, s, n, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"Per-head dim must be even for RoPE, got {d}")

    freqs_t, freqs_h, freqs_w = _rope_split_freqs(freqs, d)

    seq_len = f * h * w
    if seq_len != s:
        raise AssertionError(f"seq_len mismatch: expected {seq_len}, got {s}")

    if chunk_size is None or chunk_size <= 0:
        chunk_size = s

    out = torch.empty((b, s, n, d), device=x.device, dtype=torch.float32)

    freqs_t = freqs_t[shift_f : shift_f + f]
    freqs_h = freqs_h[shift_h : shift_h + h]
    freqs_w = freqs_w[shift_w : shift_w + w]
    if freqs_t.device != x.device:
        freqs_t = freqs_t.to(x.device)
    if freqs_h.device != x.device:
        freqs_h = freqs_h.to(x.device)
    if freqs_w.device != x.device:
        freqs_w = freqs_w.to(x.device)

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

        # Complex multiply in float64 for numerical stability.
        x_chunk = x[:, start:end].to(torch.float64).reshape(b, end - start, n, -1, 2)
        x_chunk = torch.view_as_complex(x_chunk)  # [B, chunk, N, D/2]
        y_chunk = x_chunk * mult.view(1, end - start, 1, -1)
        y_chunk = torch.view_as_real(y_chunk).flatten(-2)  # [B, chunk, N, D]
        out[:, start:end] = y_chunk.float()

    return out


def rope_apply_video_inplace(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    In-place RoPE for HuMo's video token stream.

    x: [B, S, N, D] (padded to max seq); RoPE is applied to the first (F*H*W) tokens per sample.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D [B,S,N,D], got shape {tuple(x.shape)}")

    _, _, _, d = x.shape
    freqs_t, freqs_h, freqs_w = _rope_split_freqs(freqs, d)
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = int(f) * int(h) * int(w)
        if seq_len <= 0:
            continue
        _rope_apply_3d_inplace_chunked(
            x[i : i + 1, :seq_len],
            freqs_t,
            freqs_h,
            freqs_w,
            f=int(f),
            h=int(h),
            w=int(w),
            shift_f=0,
            shift_h=0,
            shift_w=0,
            chunk_size=chunk_size,
        )
    return x


def rope_apply_video(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Out-of-place RoPE for HuMo (safer for autograd than the in-place path).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D [B,S,N,D], got shape {tuple(x.shape)}")

    b, s, n, d = x.shape
    out = torch.empty((b, s, n, d), device=x.device, dtype=torch.float32)
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = int(f) * int(h) * int(w)
        if seq_len > 0:
            out[i : i + 1, :seq_len] = _rope_apply_3d_chunked(
                x[i : i + 1, :seq_len],
                freqs,
                f=int(f),
                h=int(h),
                w=int(w),
                shift_f=0,
                shift_h=0,
                shift_w=0,
                chunk_size=chunk_size,
            )
        if seq_len < s:
            out[i, seq_len:] = x[i, seq_len:].to(dtype=torch.float32)
    return out


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
            x(Tensor): Shape [B, L, num_heads, C / num_heads], torch.Size([1, 9360, 5120])
            seq_lens(Tensor): Shape [B], tensor([9360])
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W), tensor([[ 6, 30, 52]])
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # Enable list-based early ref dropping (caller may pass `[tensor]`).
        x = unwrap_single_item_list(x)
        seq_lens = unwrap_single_item_list(seq_lens)

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

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
            key="flash",
        ).transpose(1, 2)
        del q, k, v
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanSelfAttentionSepKVDim(nn.Module):

    def __init__(
        self, kv_dim, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6
    ):
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
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
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
            x(Tensor): Shape [B, L, num_heads, C / num_heads], torch.Size([1, 9360, 5120])
            seq_lens(Tensor): Shape [B], tensor([9360])
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W), tensor([[ 6, 30, 52]])
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        x = unwrap_single_item_list(x)
        seq_lens = unwrap_single_item_list(seq_lens)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        del x

        self.norm_q(q)
        self.norm_k(k)
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)

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
            key="flash",
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
            key="flash",
        ).transpose(1, 2)
        del q, k, v

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttentionGather(WanSelfAttentionSepKVDim):

    def forward(self, x, context, context_lens, grid_sizes, freqs, audio_seq_len):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C] - video tokens
            context(Tensor): Shape [B, L2, C] - audio tokens with shape [B, frames*16, 1536]
            context_lens(Tensor): Shape [B] - actually seq_lens from call (video sequence length)
            grid_sizes(Tensor): Shape [B, 3] - video grid dimensions (F, H, W)
            freqs(Tensor): RoPE frequencies
            audio_seq_len(Tensor): Actual audio sequence length (frames * 16)
        """
        x = unwrap_single_item_list(x)
        context = unwrap_single_item_list(context)
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Handle video spatial structure: group tokens by frame
        hlen_wlen = int(grid_sizes[0][1] * grid_sizes[0][2])  # H * W
        q = q.reshape(-1, hlen_wlen, n, d)  # [B * F_video, H * W, n, d]

        # Handle audio temporal structure: 16 tokens per frame
        k = k.reshape(-1, 16, n, d)  # [B * F_audio, 16, n, d]
        v = v.reshape(-1, 16, n, d)  # [B * F_audio, 16, n, d]

        # attention_register expects [B, H, S, D]; put heads in 2nd dim
        q = q.transpose(1, 2)  # [B * F_video, n, H * W, d]
        k = k.transpose(1, 2)  # [B * F_audio, n, 16, d]
        v = v.transpose(1, 2)  # [B * F_audio, n, 16, d]

        # Match the original HuMo behaviour: if audio has more frames than video,
        # drop the extra audio frames so that batch sizes line up for attention.
        Bq = q.size(0)
        Bk = k.size(0)
        if Bk > Bq:
            k = k[:Bq]
            v = v[:Bq]
        elif Bk < Bq:
            q = q[:Bk]

        x = attention_register.call(q, k, v, k_lens=None, window_size=self.window_size)
        x = x.transpose(1, 2).contiguous()  # [B * F_common, H * W, n, d]
        x = x.view(b, -1, n, d).flatten(2)
        x = self.o(x)
        return x


class AudioCrossAttentionWrapper(nn.Module):
    def __init__(
        self,
        dim,
        kv_dim,
        num_heads,
        qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        self.audio_cross_attn = WanT2VCrossAttentionGather(
            kv_dim, dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm1_audio = WanLayerNorm(dim, eps, elementwise_affine=True)

    def forward(self, x, audio, seq_lens, grid_sizes, freqs, audio_seq_len):
        delta = self.audio_cross_attn(
            self.norm1_audio(x), audio, seq_lens, grid_sizes, freqs, audio_seq_len
        )
        if (not torch.is_grad_enabled()) and (not x.requires_grad) and (not delta.requires_grad):
            x.add_(delta)
            return x
        return x + delta


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

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

        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        self.norm_q(q)
        self.norm_k(k)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)
        x = attention_register.call(
            q=q.transpose(1, 2),
            k=k.transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=context_lens,
            window_size=self.window_size,
            key="flash",
        )
        x = x.transpose(1, 2)
        del q, k, v

        # output
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
        use_audio=True,
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

        self.use_audio = use_audio
        if use_audio:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(
                dim, 1536, num_heads, qk_norm, eps
            )

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
        x,  # torch.Size([1, 9360, 5120])
        e,  # torch.Size([1, 6, 5120])
        seq_lens,  # tensor([9360])
        grid_sizes,  # tensor([[ 6, 30, 52]])
        freqs,  # torch.Size([1024, 64])
        context,  # torch.Size([1, 512, 5120])
        context_lens,  # None
        audio=None,  # None
        audio_seq_len=None,
        ref_num_list=None,
        **kwargs,  # contains rope_apply_func
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, C]
            audio(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ref_num_list: 配合seq_lens可以查到reference image在倒数第几个
        """
        x = unwrap_single_item_list(x)
        e = unwrap_single_item_list(e)
        seq_lens = unwrap_single_item_list(seq_lens)
        context = unwrap_single_item_list(context)
        context_lens = unwrap_single_item_list(context_lens)

        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)

        # self-attention with chunked modulated norm
        norm_x = _chunked_modulated_norm(
            self.norm1, x, e[1], e[0], chunk_size=self._mod_norm_chunk_size
        )
        # Pass as single-item lists so `unwrap_single_item_list` clears references early.
        norm_x_list = [norm_x]
        seq_lens_list = [seq_lens]
        del norm_x
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

            if self.use_audio:
                x = self.audio_cross_attn_wrapper(
                    x, audio, seq_lens, grid_sizes, freqs, audio_seq_len
                )

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
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            norm_x = _chunked_modulated_norm(
                self.norm, x, e[1], e[0], chunk_size=self._mod_norm_chunk_size
            )
            x = self.head(norm_x)
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class HumoWanTransformerModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    gradient_checkpointing = False

    # Chunking profile presets
    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        "none": {
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
            "rope_chunk_size": None,
        },
        "light": {
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
            "rope_chunk_size": 8192,
        },
        "balanced": {
            "ffn_chunk_size": 512,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
            "rope_chunk_size": 4096,
        },
        "aggressive": {
            "ffn_chunk_size": 256,
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
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        audio_token_num=16,
        insert_audio=True,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
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

        assert model_type in ["t2v", "i2v"]
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
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
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
        self.insert_audio = insert_audio
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
                    use_audio=self.insert_audio,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if self.insert_audio:
            self.audio_proj = AudioProjModel(
                seq_len=8,
                blocks=5,
                channels=1280,
                intermediate_dim=512,
                output_dim=1536,
                context_tokens=audio_token_num,
            )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

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

        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

        # Also configure head chunking
        self.head.set_chunk_norms(
            modulated_norm_chunk_size=p.get("out_modulated_norm_chunk_size", None)
        )

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        audio=None,
        y=None,
        rope_on_cpu: Optional[bool] = None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            # assert clip_fea is not None and y is not None
            assert y is not None

        # params
        device = self.patch_embedding.weight.device

        # If True, keep RoPE frequency tables on CPU and only move the small slices
        # needed per chunk to GPU. This reduces peak VRAM usage at the cost of some
        # extra CPU↔GPU transfers.
        if rope_on_cpu is None:
            rope_on_cpu = False

        freqs = self.freqs
        if not rope_on_cpu:
            if not hasattr(self, "_freqs_device_cache"):
                self._freqs_device_cache = {}
            cache_key = str(device)
            if cache_key not in self._freqs_device_cache:
                self._freqs_device_cache[cache_key] = self.freqs.to(device)
            freqs = self._freqs_device_cache[cache_key]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len

        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            ).float()
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if self.insert_audio:
            audio = [
                self.audio_proj(au.unsqueeze(0)).permute(0, 3, 1, 2) for au in audio
            ]

            audio_seq_len = torch.tensor(
                max([au.shape[2] for au in audio]) * audio[0].shape[3], device=t.device
            )
            audio = [au.flatten(2).transpose(1, 2) for au in audio]  # [1, t*32, 1536]
            audio = torch.cat(
                [
                    torch.cat(
                        [au, au.new_zeros(1, audio_seq_len - au.size(1), au.size(2))],
                        dim=1,
                    )
                    for au in audio
                ]
            )
        else:
            audio = None
            audio_seq_len = None

        # arguments
        rope_chunk_size = getattr(self, "_rope_chunk_size", None)

        def apply_rope_humo(x_rope: torch.Tensor) -> torch.Tensor:
            """
            x_rope: [B, S, N, D]
            """
            if torch.is_grad_enabled() or x_rope.requires_grad:
                return rope_apply_video(
                    x_rope, grid_sizes=grid_sizes, freqs=freqs, chunk_size=rope_chunk_size
                )
            return rope_apply_video_inplace(
                x_rope, grid_sizes=grid_sizes, freqs=freqs, chunk_size=rope_chunk_size
            )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=context_lens,
            audio=audio,
            audio_seq_len=audio_seq_len,
            rope_apply_func=apply_rope_humo,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return Transformer2DModelOutput(sample=torch.stack(x).float())

    def unpatchify(self, x, grid_sizes):
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
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
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
