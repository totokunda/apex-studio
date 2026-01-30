"""
Code adapted from DiffSynth-Studio's Wan DiT implementation:
https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/wan_video_dit.py
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import PeftAdapterMixin

from einops import rearrange
from torch.distributed.tensor import DTensor
from src.attention import attention_register
from src.transformer.efficiency.ops import (
    apply_gate_inplace,
    apply_scale_shift_inplace,
    apply_wan_rope_inplace,
    chunked_feed_forward_inplace,
)
from src.transformer.efficiency.mod import InplaceRMSNorm

try:
    from yunchang import LongContextAttention
    from yunchang.kernels import AttnType
    LONG_CONTEXT_ATTN_AVAILABLE = True
except Exception:
    LONG_CONTEXT_ATTN_AVAILABLE = False
    LongContextAttention = None
    # Fallback so module import succeeds even when yunchang isn't available.
    # `USPAttention` will error if instantiated without LongContextAttention.
    class AttnType:  # type: ignore
        FA3 = None



@torch.compile(fullgraph=True)
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """
    Modulated norm with optional chunking along the sequence dimension to reduce peak memory.

    Expects `hidden_states` to be [B, S, D]. `scale`/`shift` can be broadcastable or per-token [B, S, D].
    """
    if hidden_states.ndim != 3:
        # Fallback for unexpected shapes.
        out = norm_layer(hidden_states)
        out = out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out
        apply_scale_shift_inplace(out, scale, shift)
        return out

    B, S, _ = hidden_states.shape
    in_dtype = hidden_states.dtype

    if chunk_size is None or S <= chunk_size:
        out = norm_layer(hidden_states)
        out = out.to(in_dtype) if out.dtype != in_dtype else out
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


def _chunked_norm(
    norm_layer: nn.Module, hidden_states: torch.Tensor, chunk_size: Optional[int] = 8192
) -> torch.Tensor:
    """
    Norm in chunks along the sequence dimension to reduce peak memory.

    Expects `hidden_states` to be [B, S, D].
    """
    if isinstance(norm_layer, nn.Identity):
        return hidden_states

    if hidden_states.ndim != 3:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    if chunk_size is None:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    _, S, _ = hidden_states.shape
    if S <= chunk_size:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    out = torch.empty_like(hidden_states)
    for i in range(0, S, chunk_size):
        end = min(i + chunk_size, S)
        out[:, i:end, :] = norm_layer(hidden_states[:, i:end, :])
    return out


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
) -> torch.Tensor:
    """
    Chunked feed-forward that does not require divisibility by `chunk_size`.

    Uses an inference-only in-place path when safe to reduce allocations.
    """
    return chunked_feed_forward_inplace(
        ff, hidden_states, chunk_dim=chunk_dim, chunk_size=chunk_size
    )


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def rope_apply_head_dim(x, freqs, head_dim):
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    # print(f"{x_out.shape = }, {freqs.shape = }")
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class SlowRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return (self.norm(x.float()) * self.weight).to(dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = attention_register.call(q=q, k=k, v=v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # In-place RMSNorm to reduce allocations (safe for Q/K projections).
        self.norm_q = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        *,
        rotary_emb_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # q/k/v: [B, S, D]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        self.norm_q(q)
        self.norm_k(k)

        # freqs can be cached on CPU; we move per-chunk inside apply_wan_rope_inplace.
        if isinstance(freqs, DTensor):
            freqs = freqs.to_local()

        # Normalize freqs shape to the canonical [T, D/2] complex layout.
        # Older callers used [T, 1, D/2]; caching paths may also hand us [1, T, 1, D/2].
        if freqs.ndim == 4 and freqs.shape[0] == 1 and freqs.shape[2] == 1:
            freqs = freqs.squeeze(0).squeeze(1)
        elif freqs.ndim == 3 and freqs.shape[1] == 1:
            freqs = freqs.squeeze(1)

        # reshape to [B, H, S, Dh] for in-place RoPE + attention kernel
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        apply_wan_rope_inplace(
            q, freqs, chunk_size=rotary_emb_chunk_size, freqs_may_be_cpu=True
        )
        apply_wan_rope_inplace(
            k, freqs, chunk_size=rotary_emb_chunk_size, freqs_may_be_cpu=True
        )

        x = attention_register.call(q=q, k=k, v=v)
        x = rearrange(x, "b h s d -> b s (h d)")
        return self.o(x)


class USPAttention(nn.Module):
    def __init__(self, num_heads: int, attn_type=AttnType.FA):
        super().__init__()
        if not LONG_CONTEXT_ATTN_AVAILABLE or LongContextAttention is None:
            raise RuntimeError(
                "USPAttention requires `yunchang` (LongContextAttention). "
                "Please install/enable yunchang or avoid using USPAttention."
            )
        self.num_heads = num_heads
        self.attn = LongContextAttention(ring_impl_type="basic", attn_type=attn_type)
        
    def forward(self, q, k, v):
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        x = self.attn(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)")


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # In-place RMSNorm to reduce allocations (safe for Q/K projections).
        self.norm_q = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.q(x)
        k = self.k(ctx)
        v = self.v(ctx)
        self.norm_q(q)
        self.norm_k(k)
        
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.k_img(img)
            self.norm_k_img(k_img)
            v_img = self.v_img(img)
            # einops to correct shape
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k_img = rearrange(k_img, "b s (n d) -> b n s d", n=self.num_heads)
            v_img = rearrange(v_img, "b s (n d) -> b n s d", n=self.num_heads)
            y = attention_register.call(q=q, k=k_img, v=v_img)
            y = rearrange(y, "b n s d -> b s (n d)", n=self.num_heads)
            x = x + y
        return self.o(x)

class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


def _deterministic_chunked_feed_forward(
    ff: nn.Module,
    hidden_states: torch.Tensor,
    chunk_dim: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Chunked feed-forward with deterministic CUDA operations enabled.

    This helps reduce (but may not fully eliminate) numerical differences from chunking
    by forcing consistent algorithm selection across different input sizes.

    Args:
        ff: The feed-forward module
        hidden_states: Input tensor [B, S, D]
        chunk_dim: Dimension to chunk along (typically 1 for sequence)
        chunk_size: Size of each chunk
    """
    dim_len = hidden_states.shape[chunk_dim]
    if dim_len <= chunk_size:
        return ff(hidden_states)

    # Save current deterministic settings
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        # Enable deterministic mode to reduce algorithm variation
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Process in chunks - use in-place path when safe
        if (
            not torch.is_grad_enabled()
            and not hidden_states.requires_grad
            and hidden_states.ndim == 3
            and chunk_dim == 1
        ):
            for i in range(0, dim_len, chunk_size):
                end = min(i + chunk_size, dim_len)
                chunk = hidden_states[:, i:end, :]
                hidden_states[:, i:end, :] = ff(chunk)
            return hidden_states
        else:
            outputs = []
            for i in range(0, dim_len, chunk_size):
                end = min(i + chunk_size, dim_len)
                chunk = hidden_states.narrow(chunk_dim, i, end - i)
                outputs.append(ff(chunk))
            return torch.cat(outputs, dim=chunk_dim)
    finally:
        # Restore previous settings
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

        # Chunking knobs (disabled by default, enabled via model chunking profiles)
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1
        self._mod_norm_chunk_size: Optional[int] = None
        self._norm_chunk_size: Optional[int] = None
        self._use_deterministic_ffn_chunking: bool = False

    def set_chunk_feed_forward(
        self, chunk_size: Optional[int], dim: int = 1, deterministic: bool = False
    ) -> None:
        """
        Enable/disable chunked feed-forward.

        Args:
            chunk_size: Tokens per chunk. None to disable chunking.
            dim: Dimension to chunk along (typically 1 for sequence).
            deterministic: If True, use deterministic CUDA settings during chunking.
                This can reduce (but may not fully eliminate) numerical differences.
        """
        self._ff_chunk_size = chunk_size
        self._ff_chunk_dim = dim
        self._use_deterministic_ffn_chunking = deterministic

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
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        *,
        rotary_emb_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        # Compute scale/shift/gates in fp32 for stability, then cast to x dtype to avoid fp32 intermediates.
        hs_dtype = x.dtype
        if has_seq:
            # t_mod: [B, S, 6, D]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                (self.modulation.unsqueeze(0) + t_mod.float()).to(hs_dtype).chunk(6, dim=2)
            )
        else:
            # t_mod: [B, 6, D]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                (self.modulation + t_mod.float()).to(hs_dtype).chunk(6, dim=1)
            )
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        # 1) Self-attention (modulated norm + gated residual)
        norm_x = _chunked_modulated_norm(
            self.norm1,
            x,
            scale_msa,
            shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )
        attn_out = self.self_attn(
            norm_x, freqs, rotary_emb_chunk_size=rotary_emb_chunk_size
        )
        del norm_x

        inference_mode = not torch.is_grad_enabled()
        if (
            inference_mode
            and (not x.requires_grad)
            and (not attn_out.requires_grad)
        ):
            apply_gate_inplace(attn_out, gate_msa.to(dtype=attn_out.dtype))
            x.add_(attn_out)
        else:
            x = x + attn_out * gate_msa
        del attn_out, shift_msa, scale_msa, gate_msa

        # 2) Cross-attention (+ residual)
        norm_x = _chunked_norm(self.norm3, x, chunk_size=self._norm_chunk_size)
        cross_out = self.cross_attn(norm_x, context)
        del norm_x

        if (
            inference_mode
            and (not x.requires_grad)
            and (not cross_out.requires_grad)
        ):
            x.add_(cross_out)
        else:
            x = x + cross_out
        del cross_out

        # 3) Feed-forward (modulated norm + gated residual, optionally chunked)
        norm_x = _chunked_modulated_norm(
            self.norm2,
            x,
            scale_mlp,
            shift_mlp,
            chunk_size=self._mod_norm_chunk_size,
        )
        if self._ff_chunk_size is not None:
            if self._use_deterministic_ffn_chunking:
                ff_out = _deterministic_chunked_feed_forward(
                    self.ffn, norm_x, self._ff_chunk_dim, self._ff_chunk_size
                )
            else:
                ff_out = _chunked_feed_forward(
                    self.ffn, norm_x, self._ff_chunk_dim, self._ff_chunk_size
                )
        else:
            ff_out = self.ffn(norm_x)
        del norm_x

        if (
            inference_mode
            and (not x.requires_grad)
            and (not ff_out.requires_grad)
        ):
            apply_gate_inplace(ff_out, gate_mlp.to(dtype=ff_out.dtype))
            x.add_(ff_out)
        else:
            x = x + ff_out * gate_mlp
        del ff_out, shift_mlp, scale_mlp, gate_mlp
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        *,
        modulated_norm_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        # Align dtype with x for modulation to avoid fp32 intermediates.
        shift = shift.to(dtype=x.dtype, device=x.device)
        scale = scale.to(dtype=x.dtype, device=x.device)

        x = _chunked_modulated_norm(
            self.norm,
            x,
            scale,
            shift,
            chunk_size=modulated_norm_chunk_size,
        )
        del scale, shift
        return self.head(x)


class MOVAWanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _repeated_blocks = ("DiTBlock",)
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        # Default: no chunking unless explicitly enabled.
        self.set_chunking_profile("none")

    # ----------------------------
    # Chunking profile presets
    # ----------------------------

    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        "none": {
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
            "rotary_emb_chunk_size": None,
        },
        "light": {
            # NOTE: FFN chunking changes matmul shapes and can change numerics enough
            # to diverge diffusion outputs. Keep it off by default; use
            # `set_chunk_feed_forward(...)` explicitly if you accept small differences.
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
            "rotary_emb_chunk_size": None,
        },
        "balanced": {
            # See note above.
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
            "rotary_emb_chunk_size": 1024,
        },
        "aggressive": {
            # See note above.
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
            "rotary_emb_chunk_size": 256,
        },
    }

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunk_feed_forward(
        self, chunk_size: Optional[int], dim: int = 1, deterministic: bool = False
    ) -> None:
        """
        Enable/disable chunked feed-forward on all transformer blocks.

        Args:
            chunk_size: Tokens per chunk. None to disable chunking.
            dim: Dimension to chunk along (typically 1 for sequence).
            deterministic: If True, use deterministic CUDA settings during chunking.
                This can reduce (but may not fully eliminate) numerical differences
                compared to non-chunked execution.
        """
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim, deterministic=deterministic)

    def set_chunking_profile(self, profile_name: str) -> None:
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. "
                f"Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )
        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name

        self._rotary_emb_chunk_size_default = p.get("rotary_emb_chunk_size", None)
        self._out_modulated_norm_chunk_size = p.get(
            "out_modulated_norm_chunk_size", None
        )

        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

    # ----------------------------
    # Memory-efficient inference setup
    # ----------------------------

    def enable_memory_efficient_inference(
        self,
        chunking_profile: str = "balanced",
        rope_on_cpu: bool = True,
        ffn_chunk_size: Optional[int] = None,
        ffn_deterministic: bool = True,
    ) -> None:
        """
        Enable memory optimizations for inference.

        Args:
            chunking_profile: One of "none", "light", "balanced", "aggressive".
                Controls norm chunking and RoPE chunking. Default "balanced".
            rope_on_cpu: If True, cache RoPE frequencies on CPU and transfer
                per-chunk during attention. Saves VRAM. Default True.
            ffn_chunk_size: If set, chunk the FFN to reduce peak memory from the
                4× intermediate tensor. WARNING: This may cause small numerical
                differences that can compound over diffusion steps. Use
                ffn_deterministic=True to minimize (but not eliminate) this.
                Set to None (default) for identical results.
            ffn_deterministic: If True and ffn_chunk_size is set, use deterministic
                CUDA settings during FFN chunking to reduce numerical variation.

        Note:
            The FFN intermediate tensor (4× model dim) is a major memory consumer.
            Without chunking, this cannot be reduced. If you need to save FFN memory
            and can accept small output differences, set ffn_chunk_size (e.g., 2048).

        Example:
            # Maximum memory savings with identical results (no FFN chunking)
            model.enable_memory_efficient_inference()

            # Maximum memory savings, accepting potential small output differences
            model.enable_memory_efficient_inference(ffn_chunk_size=2048)
        """
        self.set_chunking_profile(chunking_profile)

        if ffn_chunk_size is not None:
            self.set_chunk_feed_forward(
                ffn_chunk_size, dim=1, deterministic=ffn_deterministic
            )

        # Store rope_on_cpu preference as a default for forward()
        if not hasattr(self, "_apex_forward_kwargs_defaults"):
            self._apex_forward_kwargs_defaults = {}
        self._apex_forward_kwargs_defaults["rope_on_cpu"] = rope_on_cpu

    # ----------------------------
    # RoPE caching (CPU) to save VRAM
    # ----------------------------

    def _get_rope_cpu_cache(self) -> Dict[tuple, torch.Tensor]:
        if not hasattr(self, "_rope_cpu_cache"):
            self._rope_cpu_cache = {}
        return self._rope_cpu_cache

    def _rope_cache_key(self, *, f: int, h: int, w: int) -> tuple:
        return (int(f), int(h), int(w))

    def _build_rope_3d(self, *, f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        # Build WAN-style complex freqs for [S, D/2], where S = f*h*w.
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, -1)
        return freqs.to(device)

    def _build_rope_cached(
        self,
        *,
        f: int,
        h: int,
        w: int,
        device: torch.device,
        rope_on_cpu: bool = False,
    ) -> torch.Tensor:
        if not rope_on_cpu:
            return self._build_rope_3d(f=f, h=h, w=w, device=device)

        cache = self._get_rope_cpu_cache()
        key = self._rope_cache_key(f=f, h=h, w=w)
        if key not in cache:
            cache[key] = self._build_rope_3d(
                f=f, h=h, w=w, device=torch.device("cpu")
            )
        return cache[key]

    def patchify(self, x: torch.Tensor,control_camera_latents_input: torch.Tensor = None):
        # NOTE(dhyu): avoid slow_conv
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                rope_on_cpu: Optional[bool] = None,
                return_prepared: bool = False,
                **kwargs,
                ):
        if rope_on_cpu is None:
            rope_on_cpu = (
                getattr(self, "_apex_forward_kwargs_defaults", {}) or {}
            ).get("rope_on_cpu", False)

        rotary_emb_chunk_size = kwargs.pop("rotary_emb_chunk_size", None)
        if rotary_emb_chunk_size is None:
            rotary_emb_chunk_size = getattr(self, "_rotary_emb_chunk_size_default", None)

        # Keep timestep embedding in fp32 for stability, then cast back to model dtype.
        model_dtype = self.dtype
        with torch.autocast(x.device.type, dtype=torch.float32):
            t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep)
            )
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        t = t.to(model_dtype)
        t_mod = t_mod.to(model_dtype)

        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
            del clip_embdding
        
        if x.dtype != model_dtype:
            x = x.to(model_dtype)

        x, (f, h, w) = self.patchify(x)

        freqs = self._build_rope_cached(
            f=f, h=h, w=w, device=x.device, rope_on_cpu=bool(rope_on_cpu)
        )

        if return_prepared:
            return {
                "x": x,
                "context": context,
                "t": t,
                "t_mod": t_mod,
                "freqs": freqs,
                "grid_size": (f, h, w),
                "rotary_emb_chunk_size": rotary_emb_chunk_size,
            }
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, rotary_emb_chunk_size=rotary_emb_chunk_size)
            return custom_forward
        

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    rotary_emb_chunk_size=rotary_emb_chunk_size,
                )

        x = self.head(
            x,
            t,
            modulated_norm_chunk_size=getattr(
                self, "_out_modulated_norm_chunk_size", None
            ),
        )
        del t_mod, t, context, freqs
        x = self.unpatchify(x, (f, h, w))
        return x
