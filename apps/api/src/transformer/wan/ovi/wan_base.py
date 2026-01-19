# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Dict, Optional, Tuple
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.checkpoint import checkpoint
from src.attention import attention_register
from .attention import flash_attention
from .easy_cache import easycache_forward
import types


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
        out = norm_layer(hidden_states) * (1 + scale) + shift
        return out.to(in_dtype) if out.dtype != in_dtype else out

    if S <= chunk_size:
        out = norm_layer(hidden_states) * (1 + scale) + shift
        return out.to(in_dtype) if out.dtype != in_dtype else out

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

        normed = norm_layer(hs_chunk)
        out[:, i:end, :] = normed * (1 + scale_chunk) + shift_chunk
        del normed

    return out


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
) -> torch.Tensor:
    """
    Chunked feed-forward to reduce peak memory.
    """
    if chunk_size is None:
        return ff(hidden_states)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    dim_len = hidden_states.shape[chunk_dim]
    if dim_len <= chunk_size:
        return ff(hidden_states)

    outputs = []
    for start in range(0, dim_len, chunk_size):
        end = min(start + chunk_size, dim_len)
        hs_chunk = hidden_states.narrow(chunk_dim, start, end - start)
        outputs.append(ff(hs_chunk))
    return torch.cat(outputs, dim=chunk_dim)


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


def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


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


def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos = torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2  ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"

    # loop over samples
    output = []
    for i, (l,) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )  # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()


@amp.autocast("cuda", enabled=False)
def rope_apply_3d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()


def rope_apply(x, grid_sizes, freqs):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)


class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(
            dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding
        )
        self.w2 = ChannelLastConv1d(
            hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding
        )
        self.w3 = ChannelLastConv1d(
            dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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
        return self._norm(x.bfloat16()).type_as(x) * self.weight.bfloat16()

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
        return super().forward(x.bfloat16()).type_as(x)


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
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # optional sequence parallelism
        # self.world_size = get_world_size()

    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x)

        x = attention_register.call(
            q=rope_apply(q, grid_sizes, freqs).transpose(1, 2),
            k=rope_apply(k, grid_sizes, freqs).transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=seq_lens,
            window_size=self.window_size,
        ).transpose(1, 2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context)

        # compute attention
        x = attention_register.call(
            q=q.transpose(1, 2),
            k=k.transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=context_lens,
        ).transpose(1, 2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        additional_emb_length=None,
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.additional_emb_length = additional_emb_length

    def qkv_fn(self, x, context):
        context_img = context[:, : self.additional_emb_length]
        context = context[:, self.additional_emb_length :]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        return q, k, v, k_img, v_img

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v, k_img, v_img = self.qkv_fn(x, context)

        # [B, L, H/P, C/H]
        # k_img: [B, L, H, C/H]
        img_x = attention_register.call(
            q=q.transpose(1, 2),
            k=k_img.transpose(1, 2),
            v=v_img.transpose(1, 2),
            k_lens=None,
        ).transpose(1, 2)
        # compute attention
        x = attention_register.call(
            q=q.transpose(1, 2),
            k=k.transpose(1, 2),
            v=v.transpose(1, 2),
            k_lens=context_lens,
        ).transpose(1, 2)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class ModulationAdd(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(1, num, dim) / dim**0.5)

    def forward(self, e):
        return self.modulation.bfloat16() + e.bfloat16()


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
        additional_emb_length=None,
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
        if cross_attn_type == "i2v_cross_attn":
            assert (
                additional_emb_length is not None
            ), "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WanI2VCrossAttention(
                dim, num_heads, (-1, -1), qk_norm, eps, additional_emb_length
            )
        else:
            assert (
                additional_emb_length is None
            ), "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WanT2VCrossAttention(
                dim,
                num_heads,
                (-1, -1),
                qk_norm,
                eps,
            )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation = ModulationAdd(dim, 6)

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
        grid_sizes,
        freqs,
        context,
        context_lens,
        mode: str = "full",
        # Fusion mode kwargs (only used when mode != "full")
        target_seq=None,
        target_seq_lens=None,
        target_grid_sizes=None,
        target_freqs=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C] for full/modulation_self_attn modes,
                       or tuple of chunked embeddings for fusion_cross_attn_ffn mode
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Context for cross-attention
            context_lens(Tensor): Context lengths
            mode(str): One of:
                - "full": Normal forward (modulation + self-attn + cross-attn + ffn)
                - "modulation_self_attn": Only modulation + self-attention, returns (x, e_chunked)
                - "fusion_cross_attn_ffn": Fusion cross-attention + FFN, expects e as pre-chunked tuple
            target_seq: Target sequence for fusion cross-attention (fusion modes only)
            target_seq_lens: Target sequence lengths (fusion modes only)
            target_grid_sizes: Target grid sizes (fusion modes only)
            target_freqs: Target freqs (fusion modes only)
        """

        if mode == "modulation_self_attn":
            # Fusion mode: modulation + self-attention only
            assert (
                len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1]
            ), f"{e.shape}, {x.shape}"
            with amp.autocast("cuda", dtype=torch.bfloat16):
                e_chunked = self.modulation(e).chunk(6, dim=2)
            assert e_chunked[0].dtype == torch.bfloat16

            norm_x = _chunked_modulated_norm(
                self.norm1,
                x,
                e_chunked[1].squeeze(2),
                e_chunked[0].squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )
            y = self.self_attn(norm_x.bfloat16(), seq_lens, grid_sizes, freqs)
            with amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + y * e_chunked[2].squeeze(2)

            return x, e_chunked

        elif mode == "fusion_cross_attn_ffn":
            # Fusion mode: cross-attention with target modality + FFN
            # e is expected to be pre-chunked tuple from modulation_self_attn
            e_chunked = e
            b, n, d = x.size(0), self.cross_attn.num_heads, self.cross_attn.head_dim

            # Standard cross-attention with context
            if hasattr(self.cross_attn, "k_img"):
                q, k, v, k_img, v_img = self.cross_attn.qkv_fn(
                    _chunked_norm(self.norm3, x, chunk_size=self._norm_chunk_size),
                    context,
                )
            else:
                q, k, v = self.cross_attn.qkv_fn(
                    _chunked_norm(self.norm3, x, chunk_size=self._norm_chunk_size),
                    context,
                )
                k_img = v_img = None

            attn_out = attention_register.call(
                q=q.transpose(1, 2),
                k=k.transpose(1, 2),
                v=v.transpose(1, 2),
                k_lens=context_lens,
            ).transpose(1, 2)

            if k_img is not None:
                img_attn = attention_register.call(
                    q=q.transpose(1, 2),
                    k=k_img.transpose(1, 2),
                    v=v_img.transpose(1, 2),
                    k_lens=None,
                ).transpose(1, 2)
                attn_out = attn_out + img_attn

            # Fusion cross-attention with target modality
            target_seq_normed = self.cross_attn.pre_attn_norm_fusion(target_seq)
            k_target = self.cross_attn.norm_k_fusion(
                self.cross_attn.k_fusion(target_seq_normed)
            ).view(b, -1, n, d)
            v_target = self.cross_attn.v_fusion(target_seq_normed).view(b, -1, n, d)

            q_rope = rope_apply(q, grid_sizes, freqs)
            k_target_rope = rope_apply(k_target, target_grid_sizes, target_freqs)

            target_attn = attention_register.call(
                q=q_rope.transpose(1, 2),
                k=k_target_rope.transpose(1, 2),
                v=v_target.transpose(1, 2),
                k_lens=target_seq_lens,
            ).transpose(1, 2)

            combined_attn = (attn_out + target_attn).flatten(2)
            x = x + self.cross_attn.o(combined_attn)

            # FFN
            ffn_x = _chunked_modulated_norm(
                self.norm2,
                x,
                e_chunked[4].squeeze(2),
                e_chunked[3].squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )
            if self._ff_chunk_size is not None:
                y = _chunked_feed_forward(
                    self.ffn, ffn_x.bfloat16(), self._ff_chunk_dim, self._ff_chunk_size
                )
            else:
                y = self.ffn(ffn_x.bfloat16())
            with amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + y * e_chunked[5].squeeze(2)

            return x

        else:
            # Default "full" mode: normal forward
            assert (
                len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1]
            ), f"{e.shape}, {x.shape}"
            with amp.autocast("cuda", dtype=torch.bfloat16):
                e = self.modulation(e).chunk(6, dim=2)
            assert e[0].dtype == torch.bfloat16

            # self-attention with chunked modulated norm
            norm_x = _chunked_modulated_norm(
                self.norm1,
                x,
                e[1].squeeze(2),
                e[0].squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )
            y = self.self_attn(norm_x.bfloat16(), seq_lens, grid_sizes, freqs)
            with amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + y * e[2].squeeze(2)

            # cross-attention & ffn
            x = x + self.cross_attn(
                _chunked_norm(self.norm3, x, chunk_size=self._norm_chunk_size),
                context,
                context_lens,
            )
            ffn_x = _chunked_modulated_norm(
                self.norm2,
                x,
                e[4].squeeze(2),
                e[3].squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )
            if self._ff_chunk_size is not None:
                y = _chunked_feed_forward(
                    self.ffn, ffn_x.bfloat16(), self._ff_chunk_dim, self._ff_chunk_size
                )
            else:
                y = self.ffn(ffn_x.bfloat16())
            with amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)

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
            e(Tensor): Shape [B, L, C]
        """
        with amp.autocast(x.device.type, dtype=torch.bfloat16):
            e = (self.modulation.bfloat16().unsqueeze(0) + e.unsqueeze(2)).chunk(
                2, dim=2
            )  # 1 1 2 D, B L 1 D -> B L 2 D -> 2 * (B L 1 D)
            norm_x = _chunked_modulated_norm(
                self.norm,
                x,
                e[1].squeeze(2),
                e[0].squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
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


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video, text-to-audio.
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
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
        },
        "light": {
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
        },
        "balanced": {
            "ffn_chunk_size": 512,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
        },
        "aggressive": {
            "ffn_chunk_size": 256,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
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
        additional_emb_dim=None,
        additional_emb_length=None,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        gradient_checkpointing=False,
        temporal_rope_scaling_factor=1.0,
        eps=1e-6,
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

        assert model_type in [
            "t2v",
            "i2v",
            "t2a",
            "tt2a",
            "ti2v",
        ]  ## tt2a means text transcript + text description to audio (to support both TTS and T2A
        self.model_type = model_type
        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert (
            is_audio_type ^ is_video_type
        ), "Either audio or video model should be specified"
        if is_audio_type:
            ## audio model
            assert (
                len(patch_size) == 1 and patch_size[0] == 1
            ), "Audio model should only accept 1 dimensional input, and we dont do patchify"

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
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        self.is_audio_type = is_audio_type
        self.is_video_type = is_video_type
        # embeddings
        if is_audio_type:
            ## hardcoded to MMAudio
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        else:
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

        ## so i2v and tt2a share the same cross attention while t2v and t2a share the same cross attention
        cross_attn_type = (
            "t2v_cross_attn"
            if model_type in ["t2v", "t2a", "ti2v"]
            else "i2v_cross_attn"
        )

        if cross_attn_type == "t2v_cross_attn":
            assert (
                additional_emb_dim is None and additional_emb_length is None
            ), "additional_emb_length should be None for t2v and t2a model"
        else:
            assert (
                additional_emb_dim is not None and additional_emb_length is not None
            ), "additional_emb_length should be specified for i2v and tt2a model"

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
                    additional_emb_length,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        self.set_gradient_checkpointing(enable=gradient_checkpointing)
        self.set_rope_params()

        if model_type in ["i2v", "tt2a"]:
            self.img_emb = MLPProj(additional_emb_dim, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

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

    def enable_easy_cache(
        self,
        num_steps: int,
        thresh: float,
        ret_steps: int = 10,
        cutoff_steps: int | None = None,
    ):

        self.forward = types.MethodType(easycache_forward, self)
        self.cnt = 0
        self.skip_cond_step = []
        self.skip_uncond_step = []
        self.num_steps = num_steps
        self.thresh = thresh
        self.accumulated_error_even = 0
        self.should_calc_current_pair = True
        self.k = None

        self.previous_raw_input_even = None
        self.previous_raw_output_even = None
        self.previous_raw_output_odd = None
        self.prev_prev_raw_input_even = None
        self.cache_even = None
        self.cache_odd = None

        self.cost_time = 0
        self.ret_steps = ret_steps * 2
        self.cutoff_steps = (
            cutoff_steps if cutoff_steps is not None else num_steps * 2 - 2
        )

    def set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        if self.is_audio_type:
            ## to be determined
            # self.freqs = rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
            self.freqs = rope_params(
                1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor
            )
        else:
            self.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        first_frame_is_clean=False,
    ):

        # params
        ## need to change!
        device = next(self.patch_embedding.parameters()).device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        self.patch_embedding.to(x[0].device)
        # embeddings
        x = [
            self.patch_embedding(u.unsqueeze(0)) for u in x
        ]  ## x is list of [B L D] or [B C F H W]
        if self.is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x]
            )
        else:
            # [B, 3]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
            )
            x = [
                u.flatten(2).transpose(1, 2) for u in x
            ]  # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert (
            seq_lens.max() <= seq_len
        ), f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )  # single [B, L, C]

        # time embeddings
        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones(
                    (t.size(0), seq_len), device=t.device, dtype=t.dtype
                ) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)):
                    t[i, : _first_images_seq_len[i]] = 0
                # print(f"zeroing out first {_first_images_seq_len} from t: {t.shape}, {t}")
            else:
                t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with amp.autocast("cuda", dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t)
                .unflatten(0, (bt, seq_len))
                .bfloat16()
            )
            e0 = self.time_projection(e).unflatten(
                2, (6, self.dim)
            )  # [1, 26784, 6, 3072] - B, seq_len, 6, dim
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

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

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        return x, e, kwargs

    def post_transformer_block_out(self, x, grid_sizes, e):
        # head
        x = self.head(x, e)
        if self.is_audio_type:
            ## grid_sizes is [B 1] where 1 is L,
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self.unpatchify(x, grid_sizes)

        return [u.bfloat16() for u in x]

    def forward(
        self, x, t, context, seq_len, clip_fea=None, y=None, first_frame_is_clean=False
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
                OR
                List of input audio tensors, each with shape [L, C_in]
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
                OR
                List of denoised audio tensors with original input shapes [L, C_in]
        """
        x, e, kwargs = self.prepare_transformer_block_kwargs(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
        )

        for block in self.blocks:
            x = gradient_checkpointing(
                enabled=(self.training and self.gradient_checkpointing),
                module=block,
                x=x,
                **kwargs,
            )

        return self.post_transformer_block_out(x, kwargs["grid_sizes"], e)

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
            # v is [F H w] F * H * 80, 100, it was right padded by 20.
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
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
        if self.is_video_type:
            assert isinstance(
                self.patch_embedding, nn.Conv3d
            ), f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
