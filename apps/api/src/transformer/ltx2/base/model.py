# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.models._modeling_parallel import (
    ContextParallelInput,
    ContextParallelOutput,
)
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
    PixArtAlphaTextProjection,
)
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import RMSNorm
from src.attention import attention_register
from src.transformer.efficiency.mod import InplaceRMSNorm
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from src.utils.step_mem import step_mem
import gc
def _reshape_hidden_states_for_frames(
    hidden_states: torch.Tensor, frames: int
) -> torch.Tensor:
    """
    Reshape `[B, tokens, D]` -> `[B, frames, tokens_per_frame, D]` for per-frame modulation/gating.
    """
    b, tokens, d = hidden_states.shape
    if frames <= 0 or tokens % frames != 0:
        raise ValueError(
            f"Cannot reshape tokens={tokens} into frames={frames} evenly."
        )
    return hidden_states.reshape(b, frames, -1, d)


def _apply_gate_inplace(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    In-place gating helper.

    Expects `x` and `gate` to be broadcastable to the same shape, typically `[B, S, D]`.
    """
    # Wan2GP-style: allow per-frame gates shaped `[B, F, D]` applied to token sequences `[B, (F*T), D]`.
    if x.ndim == 3 and gate.ndim == 3 and gate.shape[1] != x.shape[1]:
        frames = int(gate.shape[1])
        try:
            x_view = _reshape_hidden_states_for_frames(x, frames)
            x_view.mul_(gate.unsqueeze(2))
            return x
        except Exception:
            # Fall back to broadcasting / default behavior.
            pass
    x.mul_(gate)
    return x


def _apply_scale_shift_inplace(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    In-place `x = x * (1 + scale) + shift` without allocating `(1 + scale)`.
    """
    # Wan2GP-style: allow per-frame scale/shift shaped `[B, F, D]` applied to token sequences `[B, (F*T), D]`.
    if x.ndim == 3 and scale.ndim == 3 and scale.shape[1] != x.shape[1]:
        frames = int(scale.shape[1])
        try:
            x_view = _reshape_hidden_states_for_frames(x, frames)
            scale_u = scale.unsqueeze(2)
            shift_u = shift.unsqueeze(2)
            x_view.addcmul_(x_view, scale_u)
            x_view.add_(shift_u)
            return x
        except Exception:
            pass

    # Default: x = x + x*scale; x += shift
    x.addcmul_(x, scale)
    x.add_(shift)
    return x


def _apply_scale_shift_table(
    temb: torch.Tensor,
    table: torch.Tensor,
    *,
    batch_size: int,
    num_params: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Apply a per-layer scale/shift table to a timestep embedding tensor, returning the per-parameter tensors.

    This mirrors Wan2GP's `get_ada_values` behavior but avoids creating an intermediate `[B, S, P, D]` sum tensor.

    Args:
        temb: `[B, S, P*D]` (or compatible) tensor produced by adaLN single.
        table: `[P, D]` parameter tensor.
        batch_size: Batch size (for `.view` safety).
        num_params: Optional override for `P`.

    Returns:
        Tuple of `P` tensors, each shaped `[B, S, D]`.
    """
    p = int(num_params if num_params is not None else table.shape[0])
    # `view` to avoid allocations; expects the last dim to be `p * D`.
    temb_view = temb.view(batch_size, temb.size(1), p, -1)
    out: List[torch.Tensor] = []
    for i in range(p):
        # Cast only the small table slice to temb dtype to avoid upcasting big activations.
        out.append(temb_view[:, :, i, :] + table[i].to(device=temb.device, dtype=temb.dtype))
    return tuple(out)




def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """
    Chunked `norm_layer(hidden_states) * (1 + scale) + shift` along the sequence dimension (dim=1).

    This mirrors the Wan chunking utilities, but is generic over the norm type (RMSNorm / LayerNorm).
    Expects `hidden_states` to be `[batch, seq_len, dim]`.
    """
    if chunk_size is None:
        # Non-chunked path: still needs to support per-frame modulation tensors shaped `[B, F, D]`
        # applied to token sequences `[B, (F*T), D]`.
        out = norm_layer(hidden_states)
        if out.ndim == 3:
            _apply_scale_shift_inplace(out, scale, shift)
            return out
        return out * (1 + scale) + shift
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")

    if hidden_states.ndim != 3:
        # Fallback for unexpected shapes.
        return norm_layer(hidden_states) * (1 + scale) + shift

    b, s, _ = hidden_states.shape
    # If scale/shift are per-frame (shape[1] != s), chunking by token slices is not shape-safe.
    # Fall back to full-tensor computation in that case.
    if scale.ndim == 3 and scale.shape[1] != s:
        out = norm_layer(hidden_states)
        _apply_scale_shift_inplace(out, scale, shift)
        return out

    if s <= chunk_size:
        return norm_layer(hidden_states) * (1 + scale) + shift

    out = torch.empty_like(hidden_states)
    scale_per_token = scale.ndim == 3 and scale.shape[1] == s
    for i in range(0, s, chunk_size):
        end = min(i + chunk_size, s)
        hs_chunk = hidden_states[:, i:end, :]
        if scale_per_token:
            scale_chunk = scale[:, i:end, :]
            shift_chunk = shift[:, i:end, :]
        else:
            scale_chunk = scale
            shift_chunk = shift
        # 1) write norm output, 2) apply scale/shift in-place (avoid allocating (1 + scale_chunk))
        out[:, i:end, :].copy_(norm_layer(hs_chunk))
        _apply_scale_shift_inplace(out[:, i:end, :], scale_chunk, shift_chunk)
    return out


def _chunked_norm(
    norm_layer: nn.Module, hidden_states: torch.Tensor, chunk_size: Optional[int] = 8192
) -> torch.Tensor:
    """
    Chunked `norm_layer(hidden_states)` along the sequence dimension (dim=1).
    Expects `hidden_states` to be `[batch, seq_len, dim]`.
    """
    if chunk_size is None:
        return norm_layer(hidden_states)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")

    if hidden_states.ndim != 3:
        return norm_layer(hidden_states)

    _, s, _ = hidden_states.shape
    if s <= chunk_size:
        return norm_layer(hidden_states)

    out = torch.empty_like(hidden_states)
    for i in range(0, s, chunk_size):
        end = min(i + chunk_size, s)
        out[:, i:end, :] = norm_layer(hidden_states[:, i:end, :])
    return out


def _chunked_feed_forward(
    ff: nn.Module,
    hidden_states: torch.Tensor,
    chunk_dim: int,
    chunk_size: Optional[int],
):
    """
    Chunked feed-forward that does NOT require divisibility by chunk_size.
    """
    if chunk_size is None:
        return ff(hidden_states)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")

    # Wan2GP-style: in inference, apply FFN chunked *in-place* to avoid concat allocations.
    # Only do this for the common `[B, S, D]` case chunked along S.
    if (
        not torch.is_grad_enabled()
        and hidden_states.ndim == 3
        and chunk_dim == 1
        and hidden_states.shape[1] > chunk_size
    ):
        x = hidden_states
        x_flat = x.view(-1, x.shape[-1])
        # Split along the flattened token axis; assignment writes back into `x`.
        for chunk in torch.split(x_flat, int(chunk_size)):
            chunk[...] = ff(chunk)
        return x

    dim_len = hidden_states.shape[chunk_dim]
    if dim_len <= chunk_size:
        return ff(hidden_states)

    outputs = []
    for start in range(0, dim_len, chunk_size):
        end = min(start + chunk_size, dim_len)
        hs_chunk = hidden_states.narrow(chunk_dim, start, end - start)
        outputs.append(ff(hs_chunk))
    return torch.cat(outputs, dim=chunk_dim)


def _chunked_apply_scale_shift(
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """
    Chunked `hidden_states * (1 + scale) + shift` along sequence dim (dim=1).
    """
    if chunk_size is None:
        # Non-chunked path: still needs to support per-frame scale/shift tensors shaped `[B, F, D]`
        # applied to token sequences `[B, (F*T), D]`.
        if hidden_states.ndim == 3:
            out = hidden_states.clone()
            _apply_scale_shift_inplace(out, scale, shift)
            return out
        return hidden_states * (1 + scale) + shift
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")

    if hidden_states.ndim != 3:
        return hidden_states * (1 + scale) + shift

    _, s, _ = hidden_states.shape
    # If scale/shift are per-frame (shape[1] != s), chunking by token slices is not shape-safe.
    # Fall back to full-tensor computation in that case.
    if scale.ndim == 3 and scale.shape[1] != s:
        out = hidden_states.clone()
        _apply_scale_shift_inplace(out, scale, shift)
        return out

    if s <= chunk_size:
        return hidden_states * (1 + scale) + shift

    out = torch.empty_like(hidden_states)
    scale_per_token = scale.ndim == 3 and scale.shape[1] == s
    for i in range(0, s, chunk_size):
        end = min(i + chunk_size, s)
        hs_chunk = hidden_states[:, i:end, :]
        if scale_per_token:
            scale_chunk = scale[:, i:end, :]
            shift_chunk = shift[:, i:end, :]
        else:
            scale_chunk = scale
            shift_chunk = shift
        out[:, i:end, :].copy_(hs_chunk)
        _apply_scale_shift_inplace(out[:, i:end, :], scale_chunk, shift_chunk)
    return out


def apply_interleaved_rotary_emb_inplace(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    In-place interleaved RoPE.

    Preserves the previous "compute in fp32 then cast back" behavior when freqs are fp32 and x is lower precision.
    """
    cos, sin = freqs
    # Keep prior numerical behavior: if freqs are fp32 but x isn't, compute in fp32 and copy back.
    if cos.dtype == torch.float32 and x.dtype != torch.float32:
        x_work = x.float()
        apply_interleaved_rotary_emb_inplace(x_work, (cos, sin))
        x.copy_(x_work.to(dtype=x.dtype))
        return x

    # x: [B, S, C] where C is even and grouped as (..., 2)
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for interleaved rotary, got {x.shape[-1]}."
        )
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]

    # Support both common interleaved RoPE layouts:
    # - full layout: cos/sin last dim == D (values duplicated per pair)
    # - half layout: cos/sin last dim == D/2 (matches x_real/x_imag)
    cos_t = cos.to(device=x.device, dtype=x.dtype)
    sin_t = sin.to(device=x.device, dtype=x.dtype)
    if cos_t.shape[-1] == x.shape[-1]:
        cos_e = cos_t[..., 0::2]
        sin_e = sin_t[..., 0::2]
    elif cos_t.shape[-1] == x_real.shape[-1]:
        cos_e = cos_t
        sin_e = sin_t
    else:
        raise ValueError(
            "Interleaved RoPE shape mismatch: expected cos/sin last dim to be "
            f"{x.shape[-1]} (full) or {x_real.shape[-1]} (half), got cos={tuple(cos_t.shape)} "
            f"sin={tuple(sin_t.shape)} for x={tuple(x.shape)}"
        )

    real_tmp = x_real.clone()
    x_real.mul_(cos_e).addcmul_(x_imag, sin_e, value=-1.0)
    x_imag.mul_(cos_e).addcmul_(real_tmp, sin_e, value=1.0)
    return x


def apply_split_rotary_emb_inplace(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    In-place split RoPE.

    This matches `apply_split_rotary_emb` semantics (including reshaping when cos/sin are `[B,H,T,R]`),
    but mutates `x` in-place.
    """
    cos, sin = freqs

    # Keep prior numerical behavior: `apply_split_rotary_emb` upcasts to fp32.
    if cos.dtype == torch.float32 and x.dtype != torch.float32:
        x_work = x.float()
        apply_split_rotary_emb_inplace(x_work, (cos, sin))
        x.copy_(x_work.to(dtype=x.dtype))
        return x

    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        # cos is (B, H, T, R) -> reshape x to (B, H, T, dim_per_head)
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    r = last // 2

    x0 = x[..., :r]
    x1 = x[..., r:]

    # Ensure cos/sin match dtype/device for in-place ops.
    cos_t = cos.to(device=x.device, dtype=x.dtype)
    sin_t = sin.to(device=x.device, dtype=x.dtype)

    x0_orig = x0.clone()
    x0.mul_(cos_t).addcmul_(x1, sin_t, value=-1.0)
    x1.mul_(cos_t).addcmul_(x0_orig, sin_t, value=1.0)

    if needs_reshape:
        # restore original shape: (B, T, H, D) -> (B, T, H*D)
        x = x.swapaxes(1, 2).reshape(b, t, -1)
    return x


def apply_interleaved_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, C // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


def apply_split_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs

    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        # cos is (#b, h, t, r) -> reshape x to (b, h, t, dim_per_head)
        # The cos/sin batch dim may only be broadcastable, so take batch size from x
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    # Split last dim (2*r) into (d=2, r)
    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    # (..., 2, r)
    split_x = x.reshape(*x.shape[:-1], 2, r).float()  # Explicitly upcast to float

    first_x = split_x[..., :1, :]  # (..., 1, r)
    second_x = split_x[..., 1:, :]  # (..., 1, r)

    cos_u = cos.unsqueeze(-2)  # broadcast to (..., 1, r) against (..., 2, r)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]

    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)

    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)

    out = out.to(dtype=x_dtype)
    return out


@dataclass
class AudioVisualModelOutput(BaseOutput):
    r"""
    Holds the output of an audiovisual model which produces both visual (e.g. video) and audio outputs.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input, representing the visual output
            of the model. This is typically a video (spatiotemporal) output.
        audio_sample (`torch.Tensor` of shape `(batch_size, TODO)`):
            The audio output of the audiovisual model.
    """

    sample: "torch.Tensor"  # noqa: F821
    audio_sample: "torch.Tensor"  # noqa: F821


class LTX2AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://huggingface.co/papers/2310.00426; Section 2.3) and adapted by the LTX-2.0
    model. In particular, the number of modulation parameters to be calculated is now configurable.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_mod_params (`int`, *optional*, defaults to `6`):
            The number of modulation parameters which will be calculated in the first return argument. The default of 6
            is standard, but sometimes we may want to have a different (usually smaller) number of modulation
            parameters.
        use_additional_conditions (`bool`, *optional*, defaults to `False`):
            Whether to use additional conditions for normalization or not.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_mod_params: int = 6,
        use_additional_conditions: bool = False,
    ):
        super().__init__()
        self.num_mod_params = num_mod_params

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            embedding_dim, self.num_mod_params * embedding_dim, bias=True
        )

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {
            "resolution": None,
            "aspect_ratio": None,
        }

        embedded_timestep = self.emb(
            timestep,
            **added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class LTX2AudioVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0) for the LTX-2.0 model.
    Compared to the LTX-1.0 model, we allow the RoPE embeddings for the queries and keys to be separate so that we can
    support audio-to-video (a2v) and video-to-audio (v2a) cross attention.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTX2Attention",
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_hidden_states: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Wan2GP-style: accept single-item lists and clear them to drop references early.
        if isinstance(hidden_states, list):
            x = hidden_states[0]
            hidden_states.clear()
            hidden_states = x
            x = None

        if isinstance(encoder_hidden_states, list):
            ctx = encoder_hidden_states[0]
            encoder_hidden_states.clear()
            encoder_hidden_states = ctx
            ctx = None

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # In-place Q/K RMSNorm (query/key are fresh projections).
        attn.norm_q(query)
        attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                apply_interleaved_rotary_emb_inplace(query, query_rotary_emb)
                apply_interleaved_rotary_emb_inplace(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                apply_split_rotary_emb_inplace(query, query_rotary_emb)
                apply_split_rotary_emb_inplace(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTX2Attention(torch.nn.Module, AttentionModuleMixin):
    r"""
    Attention class for all LTX-2.0 attention layers. Compared to LTX-1.0, this supports specifying the query and key
    RoPE embeddings separately for audio-to-video (a2v) and video-to-audio (v2a) cross-attention.
    """

    _default_processor_cls = LTX2AudioVideoAttnProcessor
    _available_processors = [LTX2AudioVideoAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = True,
        rope_type: str = "interleaved",
        processor=None,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError(
                "Only 'rms_norm_across_heads' is supported as a valid value for `qk_norm`."
            )

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads
        self.rope_type = rope_type

        # Use in-place RMSNorm for Q/K (saves memory; Q/K are fresh projections).
        self.norm_q = InplaceRMSNorm(
            dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )
        self.norm_k = InplaceRMSNorm(
            dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(
            self.cross_attention_dim, self.inner_kv_dim, bias=bias
        )
        self.to_v = torch.nn.Linear(
            self.cross_attention_dim, self.inner_kv_dim, bias=bias
        )
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        # Pass lists to the attention processor so it can clear references early (Wan2GP-style).
        if isinstance(self.processor, LTX2AudioVideoAttnProcessor):
            inference_mode = not torch.is_grad_enabled()
            if inference_mode:
                hs_list: List[torch.Tensor] = [hidden_states]
                ehs_list: Optional[List[torch.Tensor]] = (
                    [encoder_hidden_states]
                    if encoder_hidden_states is not None
                    else None
                )
                # Drop direct tensor references; let the processor clear the lists.
                hidden_states = hs_list
                encoder_hidden_states = ehs_list
                hidden_states = self.processor(
                    self,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    query_rotary_emb,
                    key_rotary_emb,
                    **kwargs,
                )
            else:
                hidden_states = self.processor(
                    self,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    query_rotary_emb,
                    key_rotary_emb,
                    **kwargs,
                )
        else:
            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                query_rotary_emb,
                key_rotary_emb,
                **kwargs,
            )
        return hidden_states


class LTX2VideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in [LTX-2.0](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim,
        audio_cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
    ):
        super().__init__()

        # 1. Self-Attention (video and audio)
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_norm1 = RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 2. Prompt Cross-Attention
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_norm2 = RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        # Audio-to-Video (a2v) Attention --> Q: Video; K,V: Audio
        self.audio_to_video_norm = RMSNorm(
            dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_to_video_attn = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # Video-to-Audio (v2a) Attention --> Q: Audio; K,V: Video
        self.video_to_audio_norm = RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 4. Feedforward layers
        self.norm3 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.audio_norm3 = RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_ff = FeedForward(audio_dim, activation_fn=activation_fn)

        # 5. Per-Layer Modulation Parameters
        # Self-Attention / Feedforward AdaLayerNorm-Zero mod params
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(6, audio_dim) / audio_dim**0.5
        )

        # Per-layer a2v, v2a Cross-Attention mod params
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, dim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, audio_dim)
        )

        # ----------------------------
        # Chunking controls (disabled by default)
        # ----------------------------
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1
        self._mod_norm_chunk_size: Optional[int] = (
            None  # for norms followed by modulation
        )
        self._norm_chunk_size: Optional[int] = None  # for plain norms

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
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        a2v_cross_attention_mask: Optional[torch.Tensor] = None,
        v2a_cross_attention_mask: Optional[torch.Tensor] = None,
        skip_video_self_attn: bool = False,
        skip_audio_self_attn: bool = False,
        skip_a2v_cross_attn: bool = False,
        skip_v2a_cross_attn: bool = False,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        inference_mode = not torch.is_grad_enabled()

        # 1. Video and Audio Self-Attention
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = _apply_scale_shift_table(temb, self.scale_shift_table, batch_size=batch_size)
        if not skip_video_self_attn:
            norm_hidden_states = _chunked_modulated_norm(
                self.norm1,
                hidden_states,
                scale_msa,
                shift_msa,
                chunk_size=self._mod_norm_chunk_size,
            )

            attn_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=None,
                query_rotary_emb=video_rotary_emb,
            )
            _apply_gate_inplace(attn_hidden_states, gate_msa)
            if inference_mode:
                hidden_states.add_(attn_hidden_states)
            else:
                hidden_states = hidden_states + attn_hidden_states

        (
            audio_shift_msa,
            audio_scale_msa,
            audio_gate_msa,
            audio_shift_mlp,
            audio_scale_mlp,
            audio_gate_mlp,
        ) = _apply_scale_shift_table(
            temb_audio, self.audio_scale_shift_table, batch_size=batch_size
        )
        if not skip_audio_self_attn:
            norm_audio_hidden_states = _chunked_modulated_norm(
                self.audio_norm1,
                audio_hidden_states,
                audio_scale_msa,
                audio_shift_msa,
                chunk_size=self._mod_norm_chunk_size,
            )

            attn_audio_hidden_states = self.audio_attn1(
                hidden_states=norm_audio_hidden_states,
                encoder_hidden_states=None,
                query_rotary_emb=audio_rotary_emb,
            )
            _apply_gate_inplace(attn_audio_hidden_states, audio_gate_msa)
            if inference_mode:
                audio_hidden_states.add_(attn_audio_hidden_states)
            else:
                audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        # 2. Video and Audio Cross-Attention with the text embeddings
        norm_hidden_states = _chunked_norm(
            self.norm2, hidden_states, chunk_size=self._norm_chunk_size
        )
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        if inference_mode:
            hidden_states.add_(attn_hidden_states)
        else:
            hidden_states = hidden_states + attn_hidden_states

        norm_audio_hidden_states = _chunked_norm(
            self.audio_norm2, audio_hidden_states, chunk_size=self._norm_chunk_size
        )
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=audio_encoder_attention_mask,
        )
        if inference_mode:
            audio_hidden_states.add_(attn_audio_hidden_states)
        else:
            audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        if not (skip_a2v_cross_attn and skip_v2a_cross_attn):
            norm_hidden_states = _chunked_norm(
                self.audio_to_video_norm, hidden_states, chunk_size=self._norm_chunk_size
            )
            norm_audio_hidden_states = _chunked_norm(
                self.video_to_audio_norm,
                audio_hidden_states,
                chunk_size=self._norm_chunk_size,
            )

            # Combine global and per-layer cross attention modulation parameters
            # Video
            video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
            video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]
            (
                video_a2v_ca_scale,
                video_a2v_ca_shift,
                video_v2a_ca_scale,
                video_v2a_ca_shift,
            ) = _apply_scale_shift_table(
                temb_ca_scale_shift,
                video_per_layer_ca_scale_shift,
                batch_size=batch_size,
                num_params=4,
            )
            (a2v_gate,) = _apply_scale_shift_table(
                temb_ca_gate, video_per_layer_ca_gate, batch_size=batch_size, num_params=1
            )

            # Audio
            audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
            audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]
            (
                audio_a2v_ca_scale,
                audio_a2v_ca_shift,
                audio_v2a_ca_scale,
                audio_v2a_ca_shift,
            ) = _apply_scale_shift_table(
                temb_ca_audio_scale_shift,
                audio_per_layer_ca_scale_shift,
                batch_size=batch_size,
                num_params=4,
            )
            (v2a_gate,) = _apply_scale_shift_table(
                temb_ca_audio_gate, audio_per_layer_ca_gate, batch_size=batch_size, num_params=1
            )

            # Audio-to-Video Cross Attention: Q: Video; K,V: Audio
            mod_norm_hidden_states = _chunked_apply_scale_shift(
                norm_hidden_states,
                video_a2v_ca_scale.squeeze(2),
                video_a2v_ca_shift.squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )

            mod_norm_audio_hidden_states = _chunked_apply_scale_shift(
                norm_audio_hidden_states,
                audio_a2v_ca_scale.squeeze(2),
                audio_a2v_ca_shift.squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )

            if not skip_a2v_cross_attn:
                a2v_attn_hidden_states = self.audio_to_video_attn(
                    mod_norm_hidden_states,
                    encoder_hidden_states=mod_norm_audio_hidden_states,
                    query_rotary_emb=ca_video_rotary_emb,
                    key_rotary_emb=ca_audio_rotary_emb,
                    attention_mask=a2v_cross_attention_mask,
                )

                _apply_gate_inplace(a2v_attn_hidden_states, a2v_gate)
                if inference_mode:
                    hidden_states.add_(a2v_attn_hidden_states)
                else:
                    hidden_states = hidden_states + a2v_attn_hidden_states

            # Video-to-Audio Cross Attention: Q: Audio; K,V: Video
            mod_norm_hidden_states = _chunked_apply_scale_shift(
                norm_hidden_states,
                video_v2a_ca_scale.squeeze(2),
                video_v2a_ca_shift.squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )
            mod_norm_audio_hidden_states = _chunked_apply_scale_shift(
                norm_audio_hidden_states,
                audio_v2a_ca_scale.squeeze(2),
                audio_v2a_ca_shift.squeeze(2),
                chunk_size=self._mod_norm_chunk_size,
            )

            if not skip_v2a_cross_attn:
                v2a_attn_hidden_states = self.video_to_audio_attn(
                    mod_norm_audio_hidden_states,
                    encoder_hidden_states=mod_norm_hidden_states,
                    query_rotary_emb=ca_audio_rotary_emb,
                    key_rotary_emb=ca_video_rotary_emb,
                    attention_mask=v2a_cross_attention_mask,
                )

                _apply_gate_inplace(v2a_attn_hidden_states, v2a_gate)
                if inference_mode:
                    audio_hidden_states.add_(v2a_attn_hidden_states)
                else:
                    audio_hidden_states = audio_hidden_states + v2a_attn_hidden_states

        # 4. Feedforward
        norm_hidden_states = _chunked_modulated_norm(
            self.norm3,
            hidden_states,
            scale_mlp,
            shift_mlp,
            chunk_size=self._mod_norm_chunk_size,
        )
        ff_output = _chunked_feed_forward(
            self.ff, norm_hidden_states, self._ff_chunk_dim, self._ff_chunk_size
        )
        _apply_gate_inplace(ff_output, gate_mlp)
        if inference_mode:
            hidden_states.add_(ff_output)
        else:
            hidden_states = hidden_states + ff_output

        norm_audio_hidden_states = _chunked_modulated_norm(
            self.audio_norm3,
            audio_hidden_states,
            audio_scale_mlp,
            audio_shift_mlp,
            chunk_size=self._mod_norm_chunk_size,
        )
        audio_ff_output = _chunked_feed_forward(
            self.audio_ff,
            norm_audio_hidden_states,
            self._ff_chunk_dim,
            self._ff_chunk_size,
        )
        _apply_gate_inplace(audio_ff_output, audio_gate_mlp)
        if inference_mode:
            audio_hidden_states.add_(audio_ff_output)
        else:
            audio_hidden_states = audio_hidden_states + audio_ff_output

        return hidden_states, audio_hidden_states


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    """
    Video and audio rotary positional embeddings (RoPE) for the LTX-2.0 model.

    Args:
        causal_offset (`int`, *optional*, defaults to `1`):
            Offset in the temporal axis for causal VAE modeling. This is typically 1 (for causal modeling where the VAE
            treats the very first frame differently), but could also be 0 (for non-causal modeling).
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )
        self.rope_type = rope_type

        self.base_num_frames = base_num_frames
        self.num_attention_heads = num_attention_heads

        # Video-specific
        self.base_height = base_height
        self.base_width = base_width

        # Audio-specific
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.audio_latents_per_second = (
            float(sampling_rate) / float(hop_length) / float(scale_factors[0])
        )

        self.scale_factors = scale_factors
        self.theta = theta
        self.causal_offset = causal_offset

        self.modality = modality
        if self.modality not in ["video", "audio"]:
            raise ValueError(
                f"Modality {modality} is not supported. Supported modalities are `video` and `audio`."
            )
        self.double_precision = double_precision if torch.cuda.is_available() else False

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 25.0,
    ) -> torch.Tensor:
        """
        Create per-dimension bounds [inclusive start, exclusive end) for each patch with respect to the original pixel
        space video grid (num_frames, height, width). This will ultimately have shape (batch_size, 3, num_patches, 2)
        where
            - axis 1 (size 3) enumerates (frame, height, width) dimensions (e.g. idx 0 corresponds to frames)
            - axis 3 (size 2) stores `[start, end)` indices within each dimension

        Args:
            batch_size (`int`):
                Batch size of the video latents.
            num_frames (`int`):
                Number of latent frames in the video latents.
            height (`int`):
                Latent height of the video latents.
            width (`int`):
                Latent width of the video latents.
            device (`torch.device`):
                Device on which to create the video grid.

        Returns:
            `torch.Tensor`:
                Per-dimension patch boundaries tensor of shape [batch_size, 3, num_patches, 2].
        """

        # 1. Generate grid coordinates for each spatiotemporal dimension (frames, height, width)
        # Always compute rope in fp32
        grid_f = torch.arange(
            start=0,
            end=num_frames,
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )
        grid_h = torch.arange(
            start=0,
            end=height,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid_w = torch.arange(
            start=0, end=width, step=self.patch_size, dtype=torch.float32, device=device
        )
        # indexing='ij' ensures that the dimensions are kept in order as (frames, height, width)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(
            grid, dim=0
        )  # [3, N_F, N_H, N_W], where e.g. N_F is the number of temporal patches

        # 2. Get the patch boundaries with respect to the latent video grid
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(
            patch_size, dtype=grid.dtype, device=grid.device
        )
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        # Combine the start (grid) and end (patch_ends) coordinates along new trailing dimension
        latent_coords = torch.stack([grid, patch_ends], dim=-1)  # [3, N_F, N_H, N_W, 2]
        # Reshape to (batch_size, 3, num_patches, 2)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # 3. Calculate the pixel space patch boundaries from the latent boundaries.
        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        # Broadcast the VAE scale factors such that they are compatible with latent_coords's shape
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1  # This is the (frame, height, width) dim
        # Apply per-axis scaling to convert latent coordinates to pixel space coordinates
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # As the VAE temporal stride for the first frame is 1 instead of self.vae_scale_factors[0], we need to shift
        # and clamp to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clamp(min=0)

        # Scale the temporal coordinates by the video FPS
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        fps: float = 25.0,
        shift: int = 0,
    ) -> torch.Tensor:
        """
        Create per-dimension bounds [inclusive start, exclusive end) of start and end timestamps for each latent frame.
        This will ultimately have shape (batch_size, 3, num_patches, 2) where
            - axis 1 (size 1) represents the temporal dimension
            - axis 3 (size 2) stores `[start, end)` indices within each dimension

        Args:
            batch_size (`int`):
                Batch size of the audio latents.
            num_frames (`int`):
                Number of latent frames in the audio latents.
            device (`torch.device`):
                Device on which to create the audio grid.
            shift (`int`, *optional*, defaults to `0`):
                Offset on the latent indices. Different shift values correspond to different overlapping windows with
                respect to the same underlying latent grid.

        Returns:
            `torch.Tensor`:
                Per-dimension patch boundaries tensor of shape [batch_size, 1, num_patches, 2].
        """

        # 1. Generate coordinates in the frame (time) dimension.
        grid_f = torch.arange(
            start=shift,
            end=num_frames + shift,
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )

        # 2. Calculate start timstamps in seconds with respect to the original spectrogram grid
        audio_scale_factor = self.scale_factors[0]
        # Scale back to mel spectrogram space
        grid_start_mel = grid_f * audio_scale_factor
        # Handle first frame causal offset, ensuring non-negative timestamps
        grid_start_mel = (
            grid_start_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        # Convert mel bins back into seconds
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        # 3. Calculate start timstamps in seconds with respect to the original spectrogram grid
        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(
            min=0
        )
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack(
            [grid_start_s, grid_end_s], dim=-1
        )  # [num_patches, 2]
        audio_coords = audio_coords.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, num_patches, 2]
        audio_coords = audio_coords.unsqueeze(1)  # [batch_size, 1, num_patches, 2]
        return audio_coords

    def prepare_coords(self, *args, **kwargs):
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        elif self.modality == "audio":
            return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self,
        coords: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 25.0,
        shift: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coords is not None:
            device = device or coords.device
            batch_size = batch_size or coords.size(0)
        else:
            device = device or "cpu"
            batch_size = batch_size or 1

        # 1. Calculate the coordinate grid with respect to data space for the given modality (video, audio).
        if coords is None and self.modality == "video":
            coords = self.prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                device=device,
                fps=fps,
            )
        elif coords is None and self.modality == "audio":
            coords = self.prepare_audio_coords(
                batch_size,
                num_frames,
                device=device,
                shift=shift,
                fps=fps,
            )
        # Number of spatiotemporal dimensions (3 for video, 1 (temporal) for audio and cross attn)
        num_pos_dims = coords.shape[1]

        # 2. If the coords are patch boundaries [start, end), use the midpoint of these boundaries as the patch
        # position index
        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)  # [B, num_pos_dims, num_patches]

        # 3. Get coordinates as a fraction of the base data shape
        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        elif self.modality == "audio":
            max_positions = (self.base_num_frames,)
        # [B, num_pos_dims, num_patches] --> [B, num_patches, num_pos_dims]
        grid = torch.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1
        ).to(device)
        # Number of spatiotemporal dimensions (3 for video, 1 for audio and cross attn) times 2 for cos, sin
        num_rope_elems = num_pos_dims * 2

        # 4. Create a 1D grid of frequencies for RoPE
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        # 5. Tensor-vector outer product between pos ids tensor of shape (B, 3, num_patches) and freqs vector of shape
        # (self.dim // num_elems,)
        freqs = (
            grid.unsqueeze(-1) * 2 - 1
        ) * freqs  # [B, num_patches, num_pos_dims, self.dim // num_elems]
        freqs = freqs.transpose(-1, -2).flatten(2)  # [B, num_patches, self.dim // 2]

        # 6. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        # TODO: consider implementing this as a utility and reuse in `connectors.py`.
        # src/diffusers/pipelines/ltx2/connectors.py
        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = torch.zeros_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

                cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

            # Reshape freqs to be compatible with multi-head attention
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]

            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)

            cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class LTX2VideoTransformer3DModel(
    ModelMixin,
    ConfigMixin,
    AttentionMixin,
    FromOriginalModelMixin,
    PeftAdapterMixin,
    CacheMixin,
):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTX2VideoTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(
                split_dim=1, expected_dims=3, split_output=False
            ),
            "encoder_hidden_states": ContextParallelInput(
                split_dim=1, expected_dims=3, split_output=False
            ),
            "encoder_attention_mask": ContextParallelInput(
                split_dim=1, expected_dims=2, split_output=False
            ),
        },
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    # ----------------------------
    # Chunking profile presets
    # ----------------------------
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
        in_channels: int = 128,  # Video Arguments
        out_channels: Optional[int] = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: Optional[int] = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        chunking_profile: str = "none",
        ffn_chunk_size: Optional[int] = None,
        ffn_chunk_dim: int = 1,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        # 2. Prompt embeddings
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=inner_dim
        )
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=audio_inner_dim
        )

        # 3. Timestep Modulation Params and Embedding
        # 3.1. Global Timestep Modulation Parameters (except for cross-attention) and timestep + size embedding
        # time_embed and audio_time_embed calculate both the timestep embedding and (global) modulation parameters
        self.time_embed = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=6, use_additional_conditions=False
        )
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=6, use_additional_conditions=False
        )

        # 3.2. Global Cross Attention Modulation Parameters
        # Used in the audio-to-video and video-to-audio cross attention layers as a global set of modulation params,
        # which are then further modified by per-block modulaton params in each transformer block.
        # There are 2 sets of scale/shift parameters for each modality, 1 each for audio-to-video (a2v) and
        # video-to-audio (v2a) cross attention
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        # Gate param for audio-to-video (a2v) cross attn (where the video is the queries (Q) and the audio is the keys
        # and values (KV))
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=1, use_additional_conditions=False
        )
        # Gate param for video-to-audio (v2a) cross attn (where the audio is the queries (Q) and the video is the keys
        # and values (KV))
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=1, use_additional_conditions=False
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5
        )

        # 4. Rotary Positional Embeddings (RoPE)
        # Self-Attention
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=[audio_scale_factor],
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # Audio-to-Video, Video-to-Audio Cross-Attention
        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2VideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.audio_norm_out = nn.LayerNorm(
            audio_inner_dim, eps=1e-6, elementwise_affine=False
        )
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

        self.gradient_checkpointing = False

        # Default: no chunking unless enabled explicitly.
        self._out_modulated_norm_chunk_size: Optional[int] = None
        self.set_chunking_profile(chunking_profile)

        # Back-compat / override: allow enabling FFN chunking directly via init kwargs.
        if ffn_chunk_size is not None:
            self.set_chunk_feed_forward(ffn_chunk_size, dim=ffn_chunk_dim)

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        for block in self.transformer_blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)

    def set_chunk_norms(
        self,
        *,
        modulated_norm_chunk_size: Optional[int] = None,
        norm_chunk_size: Optional[int] = None,
    ) -> None:
        for block in self.transformer_blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=modulated_norm_chunk_size,
                norm_chunk_size=norm_chunk_size,
            )

    def set_chunking_profile(self, profile_name: str) -> None:
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )
        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name

        self._out_modulated_norm_chunk_size = p.get(
            "out_modulated_norm_chunk_size", None
        )
        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        self.set_chunk_norms(
            modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
            norm_chunk_size=p.get("norm_chunk_size", None),
        )

    # ----------------------------
    # Forward helpers (scoping / memory)
    # ----------------------------
    @staticmethod
    def _mask_to_attention_bias(
        mask: Optional[torch.Tensor], *, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        # Convert encoder attention masks to biases the same way we do for attention_mask.
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask.unsqueeze(1)
        return mask

    @staticmethod
    def _rope_cpu_key(tag: str, coords: torch.Tensor, fps_val: float) -> Tuple[Any, ...]:
        # Use the coords tensor identity + shape + fps.
        # Works well when callers pass the same CPU coords tensor across steps.
        return (
            tag,
            int(coords.data_ptr()),
            tuple(coords.shape),
            str(coords.dtype),
            float(fps_val),
        )

    @staticmethod
    def _coords_to_cpu(coords: torch.Tensor) -> torch.Tensor:
        coords_cpu = coords.detach()
        if coords_cpu.device.type != "cpu":
            coords_cpu = coords_cpu.to("cpu")
        return coords_cpu

    def _ensure_rope_cpu_cache(self) -> Dict[Tuple[Any, ...], Any]:
        if not hasattr(self, "_rope_cpu_cache"):
            self._rope_cpu_cache = {}
        return self._rope_cpu_cache

    def _prepare_rope_coords(
        self,
        *,
        batch_size: int,
        num_frames: Optional[int],
        height: Optional[int],
        width: Optional[int],
        fps: float,
        audio_num_frames: Optional[int],
        video_coords: Optional[torch.Tensor],
        audio_coords: Optional[torch.Tensor],
        hidden_device: torch.device,
        audio_hidden_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vcoords = video_coords
        acoords = audio_coords
        if vcoords is None:
            vcoords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_device, fps=fps
            )
        if acoords is None:
            acoords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_device, fps=fps
            )
        return vcoords, acoords

    def _rope_embs_from_coords(
        self,
        *,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        fps: float,
        hidden_device: torch.device,
        audio_hidden_device: torch.device,
        rope_on_cpu: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Memory optimization (Wan2GP-style): keep RoPE cached on CPU across steps,
        # and only materialize GPU copies for the current forward pass.
        if rope_on_cpu:
            cache = self._ensure_rope_cpu_cache()
            vcoords_cpu = self._coords_to_cpu(video_coords)
            acoords_cpu = self._coords_to_cpu(audio_coords)

            v_key = self._rope_cpu_key("video", vcoords_cpu, fps)
            a_key = self._rope_cpu_key("audio", acoords_cpu, fps)

            vca_coords_cpu = vcoords_cpu[:, 0:1, :]
            aca_coords_cpu = acoords_cpu[:, 0:1, :]
            vca_key = self._rope_cpu_key("video_ca", vca_coords_cpu, fps)
            aca_key = self._rope_cpu_key("audio_ca", aca_coords_cpu, fps)

            if v_key not in cache:
                cache[v_key] = self.rope(vcoords_cpu, fps=fps, device=torch.device("cpu"))
            if a_key not in cache:
                cache[a_key] = self.audio_rope(acoords_cpu, device=torch.device("cpu"))
            if vca_key not in cache:
                cache[vca_key] = self.cross_attn_rope(vca_coords_cpu, device=torch.device("cpu"))
            if aca_key not in cache:
                cache[aca_key] = self.cross_attn_audio_rope(
                    aca_coords_cpu, device=torch.device("cpu")
                )

            # Keep freqs on CPU; apply_*_rotary_emb_inplace will move to the right device as needed.
            v_rot = cache[v_key]
            a_rot = cache[a_key]
            v_ca_rot = cache[vca_key]
            a_ca_rot = cache[aca_key]
            del cache, vcoords_cpu, acoords_cpu, vca_coords_cpu, aca_coords_cpu
            return v_rot, a_rot, v_ca_rot, a_ca_rot

        v_rot = self.rope(video_coords, fps=fps, device=hidden_device)
        a_rot = self.audio_rope(audio_coords, device=audio_hidden_device)
        v_ca_rot = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_device)
        a_ca_rot = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_device
        )
        return v_rot, a_rot, v_ca_rot, a_ca_rot

    def _project_inputs(
        self, hidden_states: torch.Tensor, audio_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.proj_in(hidden_states), self.audio_proj_in(audio_hidden_states)

    @staticmethod
    def _view_timestep_outputs(
        x: torch.Tensor, embedded: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(batch_size, -1, x.size(-1))
        embedded = embedded.view(batch_size, -1, embedded.size(-1))
        return x, embedded

    def _compute_time_embed(
        self, timestep: torch.LongTensor, *, batch_size: int, hidden_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temb, embedded = self.time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_dtype
        )
        return self._view_timestep_outputs(temb, embedded, batch_size)

    def _compute_audio_time_embed(
        self, timestep: torch.LongTensor, *, batch_size: int, hidden_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temb, embedded = self.audio_time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_dtype
        )
        return self._view_timestep_outputs(temb, embedded, batch_size)

    def _cross_attn_gate_scale_factor(self) -> float:
        return float(
            self.config.cross_attn_timestep_scale_multiplier
            / self.config.timestep_scale_multiplier
        )

    def _compute_video_cross_attn_modulation(
        self,
        timestep: torch.LongTensor,
        *,
        batch_size: int,
        hidden_dtype: torch.dtype,
        gate_scale_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_flat = timestep.flatten()
        scale_shift, _ = self.av_cross_attn_video_scale_shift(
            t_flat, batch_size=batch_size, hidden_dtype=hidden_dtype
        )
        gate, _ = self.av_cross_attn_video_a2v_gate(
            t_flat * gate_scale_factor, batch_size=batch_size, hidden_dtype=hidden_dtype
        )
        scale_shift = scale_shift.view(batch_size, -1, scale_shift.shape[-1])
        gate = gate.view(batch_size, -1, gate.shape[-1])
        return scale_shift, gate

    def _compute_audio_cross_attn_modulation(
        self,
        timestep: torch.LongTensor,
        *,
        batch_size: int,
        hidden_dtype: torch.dtype,
        gate_scale_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_flat = timestep.flatten()
        scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            t_flat, batch_size=batch_size, hidden_dtype=hidden_dtype
        )
        gate, _ = self.av_cross_attn_audio_v2a_gate(
            t_flat * gate_scale_factor, batch_size=batch_size, hidden_dtype=hidden_dtype
        )
        scale_shift = scale_shift.view(batch_size, -1, scale_shift.shape[-1])
        gate = gate.view(batch_size, -1, gate.shape[-1])
        return scale_shift, gate

    def _project_prompt_embeddings(
        self,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        *,
        batch_size: int,
        video_hidden_dim: int,
        audio_hidden_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, video_hidden_dim
        )
        audio_encoder_hidden_states = self.audio_caption_projection(
            audio_encoder_hidden_states
        )
        audio_encoder_hidden_states = audio_encoder_hidden_states.view(
            batch_size, -1, audio_hidden_dim
        )
        return encoder_hidden_states, audio_encoder_hidden_states

    def _run_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        video_rotary_emb: torch.Tensor,
        audio_rotary_emb: torch.Tensor,
        ca_video_rotary_emb: torch.Tensor,
        ca_audio_rotary_emb: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        audio_encoder_attention_mask: Optional[torch.Tensor],
        skip_video_self_attn_blocks: Optional[set[int]] = None,
        skip_audio_self_attn_blocks: Optional[set[int]] = None,
        skip_a2v_cross_attn: bool = False,
        skip_v2a_cross_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block_idx, block in enumerate(self.transformer_blocks):
            skip_video_self_attn = (
                skip_video_self_attn_blocks is not None
                and block_idx in skip_video_self_attn_blocks
            )
            skip_audio_self_attn = (
                skip_audio_self_attn_blocks is not None
                and block_idx in skip_audio_self_attn_blocks
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    temb_ca_scale_shift,
                    temb_ca_audio_scale_shift,
                    temb_ca_gate,
                    temb_ca_audio_gate,
                    video_rotary_emb,
                    audio_rotary_emb,
                    ca_video_rotary_emb,
                    ca_audio_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                    skip_video_self_attn,
                    skip_audio_self_attn,
                    skip_a2v_cross_attn,
                    skip_v2a_cross_attn,
                )
            else:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=temb_ca_scale_shift,
                    temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
                    temb_ca_gate=temb_ca_gate,
                    temb_ca_audio_gate=temb_ca_audio_gate,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=ca_video_rotary_emb,
                    ca_audio_rotary_emb=ca_audio_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    skip_video_self_attn=skip_video_self_attn,
                    skip_audio_self_attn=skip_audio_self_attn,
                    skip_a2v_cross_attn=skip_a2v_cross_attn,
                    skip_v2a_cross_attn=skip_v2a_cross_attn,
                )
        return hidden_states, audio_hidden_states

    def _apply_output_layers(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        embedded_timestep: torch.Tensor,
        audio_embedded_timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Avoid allocating a `[B, S, 2, D]` intermediate for scale/shift.
        _sst = self.scale_shift_table.to(
            device=embedded_timestep.device, dtype=embedded_timestep.dtype
        )
        shift = embedded_timestep + _sst[0]
        scale = embedded_timestep + _sst[1]

        hidden_states = _chunked_modulated_norm(
            self.norm_out,
            hidden_states,
            scale,
            shift,
            chunk_size=self._out_modulated_norm_chunk_size,
        )
        output = self.proj_out(hidden_states)

        _asst = self.audio_scale_shift_table.to(
            device=audio_embedded_timestep.device, dtype=audio_embedded_timestep.dtype
        )
        audio_shift = audio_embedded_timestep + _asst[0]
        audio_scale = audio_embedded_timestep + _asst[1]

        audio_hidden_states = _chunked_modulated_norm(
            self.audio_norm_out,
            audio_hidden_states,
            audio_scale,
            audio_shift,
            chunk_size=self._out_modulated_norm_chunk_size,
        )
        audio_output = self.audio_proj_out(audio_hidden_states)
        return output, audio_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 25.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        rope_on_cpu: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_video_self_attn_blocks: Optional[List[int]] = None,
        skip_audio_self_attn_blocks: Optional[List[int]] = None,
        skip_a2v_cross_attn: bool = False,
        skip_v2a_cross_attn: bool = False,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for LTX-2.0 audiovisual video transformer.

        Args:
            hidden_states (`torch.Tensor`):
                Input patchified video latents of shape (batch_size, num_video_tokens, in_channels).
            audio_hidden_states (`torch.Tensor`):
                Input patchified audio latents of shape (batch_size, num_audio_tokens, audio_in_channels).
            encoder_hidden_states (`torch.Tensor`):
                Input text embeddings of shape TODO.
            TODO for the rest.

        Returns:
            `AudioVisualModelOutput` or `tuple`:
                If `return_dict` is `True`, returns a structured output of type `AudioVisualModelOutput`, otherwise a
                `tuple` is returned where the first element is the denoised video latent patch sequence and the second
                element is the denoised audio latent patch sequence.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        # Optional memory debugging (disabled by default).
        # Enable with `APEX_STEP_MEM=1` in the environment.

        batch_size = hidden_states.size(0)

        encoder_attention_mask = self._mask_to_attention_bias(
            encoder_attention_mask, dtype=hidden_states.dtype
        )
        
        audio_encoder_attention_mask = self._mask_to_attention_bias(
            audio_encoder_attention_mask, dtype=audio_hidden_states.dtype
        )
  
        video_coords, audio_coords = self._prepare_rope_coords(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            hidden_device=hidden_states.device,
            audio_hidden_device=audio_hidden_states.device,
        )
        (
            video_rotary_emb,
            audio_rotary_emb,
            video_cross_attn_rotary_emb,
            audio_cross_attn_rotary_emb,
        ) = self._rope_embs_from_coords(
            video_coords=video_coords,
            audio_coords=audio_coords,
            fps=fps,
            hidden_device=hidden_states.device,
            audio_hidden_device=audio_hidden_states.device,
            rope_on_cpu=rope_on_cpu,
        )
        
        # coords are no longer needed after RoPE is materialized/cached
        del video_coords, audio_coords

        hidden_states, audio_hidden_states = self._project_inputs(
            hidden_states, audio_hidden_states
        )
        

        temb, embedded_timestep = self._compute_time_embed(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )
        temb_audio, audio_embedded_timestep = self._compute_audio_time_embed(
            audio_timestep, batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
        )
        

        gate_scale_factor = self._cross_attn_gate_scale_factor()
        video_cross_attn_scale_shift, video_cross_attn_a2v_gate = (
            self._compute_video_cross_attn_modulation(
                timestep,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
                gate_scale_factor=gate_scale_factor,
            )
        )
        
        audio_cross_attn_scale_shift, audio_cross_attn_v2a_gate = (
            self._compute_audio_cross_attn_modulation(
                audio_timestep,
                batch_size=batch_size,
                hidden_dtype=audio_hidden_states.dtype,
                gate_scale_factor=gate_scale_factor,
            )
        )
        del gate_scale_factor
  
        encoder_hidden_states, audio_encoder_hidden_states = self._project_prompt_embeddings(
            encoder_hidden_states,
            audio_encoder_hidden_states,
            batch_size=batch_size,
            video_hidden_dim=hidden_states.size(-1),
            audio_hidden_dim=audio_hidden_states.size(-1),
        )
        
        hidden_states, audio_hidden_states = self._run_transformer_blocks(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            temb=temb,
            temb_audio=temb_audio,
            temb_ca_scale_shift=video_cross_attn_scale_shift,
            temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
            temb_ca_gate=video_cross_attn_a2v_gate,
            temb_ca_audio_gate=audio_cross_attn_v2a_gate,
            video_rotary_emb=video_rotary_emb,
            audio_rotary_emb=audio_rotary_emb,
            ca_video_rotary_emb=video_cross_attn_rotary_emb,
            ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            skip_video_self_attn_blocks=(
                set(skip_video_self_attn_blocks)
                if skip_video_self_attn_blocks is not None
                else None
            ),
            skip_audio_self_attn_blocks=(
                set(skip_audio_self_attn_blocks)
                if skip_audio_self_attn_blocks is not None
                else None
            ),
            skip_a2v_cross_attn=skip_a2v_cross_attn,
            skip_v2a_cross_attn=skip_v2a_cross_attn,
        )

        # Free big intermediates before the output projection allocates new buffers.
        del (
            encoder_hidden_states,
            audio_encoder_hidden_states,
            temb,
            temb_audio,
            video_cross_attn_scale_shift,
            audio_cross_attn_scale_shift,
            video_cross_attn_a2v_gate,
            audio_cross_attn_v2a_gate,
            video_rotary_emb,
            audio_rotary_emb,
            video_cross_attn_rotary_emb,
            audio_cross_attn_rotary_emb,
            encoder_attention_mask,
            audio_encoder_attention_mask,
        )

        output, audio_output = self._apply_output_layers(
            hidden_states,
            audio_hidden_states,
            embedded_timestep=embedded_timestep,
            audio_embedded_timestep=audio_embedded_timestep,
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)
