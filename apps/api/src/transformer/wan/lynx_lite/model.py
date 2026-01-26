# Copyright (c) 2025 The Wan Team and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.

# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/v0.30.3/LICENSE and https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from src.attention import attention_register
from src.transformer.efficiency.list_clear import unwrap_single_item_list
from src.transformer.efficiency.mod import InplaceRMSNorm
from src.transformer.efficiency.ops import (
    apply_gate_inplace,
    apply_scale_shift_inplace,
    apply_wan_rope_inplace,
    chunked_feed_forward_inplace,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """
    Modulated LayerNorm with chunking to reduce peak memory.

    FP32LayerNorm internally casts to fp32 (full-size alloc). Chunking along the
    sequence dim caps peak usage.
    """
    if hidden_states.ndim != 3:
        out = norm_layer(hidden_states)
        out = out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out
        apply_scale_shift_inplace(out, scale, shift)
        return out

    b, s, d = hidden_states.shape
    in_dtype = hidden_states.dtype

    if chunk_size is None or s <= chunk_size:
        out = norm_layer(hidden_states).to(in_dtype)
        apply_scale_shift_inplace(out, scale, shift)
        return out

    out = torch.empty_like(hidden_states)
    scale_per_token = scale.dim() == 3 and scale.shape[1] == s
    for i in range(0, s, chunk_size):
        end = min(i + chunk_size, s)
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
    """LayerNorm in chunks along the sequence dimension to reduce peak memory."""
    if isinstance(norm_layer, nn.Identity):
        return hidden_states

    if hidden_states.ndim != 3:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    if chunk_size is None:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    b, s, d = hidden_states.shape
    if s <= chunk_size:
        out = norm_layer(hidden_states)
        return out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out

    out = torch.empty_like(hidden_states)
    for i in range(0, s, chunk_size):
        end = min(i + chunk_size, s)
        out[:, i:end, :] = norm_layer(hidden_states[:, i:end, :])
    return out


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
) -> torch.Tensor:
    """Chunked FFN without requiring divisibility."""
    return chunked_feed_forward_inplace(
        ff, hidden_states, chunk_dim=chunk_dim, chunk_size=chunk_size
    )


class WanAttnProcessor2_0:

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rope_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = unwrap_single_item_list(hidden_states)
        encoder_hidden_states = unwrap_single_item_list(encoder_hidden_states)
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Drop large refs early to reduce peak memory.
        del hidden_states, encoder_hidden_states

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            apply_wan_rope_inplace(
                query, rotary_emb, chunk_size=rope_chunk_size, freqs_may_be_cpu=True
            )
            apply_wan_rope_inplace(
                key, rotary_emb, chunk_size=rope_chunk_size, freqs_may_be_cpu=True
            )

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            del encoder_hidden_states_img

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = attention_register.call(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        del query, key, value

        if hidden_states_img is not None:
            if (
                (not torch.is_grad_enabled())
                and (not hidden_states.requires_grad)
                and (not hidden_states_img.requires_grad)
            ):
                hidden_states.add_(hidden_states_img)
            else:
                hidden_states = hidden_states + hidden_states_img
            del hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        # Keep base freqs on CPU; optionally cache a device copy for speed.
        self.freqs = torch.cat(freqs, dim=1).to(device=torch.device("cpu"))
        self._freqs_device_cache: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        rope_on_cpu: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        target_device = device or hidden_states.device
        if rope_on_cpu:
            freqs_base = self.freqs
        else:
            cache_key = str(target_device)
            if cache_key not in self._freqs_device_cache:
                self._freqs_device_cache[cache_key] = self.freqs.to(target_device)
            freqs_base = self._freqs_device_cache[cache_key]

        freqs = freqs_base.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )
        # If rope_on_cpu, keep this on CPU; RoPE apply will move slices as needed.
        return freqs if rope_on_cpu else freqs.to(target_device)


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        # In-place RMSNorm for Q/K (saves one buffer vs out-of-place normalization).
        old_norm_q = getattr(self.attn1, "norm_q", None)
        if old_norm_q is not None:
            self.attn1.norm_q = InplaceRMSNorm(
                dim,
                eps=eps,
                elementwise_affine=getattr(old_norm_q, "weight", None) is not None,
            )
        old_norm_k = getattr(self.attn1, "norm_k", None)
        if old_norm_k is not None:
            self.attn1.norm_k = InplaceRMSNorm(
                dim,
                eps=eps,
                elementwise_affine=getattr(old_norm_k, "weight", None) is not None,
            )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        old_norm_q = getattr(self.attn2, "norm_q", None)
        if old_norm_q is not None:
            self.attn2.norm_q = InplaceRMSNorm(
                dim,
                eps=eps,
                elementwise_affine=getattr(old_norm_q, "weight", None) is not None,
            )
        old_norm_k = getattr(self.attn2, "norm_k", None)
        if old_norm_k is not None:
            self.attn2.norm_k = InplaceRMSNorm(
                dim,
                eps=eps,
                elementwise_affine=getattr(old_norm_k, "weight", None) is not None,
            )
        old_norm_added_k = getattr(self.attn2, "norm_added_k", None)
        if old_norm_added_k is not None:
            self.attn2.norm_added_k = InplaceRMSNorm(
                dim,
                eps=eps,
                elementwise_affine=getattr(old_norm_added_k, "weight", None) is not None,
            )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # Chunking configuration (disabled by default).
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
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        rotary_emb_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        inference_mode = not torch.is_grad_enabled()
        hs_dtype = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            (self.scale_shift_table + temb.float()).to(hs_dtype).chunk(6, dim=1)
        )
        del temb

        # 1. Self-attention
        norm_hidden_states = _chunked_modulated_norm(
            self.norm1,
            hidden_states,
            scale_msa,
            shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )
        if inference_mode and isinstance(self.attn1.processor, WanAttnProcessor2_0):
            hs_list = [norm_hidden_states]
            attn_output = self.attn1.processor(
                self.attn1,
                hs_list,
                None,
                None,
                rotary_emb,
                rope_chunk_size=rotary_emb_chunk_size,
            )
            del hs_list
        else:
            attn_output = self.attn1(
                hidden_states=norm_hidden_states,
                rotary_emb=rotary_emb,
                rope_chunk_size=rotary_emb_chunk_size,
            )
        del norm_hidden_states
        if (
            (not torch.is_grad_enabled())
            and (not hidden_states.requires_grad)
            and (not attn_output.requires_grad)
        ):
            apply_gate_inplace(attn_output, gate_msa.to(dtype=attn_output.dtype))
            hidden_states.add_(attn_output)
        else:
            hidden_states = hidden_states + attn_output * gate_msa
        del attn_output, shift_msa, scale_msa, gate_msa

        # 2. Cross-attention
        norm_hidden_states = _chunked_norm(
            self.norm2, hidden_states, chunk_size=self._norm_chunk_size
        )
        if isinstance(self.attn2.processor, WanAttnProcessor2_0):
            if inference_mode:
                hs_list = [norm_hidden_states]
                ehs_list = [encoder_hidden_states]
                attn_output = self.attn2.processor(
                    self.attn2,
                    hs_list,
                    ehs_list,
                    None,
                    None,
                    rope_chunk_size=rotary_emb_chunk_size,
                )
                del hs_list, ehs_list
            else:
                attn_output = self.attn2(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    rope_chunk_size=rotary_emb_chunk_size,
                )
        else:
            attn_output = self.attn2(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                **(attention_kwargs or {}),
            )

        del norm_hidden_states
        if (
            (not torch.is_grad_enabled())
            and (not hidden_states.requires_grad)
            and (not attn_output.requires_grad)
        ):
            hidden_states.add_(attn_output)
        else:
            hidden_states = hidden_states + attn_output
        del attn_output

        # 3. Feed-forward
        norm_hidden_states = _chunked_modulated_norm(
            self.norm3,
            hidden_states,
            c_scale_msa,
            c_shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )
        if self._ff_chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ffn, norm_hidden_states, self._ff_chunk_dim, self._ff_chunk_size
            )
        else:
            ff_output = self.ffn(norm_hidden_states)
        del norm_hidden_states

        if (
            (not torch.is_grad_enabled())
            and (not hidden_states.requires_grad)
            and (not ff_output.requires_grad)
        ):
            apply_gate_inplace(ff_output, c_gate_msa.to(dtype=ff_output.dtype))
            hidden_states.add_(ff_output)
        else:
            hidden_states = hidden_states + ff_output * c_gate_msa
        del ff_output, c_shift_msa, c_scale_msa, c_gate_msa

        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

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
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
            "rotary_emb_chunk_size": None,
        },
        "balanced": {
            "ffn_chunk_size": 512,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
            "rotary_emb_chunk_size": 1024,
        },
        "aggressive": {
            "ffn_chunk_size": 256,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
            "rotary_emb_chunk_size": 256,
        },
    }

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)

    def set_chunking_profile(self, profile_name: str) -> None:
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. "
                f"Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )
        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name

        self._rotary_emb_chunk_size_default = p.get("rotary_emb_chunk_size", None)
        self._out_modulated_norm_chunk_size = p.get("out_modulated_norm_chunk_size", None)

        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

    def _get_rope_cpu_cache(self) -> Dict[Tuple[int, int, int], torch.Tensor]:
        if not hasattr(self, "_rope_cpu_cache"):
            self._rope_cpu_cache = {}
        return self._rope_cpu_cache

    def _build_rope_cached(
        self, hidden_states: torch.Tensor, *, rope_on_cpu: bool = False
    ) -> torch.Tensor:
        """
        Build RoPE table for a single video volume. If rope_on_cpu=True, cache the CPU
        table and rely on chunked CPU->GPU transfers during RoPE application.
        """
        _, _, t, h, w = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        key = (t // p_t, h // p_h, w // p_w)

        if rope_on_cpu:
            cache = self._get_rope_cpu_cache()
            if key not in cache:
                cache[key] = self.rope(
                    hidden_states, device=torch.device("cpu"), rope_on_cpu=True
                )
            return cache[key]

        return self.rope(hidden_states, device=hidden_states.device, rope_on_cpu=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        rope_on_cpu: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if rope_on_cpu is None:
            rope_on_cpu = (
                getattr(self, "_apex_forward_kwargs_defaults", {}) or {}
            ).get("rope_on_cpu", False)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
            rotary_emb_chunk_size = attention_kwargs.pop("rotary_emb_chunk_size", None)
        else:
            lora_scale = 1.0
            rotary_emb_chunk_size = None

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

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self._build_rope_cached(hidden_states, rope_on_cpu=bool(rope_on_cpu))

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )
        del encoder_hidden_states_image

        if rotary_emb_chunk_size is None:
            rotary_emb_chunk_size = getattr(self, "_rotary_emb_chunk_size_default", None)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_kwargs,
                    rotary_emb_chunk_size,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    attention_kwargs,
                    rotary_emb_chunk_size,
                )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        del temb
        hidden_states = _chunked_modulated_norm(
            self.norm_out,
            hidden_states,
            scale,
            shift,
            chunk_size=getattr(self, "_out_modulated_norm_chunk_size", None),
        )
        del scale, shift
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
