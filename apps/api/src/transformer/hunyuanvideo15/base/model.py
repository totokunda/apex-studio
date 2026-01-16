# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.loaders import FromOriginalModelMixin

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from src.attention import attention_register

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _apply_rotary_emb_chunked(
    x: torch.Tensor,
    freqs_cis: tuple,
    chunk_size: Optional[int] = 8192,
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Apply rotary embeddings in chunks to reduce peak memory usage."""
    cos, sin = freqs_cis
    seq_len = x.shape[sequence_dim]
    
    # Treat None (or non-positive) as "no chunking"
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    # Small sequences don't need chunking
    if chunk_size is None or seq_len <= chunk_size:
        if sequence_dim == 1:
            cos_exp = cos[None, :, None, :]
            sin_exp = sin[None, :, None, :]
        else:
            cos_exp = cos[None, None, :, :]
            sin_exp = sin[None, None, :, :]
        
        cos_exp, sin_exp = cos_exp.to(x.device), sin_exp.to(x.device)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        return (x.float() * cos_exp + x_rotated.float() * sin_exp).to(x.dtype)
    
    # Pre-allocate output tensor to avoid list accumulation and final cat
    output = torch.empty_like(x)
    
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        
        if sequence_dim == 1:
            x_chunk = x[:, start:end]
            cos_chunk = cos[start:end][None, :, None, :]
            sin_chunk = sin[start:end][None, :, None, :]
        else:
            x_chunk = x[:, :, start:end]
            cos_chunk = cos[start:end][None, None, :, :]
            sin_chunk = sin[start:end][None, None, :, :]
        
        cos_chunk, sin_chunk = cos_chunk.to(x.device), sin_chunk.to(x.device)
        x_real, x_imag = x_chunk.reshape(*x_chunk.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        out_chunk = (x_chunk.float() * cos_chunk + x_rotated.float() * sin_chunk).to(x.dtype)
        
        # Write directly to pre-allocated output
        if sequence_dim == 1:
            output[:, start:end] = out_chunk
        else:
            output[:, :, start:end] = out_chunk
    
    return output


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: Optional[int]
) -> torch.Tensor:
    """
    Chunked feed-forward to reduce peak activation memory.

    Does NOT require `hidden_states.shape[chunk_dim]` to be divisible by `chunk_size`.
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


class HunyuanVideo15AttnProcessor2_0:
    _attention_backend = None
    _parallel_config = None
    _rope_chunk_size = 8192  # Configurable chunk size for RoPE

    def __init__(self, rope_chunk_size: Optional[int] = 8192):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideo15AttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self._rope_chunk_size = rope_chunk_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream (chunked)
        if image_rotary_emb is not None:
            query = _apply_rotary_emb_chunked(
                query, image_rotary_emb, 
                chunk_size=self._rope_chunk_size, 
                sequence_dim=1
            )
            key = _apply_rotary_emb_chunked(
                key, image_rotary_emb, 
                chunk_size=self._rope_chunk_size, 
                sequence_dim=1
            )

        # 4. Encoder condition QKV projection and normalization
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)


        hidden_states = attention_register.call(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,  # Already precomputed, no rebuild needed
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 7. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideo15PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideo15AdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15TimeEmbedding(nn.Module):
    r"""
    Time embedding for HunyuanVideo 1.5.

    Supports standard timestep embedding and optional reference timestep embedding for MeanFlow-based super-resolution
    models.

    Args:
        embedding_dim (`int`):
            The dimension of the output embedding.
    """

    def __init__(self, embedding_dim: int, use_meanflow: bool = False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_meanflow = use_meanflow
        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder_r = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(timesteps_proj_r.to(dtype=timestep.dtype))
            timesteps_emb = timesteps_emb + timesteps_emb_r

        return timesteps_emb


class HunyuanVideo15IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_width_ratio, activation_fn="linear-silu", dropout=mlp_drop_rate)

        self.norm_out = HunyuanVideo15AdaNorm(hidden_size, 2 * hidden_size)

        # Chunked FFN (disabled by default). Chunk along dim=1 for [B, seq, C].
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        self._ff_chunk_size = chunk_size
        self._ff_chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        normed = self.norm2(hidden_states)
        if self._ff_chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, normed, self._ff_chunk_dim, self._ff_chunk_size)
        else:
            ff_output = self.ff(normed)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideo15IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideo15IndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            
            # Fast path: skip mask if all positions valid
            if not attention_mask.all():
                # Use 1D additive mask (B, 1, 1, S) instead of full (B, 1, S, S)
                # SDPA broadcasts this efficiently
                self_attn_mask = torch.zeros(
                    batch_size, 1, 1, seq_len, 
                    dtype=hidden_states.dtype, 
                    device=hidden_states.device
                )
                self_attn_mask = self_attn_mask.masked_fill(
                    ~attention_mask.view(batch_size, 1, 1, seq_len), 
                    float('-inf')
                )

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        for block in self.refiner_blocks:
            if hasattr(block, "set_chunk_feed_forward"):
                block.set_chunk_feed_forward(chunk_size, dim=dim)


class HunyuanVideo15TokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        if hasattr(self.token_refiner, "set_chunk_feed_forward"):
            self.token_refiner.set_chunk_feed_forward(chunk_size, dim=dim)


class HunyuanVideo15RotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: List[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


class HunyuanVideo15ByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_features)
        self.act_fn = nn.GELU()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class HunyuanVideo15ImageProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(in_channels, in_channels)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(in_channels, hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class HunyuanVideo15TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideo15AttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        # Chunked FFN (disabled by default). Chunk along dim=1 for [B, seq, C].
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        self._ff_chunk_size = chunk_size
        self._ff_chunk_dim = dim

    def set_rope_chunk_size(self, rope_chunk_size: Optional[int]) -> None:
        # The attention processor stores the RoPE chunk size.
        proc = getattr(self.attn, "processor", None)
        if proc is not None and hasattr(proc, "_rope_chunk_size"):
            proc._rope_chunk_size = rope_chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        if self._ff_chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._ff_chunk_dim, self._ff_chunk_size)
            context_ff_output = _chunked_feed_forward(
                self.ff_context, norm_encoder_hidden_states, self._ff_chunk_dim, self._ff_chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)
            context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideo15Transformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in [HunyuanVideo1.5](https://huggingface.co/tencent/HunyuanVideo1.5).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_refiner_layers (`int`, defaults to `2`):
            The number of layers of refiner blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings in the model.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        pooled_projection_dim (`int`, defaults to `768`):
            The dimension of the pooled projection of the text embeddings.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideo15TransformerBlock",
        "HunyuanVideo15PatchEmbed",
        "HunyuanVideo15TokenRefiner",
    ]
    _repeated_blocks = [
        "HunyuanVideo15TransformerBlock",
        "HunyuanVideo15PatchEmbed",
        "HunyuanVideo15TokenRefiner",
    ]

    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        # No chunking.
        "none": {
            "ffn_chunk_size": None,
            "rope_chunk_size": None,
            "refiner_ffn_chunk_size": None,
        },
        # Light chunking: only kicks in for long sequences.
        "light": {
            "ffn_chunk_size": 2048,
            # Match historical default RoPE chunking to avoid regressions.
            "rope_chunk_size": 8192,
            "refiner_ffn_chunk_size": 2048,
        },
        # Balanced memory saver.
        "balanced": {
            "ffn_chunk_size": 512,
            "rope_chunk_size": 4096,
            "refiner_ffn_chunk_size": 512,
        },
        # Aggressive memory saver.
        "aggressive": {
            "ffn_chunk_size": 256,
            "rope_chunk_size": 2048,
            "refiner_ffn_chunk_size": 256,
        },
    }

    @register_to_config
    def __init__(
        self,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        # YiYi Notes: config based on target_size_config https://github.com/yiyixuxu/hy15/blob/main/hyvideo/pipelines/hunyuan_video_pipeline.py#L205
        target_size: int = 640,  # did not name sample_size since it is in pixel spaces
        task_type: str = "i2v",
        use_meanflow: bool = False,
        # Chunking profile controls memory-reducing chunking across RoPE + FFN (and refiner FFN).
        chunking_profile: str = "none",
        # Overrides (optional): allow forcing sizes regardless of profile.
        ffn_chunk_size: Optional[int] = None,
        ffn_chunk_dim: int = 1,
        rope_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideo15PatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.image_embedder = HunyuanVideo15ImageProjection(image_embed_dim, inner_dim)

        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(text_embed_2_dim, 2048, inner_dim)

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=use_meanflow)

        self.cond_type_embed = nn.Embedding(3, inner_dim)

        # 2. RoPE
        self.rope = HunyuanVideo15RotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideo15TransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        # Default: no chunking unless enabled explicitly.
        self._chunking_profile_name: str = "none"
        self._rope_chunk_size_default: Optional[int] = None

        # Apply profile first, then apply explicit overrides.
        self.set_chunking_profile(chunking_profile)
        if ffn_chunk_size is not None:
            self.set_chunk_feed_forward(ffn_chunk_size, dim=ffn_chunk_dim)
            # Keep refiner aligned with main FFN override by default.
            self.set_refiner_chunk_feed_forward(ffn_chunk_size, dim=ffn_chunk_dim)
        if rope_chunk_size is not None:
            self.set_rope_chunk_size(rope_chunk_size)

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        """Return available chunking profile names."""
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        """Enable/disable chunked FFN across all main transformer blocks."""
        for block in self.transformer_blocks:
            if hasattr(block, "set_chunk_feed_forward"):
                block.set_chunk_feed_forward(chunk_size, dim=dim)

    def set_refiner_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        """Enable/disable chunked FFN inside the text token refiner."""
        if hasattr(self.context_embedder, "set_chunk_feed_forward"):
            self.context_embedder.set_chunk_feed_forward(chunk_size, dim=dim)

    def set_rope_chunk_size(self, rope_chunk_size: Optional[int]) -> None:
        """Set RoPE application chunk size for all attention processors."""
        self._rope_chunk_size_default = rope_chunk_size
        for block in self.transformer_blocks:
            if hasattr(block, "set_rope_chunk_size"):
                block.set_rope_chunk_size(rope_chunk_size)

    def set_chunking_profile(self, profile_name: str) -> None:
        """
        Apply a predefined chunking profile across the whole model.

        Controls:
        - RoPE application chunk size inside attention processors
        - FFN chunking inside each main transformer block
        - FFN chunking inside the text token refiner
        """
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )
        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name

        self.set_rope_chunk_size(p.get("rope_chunk_size", None))
        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        self.set_refiner_chunk_feed_forward(p.get("refiner_ffn_chunk_size", None), dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        # qwen text embedding
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        # byt5 text embedding
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        # image embed
        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # reorder and combine text tokens: combine valid tokens first, then padding
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            # Concatenate: [valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],  # valid image
                        text_2[text_mask_2],  # valid byt5
                        text[text_mask],  # valid mllm
                        image[~image_mask],  # invalid image
                        torch.zeros_like(text_2[~text_mask_2]),  # invalid byt5 (zeroed)
                        torch.zeros_like(text[~text_mask]),  # invalid mllm (zeroed)
                    ],
                    dim=0,
                )
            )

            # Apply same reordering to attention masks
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)
        
        # Free intermediate lists immediately
        del new_encoder_hidden_states, new_encoder_attention_mask
        
        # CRITICAL: Skip masking entirely to enable Flash Attention path in SDPA
        # The encoder padding tokens are already zeroed out, so attending to them has minimal impact
        # Creating a mask forces SDPA to fall back to the math kernel which materializes 
        # the full S×S attention matrix (e.g., 93k×93k = 35GB+ per attention layer)
        # By passing None, SDPA uses Flash Attention which never materializes the full matrix
        precomputed_attn_mask = None
        # 4. Transformer blocks (use precomputed mask to avoid repeated allocation)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    precomputed_attn_mask,
                    image_rotary_emb,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    precomputed_attn_mask,
                    image_rotary_emb,
                )
        
        # Free the precomputed mask after all blocks
        del precomputed_attn_mask

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
