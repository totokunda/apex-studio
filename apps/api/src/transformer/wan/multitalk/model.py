# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from src.transformer.wan.base.attention import WanAttnProcessor2_0
from .attention import (
    MultiTalkWanAttnProcessor2_0,
)
from src.attention import attention_register
from einops import repeat
from functools import lru_cache
from einops import rearrange
from src.transformer.base import TRANSFORMERS_REGISTRY

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """
    Modulated layer norm with chunking to reduce peak memory.

    FP32LayerNorm internally does inputs.float() which creates a full fp32 copy.
    By processing in chunks along the sequence dimension, we reduce peak memory
    from O(seq_len) to O(chunk_size).
    """
    B, S, D = hidden_states.shape
    in_dtype = hidden_states.dtype

    # If disabled, run directly and return in the original dtype.
    if chunk_size is None:
        out = norm_layer(hidden_states) * (1 + scale) + shift
        return out.to(in_dtype) if out.dtype != in_dtype else out

    # If sequence is small enough, just do it directly
    if S <= chunk_size:
        out = norm_layer(hidden_states) * (1 + scale) + shift
        return out.to(in_dtype) if out.dtype != in_dtype else out

    # Pre-allocate output to avoid holding all chunks in memory
    out = torch.empty_like(hidden_states)

    # Check if scale/shift need per-token slicing
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

        # Norm + modulate directly into pre-allocated output
        normed = norm_layer(hs_chunk)
        out[:, i:end, :] = normed * (1 + scale_chunk) + shift_chunk
        del normed  # Free fp32 intermediate immediately

    return out


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
) -> torch.Tensor:
    """
    Chunked feed-forward that reduces peak memory when FFN intermediate activations are very large.
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
    Expects `hidden_states` to be `[batch, seq_len, dim]`.
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


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)

        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape=None,
        enable_sp=False,
        kv_seq=None,
    ) -> torch.Tensor:

        N_t, N_h, N_w = shape
        if not enable_sp:
            hidden_states = rearrange(
                hidden_states, "B (N_t S) C -> (B N_t) S C", N_t=N_t
            )

        # get q for hidden_state
        B, N, C = hidden_states.shape
        q = self.q_linear(hidden_states)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)

        # get kv from encoder_hidden_states
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))
        encoder_k, encoder_v = encoder_kv.unbind(0)

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        attn_bias = None
        hidden_states = attention_register.call(
            q, encoder_k, encoder_v, attn_bias=attn_bias, op=None
        )

        # linear transform
        x_output_shape = (B, N, C)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(x_output_shape)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.proj_drop(hidden_states)

        hidden_states = rearrange(hidden_states, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return hidden_states


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(
        self,
        head_dim,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    @lru_cache(maxsize=32)
    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float()
                / self.head_dim
            )
        )
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def rotate_half(self, x):
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, "n d -> 1 1 n d"), rearrange(sin, "n d -> 1 1 n d")
        x_ = (x_ * cos) + (self.rotate_half(x_) * sin)

        return x_.type_as(x)


class SingleStreamMutiAttention(SingleStreamAttention):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ) -> None:
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, self.class_interval)
        self.rope_h2 = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def normalize_and_scale(self, column, source_range, target_range, epsilon=1e-8):
        source_min, source_max = source_range
        new_min, new_max = target_range
        normalized = (column - source_min) / (source_max - source_min + epsilon)
        scaled = normalized * (new_max - new_min) + new_min
        return scaled

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape=None,
        x_ref_attn_map=None,
        human_num=None,
    ) -> torch.Tensor:

        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        if human_num == 1:
            return super().forward(hidden_states, encoder_hidden_states, shape)

        N_t, _, _ = shape
        hidden_states = rearrange(hidden_states, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # get q for hidden_state
        B, N, C = hidden_states.shape
        q = self.q_linear(hidden_states)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)

        max_values = x_ref_attn_map.max(1).values[:, None, None]
        min_values = x_ref_attn_map.min(1).values[:, None, None]
        max_min_values = torch.cat([max_values, min_values], dim=2)

        human1_max_value, human1_min_value = (
            max_min_values[0, :, 0].max(),
            max_min_values[0, :, 1].min(),
        )
        human2_max_value, human2_min_value = (
            max_min_values[1, :, 0].max(),
            max_min_values[1, :, 1].min(),
        )

        human1 = self.normalize_and_scale(
            x_ref_attn_map[0],
            (human1_min_value, human1_max_value),
            (self.rope_h1[0], self.rope_h1[1]),
        )
        human2 = self.normalize_and_scale(
            x_ref_attn_map[1],
            (human2_min_value, human2_max_value),
            (self.rope_h2[0], self.rope_h2[1]),
        )
        back = torch.full(
            (x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype
        ).to(human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices]  # N

        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))
        encoder_k, encoder_v = encoder_kv.unbind(0)

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
        per_frame[: per_frame.size(0) // 2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[per_frame.size(0) // 2 :] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame] * N_t, dim=0)
        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        hidden_states = attention_register.call(
            q,
            encoder_k,
            encoder_v,
            attn_bias=None,
            op=None,
        )

        # linear transform
        hidden_states_output_shape = (B, N, C)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(hidden_states_output_shape)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.proj_drop(hidden_states)

        # reshape x to origin shape
        hidden_states = rearrange(hidden_states, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return hidden_states


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(
            batch_size_vf, window_size_vf * blocks_vf * channels_vf
        )

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(
            batch_size_c * N_t, self.context_tokens, self.output_dim
        )

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_seq_len, in_features)
            )
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            # Only add positional embedding if sequence length matches
            if seq_len == self.pos_embed.shape[1]:
                encoder_hidden_states_image = (
                    encoder_hidden_states_image + self.pos_embed
                )
            else:
                # If sequence length doesn't match, interpolate or truncate pos_embed
                if seq_len < self.pos_embed.shape[1]:
                    pos_embed = self.pos_embed[:, :seq_len, :]
                else:
                    # Simple duplication for longer sequences (can be improved with interpolation)
                    repeat_factor = (
                        seq_len + self.pos_embed.shape[1] - 1
                    ) // self.pos_embed.shape[1]
                    pos_embed = self.pos_embed.repeat(1, repeat_factor, 1)[
                        :, :seq_len, :
                    ]
                encoder_hidden_states_image = encoder_hidden_states_image + pos_embed

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
        pos_embed_seq_len: Optional[int] = None,
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
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

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
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=freqs_dtype,
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
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
        return freqs


class WanMultiTalkTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        output_dim: int = 768,
        norm_input_visual: bool = True,
        class_range: int = 24,
        class_interval: int = 4,
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
            processor=MultiTalkWanAttnProcessor2_0(),
        )

        # 2. Cross-attention for text/image
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
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Audio cross-attention
        self.audio_attn2 = SingleStreamMutiAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
            class_range=class_range,
            class_interval=class_interval,
        )
        self.norm_x = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if norm_input_visual
            else nn.Identity()
        )

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # Chunked FFN (disabled by default). Chunking along dim=1 is typical for [B, seq, C].
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1

        # Chunked norms (disabled by default). These mainly mitigate FP32LayerNorm fp32-copy spikes.
        self._mod_norm_chunk_size: Optional[int] = (
            None  # used for modulated norms (norm1/norm3 + modulation)
        )
        self._norm_chunk_size: Optional[int] = (
            None  # used for plain norms (e.g., norm2)
        )

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        self._ff_chunk_size = chunk_size
        self._ff_chunk_dim = dim

    def set_chunk_norms(
        self,
        *,
        modulated_norm_chunk_size: Optional[int] = None,
        norm_chunk_size: Optional[int] = None,
    ) -> None:
        """
        Enable/disable chunking for norm operations inside the block.
        """
        self._mod_norm_chunk_size = modulated_norm_chunk_size
        self._norm_chunk_size = norm_chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
        human_num: Optional[int] = None,
        grid_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hs_dtype = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            (self.scale_shift_table + temb.float()).to(hs_dtype).chunk(6, dim=1)
        )

        # 1. Self-attention
        norm_hidden_states = _chunked_modulated_norm(
            self.norm1,
            hidden_states,
            scale_msa,
            shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )

        attn_output, x_ref_attn_map = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            grid_sizes=grid_sizes,
            ref_target_masks=ref_target_masks,
        )

        hidden_states = hidden_states + attn_output * gate_msa

        # 2. Cross-attention for text/image
        norm_hidden_states = _chunked_norm(
            self.norm2, hidden_states, chunk_size=self._norm_chunk_size
        )
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = hidden_states + attn_output

        # 3. Audio cross-attention
        if encoder_hidden_states_audio is not None:
            x_a = self.audio_attn2(
                self.norm_x(hidden_states),
                encoder_hidden_states=encoder_hidden_states_audio,
                shape=grid_sizes,
                human_num=human_num,
                x_ref_attn_map=x_ref_attn_map,
            )
            hidden_states = hidden_states + x_a

        # 4. Feed-forward
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

        hidden_states = hidden_states + ff_output * c_gate_msa

        return hidden_states


@TRANSFORMERS_REGISTRY("wan.multitalk")
class WanMultiTalkTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    r"""
    A Transformer model for MultiTalk video generation with audio inputs.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Number of attention heads.
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
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        image_dim (`int`, *optional*, defaults to `None`):
            Image embedding dimension.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections.
        audio_window (`int`, defaults to `5`):
            Audio window size for processing.
        intermediate_dim (`int`, defaults to `512`):
            Intermediate dimension for audio projection.
        output_dim (`int`, defaults to `768`):
            Output dimension for audio features.
        context_tokens (`int`, defaults to `32`):
            Number of context tokens for audio.
        vae_scale (`int`, defaults to `4`):
            VAE time downsample scale.
        norm_input_visual (`bool`, defaults to `True`):
            Whether to normalize input visual features.
        norm_output_audio (`bool`, defaults to `True`):
            Whether to normalize output audio features.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanMultiTalkTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

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
        pos_embed_seq_len: Optional[int] = None,
        # Audio-specific parameters
        audio_window: int = 5,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,
        norm_input_visual: bool = True,
        norm_output_audio: bool = True,
        class_range: int = 24,
        class_interval: int = 4,
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
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Audio projection model
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanMultiTalkTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    output_dim,
                    norm_input_visual,
                    class_range,
                    class_interval,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False
        self.audio_window = audio_window
        self.vae_scale = vae_scale

        # Default: no chunking unless explicitly enabled via a chunking profile.
        self.set_chunking_profile("none")

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        """
        Enable/disable chunked feed-forward on all transformer blocks.
        """
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)

    # ----------------------------
    # Chunking profile presets
    # ----------------------------

    # NOTE: This is used for inference-time memory control. Values may be Optional[int]
    # for chunk sizes, and may also include booleans for feature toggles.
    _CHUNKING_PROFILES: Dict[str, Dict[str, Any]] = {
        # No chunking anywhere.
        "none": {
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
            # Attention-map chunking (x_ref_attn_map). Old behavior is no chunking.
            "x_ref_attn_use_chunks": False,
            "x_ref_attn_chunk_size_x": None,
            "x_ref_attn_chunk_size_ref": None,
        },
        # Light chunking: only kicks in for very long sequences.
        "light": {
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
            # Keep fairly large chunks to reduce overhead.
            "x_ref_attn_use_chunks": True,
            "x_ref_attn_chunk_size_x": 4096,
            "x_ref_attn_chunk_size_ref": 4096,
        },
        # Balanced (close to current behavior, but configurable + optional).
        "balanced": {
            "ffn_chunk_size": 512,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
            "x_ref_attn_use_chunks": True,
            "x_ref_attn_chunk_size_x": 2048,
            "x_ref_attn_chunk_size_ref": 2048,
        },
        # Aggressive memory-saver: smaller chunks across the board.
        "aggressive": {
            "ffn_chunk_size": 256,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
            "x_ref_attn_use_chunks": True,
            "x_ref_attn_chunk_size_x": 1024,
            "x_ref_attn_chunk_size_ref": 1024,
        },
    }

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

        # Defaults for attention-map chunking (used by MultiTalkWanAttnProcessor2_0).
        # These are applied to each block's self-attention processor.
        x_ref_use_chunks = bool(p.get("x_ref_attn_use_chunks", True))
        x_ref_chunk_size_x = p.get("x_ref_attn_chunk_size_x", None)
        x_ref_chunk_size_ref = p.get("x_ref_attn_chunk_size_ref", None)

        # Apply to blocks.
        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

            # Apply attention-map chunking defaults if the processor supports it.
            try:
                processor = getattr(block.attn1, "processor", None)
                if processor is not None and hasattr(
                    processor, "set_x_ref_attn_chunking"
                ):
                    processor.set_x_ref_attn_chunking(
                        use_chunks=x_ref_use_chunks,
                        chunk_size_x=x_ref_chunk_size_x,
                        chunk_size_ref=x_ref_chunk_size_ref,
                    )
            except Exception:
                # Be resilient if the underlying attention implementation changes.
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
        human_num: Optional[int] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Calculate grid sizes for audio processing
        grid_sizes = (post_patch_num_frames, post_patch_height, post_patch_width)

        rotary_emb = self.rope(hidden_states)

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

        # Process with audio projection model
        audio_cond = encoder_hidden_states_audio.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(
            latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale
        )
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[
            :, :, :1, : middle_index + 1, ...
        ]
        latter_first_frame_audio_emb = rearrange(
            latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )
        latter_last_frame_audio_emb = latter_frame_audio_emb[
            :, :, -1:, middle_index:, ...
        ]
        latter_last_frame_audio_emb = rearrange(
            latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )
        latter_middle_frame_audio_emb = latter_frame_audio_emb[
            :, :, 1:-1, middle_index : middle_index + 1, ...
        ]
        latter_middle_frame_audio_emb = rearrange(
            latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )
        latter_frame_audio_emb_s = torch.concat(
            [
                latter_first_frame_audio_emb,
                latter_middle_frame_audio_emb,
                latter_last_frame_audio_emb,
            ],
            dim=2,
        )
        audio_embedding = self.audio_proj(
            first_frame_audio_emb_s, latter_frame_audio_emb_s
        )
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(
            hidden_states.dtype
        )
        # Combine audio embeddings if multiple humans

        if len(audio_embedding) > 1:
            audio_embedding = torch.concat(audio_embedding.split(1), dim=2)

        audio_embedding = audio_embedding.to(hidden_states.dtype)

        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
            token_ref_target_masks = nn.functional.interpolate(
                ref_target_masks,
                size=(post_patch_height, post_patch_width),
                mode="nearest",
            )
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = token_ref_target_masks > 0
            token_ref_target_masks = token_ref_target_masks.view(
                token_ref_target_masks.shape[0], -1
            )
            token_ref_target_masks = token_ref_target_masks.to(hidden_states.dtype)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    audio_embedding,
                    token_ref_target_masks,
                    human_num,
                    grid_sizes,
                )
        else:

            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    encoder_hidden_states_audio=audio_embedding,
                    ref_target_masks=token_ref_target_masks,
                    human_num=human_num,
                    grid_sizes=grid_sizes,
                )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = _chunked_modulated_norm(
            self.norm_out,
            hidden_states,
            scale,
            shift,
            chunk_size=getattr(self, "_out_modulated_norm_chunk_size", None),
        )
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
