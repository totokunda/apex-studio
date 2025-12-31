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

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention import FeedForward
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from diffusers.models.normalization import FP32LayerNorm
from .attention import WanAttnProcessor2_0

from src.transformer.base import TRANSFORMERS_REGISTRY
import types

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Global variables for EasyCache
WANTF_GLOBAL_CNT = 0
WANTF_GLOBAL_NUM_STEPS = 100
WANTF_GLOBAL_THRESH = 0.1
WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN = 0
WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = True
WANTF_GLOBAL_K = None
WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN = None
WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN = None
WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_ODD = None
WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN = None
WANTF_GLOBAL_CACHE_EVEN = None
WANTF_GLOBAL_CACHE_ODD = None
WANTF_GLOBAL_RET_STEPS = 20


def reset_wantf_global_cache():
    """Reset all global cache variables."""
    global WANTF_GLOBAL_CNT, WANTF_GLOBAL_NUM_STEPS, WANTF_GLOBAL_THRESH
    global WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN, WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR
    global WANTF_GLOBAL_K, WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN
    global WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN, WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_ODD
    global WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN, WANTF_GLOBAL_CACHE_EVEN
    global WANTF_GLOBAL_CACHE_ODD, WANTF_GLOBAL_RET_STEPS

    WANTF_GLOBAL_CNT = 0
    WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN = 0
    WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = True
    WANTF_GLOBAL_K = None
    WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN = None
    WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN = None
    WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_ODD = None
    WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN = None
    WANTF_GLOBAL_CACHE_EVEN = None
    WANTF_GLOBAL_CACHE_ODD = None


def easycache_forward_(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ip_image_hidden_states: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    enhance_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    EasyCache-enabled forward pass for WanTransformer3DModel.
    Uses global variables to maintain caching state across calls.

    Args:
        hidden_states (Tensor): Input video tensor with shape [B, C, F, H, W]
        timestep (Tensor): Diffusion timesteps tensor of shape [B]
        encoder_hidden_states (Tensor): Text embeddings with shape [B, L, C]
        encoder_hidden_states_image (Tensor, optional): Image embeddings for I2V
        ip_image_hidden_states (Tensor, optional): IP image features
        return_dict (bool): Whether to return dict or tuple
        attention_kwargs (dict, optional): Additional attention arguments
        enhance_kwargs (dict, optional): Enhancement arguments

    Returns:
        Transformer2DModelOutput or tuple with denoised video tensor
    """
    global WANTF_GLOBAL_CNT, WANTF_GLOBAL_NUM_STEPS, WANTF_GLOBAL_THRESH
    global WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN, WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR
    global WANTF_GLOBAL_K, WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN
    global WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN, WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_ODD
    global WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN, WANTF_GLOBAL_CACHE_EVEN
    global WANTF_GLOBAL_CACHE_ODD, WANTF_GLOBAL_RET_STEPS

    # Get the number of latent channels (without condition channels)
    out_channels = self.config.out_channels

    # Store original raw input for end-to-end caching (only latent part, not condition)
    # hidden_states may be [B, C_latent + C_condition, F, H, W] when concatenated with latent_condition
    raw_input = hidden_states[:, :out_channels].clone()

    # Track which type of step (even=condition, odd=uncondition)
    is_even = WANTF_GLOBAL_CNT % 2 == 0

    # Only make decision on even (condition) steps
    if is_even:
        # Always compute first ret_steps and last steps
        if WANTF_GLOBAL_CNT < WANTF_GLOBAL_RET_STEPS or WANTF_GLOBAL_CNT >= (
            WANTF_GLOBAL_NUM_STEPS - 2
        ):
            WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = True
            WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN = 0
        else:
            if (
                WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN is not None
                and WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN is not None
            ):
                raw_input_change = (
                    (raw_input - WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN)
                    .flatten()
                    .abs()
                    .mean()
                )
                if WANTF_GLOBAL_K is not None:
                    output_norm = (
                        WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN.flatten().abs().mean()
                    )
                    pred_change = WANTF_GLOBAL_K * (raw_input_change / output_norm)
                    combined_pred_change = pred_change
                    WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN += combined_pred_change
                    if WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN < WANTF_GLOBAL_THRESH:
                        WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = False
                    else:
                        WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = True
                        WANTF_GLOBAL_ACCUMULATED_ERROR_EVEN = 0
                else:
                    WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = True
            else:
                WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR = True
        WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN = raw_input.clone()

    # Check if we can use cached output and return early
    if (
        is_even
        and not WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR
        and WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN is not None
    ):
        WANTF_GLOBAL_CNT += 1
        output = (raw_input + WANTF_GLOBAL_CACHE_EVEN).float()
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    elif (
        not is_even
        and not WANTF_GLOBAL_SHOULD_CALC_CURRENT_PAIR
        and WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_ODD is not None
    ):
        WANTF_GLOBAL_CNT += 1
        output = (raw_input + WANTF_GLOBAL_CACHE_ODD).float()
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # Continue with normal processing since we need to calculate
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if enhance_kwargs is not None:
        enhance_weight = enhance_kwargs.get("enhance_weight", None)
        num_frames = enhance_kwargs.get("num_frames", None)
        if enhance_weight is not None and num_frames is not None:
            self.set_enhance(enhance_weight, num_frames)

    if USE_PEFT_BACKEND:
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

    rotary_emb = self.rope(hidden_states)
    ip_hidden_states_len = 0
    if ip_image_hidden_states is not None:
        hidden_states_ip = self.patch_embedding(ip_image_hidden_states)
        hidden_states_ip = hidden_states_ip.flatten(2).transpose(1, 2)
        ip_hidden_states_len = hidden_states_ip.shape[1]
        rotary_emb_ip = self.rope(hidden_states, ip_image_hidden_states, time_index=0)
        rotary_emb = torch.concat([rotary_emb, rotary_emb_ip], dim=2)
    else:
        hidden_states_ip = None

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None

    (
        temb,
        timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        timestep_proj_ip,
    ) = self.condition_embedder(
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image,
        ip_image_hidden_states,
        timestep_seq_len=ts_seq_len,
    )

    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if timestep_proj_ip is not None:
        timestep_proj_ip = timestep_proj_ip.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states_image, encoder_hidden_states], dim=1
        )

    # Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                hidden_states_ip,
                timestep_proj_ip,
            )
            if hidden_states_ip is not None:
                hidden_states, hidden_states_ip = (
                    hidden_states[:, :-ip_hidden_states_len],
                    hidden_states[:, -ip_hidden_states_len:],
                )
    else:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                hidden_states_ip,
                timestep_proj_ip,
                rotary_emb_chunk_size=(
                    attention_kwargs.get("rotary_emb_chunk_size", None)
                    if attention_kwargs is not None
                    else None
                ),
            )
            if hidden_states_ip is not None:
                hidden_states, hidden_states_ip = (
                    hidden_states[:, :-ip_hidden_states_len],
                    hidden_states[:, -ip_hidden_states_len:],
                )

    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(
            2, dim=2
        )
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (
        self.norm_out(hidden_states.float()) * (1 + scale) + shift
    ).type_as(hidden_states)
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
        unscale_lora_layers(self, lora_scale)

    # Update cache and calculate change rates if needed
    if is_even:  # Condition path
        if WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN is not None:
            output_change = (
                (output - WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN).flatten().abs().mean()
            )
            if WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN is not None:
                input_change = (
                    (
                        WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN
                        - WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN
                    )
                    .flatten()
                    .abs()
                    .mean()
                )
                WANTF_GLOBAL_K = output_change / input_change
        WANTF_GLOBAL_PREV_PREV_RAW_INPUT_EVEN = WANTF_GLOBAL_PREVIOUS_RAW_INPUT_EVEN
        WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_EVEN = output.clone()
        WANTF_GLOBAL_CACHE_EVEN = output - raw_input
    else:  # Uncondition path
        WANTF_GLOBAL_PREVIOUS_RAW_OUTPUT_ODD = output.clone()
        WANTF_GLOBAL_CACHE_ODD = output - raw_input

    WANTF_GLOBAL_CNT += 1

    if not return_dict:
        return (output.float(),)

    return Transformer2DModelOutput(sample=output.float())


class LoRALinearLayerIP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor2_0
    _available_processors = [WanAttnProcessor2_0]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )

        self.norm_q = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=True
            )
            self.add_v_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=True
            )
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.kv_cache = None
        self.cond_size = None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat(
                [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
            )
            concatenated_bias = torch.cat(
                [self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]
            )
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )
        else:
            concatenated_weights = torch.cat(
                [self.to_k.weight.data, self.to_v.weight.data]
            )
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat(
                [self.add_k_proj.weight.data, self.add_v_proj.weight.data]
            )
            concatenated_bias = torch.cat(
                [self.add_k_proj.bias.data, self.add_v_proj.bias.data]
            )
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )

        self.fused_projections = True

    def init_ip_projections(
        self,
        train: bool = False,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        dim = self.inner_dim
        self.add_q_lora = LoRALinearLayerIP(
            dim, dim, rank=128, device=device, dtype=dtype
        )
        self.add_k_lora = LoRALinearLayerIP(
            dim, dim, rank=128, device=device, dtype=dtype
        )
        self.add_v_lora = LoRALinearLayerIP(
            dim, dim, rank=128, device=device, dtype=dtype
        )

        requires_grad = train
        for lora in [self.add_q_lora, self.add_k_lora, self.add_v_lora]:
            for param in lora.parameters():
                param.requires_grad = requires_grad

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            **kwargs,
        )


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
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

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
        ip_image_hidden_states: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):

        timestep_proj_ip = None
        temb_ip = None

        if ip_image_hidden_states is not None:
            timestep_ip = torch.zeros_like(timestep)
            timestep_ip = self.timesteps_proj(timestep_ip)
            temb_ip = self.time_embedder(timestep_ip).type_as(encoder_hidden_states)
            timestep_proj_ip = self.time_proj(self.act_fn(temb_ip))

        timestep = self.timesteps_proj(timestep)

        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype

        if (
            timestep.dtype != time_embedder_dtype
            and time_embedder_dtype != torch.int8
            and time_embedder_dtype != torch.uint8
            and time_embedder_dtype
            in [torch.float16, torch.float32, torch.float64, torch.bfloat16]
        ):
            timestep = timestep.to(time_embedder_dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_proj_ip,
        )


def rope_1d(
    dim: int,
    length: int,
    theta: float = 10000.0,
    start: int = 0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Return complex RoPE table of shape [length, dim//2] with positions = [start, ..., start+length-1].
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    base = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device).double() / dim)
    )
    pos = torch.arange(start, start + length, dtype=dtype, device=device)
    ang = torch.outer(pos, base)  # [length, dim//2]
    return torch.polar(torch.ones_like(ang, dtype=dtype), ang)


class WanRotaryPosEmbed(nn.Module):
    """
    3D RoPE split across (time, height, width) halves of the head-dim.
    Complex representation; returns [1, 1, num_patches, head_dim//2] table.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        time_offset: int = -1,  # use -1 to include sentinel row at t=-1
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.time_offset = time_offset

        # partition head dim: h_dim = w_dim = 2 * (head_dim // 6); rest is time
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        if any(d % 2 for d in (t_dim, h_dim, w_dim)):
            raise ValueError(
                f"t/h/w dims must be even, got t={t_dim}, h={h_dim}, w={w_dim}"
            )

        # sequence lengths
        t_len = max_seq_len + (
            1 if time_offset < 0 else 0
        )  # add sentinel if starting < 0
        h_len = max_seq_len
        w_len = max_seq_len

        # build separate tables (do NOT concat across dim=1; lengths differ)
        self.freqs_t = rope_1d(t_dim, t_len, theta=theta, start=time_offset)
        self.freqs_h = rope_1d(h_dim, h_len, theta=theta, start=0)
        self.freqs_w = rope_1d(w_dim, w_len, theta=theta, start=0)

        # cache half-dims for final concat
        self.t_half = t_dim // 2
        self.h_half = h_dim // 2
        self.w_half = w_dim // 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        ip_image_hidden_states: Optional[torch.Tensor] = None,
        time_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        If ip_image_hidden_states is provided, uses forward_ip (needs h_start, w_start; optional time_index).
        Otherwise uses forward_hidden_states.
        """
        if ip_image_hidden_states is None:
            return self.forward_hidden_states(hidden_states)
        else:
            return self.forward_ip(hidden_states, ip_image_hidden_states, time_index)

    def _patch_grid(self, T: int, H: int, W: int):
        pt, ph, pw = self.patch_size
        if (T % pt) or (H % ph) or (W % pw):
            raise ValueError(
                f"Input dims must be divisible by patch_size. "
                f"Got (T,H,W)=({T},{H},{W}), patch={self.patch_size}"
            )
        return (T // pt, H // ph, W // pw)  # (ppf, pph, ppw)

    def forward_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Build RoPE table for the current hidden_states volume (no IP crop).
        Returns: [1, 1, ppf*pph*ppw, (t_half+h_half+w_half)]
        """
        b, c, T, H, W = hidden_states.shape
        ppf, pph, ppw = self._patch_grid(T, H, W)

        # time starts at t=0 for “normal” frames; if we keep sentinel, skip index 0
        t_start = 1 if self.time_offset < 0 else 0
        if t_start + ppf > self.freqs_t.size(0):
            raise IndexError(
                f"time slice out of range: need {t_start+ppf} rows, have {self.freqs_t.size(0)}"
            )

        t = self.freqs_t[t_start : t_start + ppf]  # [ppf, t_half]
        h = self.freqs_h[:pph]  # [pph, h_half]
        w = self.freqs_w[:ppw]  # [ppw, w_half]

        # expand to 3D grid and concat along the last (feature) axis
        t3 = t.view(ppf, 1, 1, self.t_half).expand(ppf, pph, ppw, self.t_half)
        h3 = h.view(1, pph, 1, self.h_half).expand(ppf, pph, ppw, self.h_half)
        w3 = w.view(1, 1, ppw, self.w_half).expand(ppf, pph, ppw, self.w_half)

        freqs = torch.cat([t3, h3, w3], dim=-1)  # [ppf, pph, ppw, sum_halfs]
        return freqs.reshape(
            1, 1, ppf * pph * ppw, self.t_half + self.h_half + self.w_half
        ).to(hidden_states.device)

    def forward_ip(
        self,
        hidden_states: torch.Tensor,
        ip_image_hidden_states: torch.Tensor,
        time_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build RoPE table for an IP crop.
        - time_index: which time row to use for the crop.
            If None and time_offset<0, defaults to the sentinel row (index 0).
            If None and time_offset>=0, defaults to 0.
        Returns: [1, 1, ppf_ip*pph_ip*ppw_ip, (t_half+h_half+w_half)]
        """

        # main volume patch grid (not strictly needed here but kept for parity)
        _, _, T, H, W = hidden_states.shape
        ppf_main, pph_main, ppw_main = self._patch_grid(T, H, W)

        # IP volume patch grid (what we will emit)
        _, _, T_ip, H_ip, W_ip = ip_image_hidden_states.shape
        ppf_ip, pph_ip, ppw_ip = self._patch_grid(T_ip, H_ip, W_ip)

        # choose time row(s)
        if time_index is None:
            time_index = 0 if self.time_offset < 0 else 0

        if ppf_ip == 1:
            # match your manual path: take a single time row and broadcast
            if not (0 <= time_index < self.freqs_t.size(0)):
                raise IndexError(
                    f"time_index {time_index} out of range [0, {self.freqs_t.size(0)})"
                )
            t = self.freqs_t[time_index]  # [t_half]
            t3 = t.view(1, 1, 1, self.t_half).expand(
                ppf_ip, pph_ip, ppw_ip, self.t_half
            )
        else:
            # multi-frame crop: use a contiguous range
            if time_index + ppf_ip > self.freqs_t.size(0):
                raise IndexError(
                    f"time slice out of range: need {time_index+ppf_ip} rows, have {self.freqs_t.size(0)}"
                )
            t = self.freqs_t[time_index : time_index + ppf_ip]  # [ppf_ip, t_half]
            t3 = t.view(ppf_ip, 1, 1, self.t_half).expand(
                ppf_ip, pph_ip, ppw_ip, self.t_half
            )

        h = self.freqs_h[pph_main : pph_main + pph_ip]
        w = self.freqs_w[ppw_main : ppw_main + ppw_ip]

        h3 = h.view(1, pph_ip, 1, self.h_half).expand(
            ppf_ip, pph_ip, ppw_ip, self.h_half
        )
        w3 = w.view(1, 1, ppw_ip, self.w_half).expand(
            ppf_ip, pph_ip, ppw_ip, self.w_half
        )

        freqs_ip = torch.cat(
            [t3, h3, w3], dim=-1
        )  # [ppf_ip, pph_ip, ppw_ip, sum_halfs]
        return freqs_ip.reshape(
            1, 1, ppf_ip * pph_ip * ppw_ip, self.t_half + self.h_half + self.w_half
        ).to(hidden_states.device)


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
        use_enhance: bool = False,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor2_0(use_enhance=use_enhance),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor2_0(),
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        hidden_states_ip: torch.Tensor = None,
        timestep_proj_ip: torch.Tensor = None,
        rotary_emb_chunk_size: int | None = None,
    ) -> torch.Tensor:

        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)

        if hidden_states_ip is not None:
            (
                shift_msa_ip,
                scale_msa_ip,
                gate_msa_ip,
                c_shift_msa_ip,
                c_scale_msa_ip,
                c_gate_msa_ip,
            ) = (self.scale_shift_table + timestep_proj_ip.float()).chunk(6, dim=1)

            self.attn1.cond_size = hidden_states_ip.shape[1]

            norm_hidden_states_ip = (
                self.norm1(hidden_states_ip.float()) * (1 + scale_msa_ip) + shift_msa_ip
            ).type_as(hidden_states_ip)

            norm_hidden_states = torch.concat(
                [norm_hidden_states, norm_hidden_states_ip], dim=1
            )
            self.attn1.kv_cache = None

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            rotary_emb_chunk_size=rotary_emb_chunk_size,
        )

        if hidden_states_ip is not None:
            attn_output, attn_output_ip = (
                attn_output[:, : -self.attn1.cond_size],
                attn_output[:, -self.attn1.cond_size :],
            )

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            no_cache=True,
            rotary_emb_chunk_size=rotary_emb_chunk_size,
        )

        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        if hidden_states_ip is not None:
            gated_hidden_states_ip = (
                hidden_states_ip.float() + attn_output_ip.float() * gate_msa_ip
            ).type_as(hidden_states_ip)

            norm3_hidden_states_ip = (
                self.norm3(gated_hidden_states_ip.float()) * (1 + c_scale_msa_ip)
                + c_shift_msa_ip
            ).type_as(hidden_states_ip)

            ffn_output_ip = self.ffn(norm3_hidden_states_ip)
            hidden_states_ip = (
                gated_hidden_states_ip.float() + ffn_output_ip.float() * c_gate_msa_ip
            ).type_as(hidden_states_ip)
            hidden_states = torch.concat([hidden_states, hidden_states_ip], dim=1)

        return hidden_states


@TRANSFORMERS_REGISTRY("wan.base")
class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
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
        ip_adapter: bool = False,
        use_enhance: bool = False,
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
            pos_embed_seq_len=pos_embed_seq_len,
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
                    use_enhance,
                )
                for _ in range(num_layers)
            ]
        )

        if ip_adapter:
            self.init_ip_projections()

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def init_ip_projections(
        self,
        train: bool = False,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        for block in self.blocks:
            block.attn1.init_ip_projections(train=train, device=device, dtype=dtype)

    def set_enhance(self, enhance_weight: float, num_frames: int):
        for block in self.blocks:
            block.attn1.processor.set_enhance_weight(enhance_weight)
            block.attn1.processor.set_num_frames(num_frames)

    def enable_easy_cache(
        self,
        num_steps: int,
        thresh: float,
        ret_steps: int = 10,
        should_reset_global_cache: bool = True,
    ):
        """
        Enable EasyCache for accelerated inference using global state.

        Args:
            num_steps: Total number of diffusion steps (will be multiplied by 2 for cond/uncond pairs)
            thresh: Threshold for determining when to skip computation
            ret_steps: Number of initial steps to always compute (will be multiplied by 2)
        """
        global WANTF_GLOBAL_NUM_STEPS, WANTF_GLOBAL_THRESH, WANTF_GLOBAL_RET_STEPS

        # Reset global cache state
        if should_reset_global_cache:
            reset_wantf_global_cache()

        # Set global parameters
        WANTF_GLOBAL_NUM_STEPS = num_steps * 2  # Account for cond/uncond pairs
        WANTF_GLOBAL_THRESH = thresh
        WANTF_GLOBAL_RET_STEPS = ret_steps * 2  # Account for cond/uncond pairs

        # Replace forward method with cached version
        self.forward = types.MethodType(easycache_forward_, self)

    def disable_easy_cache(self):
        """
        Disable EasyCache and restore original forward method.
        """
        # Reset global cache state
        reset_wantf_global_cache()

        # Restore original forward method by rebinding the class method
        del self.forward  # Remove instance-level override, fall back to class method

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        ip_image_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        enhance_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if enhance_kwargs is not None:
            enhance_weight = enhance_kwargs.get("enhance_weight", None)
            num_frames = enhance_kwargs.get("num_frames", None)
            if enhance_weight is not None and num_frames is not None:
                self.set_enhance(enhance_weight, num_frames)

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

        rotary_emb = self.rope(hidden_states)
        ip_hidden_states_len = 0
        if ip_image_hidden_states is not None:
            hidden_states_ip = self.patch_embedding(ip_image_hidden_states)
            hidden_states_ip = hidden_states_ip.flatten(2).transpose(1, 2)
            ip_hidden_states_len = hidden_states_ip.shape[1]
            rotary_emb_ip = self.rope(
                hidden_states, ip_image_hidden_states, time_index=0
            )
            rotary_emb = torch.concat([rotary_emb, rotary_emb_ip], dim=2)
        else:
            hidden_states_ip = None

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_proj_ip,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            ip_image_hidden_states,
            timestep_seq_len=ts_seq_len,
        )

        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if timestep_proj_ip is not None:
            timestep_proj_ip = timestep_proj_ip.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    hidden_states_ip,
                    timestep_proj_ip,
                )
                if hidden_states_ip is not None:
                    hidden_states, hidden_states_ip = (
                        hidden_states[:, :-ip_hidden_states_len],
                        hidden_states[:, -ip_hidden_states_len:],
                    )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    hidden_states_ip,
                    timestep_proj_ip,
                    rotary_emb_chunk_size=(
                        attention_kwargs.get("rotary_emb_chunk_size", None)
                        if attention_kwargs is not None
                        else None
                    ),
                )
                if hidden_states_ip is not None:
                    hidden_states, hidden_states_ip = (
                        hidden_states[:, :-ip_hidden_states_len],
                        hidden_states[:, -ip_hidden_states_len:],
                    )

        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
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
