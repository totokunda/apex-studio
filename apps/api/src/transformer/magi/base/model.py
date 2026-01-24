# Copyright (c) 2025 SandAI. All Rights Reserved.
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
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from src.transformer.base import TRANSFORMERS_REGISTRY

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, Any

from .module import (
    CaptionEmbedder,
    FinalLinear,
    LearnableRotaryEmbeddingCat,
    TimestepEmbedder,
    MagiTransformerBlock,
    FusedLayerNorm,
)


@TRANSFORMERS_REGISTRY("magi.base")
class MagiTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    """MagiTransformer3D model for video diffusion.

    Args:
        config (MagiConfig): Transformer config
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
    """

    _no_split_modules = ["MagiTransformerBlock"]
    _keep_in_fp32_modules = [
        "norm_q",
        "norm_k",
        "norm2",
        "norm3",
        "norm_out",
        "patch_embedding",
        "timestep_embedding",
        "y_proj_xattn",
        "y_proj_adaln",
        "null_caption_embedding",
        "rope",
        "proj_out",
    ]

    @register_to_config
    def __init__(
        self,
        num_layers: int = 24,
        ffn_dim: int = 1024,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        eps: float = 1e-5,
        x_rescale_factor: float = 1,
        half_channel_vae: bool = False,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_size: int = 16,
        t_patch_size: int = 16,
        cond_hidden_ratio: float = 0.125,
        frequency_embedding_size: int = 256,
        xattn_cond_hidden_ratio: float = 0.125,
        num_query_groups: int = 8,
        gate_num_chunks: int = 2,
        distill: bool = False,
        zero_centered_gamma: bool = True,
        caption_channels: int = 4096,
        caption_max_length: int = 128,
        cond_gating_ratio: float = 1.0,
        gated_linear_unit: bool = False,
    ) -> None:
        super().__init__()

        hidden_dim = num_attention_heads * attention_head_dim
        cross_attention_dim = num_query_groups * attention_head_dim
        self.half_channel_vae = half_channel_vae

        self.patch_embedding = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(t_patch_size, patch_size, patch_size),
            stride=(t_patch_size, patch_size, patch_size),
            bias=False,
        )

        self.timestep_embedding = TimestepEmbedder(
            hidden_dim=hidden_dim,
            cond_hidden_ratio=cond_hidden_ratio,
            frequency_embedding_size=frequency_embedding_size,
        )

        self.caption_embedding = CaptionEmbedder(
            caption_channels=caption_channels,
            hidden_dim=hidden_dim,
            xattn_cond_hidden_ratio=xattn_cond_hidden_ratio,
            cond_hidden_ratio=cond_hidden_ratio,
            caption_max_length=caption_max_length,
        )

        self.rope = LearnableRotaryEmbeddingCat(
            hidden_dim // num_attention_heads,
            in_pixels=False,
        )

        # trm block
        self.blocks = torch.nn.ModuleList(
            [
                MagiTransformerBlock(
                    layer_number=i,
                    hidden_dim=hidden_dim,
                    cross_attention_dim=cross_attention_dim,
                    ffn_hidden_dim=ffn_dim,
                    cond_hidden_ratio=cond_hidden_ratio,
                    xattn_cond_hidden_ratio=xattn_cond_hidden_ratio,
                    num_attention_heads=num_attention_heads,
                    num_query_groups=num_query_groups,
                    gate_num_chunks=gate_num_chunks,
                    zero_centered_gamma=zero_centered_gamma,
                    cond_gating_ratio=cond_gating_ratio,
                    gated_linear_unit=gated_linear_unit,
                    eps=eps,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = FusedLayerNorm(
            zero_centered_gamma=zero_centered_gamma,
            hidden_dim=hidden_dim,
            eps=eps,
        )

        self.proj_out = FinalLinear(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            t_patch_size=t_patch_size,
            out_channels=out_channels,
        )

        self.distill = distill

    def get_null_caption_embeds(
        self, device: torch.device, num_videos_per_prompt: int = 1
    ) -> torch.Tensor:
        return self.caption_embedding.null_caption_embedding.to(device).repeat(
            num_videos_per_prompt, 1
        )

    def set_distill(self, distill: bool):
        self.distill = distill

    def generate_kv_range_for_uncondition(self, uncond_x) -> torch.Tensor:
        device = f"cuda:{torch.cuda.current_device()}"
        B, C, T, H, W = uncond_x.shape
        chunk_token_nums = (
            (T // self.config.t_patch_size)
            * (H // self.config.patch_size)
            * (W // self.config.patch_size)
        )

        k_chunk_start = torch.linspace(0, (B - 1) * chunk_token_nums, steps=B).reshape(
            (B, 1)
        )
        k_chunk_end = torch.linspace(
            chunk_token_nums, B * chunk_token_nums, steps=B
        ).reshape((B, 1))
        return (
            torch.concat([k_chunk_start, k_chunk_end], dim=1).to(torch.int32).to(device)
        )

    def unpatchify(self, x, H, W):
        return rearrange(
            x,
            "(T H W) N (pT pH pW C) -> N C (T pT) (H pH) (W pW)",
            H=H,
            W=W,
            pT=self.config.t_patch_size,
            pH=self.config.patch_size,
            pW=self.config.patch_size,
        ).contiguous()

    @torch.no_grad()
    def get_embedding_and_meta(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        caption_dropout_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_range: torch.Tensor,
        range_num: int,
        denoising_range_num: int,
        slice_point: int = 0,
        num_steps: int = 12,
        distill_interval: int = 4,
        **kwargs,
    ):
        """
        Forward embedding and meta for VideoDiT.
        NOTE: This function should only handle single card behavior.

        Input:
            x: (N, C, T, H, W). torch.Tensor of spatial inputs (images or latent representations of images)
            t: (N, denoising_range_num). torch.Tensor of diffusion timesteps
            y: (N * denoising_range_num, 1, L, C). torch.Tensor of class labels
            caption_dropout_mask: (N). torch.Tensor of whether to drop caption
            xattn_mask: (N * denoising_range_num, 1, L). torch.Tensor of xattn mask
            kv_range: (N * denoising_range_num, 2). torch.Tensor of kv range

        Output:
            x: (S, N, D). torch.Tensor of inputs embedding (images or latent representations of images)
            condition: (N, denoising_range_num, D). torch.Tensor of condition embedding
            condition_map: (S, N). torch.Tensor determine which condition to use for each token
            rope: (S, 96). torch.Tensor of rope
            y_xattn_flat: (total_token, D). torch.Tensor of y_xattn_flat
            cuda_graph_inputs: (y_xattn_flat, xattn_mask) or None. None means no cuda graph
                NOTE: y_xattn_flat and xattn_mask with static shape
            H: int. Height of the input
            W: int. Width of the input
            self_attn_params: dict. Packed sequence parameters for self_attention
            kv_cache_params: dict. Packed sequence parameters for kv_cache
            cross_attn_params: PackedCrossAttnParams. Packed sequence parameters for cross_attention
        """

        ###################################
        #          Part1: Embed x         #
        ###################################
        hidden_states = self.patch_embedding(hidden_states)  # [N, C, T, H, W]
        batch_size, _, T, H, W = hidden_states.shape

        # Prepare necessary variables
        frame_in_range = T // denoising_range_num
        prev_clean_T = frame_in_range * slice_point
        T_total = T + prev_clean_T

        ###################################
        #          Part2: rope            #
        ###################################
        # caculate rescale_factor for multi-resolution & multi aspect-ratio training
        # the base_size [16*16] is A predefined size based on data:(256x256)  vae: (8,8,4) patch size: (1,1,2)
        # This definition do not have any relationship with the actual input/model/setting.
        # ref_feat_shape is used to calculate innner rescale factor, so it can be float.
        rescale_factor = math.sqrt((H * W) / (16 * 16))
        rope = self.rope.get_embed(
            shape=[T_total, H, W],
            ref_feat_shape=[T_total, H / rescale_factor, W / rescale_factor],
        )
        # the shape of rope is (T*H*W, -1) aka (seq_length, head_dim), as T is the first dimension, we can directly cut it.
        rope = rope[-(T * H * W) :]

        ###################################
        #          Part3: Embed t         #
        ###################################
        assert (
            timestep.shape[0] == batch_size
        ), f"Invalid t shape, got {timestep.shape[0]} != {batch_size}"  # nolint
        assert (
            timestep.shape[1] == denoising_range_num
        ), f"Invalid t shape, got {timestep.shape[1]} != {denoising_range_num}"  # nolint
        t_flat = timestep.flatten()  # (N * denoising_range_num,)
        timestep_embed = self.timestep_embedding(t_flat)  # (N, D)

        if self.distill:
            distill_dt_scalar = 2
            if num_steps == 12:
                base_chunk_step = 4
                distill_dt_factor = (
                    base_chunk_step / distill_interval * distill_dt_scalar
                )
            else:
                distill_dt_factor = num_steps / 4 * distill_dt_scalar

            distill_dt = torch.ones_like(t_flat) * distill_dt_factor
            distill_dt_embed = self.timestep_embedding(distill_dt)
            timestep_embed = timestep_embed + distill_dt_embed

        timestep_embed = timestep_embed.reshape(
            batch_size, denoising_range_num, -1
        )  # (N, range_num, D)

        ######################################################
        # Part4: Embed y, prepare condition and y_xattn_flat #
        ######################################################
        # (N * denoising_range_num, 1, L, D)
        y_xattn, y_adaln = self.caption_embedding(
            encoder_hidden_states, self.training, caption_dropout_mask
        )

        assert attn_mask is not None
        attn_mask = attn_mask.squeeze(1).squeeze(1)

        # condition: (N, range_num, D)
        y_adaln = y_adaln.squeeze(1)  # (N, D)
        condition = timestep_embed + y_adaln.unsqueeze(1)

        assert condition.shape[0] == batch_size
        assert condition.shape[1] == denoising_range_num
        seqlen_per_chunk = (T * H * W) // denoising_range_num
        condition_map = torch.arange(
            batch_size * denoising_range_num, device=hidden_states.device
        )
        condition_map = torch.repeat_interleave(condition_map, seqlen_per_chunk)
        condition_map = (
            condition_map.reshape(batch_size, -1).transpose(0, 1).contiguous()
        )

        # y_xattn_flat: (total_token, D)
        encoder_hidden_states_flat = torch.masked_select(
            y_xattn.squeeze(1), attn_mask.unsqueeze(-1).bool()
        ).reshape(-1, y_xattn.shape[-1])

        ######################################################
        # Part5: Prepare cross_attn_params for cross_atten   #
        ######################################################
        # (N * denoising_range_num, L)
        attn_mask = attn_mask.reshape(attn_mask.shape[0], -1)
        y_index = torch.sum(attn_mask, dim=-1)
        clip_token_nums = H * W * frame_in_range

        cu_seqlens_q = (
            torch.Tensor([0] + ([clip_token_nums] * denoising_range_num * batch_size))
            .to(torch.int64)
            .to(hidden_states.device)
        )
        cu_seqlens_k = (
            torch.cat([y_index.new_tensor([0]), y_index])
            .to(torch.int64)
            .to(hidden_states.device)
        )
        cu_seqlens_q = cu_seqlens_q.cumsum(-1).to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.cumsum(-1).to(torch.int32)
        assert (
            cu_seqlens_q.shape == cu_seqlens_k.shape
        ), f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, cu_seqlens_k.shape: {cu_seqlens_k.shape}"

        xattn_q_ranges = torch.cat(
            [cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1
        )
        xattn_k_ranges = torch.cat(
            [cu_seqlens_k[:-1].unsqueeze(1), cu_seqlens_k[1:].unsqueeze(1)], dim=1
        )
        assert (
            xattn_q_ranges.shape == xattn_k_ranges.shape
        ), f"xattn_q_ranges.shape: {xattn_q_ranges.shape}, xattn_k_ranges.shape: {xattn_k_ranges.shape}"

        cross_attn_params = dict(
            q_range=xattn_q_ranges,
            kv_range=xattn_k_ranges,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=clip_token_nums,
            max_seqlen_kv=self.config.caption_max_length,
        )

        ##################################################
        #  Part6: Prepare core_atten related q/kv range  #
        ##################################################
        q_range = torch.cat(
            [cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1
        )
        flat_kv = torch.unique(kv_range, sorted=True)
        max_seqlen_k = (flat_kv[-1] - flat_kv[0]).cpu().item()

        self_attn_params = dict(
            q_range=q_range,
            kv_range=kv_range,
            max_seqlen_q=clip_token_nums,
            max_seqlen_k=max_seqlen_k,
            denoising_range_num=denoising_range_num,
        )

        kv_cache_params = dict(
            clip_token_nums=clip_token_nums,
            slice_point=slice_point,
            range_num=range_num,
            **kwargs,
        )

        return (
            hidden_states,
            condition,
            condition_map,
            rope,
            encoder_hidden_states_flat,
            H,
            W,
            kv_cache_params,
            cross_attn_params,
            self_attn_params,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        caption_dropout_mask: torch.Tensor | None = None,
        kv_range: torch.Tensor | None = None,
        kv_cache_params: Dict[str, Any] | None = None,
        range_num: int = 1,
        denoising_range_num: int = 1,
        slice_point: int = 0,
        num_steps: int = 12,
        distill_interval: int = 4,
        transformer_dtype: torch.dtype = torch.bfloat16,
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:

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

        hidden_states = hidden_states * self.config.x_rescale_factor

        if kv_cache_params is None:
            kv_cache_params = {}

        if self.half_channel_vae:
            assert hidden_states.shape[1] == 16
            hidden_states = torch.cat([hidden_states, hidden_states], dim=1)

        hidden_states = hidden_states.to(torch.float32)
        timestep = timestep.to(torch.float32)
        encoder_hidden_states = encoder_hidden_states.to(torch.float32)

        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            (
                hidden_states,
                condition,
                condition_map,
                rotary_pos_emb,
                encoder_hidden_states_flat,
                H,
                W,
                kv_cache_params_meta,
                cross_attn_params,
                self_attn_params,
            ) = self.get_embedding_and_meta(
                hidden_states,
                timestep,
                encoder_hidden_states,
                caption_dropout_mask,
                encoder_hidden_states_mask,
                kv_range,
                range_num=range_num,
                denoising_range_num=denoising_range_num,
                slice_point=slice_point,
                num_steps=num_steps,
                distill_interval=distill_interval,
                **kwargs,
            )

        kv_cache_params.update(kv_cache_params_meta)

        hidden_states = rearrange(
            hidden_states, "N C T H W -> (T H W) N C"
        ).contiguous()  # (thw, N, D)
        hidden_states = hidden_states.clone()

        hidden_states = hidden_states.to(transformer_dtype)
        condition = condition.to(transformer_dtype)
        encoder_hidden_states_flat = encoder_hidden_states_flat.to(transformer_dtype)

        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states_flat,
                condition=condition,
                condition_map=condition_map,
                self_attn_params=self_attn_params,
                cross_attn_params=cross_attn_params,
                kv_cache_params=kv_cache_params,
                rotary_pos_emb=rotary_pos_emb,
            )

        hidden_states = hidden_states.to(torch.float32)
        norm_hidden_states = self.norm_out(hidden_states)

        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            hidden_states = self.proj_out(norm_hidden_states)
        # N C T H W
        hidden_states = self.unpatchify(hidden_states, H, W)

        if self.config.half_channel_vae:
            assert hidden_states.shape[1] == 32
            hidden_states = hidden_states[:, :16]

        hidden_states = hidden_states / self.config.x_rescale_factor

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)
        else:
            return Transformer2DModelOutput(sample=hidden_states)
