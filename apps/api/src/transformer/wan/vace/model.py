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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from src.transformer.wan.base.model import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
    _chunked_modulated_norm,
    _chunked_norm,
    _chunked_feed_forward,
)
from src.transformer.wan.base.attention import WanAttnProcessor2_0
from src.transformer.base import TRANSFORMERS_REGISTRY

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanVACETransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
        use_enhance: bool = False,
    ):
        super().__init__()

        # 1. Input projection
        self.proj_in = None
        if apply_input_projection:
            self.proj_in = nn.Linear(dim, dim)

        # 2. Self-attention
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

        # 3. Cross-attention
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

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. Output projection
        self.proj_out = None
        if apply_output_projection:
            self.proj_out = nn.Linear(dim, dim)

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

        Args:
            modulated_norm_chunk_size: chunk size along sequence dim for modulated norms (norm1/norm3).
                Set to None to disable chunking.
            norm_chunk_size: chunk size along sequence dim for plain norms (norm2).
                Set to None to disable chunking.
        """
        self._mod_norm_chunk_size = modulated_norm_chunk_size
        self._norm_chunk_size = norm_chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        # Compute scale/shift in fp32 for numerical stability, then cast to hidden_states dtype
        # to avoid fp32 intermediates during modulation (which would double memory).
        hs_dtype = control_hidden_states.dtype
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            (self.scale_shift_table + temb.float()).to(hs_dtype).chunk(6, dim=1)
        )

        # 1. Self-attention
        norm_hidden_states = _chunked_modulated_norm(
            self.norm1,
            control_hidden_states,
            scale_msa,
            shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb
        )

        control_hidden_states = control_hidden_states + attn_output * gate_msa

        # 2. Cross-attention
        norm_hidden_states = _chunked_norm(
            self.norm2, control_hidden_states, chunk_size=self._norm_chunk_size
        )
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        control_hidden_states = control_hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = _chunked_modulated_norm(
            self.norm3,
            control_hidden_states,
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

        control_hidden_states = control_hidden_states + ff_output * c_gate_msa

        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states


@TRANSFORMERS_REGISTRY("wan.vace")
class WanVACETransformer3DModel(
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
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "vace_patch_embedding",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock", "WanVACETransformerBlock"]
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
        vace_layers: List[int] = [0, 5, 10, 15, 20, 25, 30, 35],
        vace_in_channels: int = 96,
        ip_adapter: bool = False,
        use_enhance: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        if max(vace_layers) >= num_layers:
            raise ValueError(
                f"VACE layers {vace_layers} exceed the number of transformer layers {num_layers}."
            )
        if 0 not in vace_layers:
            raise ValueError("VACE layers must include layer 0.")

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )
        self.vace_patch_embedding = nn.Conv3d(
            vace_in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
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

        self.vace_blocks = nn.ModuleList(
            [
                WanVACETransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    apply_input_projection=i
                    == 0,  # Layer 0 always has input projection and is in vace_layers
                    apply_output_projection=True,
                    use_enhance=use_enhance,
                )
                for i in range(len(vace_layers))
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

        # Default: no chunking unless explicitly enabled via a chunking profile.
        self.set_chunking_profile("none")

    # ----------------------------
    # Chunking configuration
    # ----------------------------

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        """
        Enable/disable chunked feed-forward on all transformer blocks.

        Args:
            chunk_size: number of tokens to process per chunk along `dim`. Set to `None` to disable.
            dim: dimension to chunk along (typically 1 for [B, seq, C]).
        """
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)
        for block in self.vace_blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)

    # ----------------------------
    # Chunking profile presets
    # ----------------------------

    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        # No chunking anywhere.
        "none": {
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
        },
        # Light chunking: only kicks in for very long sequences.
        "light": {
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
        },
        # Balanced (close to current behavior, but configurable + optional).
        "balanced": {
            "ffn_chunk_size": 512,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
        },
        # Aggressive memory-saver: smaller chunks across the board.
        "aggressive": {
            "ffn_chunk_size": 256,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
        },
    }

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        """Return available chunking profile names."""
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunking_profile(self, profile_name: str) -> None:
        """
        Apply a predefined chunking profile across the whole model.

        This controls:
        - FFN chunking inside each transformer block
        - Norm chunking (modulated norms and plain norms)
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

        # Apply to blocks.
        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )
        for block in self.vace_blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

    def init_ip_projections(self, train: bool = False):
        for block in self.blocks:
            block.attn1.init_ip_projections(train=train)

    def set_enhance(self, enhance_weight: float, num_frames: int):
        for block in self.blocks:
            block.attn1.processor.set_enhance_weight(enhance_weight)
            block.attn1.processor.set_num_frames(num_frames)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        control_hidden_states: torch.Tensor = None,
        control_hidden_states_scale: torch.Tensor = None,
        ip_image_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        enhance_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if enhance_kwargs is not None:
            enhance_weight = enhance_kwargs.get("enhance_weight", None)
            num_frames = enhance_kwargs.get("num_frames", None)
            if enhance_weight is not None and num_frames is not None:
                self.set_enhance(enhance_weight, num_frames)

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

        if control_hidden_states_scale is None:
            control_hidden_states_scale = control_hidden_states.new_ones(
                len(self.config.vace_layers)
            )
        control_hidden_states_scale = torch.unbind(control_hidden_states_scale)
        if len(control_hidden_states_scale) != len(self.config.vace_layers):
            raise ValueError(
                f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} should be "
                f"equal to {len(self.config.vace_layers)}."
            )

        # 1. Rotary position embedding
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

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        control_hidden_states = self.vace_patch_embedding(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
        control_hidden_states_padding = control_hidden_states.new_zeros(
            batch_size,
            hidden_states.size(1) - control_hidden_states.size(1),
            control_hidden_states.size(2),
        )
        control_hidden_states = torch.cat(
            [control_hidden_states, control_hidden_states_padding], dim=1
        )

        # 3. Time embedding
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
        )

        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if timestep_proj_ip is not None:
            timestep_proj_ip = timestep_proj_ip.unflatten(1, (6, -1))

        # 4. Image embedding
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        control_hidden_states,
                        timestep_proj,
                        rotary_emb,
                    )
                )
                control_hidden_states_list.append(
                    (conditioning_states, control_hidden_states_scale[i])
                )
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
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

                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale
        else:
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    control_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
                control_hidden_states_list.append(
                    (conditioning_states, control_hidden_states_scale[i])
                )
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = block(
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

                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale

        # 6. Output norm, projection & unpatchify
        # Compute scale/shift in fp32 for numerical stability, then cast to hidden_states dtype.
        hs_dtype = hidden_states.dtype
        shift, scale = (
            (self.scale_shift_table + temb.unsqueeze(1)).to(hs_dtype).chunk(2, dim=1)
        )

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
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
