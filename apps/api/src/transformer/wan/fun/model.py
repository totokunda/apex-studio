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
from src.transformer.base import TRANSFORMERS_REGISTRY
from src.transformer.efficiency.ops import (
    apply_gate_inplace,
    apply_scale_shift_inplace,
    chunked_feed_forward_inplace,
)
from src.transformer.efficiency.mod import InplaceRMSNorm
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
            if (not torch.is_grad_enabled()) and (not encoder_hidden_states_image.requires_grad):
                encoder_hidden_states_image.add_(self.pos_embed)
            else:
                encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanCameraAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(WanCameraAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(
            in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0
        )

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


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

    def forward(
        self,
        hidden_states: Union[torch.Tensor, Tuple[int, int, int, int, int]],
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Build RoPE table for the given 5D input shape without allocating a dummy tensor.

        Accepts either:
        - a 5D tensor shaped [B, C, F, H, W], or
        - a shape tuple/list (B, C, F, H, W).
        """
        if torch.is_tensor(hidden_states):
            _, _, num_frames, height, width = hidden_states.shape
            target_device = device or hidden_states.device
        else:
            # Shape-only path (saves a large allocation vs `torch.zeros(shape)`).
            _, _, num_frames, height, width = hidden_states
            target_device = device
            if target_device is None:
                raise ValueError(
                    "WanRotaryPosEmbed.forward(shape=...) requires `device=`."
                )
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(target_device)
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
            processor=WanAttnProcessor2_0(use_enhance=use_enhance),
        )
        
        
        self.attn1.norm_q = InplaceRMSNorm(self.attn1.norm_q.weight.shape[0], eps=self.attn1.norm_q.eps, elementwise_affine=self.attn1.norm_q.elementwise_affine)
        self.attn1.norm_k = InplaceRMSNorm(self.attn1.norm_k.weight.shape[0], eps=self.attn1.norm_k.eps, elementwise_affine=self.attn1.norm_k.elementwise_affine)

# self.attn2 created above


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
        
        
        self.attn2.norm_q = InplaceRMSNorm(self.attn2.norm_q.weight.shape[0], eps=self.attn2.norm_q.eps, elementwise_affine=self.attn2.norm_q.elementwise_affine)
        self.attn2.norm_k = InplaceRMSNorm(self.attn2.norm_k.weight.shape[0], eps=self.attn2.norm_k.eps, elementwise_affine=self.attn2.norm_k.elementwise_affine)



        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

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
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:

        # Cast modulation to hidden_states dtype early to avoid fp32 intermediates during
        # gating (e.g. `attn_output * gate`) which would otherwise promote to fp32 and
        # increase peak memory.
        hs_dtype = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            (self.scale_shift_table + temb.float()).to(hs_dtype)
        ).chunk(6, dim=1)

        # 1. Self-attention with chunked modulated norm
        norm_hidden_states = _chunked_modulated_norm(
            self.norm1,
            hidden_states,
            scale_msa,
            shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )

        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb
        )

        if (
            (not torch.is_grad_enabled())
            and (not hidden_states.requires_grad)
            and (not attn_output.requires_grad)
        ):
            apply_gate_inplace(attn_output, gate_msa)
            hidden_states.add_(attn_output)
        else:
            hidden_states = hidden_states + attn_output * gate_msa
        del norm_hidden_states, attn_output

        # 2. Cross-attention
        norm_hidden_states = _chunked_norm(
            self.norm2, hidden_states, chunk_size=self._norm_chunk_size
        )
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        if (
            (not torch.is_grad_enabled())
            and (not hidden_states.requires_grad)
            and (not attn_output.requires_grad)
        ):
            hidden_states.add_(attn_output)
        else:
            hidden_states = hidden_states + attn_output
        del norm_hidden_states, attn_output

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
        if (
            (not torch.is_grad_enabled())
            and (not hidden_states.requires_grad)
            and (not ff_output.requires_grad)
        ):
            apply_gate_inplace(ff_output, c_gate_msa)
            hidden_states.add_(ff_output)
        else:
            hidden_states = hidden_states + ff_output * c_gate_msa
        del norm_hidden_states, ff_output

        return hidden_states


@TRANSFORMERS_REGISTRY("wan.fun")
class WanFunTransformer3DModel(
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
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        in_dim: int = 16,
        out_dim: int = 16,
        dim: int = 5120,
        hidden_size: int = 5120,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        cross_attn_type: str = "cross_attn",
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        add_control_adapter=True,
        in_dim_control_adapter=24,
        add_ref_conv=True,
        in_dim_ref_conv=16,
        use_enhance: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_dim = out_dim or in_dim

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_dim, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        self.attention_head_dim = attention_head_dim

        if add_control_adapter:
            self.control_adapter = WanCameraAdapter(
                in_dim_control_adapter,
                inner_dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
                num_residual_blocks=1,
            )

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(
                in_dim_ref_conv,
                inner_dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )

        self.add_ref_conv = add_ref_conv
        self.add_control_adapter = add_control_adapter

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

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_dim * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

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
        encoder_hidden_states_camera: Optional[torch.Tensor] = None,
        encoder_hidden_states_full_ref: Optional[torch.Tensor] = None,
        encoder_hidden_states_subject_ref: Optional[torch.Tensor] = None,
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

        # Update rope!!!
        hidden_states_shape = list(hidden_states.shape)

        hidden_states = self.patch_embedding(hidden_states)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        seq_len = math.ceil((height * width) / (p_h * p_w) * num_frames)
        full_ref_shape = 0

        if encoder_hidden_states_camera is not None and self.control_adapter:

            encoder_hidden_states_camera = self.control_adapter(
                encoder_hidden_states_camera
            )
            # concat camera features to hidden_states
            encoder_hidden_states_camera = encoder_hidden_states_camera.flatten(
                2
            ).transpose(1, 2)
            if (
                (not torch.is_grad_enabled())
                and (not hidden_states.requires_grad)
                and (not encoder_hidden_states_camera.requires_grad)
            ):
                hidden_states.add_(encoder_hidden_states_camera)
            else:
                hidden_states = hidden_states + encoder_hidden_states_camera
            del encoder_hidden_states_camera

        if encoder_hidden_states_full_ref is not None and hasattr(self, "ref_conv"):
            
            encoder_hidden_states_full_ref = self.ref_conv(
                encoder_hidden_states_full_ref
            )
            hidden_states_shape[2] += encoder_hidden_states_full_ref.shape[0]
            encoder_hidden_states_full_ref = encoder_hidden_states_full_ref.flatten(
                2
            ).transpose(1, 2)

            seq_len += encoder_hidden_states_full_ref.shape[1]

            # concat full ref features to hidden_states
            hidden_states = torch.cat(
                [encoder_hidden_states_full_ref, hidden_states], dim=1
            )
            full_ref_shape = encoder_hidden_states_full_ref.shape[1]
            del encoder_hidden_states_full_ref

            if timestep.dim() != 1 and timestep.size(1) < seq_len:
                pad_size = seq_len - timestep.size(1)
                last_elements = timestep[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                timestep = torch.cat([padding, timestep], dim=1)

        if encoder_hidden_states_subject_ref is not None:
            
            hidden_states_shape[2] += encoder_hidden_states_subject_ref.shape[2]
            encoder_hidden_states_subject_ref = self.patch_embedding(
                encoder_hidden_states_subject_ref
            )

            encoder_hidden_states_subject_ref = (
                encoder_hidden_states_subject_ref.flatten(2).transpose(1, 2)
            )

            seq_len += encoder_hidden_states_subject_ref.shape[1]
            # concat subject ref features to hidden_states
            hidden_states = torch.cat(
                [hidden_states, encoder_hidden_states_subject_ref], dim=1
            )

            if timestep.dim() != 1 and timestep.size(1) < seq_len:
                pad_size = seq_len - timestep.size(1)
                last_elements = timestep[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                timestep = torch.cat([timestep, padding], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )

        rotary_emb = self.rope(tuple(hidden_states_shape), device=hidden_states.device)

        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )
            del encoder_hidden_states_image

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

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

        if self.add_ref_conv and full_ref_shape > 0:
            hidden_states = hidden_states[:, full_ref_shape:]

        if encoder_hidden_states_subject_ref is not None:
            hidden_states = hidden_states[
                :, : -encoder_hidden_states_subject_ref[0].shape[1]
            ]

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
