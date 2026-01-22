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
import math
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
from diffusers.models.cache_utils import CacheMixin

from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    FP32LayerNorm,
)
from src.attention import attention_register
from src.transformer.hunyuanvideo.base.attention import (
    HunyuanAvatarVideoAttnProcessor2_0,
)
from src.transformer.base import TRANSFORMERS_REGISTRY
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


class HunyuanAudioProjNet2(ModelMixin):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=4,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        print(
            f"input_dim: {self.input_dim}, intermediate_dim: {self.intermediate_dim}, context_tokens: {self.context_tokens}, output_dim: {self.output_dim}"
        )

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):

        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )
        context_tokens = self.norm(context_tokens)
        out_all = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return out_all


class HunyuanPerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=1024, heads=33):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head  # * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        import torch.nn.init as init

        init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            init.zeros_(self.to_out.bias)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, t, aa, D)
            latent (torch.Tensor): latent features
                shape (b, t, hw, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        return self.to_out(out)


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideoAdaNorm(nn.Module):
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


class HunyuanVideoTokenReplaceAdaLayerNormZero(nn.Module):
    def __init__(
        self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(
                embedding_dim, elementwise_affine=False, bias=False
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
        token_replace_emb: torch.Tensor,
        first_frame_num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        token_replace_emb = self.linear(self.silu(token_replace_emb))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        (
            tr_shift_msa,
            tr_scale_msa,
            tr_gate_msa,
            tr_shift_mlp,
            tr_scale_mlp,
            tr_gate_mlp,
        ) = token_replace_emb.chunk(6, dim=1)

        norm_hidden_states = self.norm(hidden_states)
        hidden_states_zero = (
            norm_hidden_states[:, :first_frame_num_tokens] * (1 + tr_scale_msa[:, None])
            + tr_shift_msa[:, None]
        )
        hidden_states_orig = (
            norm_hidden_states[:, first_frame_num_tokens:] * (1 + scale_msa[:, None])
            + shift_msa[:, None]
        )
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            tr_gate_msa,
            tr_shift_mlp,
            tr_scale_mlp,
            tr_gate_mlp,
        )


class HunyuanVideoConditionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        guidance_embeds: bool,
        image_condition_type: Optional[str] = None,
    ):
        super().__init__()

        self.image_condition_type = image_condition_type

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )

        self.guidance_embedder = None
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=embedding_dim
            )

        self.motion_exp = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim // 4
        )
        self.motion_pose = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim // 4
        )
        self.fps_proj = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        motion_exp: Optional[torch.Tensor] = None,
        motion_pose: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_proj = self.time_proj

        conditioning = self.timestep_embedder(time_proj(timestep))

        batch_size = pooled_projection.shape[0]

        if self.guidance_embedder is not None:
            guidance_emb = self.guidance_embedder(time_proj(guidance))
            conditioning = conditioning + guidance_emb

        if motion_exp is not None:
            motion_exp_emb = self.motion_exp(time_proj(motion_exp.view(-1))).view(
                batch_size, -1
            )
            conditioning = conditioning + motion_exp_emb

        if motion_pose is not None:
            motion_pose_emb = self.motion_pose(time_proj(motion_pose.view(-1))).view(
                batch_size, -1
            )
            conditioning = conditioning + motion_pose_emb

        if fps is not None:
            fps_emb = self.fps_proj(time_proj(fps))
            conditioning = conditioning + fps_emb

        conditioning = conditioning + self.text_embedder(pooled_projection)

        return conditioning


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
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
        self.ff = FeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
            dropout=mlp_drop_rate,
        )

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

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

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
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
                HunyuanVideoIndividualTokenRefinerBlock(
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
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
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
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
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

            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: List[int],
        theta: float = 256.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for i in range(3):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = torch.arange(
                0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32
            )
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(
                self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True
            )
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanAvatarVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(nn.Module):
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
            processor=HunyuanAvatarVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

        self.norm2_context = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.ff_context = FeedForward(
            hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )

        return hidden_states, encoder_hidden_states


@TRANSFORMERS_REGISTRY("hunyuanvideo.avatar")
class HunyuanAvatarVideoTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).

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
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.
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
        image_condition_type (`str`, *optional*, defaults to `None`):
            The type of image conditioning to use. If `None`, no image conditioning is used. If `latent_concat`, the
            image is concatenated to the latent stream. If `token_replace`, the image is used to replace first-frame
            tokens in the latent stream and apply conditioning.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        image_condition_type: Optional[str] = None,
        audio_seq_len: int = 10,
        audio_blocks: int = 5,
        audio_channels: int = 384,
        audio_intermediate_dim: int = 1024,
        audio_output_dim: int = 3072,
        audio_context_tokens: int = 4,
        audio_heads: int = 33,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim
        )
        self.ref_latents_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim
        )

        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim,
            num_attention_heads,
            attention_head_dim,
            num_layers=num_refiner_layers,
        )

        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim, pooled_projection_dim, guidance_embeds, image_condition_type
        )

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(
            patch_size, patch_size_t, rope_axes_dim, rope_theta
        )

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6
        )

        self.ref_latents_proj = nn.Linear(inner_dim, inner_dim)

        self.proj_out = nn.Linear(
            inner_dim, patch_size_t * patch_size * patch_size * out_channels
        )

        self.audio_projection = HunyuanAudioProjNet2(
            seq_len=audio_seq_len,
            blocks=audio_blocks,
            channels=audio_channels,
            intermediate_dim=audio_intermediate_dim,
            output_dim=audio_output_dim,
            context_tokens=audio_context_tokens,
        )

        self.gradient_checkpointing = False
        self.double_stream_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        self.single_stream_list = []
        self.double_stream_map = {
            str(i): j for j, i in enumerate(self.double_stream_list)
        }
        self.single_stream_map = {
            str(i): j + len(self.double_stream_list)
            for j, i in enumerate(self.single_stream_list)
        }

        self.audio_adapter_blocks = nn.ModuleList(
            [
                HunyuanPerceiverAttentionCA(
                    dim=audio_output_dim,
                    dim_head=audio_intermediate_dim,
                    heads=audio_heads,
                )
                for _ in range(
                    len(self.double_stream_list) + len(self.single_stream_list)
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        ref_latents: torch.Tensor = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        encoder_hidden_states_motion: Optional[torch.Tensor] = None,
        encoder_hidden_states_pose: Optional[torch.Tensor] = None,
        encoder_hidden_states_fps: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        encoder_hidden_states_face_mask: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        use_cache: bool = False,
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
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        # 1. RoPE
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 2. Conditional embeddings
        temb = self.time_text_embed(
            timestep,
            pooled_projections,
            guidance,
            encoder_hidden_states_motion,
            encoder_hidden_states_pose,
            encoder_hidden_states_fps,
        )

        audio_embeds = self.audio_projection(encoder_hidden_states_audio)

        ref_latents_first = ref_latents[:, :, :1].clone()
        hidden_states = self.x_embedder(hidden_states)
        ref_hidden_states = self.ref_latents_embedder(ref_latents)
        ref_hidden_states_first = self.x_embedder(ref_latents_first)

        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        encoder_hidden_states_length = encoder_hidden_states.shape[1]

        hidden_states = self.ref_latents_proj(ref_hidden_states) + hidden_states

        ref_length = ref_hidden_states_first.shape[1]
        hidden_states = torch.cat([ref_hidden_states_first, hidden_states], dim=1)
        hidden_states_len = hidden_states.shape[1]
        mask_len = hidden_states_len - ref_length

        if encoder_hidden_states_face_mask.shape[2] == 1:
            encoder_hidden_states_face_mask = encoder_hidden_states_face_mask.repeat(
                1, 1, num_frames, 1, 1
            )

        encoder_hidden_states_face_mask = torch.nn.functional.interpolate(
            encoder_hidden_states_face_mask,
            size=[num_frames, post_patch_height, post_patch_width],
            mode="nearest",
        )

        encoder_hidden_states_face_mask = (
            encoder_hidden_states_face_mask.view(-1, mask_len, 1)
            .repeat(1, 1, hidden_states.shape[2])
            .type_as(hidden_states)
        )

        cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, hidden_states_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = hidden_states_len + encoder_hidden_states_length
        max_seqlen_kv = max_seqlen_q

        # 4. Transformer blocks

        if not use_cache:
            for block_idx, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                )

                if block_idx in self.double_stream_list:

                    hidden_states_real = (
                        hidden_states[:, ref_length:]
                        .clone()
                        .view(batch_size, num_frames, -1, hidden_states.shape[2])
                    )
                    hidden_states_ref_real = torch.zeros_like(
                        hidden_states[:, :ref_length]
                    )

                    audio_embeds_padded = audio_embeds[:, :1].repeat(1, 3, 1, 1)
                    audio_embeds_all_insert = torch.cat(
                        [audio_embeds_padded, audio_embeds], dim=1
                    ).view(batch_size, num_frames, 16, hidden_states.shape[2])

                    double_idx = self.double_stream_map[str(block_idx)]
                    hidden_states_real = self.audio_adapter_blocks[double_idx](
                        audio_embeds_all_insert, hidden_states_real
                    ).view(batch_size, -1, hidden_states.shape[2])

                    hidden_states = hidden_states + torch.cat(
                        [
                            hidden_states_ref_real,
                            hidden_states_real * encoder_hidden_states_face_mask,
                        ],
                        dim=1,
                    )

            if len(self.single_transformer_blocks) > 0:
                for block_idx, block in enumerate(self.single_transformer_blocks):
                    if block_idx == len(self.single_transformer_blocks) - 1:
                        self.latent_cache = torch.cat(
                            (hidden_states, encoder_hidden_states), dim=1
                        )

                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,
                        image_rotary_emb,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                    )
        else:
            hidden_states = self.latent_cache
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :-encoder_hidden_states_length, ...],
                hidden_states[:, -encoder_hidden_states_length:, ...],
            )
            if len(self.single_transformer_blocks) > 0:
                for layer_num, block in enumerate(self.single_transformer_blocks):
                    if layer_num < (len(self.single_transformer_blocks) - 1):
                        continue
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,
                        image_rotary_emb,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                    )

        # 5. Output projection
        hidden_states = hidden_states[:, ref_length:]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
