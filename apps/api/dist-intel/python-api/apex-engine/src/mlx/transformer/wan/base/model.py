import mlx.core as mx
from mlx import nn
from src.mlx.transformer.base import TRANSFORMERS_REGISTRY
from typing import Optional, Tuple, Union, Dict, Any
from .attention import WanAttnProcessor2_0
from src.mlx.attention.base_attention import Attention
from src.mlx.modules.layers import FP32LayerNorm, FeedForward, PixArtAlphaTextProjection
from src.mlx.modules.embedding import Timesteps, TimestepEmbedding
from src.mlx.modules.rotary import get_1d_rotary_pos_embed
import math
import numpy as np
from dataclasses import dataclass
from diffusers.configuration_utils import ConfigMixin, register_to_config
from src.mlx.mixins.from_model_mixin import FromModelMixin
from src.mlx.modules.outputs import Transformer2DModelOutput


class WanImageEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()
        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = mx.array(mx.zeros((1, pos_embed_seq_len, in_features)))
        else:
            self.pos_embed = None

    def __call__(self, encoder_hidden_states_image: mx.array) -> mx.array:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.reshape(
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

    def __call__(
        self,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_image: Optional[mx.array] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        enc_dtype = encoder_hidden_states.dtype

        if timestep_seq_len is not None:
            timestep = mx.unflatten(timestep, 0, (1, timestep_seq_len))

        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != mx.int8:
            timestep = timestep.astype(time_embedder_dtype)

        temb = self.time_embedder(timestep).astype(dtype=enc_dtype)
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
        freqs_dtype = mx.float32
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

        self._buffers = {"freqs": mx.concatenate(freqs, axis=1)}

    def get_freqs(self):
        return self._buffers["freqs"]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.get_freqs()
        a = self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6)  # 22
        b = self.attention_head_dim // 6  # 32
        freqs = mx.split(
            freqs,
            [a, a + b],
            axis=1,
        )

        freqs_f = freqs[0][:ppf].reshape(ppf, 1, 1, -1)
        freqs_f = mx.repeat(freqs_f, pph, axis=1)
        freqs_f = mx.repeat(freqs_f, ppw, axis=2)
        freqs_h = freqs[1][:pph].reshape(1, pph, 1, -1)
        freqs_h = mx.repeat(freqs_h, ppf, axis=0)
        freqs_h = mx.repeat(freqs_h, ppw, axis=2)
        freqs_w = freqs[2][:ppw].reshape(1, 1, ppw, -1)
        freqs_w = mx.repeat(freqs_w, ppf, axis=0)
        freqs_w = mx.repeat(freqs_w, pph, axis=1)

        freqs = mx.concatenate([freqs_f, freqs_h, freqs_w], axis=-1).reshape(
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
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, affine=False)
        self.attn1 = Attention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, affine=True) if cross_attn_norm else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, affine=False)

        self.scale_shift_table = mx.array(
            mx.random.normal(shape=(1, 6, dim), stream=mx.default_device()) / dim**0.5
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        rotary_emb: mx.array,
    ) -> mx.array:

        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                mx.split(
                    mx.expand_dims(self.scale_shift_table, 0) + temb.astype(mx.float32),
                    6,
                    axis=2,
                )
            )
            # batch_size, seq_len, 1, inner_dim
            shift_msa = mx.squeeze(shift_msa, 2)
            scale_msa = mx.squeeze(scale_msa, 2)
            gate_msa = mx.squeeze(gate_msa, 2)
            c_shift_msa = mx.squeeze(c_shift_msa, 2)
            c_scale_msa = mx.squeeze(c_scale_msa, 2)
            c_gate_msa = mx.squeeze(c_gate_msa, 2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                mx.split(self.scale_shift_table + temb.astype(mx.float32), 6, axis=1)
            )

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.astype(mx.float32)) * (1 + scale_msa) + shift_msa
        ).astype(hidden_states.dtype)

        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb
        )

        hidden_states = (
            hidden_states.astype(mx.float32) + attn_output * gate_msa
        ).astype(hidden_states.dtype)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.astype(mx.float32)).astype(
            hidden_states.dtype
        )
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.astype(mx.float32)) * (1 + c_scale_msa)
            + c_shift_msa
        ).astype(hidden_states.dtype)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.astype(mx.float32) + ff_output.astype(mx.float32) * c_gate_msa
        ).astype(hidden_states.dtype)

        return hidden_states


@TRANSFORMERS_REGISTRY("wan.base")
class WanTransformer3DModel(nn.Module, ConfigMixin, FromModelMixin):
    config_name = "config.json"
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
        self.blocks = [
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

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = mx.array(
            mx.random.normal(shape=(1, 2, inner_dim), stream=mx.default_device())
            / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_image: Optional[mx.array] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Tuple[mx.array], Transformer2DModelOutput]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hs = mx.transpose(hidden_states, axes=(0, 2, 3, 4, 1))  # N,D,H,W,C
        pe = self.patch_embedding(hs)  # NDHWC -> NDHWC
        pe = mx.transpose(pe, axes=(0, 4, 1, 2, 3))  # back to N,C,D,H,W
        hidden_states = mx.flatten(pe, start_axis=2, end_axis=4).transpose(0, 2, 1)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = mx.flatten(timestep)  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )

        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = mx.unflatten(timestep_proj, 2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = mx.unflatten(timestep_proj, 1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = mx.concatenate(
                [encoder_hidden_states_image, encoder_hidden_states], axis=1
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = mx.split(
                mx.expand_dims(self.scale_shift_table, 0) + mx.expand_dims(temb, 2),
                2,
                axis=2,
            )
            shift = mx.squeeze(shift, 2)
            scale = mx.squeeze(scale, 2)
        else:
            # batch_size, inner_dim
            shift, scale = mx.split(
                self.scale_shift_table + mx.expand_dims(temb, 1), 2, axis=1
            )

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.astype(hidden_states.dtype)
        scale = scale.astype(hidden_states.dtype)
        hidden_states_dtype = hidden_states.dtype

        hidden_states = (
            self.norm_out(hidden_states.astype(mx.float32)) * (1 + scale) + shift
        ).astype(hidden_states_dtype)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = mx.reshape(
            hidden_states,
            shape=(
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                p_t,
                p_h,
                p_w,
                -1,
            ),
        )

        hidden_states = mx.transpose(hidden_states, axes=(0, 7, 1, 4, 2, 5, 3, 6))
        output = mx.flatten(hidden_states, 6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
