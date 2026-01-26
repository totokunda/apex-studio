from typing import Optional
import torch
from diffusers.models.attention import Attention
from enum import Enum, auto
from typing import Tuple
from src.attention.functions import attention_register


class SkipLayerStrategy(Enum):
    AttentionSkip = auto()
    AttentionValues = auto()
    Residual = auto()
    TransformerBlock = auto()


class LTXVideoAttentionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        skip_layer_mask: Optional[torch.FloatTensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if skip_layer_mask is not None:
            skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1)

        if (attention_mask is not None) and (not attn.use_tpu_flash_attention):
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.q_norm(query)

        if encoder_hidden_states is not None:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = attn.to_k(encoder_hidden_states)
            key = attn.k_norm(key)
        else:  # if no context provided do self-attention
            encoder_hidden_states = hidden_states
            key = attn.to_k(hidden_states)
            key = attn.k_norm(key)
            if attn.use_rope:
                key = attn.apply_rotary_emb(key, freqs_cis)
                query = attn.apply_rotary_emb(query, freqs_cis)

        value = attn.to_v(encoder_hidden_states)
        value_for_stg = value

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)

        hidden_states_a = attention_register.call(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states_a = hidden_states_a.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states_a = hidden_states_a.to(query.dtype)

        if (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.AttentionSkip
        ):
            hidden_states = hidden_states_a * skip_layer_mask + hidden_states * (
                1.0 - skip_layer_mask
            )
        elif (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.AttentionValues
        ):
            hidden_states = hidden_states_a * skip_layer_mask + value_for_stg * (
                1.0 - skip_layer_mask
            )
        else:
            hidden_states = hidden_states_a

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.Residual
            ):
                skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1, 1)

        if attn.residual_connection:
            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.Residual
            ):
                hidden_states = hidden_states + residual * skip_layer_mask
            else:
                hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
