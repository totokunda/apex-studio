import mlx.core as mx
from mlx import nn
from typing import Optional
from src.mlx.attention.base_attention import Attention
from src.mlx.modules.rotary import view_as_complex, view_as_real
import math


@mx.compile
def apply_rotary_emb(hidden_states: mx.array, freqs: mx.array):
    orig_dtype = hidden_states.dtype
    dtype = mx.float32
    hidden_states = mx.unflatten(hidden_states.astype(dtype), 3, (-1, 2))
    x_rotated = view_as_complex(hidden_states)
    x_out = view_as_real(x_rotated * freqs)
    x_out = mx.flatten(x_out, 3, 4)
    return x_out.astype(orig_dtype)


class WanAttnProcessor2_0:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None,
        attention_mask: mx.array | None = None,
        rotary_emb: mx.array | None = None,
    ):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        hidden_states = hidden_states.astype(attn.to_q.weight.dtype)
        encoder_hidden_states = encoder_hidden_states.astype(attn.to_k.weight.dtype)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape from (batch, seq, heads*dim_head) -> (batch, heads, seq, dim_head)
        query = mx.unflatten(query, 2, (attn.heads, -1)).transpose(0, 2, 1, 3)
        key = mx.unflatten(key, 2, (attn.heads, -1)).transpose(0, 2, 1, 3)
        value = mx.unflatten(value, 2, (attn.heads, -1)).transpose(0, 2, 1, 3)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        head_dim = attn.inner_dim // attn.heads
        scale = 1 / math.sqrt(head_dim)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = mx.unflatten(key_img, 2, (attn.heads, -1)).transpose(0, 2, 1, 3)
            value_img = mx.unflatten(value_img, 2, (attn.heads, -1)).transpose(
                0, 2, 1, 3
            )

            qd = query.dtype
            hidden_states_img = mx.fast.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                scale=scale,
            ).transpose(0, 2, 1, 3)
            hidden_states_img = mx.flatten(hidden_states_img, 2, 3)
            hidden_states_img = hidden_states_img.astype(query.dtype)

        # Compute attention in float32 for numerical parity, then cast back
        qd = query.dtype
        hidden_states = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
        ).transpose(0, 2, 1, 3)

        hidden_states = mx.flatten(hidden_states, 2, 3)
        hidden_states = hidden_states.astype(query.dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
