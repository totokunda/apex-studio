import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from src.attention import attention_register


# Copied from diffusers.models.transformers.transformer_wan._get_qkv_projections
def _get_qkv_projections(
    attn: "WanAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


# Copied from diffusers.models.transformers.transformer_wan._get_added_kv_projections
def _get_added_kv_projections(
    attn: "WanAttention", encoder_hidden_states_img: torch.Tensor
):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(
            2, dim=-1
        )
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(
                attn, encoder_hidden_states_img
            )
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = attention_register.call(
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = attention_register.call(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAnimateFaceBlockAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or"
                f" higher."
            )

    def __call__(
        self,
        attn: "WanAnimateFaceBlockCrossAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # encoder_hidden_states corresponds to the motion vec
        # attention_mask corresponds to the motion mask (if any)
        hidden_states = attn.pre_norm_q(hidden_states)
        encoder_hidden_states = attn.pre_norm_kv(encoder_hidden_states)

        # B --> batch_size, T --> reduced inference segment len, N --> face_encoder_num_heads + 1, C --> attn.dim
        B, T, N, C = encoder_hidden_states.shape

        query, key, value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(2, (attn.heads, -1))  # [B, S, H * D] --> [B, S, H, D]
        key = key.view(
            B, T, N, attn.heads, -1
        )  # [B, T, N, H * D_kv] --> [B, T, N, H, D_kv]
        value = value.view(B, T, N, attn.heads, -1)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # NOTE: the below line (which follows the official code) means that in practice, the number of frames T in
        # encoder_hidden_states (the motion vector after applying the face encoder) must evenly divide the
        # post-patchify sequence length S of the transformer hidden_states. Is it possible to remove this dependency?
        query = query.unflatten(1, (T, -1)).flatten(
            0, 1
        )  # [B, S, H, D] --> [B * T, S / T, H, D]
        key = key.flatten(0, 1)  # [B, T, N, H, D_kv] --> [B * T, N, H, D_kv]
        value = value.flatten(0, 1)

        hidden_states = attention_register.call(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = hidden_states.unflatten(0, (B, T)).flatten(1, 2)

        hidden_states = attn.to_out(hidden_states)

        if attention_mask is not None:
            # NOTE: attention_mask is assumed to be a multiplicative mask
            attention_mask = attention_mask.flatten(start_dim=1)
            hidden_states = hidden_states * attention_mask

        return hidden_states
