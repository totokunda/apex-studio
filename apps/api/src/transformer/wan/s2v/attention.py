from diffusers.models.attention import Attention
from src.attention.functions import attention_register
from src.transformer.efficiency.list_clear import unwrap_single_item_list
from src.transformer.efficiency.ops import apply_wan_rope_inplace
import torch
from typing import Optional, Tuple, Union
import torch.nn.functional as F


def _get_qkv_projections(
    attn: Attention, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
):
    # encoder_hidden_states is only passed  or cross-attention
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


def _get_added_kv_projections(attn: Attention, encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(
            2, dim=-1
        )
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanS2VAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanS2VAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, int]]] = None,
    ) -> torch.Tensor:
        inference_mode = not torch.is_grad_enabled()
        hidden_states = unwrap_single_item_list(hidden_states)
        encoder_hidden_states = unwrap_single_item_list(encoder_hidden_states)
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

        # [B, S, H, D] -> [B, H, S, D]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            rotary_emb_chunk_size: Optional[int] = None
            if isinstance(rotary_emb, tuple):
                rotary_emb, rotary_emb_chunk_size = rotary_emb

            # WAN RoPE freqs are complex and broadcast across heads. The model's RoPE builder
            # commonly returns [B, S, H, D/2]; compress to [B, 1, S, D/2] to match
            # `apply_wan_rope_inplace`'s expected broadcast layout.
            if rotary_emb.dim() == 4:
                freqs = rotary_emb[:, :, 0, :].unsqueeze(1)  # [B, 1, S, D/2]
            elif rotary_emb.dim() == 3:
                freqs = rotary_emb.unsqueeze(1)  # [B, 1, S, D/2]
            else:
                raise ValueError(
                    f"Unexpected rotary_emb shape {tuple(rotary_emb.shape)}; expected 3D or 4D tensor."
                )

            apply_wan_rope_inplace(
                query,
                freqs,
                chunk_size=rotary_emb_chunk_size,
                freqs_may_be_cpu=True,
            )
            apply_wan_rope_inplace(
                key,
                freqs,
                chunk_size=rotary_emb_chunk_size,
                freqs_may_be_cpu=True,
            )
            freqs = None

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(
                attn, encoder_hidden_states_img
            )
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = attention_register.call(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)  # [B, S, H, D]
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(hidden_states)

        hidden_states = attention_register.call(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)  # [B, S, H, D]
        hidden_states = hidden_states.flatten(2, 3).type_as(hidden_states)

        if hidden_states_img is not None:
            if inference_mode and not hidden_states.requires_grad:
                hidden_states.add_(hidden_states_img)
            else:
                hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
