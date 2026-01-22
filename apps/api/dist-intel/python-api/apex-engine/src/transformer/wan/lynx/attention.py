import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional, List
from src.attention import attention_register
from src.transformer.efficiency.ops import apply_wan_rope_inplace
from src.transformer.efficiency.list_clear import unwrap_single_item_list

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from flash_attn_interface import flash_attn_varlen_func  # flash attn 3
except:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
        )  # flash attn 2
    except Exception:
        flash_attn_varlen_func = None


def flash_attention(query, key, value, q_lens, kv_lens, causal=False):
    """
    Args:
        query, key, value: [B, H, T, D_h]
        q_lens: list[int] per sequence query length
        kv_lens: list[int] per sequence key/value length
        causal: whether to use causal mask

    Returns:
        output: [B, H, T_q, D_h]
    """
    B, H, T_q, D_h = query.shape
    T_k = key.shape[2]
    device = query.device

    # Flatten: [B, H, T, D] -> [total_tokens, H, D]
    q = query.permute(0, 2, 1, 3).reshape(B * T_q, H, D_h)
    k = key.permute(0, 2, 1, 3).reshape(B * T_k, H, D_h)
    v = value.permute(0, 2, 1, 3).reshape(B * T_k, H, D_h)

    # Prepare cu_seqlens: prefix sum
    q_lens_tensor = torch.tensor(q_lens, device=device, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, device=device, dtype=torch.int32)

    cu_seqlens_q = torch.zeros(len(q_lens_tensor) + 1, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(
        len(kv_lens_tensor) + 1, device=device, dtype=torch.int32
    )

    cu_seqlens_q[1:] = torch.cumsum(q_lens_tensor, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(kv_lens_tensor, dim=0)

    max_seqlen_q = int(q_lens_tensor.max().item())
    max_seqlen_k = int(kv_lens_tensor.max().item())

    # Call FlashAttention varlen kernel
    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=causal
    )
    if not torch.is_tensor(out):  # flash attn 3
        out = out[0]

    # Restore shape: [total_q, H, D_h] -> [B, H, T_q, D_h]
    out = out.view(B, T_q, H, D_h).permute(0, 2, 1, 3).contiguous()

    return out


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        q_lens: List[int] = None,
        kv_lens: List[int] = None,
        rope_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = unwrap_single_item_list(hidden_states)
        encoder_hidden_states = unwrap_single_item_list(encoder_hidden_states)
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Drop large refs early to reduce peak memory (q/k/v now own the activations).
        del hidden_states, encoder_hidden_states

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            apply_wan_rope_inplace(
                query,
                rotary_emb,
                chunk_size=rope_chunk_size,
                freqs_may_be_cpu=True,
            )
            apply_wan_rope_inplace(
                key,
                rotary_emb,
                chunk_size=rope_chunk_size,
                freqs_may_be_cpu=True,
            )

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            del encoder_hidden_states_img

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = attention_register.call(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if flash_attn_varlen_func is not None:
            hidden_states = flash_attention(
                query, key, value, q_lens=q_lens, kv_lens=kv_lens
            )
        else:
            hidden_states = attention_register.call(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        del query, key, value

        if hidden_states_img is not None:
            if (
                (not torch.is_grad_enabled())
                and (not hidden_states.requires_grad)
                and (not hidden_states_img.requires_grad)
            ):
                hidden_states.add_(hidden_states_img)
            else:
                hidden_states = hidden_states + hidden_states_img
            del hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
