# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import einops
import torch
from typing import Optional
from loguru import logger
import numpy as np
import torch.nn.functional as F
from src.attention.functions import attention_register

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    logger.warning("Could not load Sliding Tile Attention of FlexAttn.")

from .ssta_attention import ssta_3d_attention
from einops import rearrange


def flash_attn_no_pad(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv,
    key_padding_mask,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


@torch.compiler.disable
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    drop_rate: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """

    if attention_register.is_available("flash") and attn_mode == "flash":
        # flash mode (default)
        qkv = torch.stack([q, k, v], dim=2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()
        x = flash_attn_no_pad(
            qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None
        )
        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = q.transpose(1, 2)  # B * H * L * D
        key = k.transpose(1, 2)  # B * H * L * D
        value = v.transpose(1, 2)  # B * H * L * D

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), f"Integer attention mask must be between 0 and 1 for torch attention."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    f"Float attention mask is not implemented for torch attention."
                )
            attn_mask1 = einops.rearrange(attn_mask, "b l -> b 1 l 1")
            attn_mask2 = einops.rearrange(attn_mask1, "b 1 l 1 -> b 1 1 l")
            attn_mask = attn_mask1 & attn_mask2

        x = attention_register.call(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=drop_rate,
            is_causal=causal,
        )

        # transpose back
        x = x.transpose(1, 2)  # B * L * H * D
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out


@torch.compiler.disable
def parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
):
    return sequence_parallel_attention(
        q,
        k,
        v,
        img_q_len,
        img_kv_len,
        attn_mode,
        text_mask,
        attn_param=attn_param,
        block_idx=block_idx,
    )


def sequence_parallel_attention(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
):
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    if attention_register.is_available("sage") and attn_mode == "sageattn":
        from sageattention import sageattn

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = sageattn(
            query, key, value, tensor_layout="NHD", is_causal=False
        )
    elif attn_mode == "torch":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [
                torch.int64,
                torch.int32,
            ]:
                assert (
                    attn_mask.max() <= 1 and attn_mask.min() >= 0
                ), f"Integer attention mask must be between 0 and 1 for torch attention."
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(
                    f"Float attention mask is not implemented for torch attention."
                )

        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)  # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim

        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(
                attn_mask[b, q_idx] & attn_mask[b, kv_idx], score, float("-inf")
            )

        hidden_states = flex_attention(query, key, value, score_mod=score_mod)

        # transpose back
        hidden_states = hidden_states.transpose(1, 2)

    elif attention_register.is_available("flash") and attn_mode == "flash":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(
            qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
        )

    elif attention_register.is_available("flash3") and attn_mode == "flash3":
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(
            qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None
        )

    elif (
        attention_register.is_available("flex-block-attn")
        and attn_mode == "flex-block-attn"
    ):
        sparse_type = attn_param["attn_sparse_type"]  # sta/block_attn/ssta
        ssta_threshold = attn_param["ssta_threshold"]
        ssta_lambda = attn_param["ssta_lambda"]
        ssta_sampling_type = attn_param["ssta_sampling_type"]
        ssta_adaptive_pool = attn_param["ssta_adaptive_pool"]

        attn_pad_type = attn_param["attn_pad_type"]  # repeat/zero
        attn_use_text_mask = attn_param["attn_use_text_mask"]
        attn_mask_share_within_head = attn_param["attn_mask_share_within_head"]

        ssta_topk = attn_param["ssta_topk"]
        thw = attn_param["thw"]
        tile_size = attn_param["tile_size"]
        win_size = attn_param["win_size"][0].copy()

        def get_image_tile(tile_size):
            block_size = np.prod(tile_size)
            if block_size == 384:
                tile_size = (1, 16, 24)
            elif block_size == 128:
                tile_size = (1, 16, 8)
            elif block_size == 64:
                tile_size = (1, 8, 8)
            elif block_size == 16:
                tile_size = (1, 4, 4)
            else:
                raise ValueError(
                    f"Error tile_size {tile_size}, only support in [16, 64, 128, 384]"
                )
            return tile_size

        if thw[0] == 1:
            tile_size = get_image_tile(tile_size)
            win_size = [1, 1, 1]
        elif thw[0] <= 31:  # 16fps: 5 * 16 / 4 + 1 = 21; 24fps: 5 * 24 / 4 + 1 = 31
            ssta_topk = ssta_topk // 2

        # Concatenate and permute query, key, value to (B, H, S, D)
        query = torch.cat([query, encoder_query], dim=1).permute(0, 2, 1, 3)
        key = torch.cat([key, encoder_key], dim=1).permute(0, 2, 1, 3)
        value = torch.cat([value, encoder_value], dim=1).permute(0, 2, 1, 3)

        assert (
            query.shape[-1] == 128
        ), "The last dimension of query, key and value must be 128 for flex-block-attn."

        hidden_states = ssta_3d_attention(
            query,
            key,
            value,
            thw,
            topk=ssta_topk,
            tile_thw=tile_size,
            kernel_thw=win_size,
            text_len=encoder_sequence_length,
            sparse_type=sparse_type,
            threshold=ssta_threshold,
            lambda_=ssta_lambda,
            pad_type=attn_pad_type,
            text_mask=text_mask if attn_use_text_mask else None,
            sampling_type=ssta_sampling_type,
            adaptive_pool=ssta_adaptive_pool,
            mask_share_within_head=attn_mask_share_within_head,
        )
        hidden_states, sparse_ratio = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3)

    else:
        raise NotImplementedError(
            f"Unsupported attention mode: {attn_mode}. Only torch, flash, flash3, sageattn and flex-block-attn are supported."
        )

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states
