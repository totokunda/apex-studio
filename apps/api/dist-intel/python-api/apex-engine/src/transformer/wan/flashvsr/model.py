import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import time
from typing import Dict, List, Optional, Tuple
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from .utils import Buffer_LQ4x_Proj, Causal_LQ4x_Proj

try:
    from block_sparse_attn import block_sparse_attn_func

    USE_BLOCK_SPARSE_ATTN = True
    SUPPORTS_SPARSE_ATTN = True
except ImportError:
    block_sparse_attn_func = None
    try:
        from .sparse_sage import sparse_sageattn

        SUPPORTS_SPARSE_ATTN = True
    except ImportError:
        sparse_sageattn = None
        SUPPORTS_SPARSE_ATTN = False
    USE_BLOCK_SPARSE_ATTN = False

from PIL import Image
import numpy as np
from src.attention import attention_register
from src.transformer.efficiency.mod import InplaceRMSNorm
from src.transformer.efficiency.ops import (
    apply_gate_inplace,
    apply_scale_shift_inplace,
    apply_wan_rope_inplace,
    chunked_feed_forward_inplace,
)


def _chunked_norm(
    norm_layer: nn.Module, hidden_states: torch.Tensor, chunk_size: Optional[int] = None
) -> torch.Tensor:
    """
    LayerNorm in chunks along the sequence dimension to reduce peak memory.

    Expects `hidden_states` to be `[batch, seq_len, dim]`.
    """
    if isinstance(norm_layer, nn.Identity):
        return hidden_states

    if chunk_size is None or hidden_states.ndim != 3:
        return norm_layer(hidden_states)

    b, s, _ = hidden_states.shape
    if s <= chunk_size:
        return norm_layer(hidden_states)

    out = torch.empty_like(hidden_states)
    for i in range(0, s, int(chunk_size)):
        end = min(i + int(chunk_size), s)
        out[:, i:end, :] = norm_layer(hidden_states[:, i:end, :])
    return out


def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Modulated norm with optional chunking. Uses in-place scale/shift to avoid allocating `(1 + scale)`.
    """
    if chunk_size is None or hidden_states.ndim != 3:
        out = norm_layer(hidden_states)
        apply_scale_shift_inplace(out, scale, shift)
        return out

    b, s, _ = hidden_states.shape
    if s <= chunk_size:
        out = norm_layer(hidden_states)
        apply_scale_shift_inplace(out, scale, shift)
        return out

    out = torch.empty_like(hidden_states)
    scale_per_token = scale.ndim == 3 and scale.shape[1] == s
    for i in range(0, s, int(chunk_size)):
        end = min(i + int(chunk_size), s)
        out_chunk = out[:, i:end, :]
        out_chunk.copy_(norm_layer(hidden_states[:, i:end, :]))
        if scale_per_token:
            apply_scale_shift_inplace(out_chunk, scale[:, i:end, :], shift[:, i:end, :])
        else:
            apply_scale_shift_inplace(out_chunk, scale, shift)
    return out


# ----------------------------
# Local / window masks
# ----------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec(
    block_h: int,
    block_w: int,
    win_h: int = 6,
    win_w: int = 6,
    include_self: bool = True,
    device=None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = torch.clamp(r_all - r_half, 0, H - win_h)
    end_r = start_r + win_h - 1
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(
    block_h: int,
    block_w: int,
    win_h: int = 6,
    win_w: int = 6,
    include_self: bool = True,
    device=None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert (
            F % wf == 0 and H % wh == 0 and W % ww == 0
        ), "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(
        windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]
    ):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def generate_draft_block_mask(
    batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None
):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, "s (h d) -> s h d", h=nheads)
    avgpool_k = rearrange(avgpool_k, "s (h d) -> s h d", h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = (
        local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    )
    local_attn_mask = rearrange(local_attn_mask, "x a y b -> (x a) (y b)")
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(
        local_attn_mask == False, -float("inf")
    )
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(
        mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen
    )  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


@torch.no_grad()
def generate_draft_block_mask_sage(
    batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None
):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"

    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_q = rearrange(avgpool_q, "s (h d) -> s h d", h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    D = avgpool_q.shape[-1]

    k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
    avgpool_k_split = torch.mean(k_w_split, dim=2)
    avgpool_k_refined = rearrange(
        avgpool_k_split, "s two d -> (s two) d", two=2
    )  # shape: (s*2, C)
    avgpool_k_refined = rearrange(
        avgpool_k_refined, "s (h d) -> s h d", h=nheads
    )  # shape: (s*2, h, d)
    k_heads_doubled = avgpool_k_refined.permute(1, 0, 2)  # shape: (h, s*2, d)

    k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
    scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1) / math.sqrt(D)
    scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2) / math.sqrt(D)
    scores = torch.cat([scores_1, scores_2], dim=-1)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = (scores.shape[2] // 2) // local_attn_mask.shape[1]

    local_attn_mask = (
        local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    )
    local_attn_mask = rearrange(local_attn_mask, "x a y b -> (x a) (y b)")
    local_attn_mask = local_attn_mask.repeat_interleave(2, dim=1)
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)

    assert (
        scores.shape == local_attn_mask.shape
    ), f"Scores shape {scores.shape} != Mask shape {local_attn_mask.shape}"

    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(
        local_attn_mask == False, -float("inf")
    )
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, "h (it s1) s2 -> (h it) s1 s2", it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1] - 1, topk)

    if apply_topk <= 0:
        mask_new = torch.zeros_like(flat, dtype=torch.bool).reshape(loop_num, s1, s2)
    else:
        thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[
            :, -1
        ]
        thresholds = thresholds.unsqueeze(1)
        mask_new = (flat > thresholds).reshape(loop_num, s1, s2)

    mask_new = rearrange(mask_new, "(h it) s1 s2 -> h (it s1) s2", it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


@torch.no_grad()
def generate_causal_block_mask(
    batch_size, nheads, seqlen, local_num, window_size, device="cuda", train_img=False
):
    i = torch.arange(seqlen, device=device).view(-1, 1)
    j = torch.arange(seqlen, device=device).view(1, -1)
    causal_mask = (j <= i) & (j >= i - local_num + 1)
    causal_mask[0, 1] = True
    causal_mask[:2, 2] = True
    if train_img:
        causal_mask[-1, :-1] = False
    causal_mask = (
        causal_mask.unsqueeze(1).unsqueeze(-1).repeat(1, window_size, 1, window_size)
    )
    causal_mask = rearrange(causal_mask, "a n1 b n2 -> (a n1) (b n2)")
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, nheads, 1, 1)
    return causal_mask


# ----------------------------
# Attention kernels
# ----------------------------
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
    attention_mask=None,
    return_KV=False,
):
    if (
        attention_mask is not None
        and attention_register.get_default() == "block_sparse"
        and SUPPORTS_SPARSE_ATTN
    ):
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        if USE_BLOCK_SPARSE_ATTN:
            q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
        head_mask_type = torch.tensor(
            [1] * num_heads, device=q.device, dtype=torch.int32
        )
        streaming_info = None
        base_blockmask = attention_mask
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        if USE_BLOCK_SPARSE_ATTN:
            x = block_sparse_attn_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_,
                max_seqlen_k_,
                p_dropout,
                deterministic=False,
                softmax_scale=None,
                is_causal=False,
                exact_streaming=False,
                return_attn_probs=False,
            ).unsqueeze(0)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            x = sparse_sageattn(
                q,
                k,
                v,
                mask_id=base_blockmask.to(torch.int8),
                is_causal=False,
                tensor_layout="HND",
            )
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = attention_register.call(q, k, v, attention_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    # Legacy (allocating) implementation kept for back-compat / debugging.
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(
            q=q, k=k, v=v, num_heads=self.num_heads, attention_mask=attention_mask
        )
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)

        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(
        self,
        x,
        freqs,
        f=None,
        h=None,
        w=None,
        local_num=None,
        topk=None,
        train_img=False,
        block_id=None,
        kv_len=None,
        is_full_block=False,
        is_stream=False,
        pre_cache_k=None,
        pre_cache_v=None,
        local_range=9,
        rotary_emb_chunk_size: Optional[int] = None,
        freqs_may_be_cpu: bool = True,
    ):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f == 2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f == 6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        # Canonicalize RoPE freqs to `[1, 1, S, D/2]` for `apply_wan_rope_inplace`.
        # Some call sites still provide the legacy shape `[S, 1, D/2]`.
        if freqs is not None:
            if freqs.dim() == 3:
                # [S, 1, D/2] -> [1, S, 1, D/2]
                freqs = freqs.unsqueeze(0)
            if freqs.dim() == 4 and freqs.shape[1] != 1 and freqs.shape[2] == 1:
                # [1, S, 1, D/2] -> [1, 1, S, D/2]
                freqs = freqs.permute(0, 2, 1, 3)

        # Project + normalize Q/K in-place (fresh tensors, safe to mutate).
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        self.norm_q(q)
        self.norm_k(k)

        # Apply RoPE in-place on [B, H, S, Dh] with optional chunking and CPU freqs support.
        q_4d = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_4d = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        apply_wan_rope_inplace(
            q_4d,
            freqs,
            chunk_size=rotary_emb_chunk_size,
            freqs_may_be_cpu=freqs_may_be_cpu,
        )
        apply_wan_rope_inplace(
            k_4d,
            freqs,
            chunk_size=rotary_emb_chunk_size,
            freqs_may_be_cpu=freqs_may_be_cpu,
        )

        # Back to [B, S, D] for the rest of this attention path.
        q = q_4d.transpose(1, 2).reshape(B, L, D)
        k = k_4d.transpose(1, 2).reshape(B, L, D)
        del q_4d, k_4d

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = WindowPartition3D.partition(q, win)
        k_w = WindowPartition3D.partition(k, win)
        v_w = WindowPartition3D.partition(v, win)
        del q, k, v

        seqlen = f // win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(
            q_w,
            "(b block_n) (block_s) d -> b (block_n block_s) d",
            block_n=block_n,
            block_s=block_s,
        )
        reorder_k = rearrange(
            k_w,
            "(b block_n) (block_s) d -> b (block_n block_s) d",
            block_n=block_n_kv,
            block_s=block_s,
        )
        reorder_v = rearrange(
            v_w,
            "(b block_n) (block_s) d -> b (block_n block_s) d",
            block_n=block_n_kv,
            block_s=block_s,
        )

        window_size = win[0] * h * w // 128

        if (
            self.local_attn_mask is None
            or self.local_attn_mask_h != h // 8
            or self.local_attn_mask_w != w // 8
            or self.local_range != local_range
        ):
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(
                h // 8,
                w // 8,
                local_range,
                local_range,
                include_self=True,
                device=k_w.device,
            )
            self.local_attn_mask_h = h // 8
            self.local_attn_mask_w = w // 8
            self.local_range = local_range

        if USE_BLOCK_SPARSE_ATTN:
            attention_mask = generate_draft_block_mask(
                B,
                self.num_heads,
                seqlen,
                q_w,
                k_w,
                topk=topk,
                local_attn_mask=self.local_attn_mask,
            )
        else:
            attention_mask = generate_draft_block_mask_sage(
                B,
                self.num_heads,
                seqlen,
                q_w,
                k_w,
                topk=topk,
                local_attn_mask=self.local_attn_mask,
            )

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)
        del reorder_q, reorder_k, reorder_v, attention_mask

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = rearrange(
            x,
            "b (block_n block_s) d -> (b block_n) (block_s) d",
            block_n=block_n,
            block_s=block_s,
        )
        x = WindowPartition3D.reverse(x, win, (f, h, w))
        x = x.view(B, f * h * w, D)

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)


class CrossAttention(nn.Module):
    """
    仅考虑文本 context；提供持久 KV 缓存。
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = InplaceRMSNorm(dim, eps=eps, elementwise_affine=True)

        self.attn = AttentionModule(self.num_heads)

        # 持久缓存
        self.cache_k = None
        self.cache_v = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        """ctx: [B, S_ctx, dim] —— 经过 text_embedding 之后的上下文"""
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_stream: bool = False):
        """
        y 即文本上下文（未做其他分支）。
        """
        q = self.norm_q(self.q(x))
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v

        x = self.attn(q, k, v)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

        # Memory/VRAM controls (disabled by default; configured via `FlashVSRModel.set_chunking_profile()`).
        self._ff_chunk_size: Optional[int] = None
        self._ff_chunk_dim: int = 1
        self._mod_norm_chunk_size: Optional[int] = None
        self._norm_chunk_size: Optional[int] = None
        self._rotary_emb_chunk_size: Optional[int] = None

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

    def set_rotary_emb_chunk_size(self, chunk_size: Optional[int]) -> None:
        self._rotary_emb_chunk_size = chunk_size

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        f,
        h,
        w,
        local_num=None,
        topk=None,
        train_img=False,
        block_id=None,
        kv_len=None,
        is_full_block=False,
        is_stream=False,
        pre_cache_k=None,
        pre_cache_v=None,
        local_range=9,
        rotary_emb_chunk_size: Optional[int] = None,
        freqs_may_be_cpu: bool = True,
    ):
        # Prefer a per-call override; otherwise use the per-block configured default.
        if rotary_emb_chunk_size is None:
            rotary_emb_chunk_size = self._rotary_emb_chunk_size

        x_dtype = x.dtype
        # Compute modulation in fp32, then cast to activation dtype to avoid fp32 intermediates.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            (self.modulation + t_mod.float()).to(dtype=x_dtype, device=x.device).chunk(
                6, dim=1
            )
        )
        del t_mod

        # 1) Self-attention with modulated norm (chunked) + in-place gating/add in inference.
        norm_x = _chunked_modulated_norm(
            self.norm1,
            x,
            scale_msa,
            shift_msa,
            chunk_size=self._mod_norm_chunk_size,
        )
        del scale_msa, shift_msa

        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            norm_x,
            freqs,
            f,
            h,
            w,
            local_num,
            topk,
            train_img,
            block_id,
            kv_len=kv_len,
            is_full_block=is_full_block,
            is_stream=is_stream,
            pre_cache_k=pre_cache_k,
            pre_cache_v=pre_cache_v,
            local_range=local_range,
            rotary_emb_chunk_size=rotary_emb_chunk_size,
            freqs_may_be_cpu=freqs_may_be_cpu,
        )
        del norm_x

        can_inplace = (
            (not torch.is_grad_enabled())
            and (not x.requires_grad)
            and (not self_attn_output.requires_grad)
        )
        if can_inplace:
            apply_gate_inplace(self_attn_output, gate_msa.to(dtype=self_attn_output.dtype))
            x.add_(self_attn_output)
        else:
            x = x + self_attn_output * gate_msa
        del self_attn_output, gate_msa

        # 2) Cross-attention with chunked norm, add in-place when safe.
        norm3_x = _chunked_norm(self.norm3, x, chunk_size=self._norm_chunk_size)
        cross_out = self.cross_attn(norm3_x, context, is_stream=is_stream)
        del norm3_x

        can_inplace2 = (
            (not torch.is_grad_enabled())
            and (not x.requires_grad)
            and (not cross_out.requires_grad)
        )
        if can_inplace2:
            x.add_(cross_out)
        else:
            x = x + cross_out
        del cross_out

        # 3) Feed-forward: modulated norm (chunked) -> optional chunked FF -> gated residual add.
        ff_x = _chunked_modulated_norm(
            self.norm2,
            x,
            scale_mlp,
            shift_mlp,
            chunk_size=self._mod_norm_chunk_size,
        )
        del scale_mlp, shift_mlp

        if self._ff_chunk_size is not None:
            ff_out = chunked_feed_forward_inplace(
                self.ffn, ff_x, chunk_dim=self._ff_chunk_dim, chunk_size=self._ff_chunk_size
            )
        else:
            ff_out = self.ffn(ff_x)
        # If we used the inference-only in-place FF path, `ff_out` is `ff_x` mutated.
        if ff_out is ff_x:
            del ff_x
        else:
            del ff_x

        can_inplace3 = (
            (not torch.is_grad_enabled())
            and (not x.requires_grad)
            and (not ff_out.requires_grad)
        )
        if can_inplace3:
            apply_gate_inplace(ff_out, gate_mlp.to(dtype=ff_out.dtype))
            x.add_(ff_out)
        else:
            x = x + ff_out * gate_mlp
        del ff_out, gate_mlp

        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        x_dtype = x.dtype
        shift, scale = (self.modulation + t_mod.float()).to(
            dtype=x_dtype, device=x.device
        ).chunk(2, dim=1)
        del t_mod

        norm_x = _chunked_norm(
            self.norm, x, chunk_size=getattr(self, "_norm_chunk_size", None)
        )
        apply_scale_shift_inplace(norm_x, scale, shift)
        del scale, shift
        out = self.head(norm_x)
        del norm_x
        return out


# ----------------------------
# FlashVSRModel (no image branch) — init 时即产生 KV 缓存
# ----------------------------
class FlashVSRModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        # init_context: torch.Tensor,     # <<<< 必填：在 __init__ 里用它生成 cross-attn KV 缓存
        has_image_input: bool = False,
        use_causal_lq4x_proj: bool = True,
        use_buffer_lq4x_proj: bool = False,
        lq4x_proj_in_dim: int = 1536,
        lq4x_proj_out_dim: int = 1536,
        lq4x_proj_layer_num: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        # patch embed
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        # text / time embed
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False

        # Chunking defaults (match WAN base: off unless explicitly enabled).
        self.set_chunking_profile("none")

        if use_causal_lq4x_proj:
            self.LQ_proj_in = Causal_LQ4x_Proj(
                in_dim=lq4x_proj_in_dim,
                out_dim=lq4x_proj_out_dim,
                layer_num=lq4x_proj_layer_num,
            )
        elif use_buffer_lq4x_proj:
            self.LQ_proj_in = Buffer_LQ4x_Proj(
                in_dim=lq4x_proj_in_dim,
                out_dim=lq4x_proj_out_dim,
                layer_num=lq4x_proj_layer_num,
            )
        else:
            self.LQ_proj_in = None

    # ----------------------------
    # Chunking profile presets
    # ----------------------------

    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        "none": {
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "head_norm_chunk_size": None,
            "rotary_emb_chunk_size": None,
        },
        "light": {
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "head_norm_chunk_size": 16384,
            "rotary_emb_chunk_size": None,
        },
        "balanced": {
            "ffn_chunk_size": 512,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "head_norm_chunk_size": 8192,
            "rotary_emb_chunk_size": 1024,
        },
        "aggressive": {
            "ffn_chunk_size": 256,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "head_norm_chunk_size": 4096,
            "rotary_emb_chunk_size": 256,
        },
    }

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunking_profile(self, profile_name: str) -> None:
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. "
                f"Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )

        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name

        self._rotary_emb_chunk_size_default = p.get("rotary_emb_chunk_size", None)

        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        self.set_chunk_norms(
            modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
            norm_chunk_size=p.get("norm_chunk_size", None),
            head_norm_chunk_size=p.get("head_norm_chunk_size", None),
        )

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 1) -> None:
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim)

    def set_chunk_norms(
        self,
        *,
        modulated_norm_chunk_size: Optional[int] = None,
        norm_chunk_size: Optional[int] = None,
        head_norm_chunk_size: Optional[int] = None,
    ) -> None:
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=modulated_norm_chunk_size,
                norm_chunk_size=norm_chunk_size,
            )
        # Store on head for `_chunked_norm` in `Head.forward`.
        self.head._norm_chunk_size = head_norm_chunk_size

    def set_rotary_emb_chunk_size(self, chunk_size: Optional[int]) -> None:
        self._rotary_emb_chunk_size_default = chunk_size
        for block in self.blocks:
            block.set_rotary_emb_chunk_size(chunk_size)

    # ----------------------------
    # RoPE CPU caching
    # ----------------------------

    def _get_rope_cpu_cache(self) -> Dict[Tuple[int, int, int, int], torch.Tensor]:
        if not hasattr(self, "_rope_cpu_cache"):
            self._rope_cpu_cache = {}
        return self._rope_cpu_cache

    def _build_rope_cached(
        self,
        *,
        f: int,
        h: int,
        w: int,
        device: torch.device,
        rope_on_cpu: bool,
        f_start: int = 0,
    ) -> torch.Tensor:
        """
        Build RoPE freqs for current patch-grid (f,h,w).

        Returns complex freqs shaped [1, 1, S, Dh/2] where S=f*h*w.
        When `rope_on_cpu=True`, the returned tensor stays on CPU and is cached.
        """
        key = (int(f), int(h), int(w), int(f_start))
        cache = self._get_rope_cpu_cache()
        if key not in cache:
            f0 = int(f_start)
            f1 = f0 + int(f)
            freqs_cpu = torch.cat(
                [
                    self.freqs[0][f0:f1].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(1, 1, f * h * w, -1)
            cache[key] = freqs_cpu

        freqs = cache[key]
        if rope_on_cpu:
            return freqs
        return freqs.to(device)

    # 可选：手动清空 / 重新初始化
    def clear_cross_kv(self):
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    @torch.no_grad()
    def reinit_cross_kv(self, new_context: torch.Tensor):
        ctx_txt = self.text_embedding(new_context)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)
        self._cross_kv_initialized = True

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        LQ_latents: Optional[List[torch.Tensor]] = None,
        train_img: bool = False,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_num: Optional[int] = None,
        is_full_block: bool = False,
        causal_idx: Optional[int] = None,
        is_stream: bool = False,
        pre_cache_k: Optional[List[torch.Tensor]] = None,
        pre_cache_v: Optional[List[torch.Tensor]] = None,
        cur_process_idx: int = 0,
        t_mod: torch.Tensor = None,
        t: torch.Tensor = None,
        local_range: int = 9,
        rope_on_cpu: Optional[bool] = None,
        rotary_emb_chunk_size: Optional[int] = None,
        **kwargs,
    ):
        if rope_on_cpu is None:
            rope_on_cpu = (
                getattr(self, "_apex_forward_kwargs_defaults", {}) or {}
            ).get("rope_on_cpu", False)
        if rotary_emb_chunk_size is None:
            rotary_emb_chunk_size = getattr(self, "_rotary_emb_chunk_size_default", None)

        # Streaming mode returns per-block caches; checkpointing doesn't handle tuple outputs.
        if is_stream:
            use_gradient_checkpointing = False

        # Back-compat: allow passing local_range via kwargs too.
        if "local_range" in kwargs:
            local_range = int(kwargs.pop("local_range", local_range))

        # time / text embeds (engine often precomputes these and passes them in)
        if t is None:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        if t_mod is None:
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 这里仍会嵌入 text（CrossAttention 若已有缓存会忽略它）
        # context = self.text_embedding(context)

        # 输入打补丁
        x, (f, h, w) = self.patchify(x)
        # window / masks hyperparams (match engine `inference_step`)
        win = (2, 8, 8)
        seqlen = f // win[0]
        if local_num is None:
            local_num = seqlen
        window_size = win[0] * h * w // 128
        square_num = window_size * window_size
        topk = max(int(square_num * float(topk_ratio)) - 1, 0)
        kv_len = int(kv_ratio)

        # RoPE 位置（分段）: match engine's segmented time indexing.
        f_start = 0 if int(cur_process_idx) == 0 else 4 + int(cur_process_idx) * 2

        # RoPE 3D (optionally cached on CPU and streamed to GPU in chunks).
        freqs = self._build_rope_cached(
            f=f,
            h=h,
            w=w,
            device=x.device,
            rope_on_cpu=bool(rope_on_cpu),
            f_start=f_start,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # blocks
        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                lq = LQ_latents[block_id]
                if (
                    (not torch.is_grad_enabled())
                    and (not x.requires_grad)
                    and (not lq.requires_grad)
                ):
                    x.add_(lq)
                else:
                    x = x + lq
                del lq

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            context,
                            t_mod,
                            freqs,
                            f,
                            h,
                            w,
                            local_num,
                            topk,
                            train_img,
                            block_id,
                            kv_len,
                            is_full_block,
                            False,
                            None,
                            None,
                            local_range,
                            rotary_emb_chunk_size,
                            bool(rope_on_cpu),
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        f,
                        h,
                        w,
                        local_num,
                        topk,
                        train_img,
                        block_id,
                        kv_len,
                        is_full_block,
                        False,
                        None,
                        None,
                        local_range,
                        rotary_emb_chunk_size,
                        bool(rope_on_cpu),
                        use_reentrant=False,
                    )
            else:
                out = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    f,
                    h,
                    w,
                    local_num,
                    topk,
                    train_img=train_img,
                    block_id=block_id,
                    kv_len=kv_len,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=(
                        pre_cache_k[block_id] if pre_cache_k is not None else None
                    ),
                    pre_cache_v=(
                        pre_cache_v[block_id] if pre_cache_v is not None else None
                    ),
                    local_range=local_range,
                    rotary_emb_chunk_size=rotary_emb_chunk_size,
                    freqs_may_be_cpu=bool(rope_on_cpu),
                )
                if is_stream:
                    x, last_k, last_v = out
                    if pre_cache_k is not None:
                        pre_cache_k[block_id] = last_k
                    if pre_cache_v is not None:
                        pre_cache_v[block_id] = last_v
                else:
                    x = out

        del freqs, t_mod
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        if is_stream:
            return x, pre_cache_k, pre_cache_v
        return x
