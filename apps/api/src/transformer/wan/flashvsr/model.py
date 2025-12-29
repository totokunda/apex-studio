import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import time
from typing import Tuple, Optional, List
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from .utils import Buffer_LQ4x_Proj, Causal_LQ4x_Proj

try:
    from block_sparse_attn import block_sparse_attn_func
    USE_BLOCK_SPARSE_ATTN = True
except ImportError:
    block_sparse_attn_func = None
    from .sparse_sage import sparse_sageattn
    USE_BLOCK_SPARSE_ATTN = False
    
from PIL import Image
import numpy as np
from src.attention import attention_register

# ----------------------------
# Local / window masks
# ----------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec(block_h: int,
                                       block_w: int,
                                       win_h: int = 6,
                                       win_w: int = 6,
                                       include_self: bool = True,
                                       device=None) -> torch.Tensor:
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
    end_r   = start_r + win_h - 1
    start_c = torch.clamp(c_all - c_half, 0, W - win_w)
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask

@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                   block_w: int,
                                                   win_h: int = 6,
                                                   win_w: int = 6,
                                                   include_self: bool = True,
                                                   device=None) -> torch.Tensor:
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
    end_r   = start_r + win_h - 1
    start_c = c_all - c_half
    end_c   = start_c + win_w - 1
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
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              q_w, k_w, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1]-1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask

@torch.no_grad()
def generate_draft_block_mask_sage(batch_size, nheads, seqlen,
                                      q_w, k_w, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    
    k_w_split = k_w.view(k_w.shape[0], 2, 64, k_w.shape[2])
    avgpool_k_split = torch.mean(k_w_split, dim=2)
    avgpool_k_refined = rearrange(avgpool_k_split, 's two d -> (s two) d', two=2) # shape: (s*2, C)
    avgpool_k_refined = rearrange(avgpool_k_refined, 's (h d) -> s h d', h=nheads) # shape: (s*2, h, d)
    k_heads_doubled = avgpool_k_refined.permute(1, 0, 2) # shape: (h, s*2, d)
    
    k_heads_1, k_heads_2 = torch.chunk(k_heads_doubled, 2, dim=1)
    scores_1 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_1) / math.sqrt(D)
    scores_2 = torch.einsum("hld,hmd->hlm", q_heads, k_heads_2) / math.sqrt(D)
    scores = torch.cat([scores_1, scores_2], dim=-1)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = (scores.shape[2] // 2) // local_attn_mask.shape[1]
    
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.repeat_interleave(2, dim=1)
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    
    assert scores.shape == local_attn_mask.shape, \
        f"Scores shape {scores.shape} != Mask shape {local_attn_mask.shape}"
    
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1]-1, topk)
    
    if apply_topk <= 0:
        mask_new = torch.zeros_like(flat, dtype=torch.bool).reshape(loop_num, s1, s2)
    else:
        thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
        thresholds = thresholds.unsqueeze(1)
        mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
        
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask

@torch.no_grad()
def generate_causal_block_mask(batch_size, nheads, seqlen, local_num, window_size, device='cuda', train_img=False):
    i = torch.arange(seqlen, device=device).view(-1, 1)
    j = torch.arange(seqlen, device=device).view(1, -1)
    causal_mask = (j <= i) & (j >= i - local_num + 1)
    causal_mask[0,1] = True
    causal_mask[:2,2] = True
    if train_img:
        causal_mask[-1, :-1] = False
    causal_mask = causal_mask.unsqueeze(1).unsqueeze(-1).repeat(1, window_size, 1, window_size)
    causal_mask = rearrange(causal_mask, 'a n1 b n2 -> (a n1) (b n2)')
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, nheads, 1, 1)
    return causal_mask


# ----------------------------
# Attention kernels
# ----------------------------
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):
    if attention_mask is not None and attention_register.get_default() == "block_sparse":
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
        head_mask_type = torch.tensor([1]*num_heads, device=q.device, dtype=torch.int32)
        streaming_info = None
        base_blockmask = attention_mask
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        if USE_BLOCK_SPARSE_ATTN:
            x = block_sparse_attn_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
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
                q, k, v,
                mask_id=base_blockmask.to(torch.int8),
                is_causal=False,
                tensor_layout="HND"
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
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


# ----------------------------
# Norms & Blocks
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attention_mask=attention_mask)
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
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f==2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f==6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = WindowPartition3D.partition(q, win)
        k_w = WindowPartition3D.partition(k, win)
        v_w = WindowPartition3D.partition(v, win)

        seqlen = f//win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n, block_s=block_s)
        reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)
        reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)

        window_size = win[0]*h*w//128

        if self.local_attn_mask is None or self.local_attn_mask_h!=h//8 or self.local_attn_mask_w!=w//8 or self.local_range!=local_range:
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=k_w.device)
            self.local_attn_mask_h = h//8
            self.local_attn_mask_w = w//8
            self.local_range = local_range
        
        if USE_BLOCK_SPARSE_ATTN:
            attention_mask = generate_draft_block_mask(B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask)
        else:
            attention_mask = generate_draft_block_mask_sage(B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask)

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d', block_n=block_n, block_s=block_s)
        x = WindowPartition3D.reverse(x, win, (f, h, w))
        x = x.view(B, f*h*w, D)

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

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

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
    def __init__(self,):
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
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            input_x, freqs, f, h, w, local_num, topk, train_img, block_id,
            kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
            pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v, local_range = local_range)

        x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x), context, is_stream=is_stream)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
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
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


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
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # text / time embed
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False
        
        if use_causal_lq4x_proj:
            self.LQ_proj_in = Causal_LQ4x_Proj(in_dim=lq4x_proj_in_dim, out_dim=lq4x_proj_out_dim, layer_num=lq4x_proj_layer_num)
        elif use_buffer_lq4x_proj:
            self.LQ_proj_in = Buffer_LQ4x_Proj(in_dim=lq4x_proj_in_dim, out_dim=lq4x_proj_out_dim, layer_num=lq4x_proj_layer_num)
        else:
            self.LQ_proj_in = None

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
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                LQ_latents: Optional[List[torch.Tensor]] = None,
                train_img: bool = False,
                topk_ratio: Optional[float] = None,
                kv_ratio: Optional[float] = None,
                local_num: Optional[int] = None,
                is_full_block: bool = False,
                causal_idx: Optional[int] = None,
                **kwargs,
                ):
        # time / text embeds
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 这里仍会嵌入 text（CrossAttention 若已有缓存会忽略它）
        # context = self.text_embedding(context)

        # 输入打补丁
        x, (f, h, w) = self.patchify(x)
        B = x.shape[0]

        # window / masks 超参
        win = (2, 8, 8)
        seqlen = f//win[0]
        if local_num is None:
            local_random = random.random()
            if local_random < 0.3:
                local_num = seqlen - 3
            elif local_random < 0.4:
                local_num = seqlen - 4
            elif local_random < 0.5:
                local_num = seqlen - 2
            else:
                local_num = seqlen

        window_size = win[0]*h*w//128
        square_num = window_size*window_size
        topk_ratio = 2.0
        topk = min(max(int(square_num*topk_ratio), 1), int(square_num*seqlen)-1)

        if kv_ratio is None:
            kv_ratio = (random.uniform(0., 1.0)**2)*(local_num-2-2)+2
        kv_len = min(max(int(window_size*kv_ratio), 1), int(window_size*seqlen)-1)

        decay_ratio = random.uniform(0.7, 1.0)

        # RoPE 3D
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # blocks
        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x += LQ_latents[block_id]

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, f, h, w, local_num, topk,
                            train_img, block_id, kv_len, is_full_block, False,
                            None, None,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, f, h, w, local_num, topk,
                        train_img, block_id, kv_len, is_full_block, False,
                        None, None, 
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, f, h, w, local_num, topk,
                          train_img, block_id, kv_len, is_full_block, False,
                          None, None)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

