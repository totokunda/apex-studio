from enum import Enum
from typing import Protocol

import torch

from src.transformer.ltx2.base2.rope import LTXRopeType, apply_rotary_emb_inplace
from src.attention import attention_register
from einops import rearrange

def wrapped_attention(qkv_list: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    return attention_register.call(q, k, v,  attn_mask=mask, dropout_p=0.0, is_causal=False)

class DBMRMSNorm(torch.nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x, in_place= True):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        y = x.float()
        y.pow_(2)
        y = y.mean(dim=-1, keepdim=True)
        y += self.eps
        y.rsqrt_()
        if in_place:
            x *=  y
        else:
            x = x * y
        x *= self.weight
        return x
        # return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    

class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.norm_q = DBMRMSNorm(inner_dim, eps=norm_eps)
        self.norm_k = DBMRMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x_list: torch.Tensor,
        context_list: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x_list[0]
        x_list.clear()
        context = None
        if context_list is not None:
            context = context_list[0]
            context_list.clear()
        
        
        q = self.to_q(x)
        context = x if context is None else context
        x = None
        k = self.to_k(context)
        v = self.to_v(context)
        del x, context
        context = None
        self.norm_q(q)
        self.norm_k(k)

        if pe is not None:
            apply_rotary_emb_inplace(q, pe, self.rope_type)
            apply_rotary_emb_inplace(k, pe if k_pe is None else k_pe, self.rope_type)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head)  
        k = rearrange(k, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.heads, d=self.dim_head)
        qkv_list = [q, k, v]
        q = k = v = None
        out = wrapped_attention(qkv_list, mask)

        out = rearrange(out, "b h s d -> b s (h d)")
        
        return self.to_out(out)
