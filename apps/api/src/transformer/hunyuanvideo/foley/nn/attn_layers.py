import importlib.metadata
import math
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func = None, None, None
    index_first_axis = None
from packaging import version
from transformers.utils.import_utils import _is_package_available

from src.attention import attention_register
from .norm_layers import get_norm_layer

def reshape_for_broadcast(freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]], x: torch.Tensor, head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


class BasicAttentionLayer(nn.Module):
    def __init__(self, attn_mode="flash", deterministic=False):
        super().__init__()
        self.attn_mode = attn_mode
        self.deterministic = deterministic

    def set_attn_mode(self, new_mode):
        self.attn_mode = new_mode

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False


MEMORY_LAYOUT = {
    "self_flash": (
        lambda x: x,
        lambda x: x,
    ),
    "cross_flash": (
        lambda x: x,
        lambda x: x,
    ),
    "flash_torch_sp": (
        lambda x: x,
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


# Copyed from https://github.com/huggingface/transformers/blob/b873234cb649a24865021f0d598627ce2b24d34a/src/transformers/modeling_flash_attention_utils.py#L33C1-L57C6
def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copyed from https://github.com/huggingface/transformers/blob/b873234cb649a24865021f0d598627ce2b24d34a/src/transformers/utils/import_utils.py#L822
def is_flash_attn_greater_or_equal(library_version: str):
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)


def get_kv_seqlens_with_mask(attn_mask, k, v):
    indices_k, cu_seqlens_k, max_seqlen_k = _get_unpad_data(attn_mask)
    b, s1, a, d = k.shape
    k = index_first_axis(k.reshape(b * s1, a, d), indices_k)
    v = index_first_axis(v.reshape(b * s1, a, d), indices_k)
    kv = torch.stack([k, v], dim=1)
    return cu_seqlens_k, max_seqlen_k, kv


def get_q_seqlens(q):
    bs, s, a, d = q.shape
    cu_seqlens_q = torch.arange(0, (bs + 1) * s, step=s, dtype=torch.int32, device=q.device)
    q = q.reshape(bs * s, a, d)
    return cu_seqlens_q, s, q

def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None
):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    # x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch
    # x_unpad, indices, cu_seqlens, max_s
    unpad_results = unpad_input(
        x, key_padding_mask
    )

    if len(unpad_results) == 4:
        x_unpad, indices, cu_seqlens, max_s = unpad_results
    elif len(unpad_results) == 5:
        x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_results
    else:
        raise ValueError

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def attention(
    q,
    k,
    v,
    mode,
    drop_rate=0,
    attn_mask=None,
    cond_mask=None,
    causal=False,
    deterministic=False,
    cu_seqlens=None,
    max_seqlen=None,
    cu_seqlens_k=None,
    max_seqlen_k=None,
    img_seq_len=None,
):
    """
    Perform (self/cross) attention via the shared `attention_register`.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Historically one of 'self_flash', 'cross_flash', 'torch', 'vanilla'.
            In this implementation, `mode` is used as a *hint* for backend selection, but all computation
            routes through `attention_register.call(...)`.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask. Common shapes:
            - (b, s1): key-padding mask for cross-attn (1/True = keep, 0/False = mask)
            - (b, s, s1) or (b, 1, s, s1) or (b, a, s, s1): bool mask (True = keep) or additive bias.
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        deterministic (bool): Whether to use deterministic attention. (default: False)
        cu_seqlens (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        max_seqlen (int): The maximum sequence length in the batch of q.
        cu_seqlens_k (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_k (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    # NOTE:
    # - This helper is used by `hunyuanvideo/foley/model.py`, which passes q/k/v as (B, S, H, D).
    # - Our shared `attention_register` expects q/k/v as (B, H, S, D).
    # - We keep the legacy signature for compatibility with upstream code, but delegate the
    #   actual computation to the shared attention registry so global backend selection works.

    if isinstance(q, tuple):
        q = torch.cat(q, dim=1)
    if isinstance(k, tuple):
        k = torch.cat(k, dim=1)
    if isinstance(v, tuple):
        v = torch.cat(v, dim=1)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected q/k/v to be rank-4 tensors, got q={q.shape}, k={k.shape}, v={v.shape}")

    if causal and attn_mask is not None:
        # Mirrors the common SDPA constraint and avoids subtle backend-dependent behaviour.
        raise ValueError("`causal=True` cannot be combined with an explicit `attn_mask`.")

    b, sq, h, d = q.shape
    _, sk, hk, dk = k.shape
    if (h, d) != (hk, dk):
        raise ValueError(f"q heads/dim {(h, d)} must match k heads/dim {(hk, dk)}")
    if v.shape[:2] != (b, sk) or v.shape[2:] != (h, d):
        raise ValueError(f"v must have shape (B, S_k, H, D)=({b},{sk},{h},{d}), got {v.shape}")

    # Normalize masks to SDPA-style (broadcastable to [B, H, S_q, S_k]).
    am = None
    if attn_mask is not None:
        m = attn_mask
        if m.ndim == 2:
            # Key padding mask: (B, S_k). 1/True = keep, 0/False = mask.
            if m.shape != (b, sk):
                raise ValueError(f"2D attn_mask must have shape (B, S_k)=({b},{sk}), got {m.shape}")
            keep = m if m.dtype == torch.bool else (m != 0)
            am = keep[:, None, None, :].expand(b, 1, sq, sk)
        elif m.ndim == 3:
            # (B, S_q, S_k) -> (B, 1, S_q, S_k)
            if m.shape != (b, sq, sk):
                raise ValueError(f"3D attn_mask must have shape (B, S_q, S_k)=({b},{sq},{sk}), got {m.shape}")
            am = m[:, None, :, :]
        elif m.ndim == 4:
            # (B, 1/H, S_q, S_k) or (B, H, S_q, S_k) - pass through.
            am = m
        else:
            raise ValueError(f"Unsupported attn_mask rank: {m.ndim} (shape={m.shape})")

        # Preserve legacy behaviour: when the mask is numeric (not bool), treat it as additive bias.
        # (If the caller intended a 0/1 keep-mask they should pass bool.)
        if am is not None and am.dtype != torch.bool and am.dtype != q.dtype:
            am = am.to(q.dtype)

    # Convert (B, S, H, D) -> (B, H, S, D) for the registry.
    q_bhsd = q.permute(0, 2, 1, 3).contiguous()
    k_bhsd = k.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v.permute(0, 2, 1, 3).contiguous()

    # Backend selection:
    # - If a mask is provided, force SDPA so masking semantics are correct across hardware.
    # - Otherwise, respect the caller's `mode` hint when possible, but still route through the registry.
    backend_key = None
    if am is not None:
        backend_key = "sdpa"
    else:
        if mode in ("torch", "vanilla"):
            backend_key = "sdpa"
        elif mode in ("self_flash", "cross_flash", "flash_torch_sp"):
            # Prefer faster flash backends when available; otherwise fall back safely.
            for candidate in ("flash3", "flash_padded", "flash_varlen"):
                if attention_register.is_available(candidate):
                    backend_key = candidate
                    break
            if backend_key is None:
                backend_key = "sdpa"
        else:
            # Unknown mode -> use the globally configured default.
            backend_key = None

    out_bhsd = attention_register.call(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        key=backend_key,
        attn_mask=am,
        dropout_p=float(drop_rate),
        is_causal=bool(causal),
        # Optional var-len args (ignored by most backends).
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        deterministic=deterministic,
        cond_mask=cond_mask,
        img_seq_len=img_seq_len,
    )

    # (B, H, S_q, D) -> (B, S_q, H*D)
    out = out_bhsd.permute(0, 2, 1, 3).contiguous().reshape(b, sq, -1)
    return out


class SelfAttentionLayer(BasicAttentionLayer):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0,
        proj_drop=0,
        dtype=None,
        device=None,
        norm_type="layer",
        attn_mode="self_flash",
        deterministic=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(attn_mode, deterministic)
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        self.attn_drop = attn_drop

        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **factory_kwargs)

        norm_layer = get_norm_layer(norm_type)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, hidden_dim) (where hidden_dim = num heads * head dim)
            freqs_cis (torch.Tensor, optional): (batch, hidden_dim // 2), RoPE for image
            attn_mask (torch.Tensor, optional): (batch, seq_len, seq_len), mask for attention
        """
        b, s, d = x.shape

        # Apply QKV projection
        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, a, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, a, d]

        # Apply QK-Norm if needed
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cis is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis)
            assert (
                qq.shape == q.shape and kk.shape == k.shape
            ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
            q, k = qq, kk

        # Apply self attention
        context = attention(
            q,
            k,
            v,
            drop_rate=self.attn_drop if self.training else 0,
            attn_mask=attn_mask,
            mode=self.attn_mode,
            deterministic=self.deterministic,
        )
        out = self.out_proj(context)
        out = self.proj_drop(out)

        return out


class CrossAttentionLayer(BasicAttentionLayer):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0,
        proj_drop=0,
        dtype=None,
        device=None,
        norm_type="layer",
        attn_mode="cross_flash",
        deterministic=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(attn_mode, deterministic)
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        self.attn_drop = attn_drop

        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        norm_layer = get_norm_layer(norm_type)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, attn_mask=None):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, hidden_dim) (where hidden_dim = num heads * head dim)
            y (torch.Tensor): (batch, seq_len1, hidden_dim1)
            attn_mask (torch.Tensor): (batch, seq_len1), mask for attention
        """
        b, s, d = x.shape
        _, s1, d1 = y.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        kv = self.kv_proj(y).view(b, s1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        # Apply QK-Norm if needed
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply cross attention
        context = attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            drop_rate=self.attn_drop if self.training else 0,
            mode=self.attn_mode,
            deterministic=self.deterministic,
        )
        out = self.out_proj(context)
        out = self.proj_drop(out)

        return out
