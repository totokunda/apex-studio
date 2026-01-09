import torch
import torch
import torch.nn.functional as F

from src.register import FunctionRegister
import math
import warnings
import torch
from torch.utils.checkpoint import checkpoint
import math
import numpy as np 
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_3
except ImportError:
    flash_attn_func_3 = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None

try:
    from torch_xla.experimental.custom_kernel import (
        flash_attention as xla_flash_attention_func,
    )
except ImportError:
    xla_flash_attention_func = None
    pass
try:
    from sageattention import sageattn
except ImportError:
    sageattn = None

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention)
except ImportError:
    flex_attention = None

try:
    import xformers.ops
except ImportError:
    xformers = None


try:
    from flex_block_attn import flex_block_attn_func
except ImportError:
    flex_block_attn_func = None

attention_register = FunctionRegister()


@attention_register("sdpa_streaming")
def sdpa_streaming(
    q,
    k,
    v,
    attn_mask=None,
    is_causal=False,
    q_chunk=1024,
    kv_chunk=4096,
    **kwargs,
):
    """
    Exact attention via streaming softmax, never forms full S_q x S_k.
    q: [B,H,S_q,D], k,v: [B,H,S_k,D]  -> out: [B,H,S_q,D]
    """
    assert q.dim() == k.dim() == v.dim() == 4, "q,k,v must be [B,H,S, D]"
    Bq, Hq, S_q, Dq = q.shape
    Bk, Hk, S_k, Dk = k.shape
    Bv, Hv, S_v, Dv = v.shape
    assert (Bq, Hq, Dq) == (Bk, Hk, Dk) == (Bv, Hv, Dv), "B,H,D must match across q,k,v"
    assert S_k == S_v, "K and V must share sequence length"

    BH = Bq * Hq
    device = q.device
    dtype = q.dtype
    logits_dtype = torch.float32  # stable logits math

    # Fold heads to batch
    qf = q.reshape(BH, S_q, Dq)
    kf = k.reshape(BH, S_k, Dq)
    vf = v.reshape(BH, S_k, Dq)

    # Optional: normalize mask to [BH,S_q,S_k]
    am = None
    if attn_mask is not None:
        if attn_mask.ndim == 2:  # [S_q,S_k]
            am = attn_mask.to(logits_dtype).expand(BH, S_q, S_k)
        elif attn_mask.ndim == 4:  # [B,H,S_q,S_k]
            Bm, Hm, Smq, Smk = attn_mask.shape
            assert (Bm, Hm, Smq, Smk) == (Bq, Hq, S_q, S_k)
            am = attn_mask.reshape(BH, S_q, S_k).to(logits_dtype)
        elif attn_mask.ndim == 3 and attn_mask.shape[0] == Bq:
            # common case: [B,1,S_q,S_k]
            Bm, _, Smq, Smk = attn_mask.shape
            assert (Bm, Smq, Smk) == (Bq, S_q, S_k)
            am = (
                attn_mask.repeat_interleave(Hq, dim=0).squeeze(1).to(logits_dtype)
            )  # [BH,S_q,S_k]
        else:
            raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")

    scale = 1.0 / math.sqrt(Dq)
    out = torch.empty_like(qf)  # [BH,S_q,D]
    pos_q = torch.arange(S_q, device=device)
    pos_k = torch.arange(S_k, device=device)

    for qs in range(0, S_q, q_chunk):
        qe = min(qs + q_chunk, S_q)
        q_block = qf[:, qs:qe, :]  # [BH,q_len,D]
        q_len = q_block.size(1)

        # Streaming state per row
        m_i = torch.full((BH, q_len), -float("inf"), device=device, dtype=logits_dtype)
        l_i = torch.zeros((BH, q_len), device=device, dtype=logits_dtype)
        out_i = torch.zeros((BH, q_len, Dq), device=device, dtype=dtype)

        for ks in range(0, S_k, kv_chunk):
            ke = min(ks + kv_chunk, S_k)
            k_blk = kf[:, ks:ke, :]  # [BH,k_len,D]
            v_blk = vf[:, ks:ke, :]  # [BH,k_len,D]
            k_len = k_blk.size(1)

            # Scores: [BH,q_len,k_len] in fp32
            scores = (
                torch.matmul(
                    q_block.to(logits_dtype), k_blk.transpose(1, 2).to(logits_dtype)
                )
                * scale
            )

            # Add masks
            if is_causal:
                qpos = pos_q[qs:qe].unsqueeze(-1)  # [q_len,1]
                kpos = pos_k[ks:ke].unsqueeze(0)  # [1,k_len]
                tri = (kpos > qpos).unsqueeze(0)  # [1,q_len,k_len]
                scores = scores.masked_fill(tri, float("-inf"))

            if am is not None:
                scores = scores + am[:, qs:qe, ks:ke]

            # Streaming softmax update
            m_ij = scores.max(dim=-1).values  # [BH,q_len]
            p = torch.exp(scores - m_ij.unsqueeze(-1))  # [BH,q_len,k_len]
            l_ij = p.sum(dim=-1)  # [BH,q_len]

            m_new = torch.maximum(m_i, m_ij)  # [BH,q_len]
            alpha = torch.exp(m_i - m_new)  # old scale
            beta = torch.exp(m_ij - m_new)  # new block scale

            out_i = out_i * alpha.unsqueeze(-1).to(out_i.dtype) + (
                p.to(out_i.dtype) @ v_blk
            )
            l_i = l_i * alpha + l_ij * beta
            m_i = m_new

        out[:, qs:qe, :] = out_i / l_i.clamp_min(1e-20).unsqueeze(-1).to(out_i.dtype)

    return out.reshape(Bq, Hq, S_q, Dq)


@attention_register("flex-block-attn", available=flex_block_attn_func is not None)
def flex_block_attn(
    q,
    k,
    v,
    block_size,
    block_stride,
    block_mask,
    **kwargs,
):
    out = flex_block_attn_func(q, k, v, block_size, block_stride, block_mask)
    return out


@attention_register("sdpa")
def sdpa(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    softmax_scale=None,
    **kwargs,
):
    # When no mask, force efficient backends (flash or mem_efficient)
    # The math kernel materializes full S×S attention matrix causing VRAM spikes
    if attn_mask is None and q.is_cuda:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=False, 
            enable_mem_efficient=True
        ):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=softmax_scale,
            )
    else:
        # With mask, let PyTorch choose (may need math kernel for some mask formats)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=softmax_scale,
        )


@attention_register("sdpa_varlen")
def sdpa_varlen(
    q,  # [total_q, n_heads, head_dim]
    k,  # [total_k, n_heads, head_dim]
    v,  # [total_k, n_heads, head_dim]
    cu_seqlens_q,  # 1-D (B+1) cumulative -> 0, sq1, sq1+sq2, …
    cu_seqlens_kv,  # 1-D (B+1)
    max_seqlen_q: int,
    max_seqlen_kv: int,
    deterministic: bool = False,  # same flag you forwarded to flash-attn
    is_causal: bool = False,  # set True for decoder causal attention
    **kwargs,
):
    """
    Drop-in replacement for flash_attn_varlen_func that calls
    torch.scaled_dot_product_attention instead.
    Returns: packed tensor with shape [total_q, n_heads, head_dim]
    """
    if deterministic:
        torch.use_deterministic_algorithms(True)

    B = cu_seqlens_q.numel() - 1
    n_heads_q, head_dim_q = q.shape[1:]
    n_heads_k, head_dim_k = k.shape[1:]
    n_heads_v, head_dim_v = v.shape[1:]

    # Validate shapes
    if head_dim_q != head_dim_k or head_dim_q != head_dim_v:
        raise ValueError(
            f"Head dimensions must match: q={head_dim_q}, k={head_dim_k}, v={head_dim_v}"
        )

    head_dim = head_dim_q

    # 1. recover individual sequence lengths
    seq_lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    seq_lens_k = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).tolist()

    # 2. pad into (B, H, L, D)
    q_pad = q.new_zeros(B, n_heads_q, max_seqlen_q, head_dim)
    k_pad = k.new_zeros(B, n_heads_k, max_seqlen_kv, head_dim)
    v_pad = v.new_zeros(B, n_heads_v, max_seqlen_kv, head_dim)

    q_splits = torch.split(q, seq_lens_q, dim=0)
    k_splits = torch.split(k, seq_lens_k, dim=0)
    v_splits = torch.split(v, seq_lens_k, dim=0)

    for b in range(B):
        # flash-attn layout is (L, H, D) –> transpose to (H, L, D)
        q_pad[b, :, : seq_lens_q[b]] = q_splits[b].transpose(0, 1)
        k_pad[b, :, : seq_lens_k[b]] = k_splits[b].transpose(0, 1)
        v_pad[b, :, : seq_lens_k[b]] = v_splits[b].transpose(0, 1)

    # 3. Handle multi-query/grouped-query attention by expanding k,v if needed
    if n_heads_k != n_heads_q:
        if n_heads_q % n_heads_k != 0:
            raise ValueError(
                f"Number of query heads ({n_heads_q}) must be divisible by key heads ({n_heads_k})"
            )
        expand_ratio = n_heads_q // n_heads_k
        k_pad = k_pad.repeat_interleave(expand_ratio, dim=1)
        v_pad = v_pad.repeat_interleave(expand_ratio, dim=1)

    # 4. SDPA call with per-batch processing to avoid large mask materialization
    out_pad = torch.zeros_like(q_pad)

    for b in range(B):
        # Create minimal mask only for this batch item
        if seq_lens_q[b] < max_seqlen_q or seq_lens_k[b] < max_seqlen_kv:
            # Only create mask if we have padding
            batch_mask = torch.full(
                (1, 1, seq_lens_q[b], seq_lens_k[b]),
                0.0,
                device=q.device,
                dtype=q.dtype,
            )
            batch_out = F.scaled_dot_product_attention(
                q_pad[b : b + 1, :, : seq_lens_q[b]],
                k_pad[b : b + 1, :, : seq_lens_k[b]],
                v_pad[b : b + 1, :, : seq_lens_k[b]],
                attn_mask=batch_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        else:
            # No padding, no mask needed
            batch_out = F.scaled_dot_product_attention(
                q_pad[b : b + 1, :, : seq_lens_q[b]],
                k_pad[b : b + 1, :, : seq_lens_k[b]],
                v_pad[b : b + 1, :, : seq_lens_k[b]],
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        out_pad[b, :, : seq_lens_q[b]] = batch_out[0]

    # 6. strip padding and repack to (∑Lq, H, D) with original query head count
    packed_out = [
        out_pad[b, :n_heads_q, : seq_lens_q[b]].transpose(0, 1)
        for b in range(B)  # (Lq, H, D)
    ]
    return torch.cat(packed_out, dim=0)


@attention_register("flash_padded", available=flash_attn_func is not None)
def flash_attention_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    default_dtype: torch.dtype = torch.bfloat16,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    FlashAttention-2 **padded** (non-var-len) wrapper that works for *self-* **and**
    *cross-attention*.

    It accepts `(B, H, S, D)` tensors.  Sequence lengths may differ between
    `q` and `k/v` (cross-attention).  Internally everything runs in **bf16 or
    fp16 only** – never float32.

    Parameters
    ----------
    q : (B, H, Sq, D) tensor
    k : (B, H, Sk, D) tensor
    v : (B, H, Sk, D) tensor
        *H* (num heads) and *D* (head dimension) must match across all three.
    softmax_scale : float, optional
        Defaults to ``1/sqrt(D)``.
    default_dtype : torch.dtype, default ``torch.bfloat16``
        Used when an input arrives in an unsupported dtype.
    is_causal : bool, default ``False``
        Apply a causal mask (only makes sense for self-attention).

    Returns
    -------
    out : (B, H, Sq, D) tensor
    """
    # ------------------------------------------------------------------ #
    if flash_attn_func is None:
        raise ImportError(
            "flash_attn is not installed or flash_attn_func missing. "
            "Install FlashAttention-2 (pip install flash-attn)."
        )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (B, H, S, D)")

    Bq, Hq, Sq, Dq = q.shape
    Bk, Hk, Sk, Dk = k.shape
    Bv, Hv, Sv, Dv = v.shape

    if not (Bq == Bk == Bv):
        raise ValueError("Batch sizes differ (q, k, v); pad or split batches first.")
    if not (Sk == Sv):
        raise ValueError("Key and value sequence lengths must match (Sk == Sv).")

    # ------------------------------------------------------------------ #
    # Make sure we are using a supported low-precision dtype
    #
    allowed = {torch.bfloat16, torch.float16}
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in allowed:
            warnings.warn(
                f"{name} has dtype {t.dtype} – casting to {default_dtype}.",
                stacklevel=2,
            )

    q = q.to(default_dtype if q.dtype not in allowed else q.dtype).contiguous()
    k = k.to(default_dtype if k.dtype not in allowed else k.dtype).contiguous()
    v = v.to(default_dtype if v.dtype not in allowed else v.dtype).contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dq)

    # ------------------------------------------------------------------ #
    # FlashAttention kernel needs (B, Sq, H, D) layout
    #
    q_in = q.permute(0, 2, 1, 3)  # (B, Sq, H, D)
    k_in = k.permute(0, 2, 1, 3)  # (B, Sk, H, D)
    v_in = v.permute(0, 2, 1, 3)  # (B, Sk, H, D)

    out = flash_attn_func(
        q_in,
        k_in,
        v_in,
        causal=is_causal,
        softmax_scale=softmax_scale,
        dropout_p=0.0,  # change if training-time dropout desired
    )

    # Back to (B, H, Sq, D) and caller’s dtype
    return out.permute(0, 2, 1, 3).to(q.dtype)


@attention_register("flash_varlen", available=flash_attn_varlen_func is not None)
def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    default_dtype: torch.dtype = torch.bfloat16,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    FlashAttention-2 var-len wrapper that supports both *self* and *cross* attention.

    Parameters
    ----------
    q : (Bq, H, Sq, D) or (T, H, D) tensor
    k : (Bk, H, Sk, D) or (T, H, D) tensor
    v : (Bk, H, Sk, D) or (T, H, D) tensor
        *H* (num heads) and *D* (head dim) must match across all three tensors.
        Usually `Bq == Bk`, but the wrapper only assumes `Bk >= Bq`
        (common case in encoder–decoder cross-attn with packed memory).
        If tensors are already in (T, H, D) format, no reshaping is performed.
    cu_seqlens_q, cu_seqlens_k : (batch+1,) int32 tensors, optional
        Cumulative sequence-length vectors:
          `[0, len₀, len₀+len₁, …]`.
        If **omitted** we assume *uniform* lengths and build them automatically.
    max_seqlen_q, max_seqlen_k : int, optional
        Maximum sequence length for q / k-v.  Inferred if not given.
    softmax_scale : float, optional
        Defaults to `1/√D` if `None`.
    default_dtype : torch.dtype, default **bfloat16**
        Used when any input arrives in an unsupported dtype.
    is_causal : bool, default **False**
        Apply a causal mask (useful for decoder self-attention).

    Returns
    -------
    out : tensor with same shape as q
    """
    # -------------------- checks & dtype sanitisation -------------------- #
    if flash_attn_varlen_func is None:
        raise ImportError(
            "flash_attn is not installed or flash_attn_varlen_func is undefined."
        )

    # Check if inputs are already in varlen format (T, H, D)
    is_varlen_format = q.ndim == 3 and k.ndim == 3 and v.ndim == 3

    if not is_varlen_format:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, v must have shape (B, H, S, D) or (T, H, D)")

        Bq, Hq, Sq, Dq = q.shape
        Bk, Hk, Sk, Dk = k.shape
        Bv, Hv, Sv, Dv = v.shape

        if not (Hq == Hk == Hv and Dq == Dk == Dv and Sk == Sv):
            raise ValueError("Mismatched head counts / head dims or K ≠ V shapes")
    else:
        # For varlen format (T, H, D)
        Tq, Hq, Dq = q.shape
        Tk, Hk, Dk = k.shape
        Tv, Hv, Dv = v.shape

    accepted = {torch.bfloat16, torch.float16}
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in accepted:
            warnings.warn(
                f"{name} is {t.dtype}. Casting to {default_dtype} (never float32).",
                stacklevel=2,
            )

    q = q.to(default_dtype if q.dtype not in accepted else q.dtype).contiguous()
    k = k.to(default_dtype if k.dtype not in accepted else k.dtype).contiguous()
    v = v.to(default_dtype if v.dtype not in accepted else v.dtype).contiguous()

    # ------------------ build (or validate) cu_seqlens ------------------- #
    device = q.device

    if not is_varlen_format:
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(
                0, (Bq + 1) * Sq, Sq, dtype=torch.int32, device=device
            )
        if cu_seqlens_k is None:
            cu_seqlens_k = torch.arange(
                0, (Bk + 1) * Sk, Sk, dtype=torch.int32, device=device
            )

        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens tensors must be int32")

        if max_seqlen_q is None:
            max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_k is None:
            max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(Dq)

        # --------------------- flatten to (T, H, D) -------------------------- #
        # FlashAttention-2 expects contiguous `(total_tokens, num_heads, head_dim)`
        #
        q_flat = q.permute(0, 2, 1, 3).reshape(-1, Hq, Dq)
        k_flat = k.permute(0, 2, 1, 3).reshape(-1, Hq, Dq)
        v_flat = v.permute(0, 2, 1, 3).reshape(-1, Hq, Dq)
    else:
        # Already in varlen format - use directly
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must be provided for varlen format inputs"
            )

        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens tensors must be int32")

        if max_seqlen_q is None:
            max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_k is None:
            max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(Dq)

        q_flat = q
        k_flat = k
        v_flat = v

    # ----------------------- kernel invocation --------------------------- #
    out_flat = flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,  # change if training-time dropout desired
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    # --------------------------- re-shape -------------------------------- #
    if not is_varlen_format:
        out = (
            out_flat.reshape(Bq, Sq, Hq, Dq)
            .permute(0, 2, 1, 3)  # (Bq, H, Sq, D)
            .to(q.dtype)  # preserve caller's dtype
        )
    else:
        # Return in same shape as query (varlen format)
        out = out_flat.to(q.dtype)

    return out


@attention_register("flash", available=flash_attn_func is not None)
def flash_attention(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    default_dtype=torch.bfloat16,
    is_causal=False,
    **kwargs,
):

    if cu_seqlens_q is None or cu_seqlens_k is None:
        return flash_attention_padded(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            default_dtype=default_dtype,
            is_causal=is_causal,
            **kwargs,
        )
    else:
        return flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            default_dtype=default_dtype,
            is_causal=is_causal,
            **kwargs,
        )


@attention_register("flash3", available=flash_attn_func_3 is not None)
def flash_attention3(
    q, k, v, softmax_scale=None, default_dtype=torch.bfloat16, is_causal=False, **kwargs
):
    if flash_attn_func_3 is None:
        raise ImportError("flash_attn_interface is not installed")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    start_dtype = q.dtype
    acceptable_dtypes = [torch.bfloat16, torch.float16]
    if q.dtype not in acceptable_dtypes:
        q = q.to(default_dtype)
    if k.dtype not in acceptable_dtypes:
        k = k.to(default_dtype)
    if v.dtype not in acceptable_dtypes:
        v = v.to(default_dtype)

    out = flash_attn_func_3(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    # check if out is a tuple of two tensors
    if isinstance(out, tuple):
        return out[0].permute(0, 2, 1, 3)
    else:
        return out.permute(0, 2, 1, 3)


@attention_register("sage", available=sageattn is not None)
def sage_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    softmax_scale=None,
    default_dtype=torch.bfloat16,
    **kwargs,
):
    """
    SageAttention backend with an SDPA-compatible calling convention.

    Mirrors the signature of `sdpa` so that `attention_register.call(...)`
    can switch between backends transparently:

        (q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, softmax_scale=None)

    Notes
    -----
    * `attn_mask` is currently **ignored** – SageAttention does not support it.
    * `dropout_p` is also ignored and must effectively be 0.0 in inference.
    * When `softmax_scale` is None we match PyTorch SDPA's default of
      `1 / sqrt(head_dim)`.
    """
    if attn_mask is not None:
        warnings.warn(
            "sage_attention currently ignores `attn_mask` – behaviour may differ "
            "from SDPA when masks are used.",
            RuntimeWarning,
        )

    if dropout_p not in (0.0, None):
        warnings.warn(
            "sage_attention does not implement dropout – `dropout_p` is ignored.",
            RuntimeWarning,
        )

    # Match PyTorch SDPA's default scaling when none is provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # ensure q, k, v are on the same device and are of either bfloat16 or float16 dont raise error, just fix it
    if q.device != k.device or q.device != v.device:
        q = q.to(k.device)
        v = v.to(k.device)
    if (
        q.dtype not in [torch.bfloat16, torch.float16]
        or k.dtype not in [torch.bfloat16, torch.float16]
        or v.dtype not in [torch.bfloat16, torch.float16]
    ):
        q = q.to(default_dtype)
        k = k.to(default_dtype)
        v = v.to(default_dtype)

    attn_output = sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=is_causal,
        sm_scale=softmax_scale,
    )
    return attn_output


@attention_register("xla_flash", available=xla_flash_attention_func is not None)
def xla_flash_attention(q, k, v, attention_mask, softmax_scale, **kwargs):
    batch_size = q.shape[0]
    q_segment_indexes = None
    if (
        attention_mask is not None
    ):  # if mask is required need to tune both segmenIds fields
        # attention_mask = torch.squeeze(attention_mask).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)
        q_segment_indexes = torch.ones(
            batch_size, q.shape[2], device=q.device, dtype=torch.float32
        )
        assert (
            attention_mask.shape[1] == k.shape[2]
        ), f"ERROR: KEY SHAPE must be same as attention mask [{k.shape[2]}, {attention_mask.shape[1]}]"
    assert (
        q.shape[2] % 128 == 0
    ), f"ERROR: QUERY SHAPE must be divisible by 128 (TPU limitation) [{q.shape[2]}]"

    assert (
        k.shape[2] % 128 == 0
    ), f"ERROR: KEY SHAPE must be divisible by 128 (TPU limitation) [{k.shape[2]}]"

    if xla_flash_attention_func is None:
        raise ImportError("xla_flash_attention is not installed")
    return xla_flash_attention_func(
        q, k, v, q_segment_indexes, attention_mask, softmax_scale
    )


@attention_register("flex", available=flex_attention is not None)
def flex_attention_func(q, k, v, attn_mask=None, softmax_scale=None, **kwargs):
    out = flex_attention(q, k, v, block_mask=attn_mask, scale=softmax_scale, **kwargs)
    return out


@attention_register("xformers", available=xformers is not None)
def xformers_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    softmax_scale=None,
    **kwargs,
):
    if xformers is None:
        raise ImportError("xformers is not installed")

    # xformers expects (B, S, H, D) format, so we need to transpose
    q = q.transpose(1, 2)  # (B, H, S, D) -> (B, S, H, D)
    k = k.transpose(1, 2)  # (B, H, S, D) -> (B, S, H, D)
    v = v.transpose(1, 2)  # (B, H, S, D) -> (B, S, H, D)

    # Handle causal mask
    if is_causal:
        attn_bias = xformers.ops.fmha.attn_bias.LowerTriangularMask()
    else:
        attn_bias = None

    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_bias is None:
            attn_bias = attn_mask
        else:
            # Combine causal mask with attention mask
            attn_bias = attn_bias & attn_mask

    output = xformers.ops.memory_efficient_attention(
        q, k, v, attn_bias=attn_bias, p=dropout_p, scale=softmax_scale
    )

    # Transpose back to (B, H, S, D) format
    return output.transpose(1, 2)





def dynamic_slice(x, starts, sizes):
    # start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - size_indices[i])
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))]
    for i, (start, size) in enumerate(zip(starts, sizes)):
        x = torch.index_select(x, i, torch.tensor(range(start, start + size), device=x.device))
    return x


def map_pt(f, xs):
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)

def _query_chunk_attention(query_idx, query, key, value,
                           mask, bias, key_chunk_size=4096,
                           mask_calc_fn=None,
                           bias_calc_fn=None,
                           weights_calc_fn=None,
                           calc_fn_data=None):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / math.sqrt(k_features)

    def summarize_chunk(key_idx, query, key, value, mask, bias):
        attn_weights = torch.einsum('...qhd,...khd->...qhk', query, key)
        if bias_calc_fn is not None:
            bias = bias_calc_fn(query_idx, key_idx, bias, attn_weights, calc_fn_data)
        if bias is not None:
            bias = torch.einsum('...hqk->...qhk', bias)
            attn_weights = attn_weights + bias
        if mask_calc_fn is not None:
            mask = mask_calc_fn(query_idx, key_idx, mask, attn_weights, calc_fn_data)
        if mask is not None:
            big_neg = torch.finfo(attn_weights.dtype).min
            big_neg = torch.tensor(big_neg, device=mask.device, dtype=torch.float32)
            mask = torch.einsum('...hqk->...qhk', mask)
            attn_weights = torch.where(mask, attn_weights, big_neg)
        if weights_calc_fn is not None:
            attn_weights = weights_calc_fn(query_idx, key_idx, attn_weights, calc_fn_data)
        max_score, _ = torch.max(attn_weights, -1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum('...vhf,...qhv->...qhf', value, exp_weights)
        max_score = torch.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights.sum(dim=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = dynamic_slice(key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
                                  tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
        value_chunk = dynamic_slice(value, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))

        if bias is None:
            bias_chunk = None
        elif bias.shape[-1] == 1:
            bias_chunk = bias
        elif bias.shape[-1] == num_kv:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(bias.shape[:-3]) + (bias.shape[-3], bias.shape[-2], key_chunk_size))
        else:
            raise TypeError(f'bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}')

        if mask is None:
            mask_chunk = None
        elif mask.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(mask.shape[:-3]) + (mask.shape[-3], mask.shape[-2], key_chunk_size))
        else:
            raise TypeError(f'bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}')

        return checkpoint(summarize_chunk, chunk_idx, query, key_chunk, value_chunk, mask_chunk, bias_chunk)

    chunk_values, chunk_weights, chunk_max = map_pt(
        chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size))

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


@attention_register("efficient_dot_product_attention", available=True)
def efficient_dot_product_attention(query, key, value,
                                    mask=None, bias=None,
                                    query_chunk_size=1024,
                                    key_chunk_size=4096,
                                    bias_calc_fn=None,
                                    mask_calc_fn=None,
                                    weights_calc_fn=None,
                                    calc_fn_data=None):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Note: query, key, value needn't have any batch dimensions.
      Args:
        query: queries for calculating attention with shape of
          `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
          `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
          `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating padding masks, proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        bias_calc_fn: a bias calculation callback for each chunk, of form
          `(q_offset, k_offset, bias_chunk, attn_weights, calc_fn_data) -> bias`.
          This can be used for incorporating causal masks, padding masks,
          proximity bias, etc.
        mask_calc_fn: a mask calculation callback for each chunk, of form
          `(q_offset, k_offset, mask_chunk, attn_weights, calc_fn_data) -> mask`.
          This can be used for incorporating causal or other large masks.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        weights_calc_fn: a general attn_weights callback for each chunk, of form
          `(q_offset, k_offset, attn_weights, calc_fn_data) -> attn_weights`.
          attn_weights has shape of
          `[batch..., q_chunk_size, num_heads, k_chunk_size]`.
          This can be used to implement complex weights processing in a memory
          efficient way.
        calc_fn_data: optional pure data to pass to each per-chunk call of
          bias_calc_fn, mask_calc_fn, and weights_calc_fn.
        weights_calc_data: pure_data to pass with each call to weights_calc_fn
      Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
      """
    # transpose 
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx, _):
        query_chunk = dynamic_slice(query, tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(query.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, q_features))

        if mask is None:
            mask_chunk = None
        elif mask.shape[-2] == 1:
            mask_chunk = mask
        elif mask.shape[-2] == num_q:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(mask.shape[:-3]) + (mask.shape[-3], min(query_chunk_size, num_q), mask.shape[-1]))
        else:
            raise TypeError(f'mask.shape[-2] == {mask.shape[-2]} must broadcast with query.shape[-3] == {num_q}')

        if bias is None:
            bias_chunk = None
        elif bias.shape[-2] == 1:
            bias_chunk = bias
        elif bias.shape[-2] == num_q:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(bias.shape[:-3]) + (bias.shape[-3], min(query_chunk_size, num_q), bias.shape[-1]))
        else:
            raise TypeError(f'bias.shape[-2] == {bias.shape[-2]} must broadcast with query.shape[-3] == {num_q}')
        return (chunk_idx + query_chunk_size,
                _query_chunk_attention(chunk_idx, query_chunk, key, value, mask_chunk, bias_chunk, key_chunk_size=key_chunk_size,
                                       bias_calc_fn=bias_calc_fn, mask_calc_fn=mask_calc_fn,
                                       weights_calc_fn=weights_calc_fn, calc_fn_data=calc_fn_data))

    _, res = scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    rl = [res[i] for i in range(res.shape[0])]
    return torch.cat(rl, dim=-3).transpose(1, 2)


attention_register.set_default("sdpa")

