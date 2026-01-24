from __future__ import annotations

from typing import Optional, Tuple

import torch


def _reshape_hidden_states_for_frames(
    hidden_states: torch.Tensor, frames: int
) -> torch.Tensor:
    b, tokens, d = hidden_states.shape
    if frames <= 0 or tokens % frames != 0:
        raise ValueError(
            f"Cannot reshape tokens={tokens} into frames={frames} evenly."
        )
    return hidden_states.reshape(b, frames, -1, d)


def apply_gate_inplace(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    In-place gating helper for inference-only paths.

    Supports per-frame gates shaped [B, F, D] applied to token sequences [B, F*T, D].
    """
    if x.ndim == 3 and gate.ndim == 3 and gate.shape[1] != x.shape[1]:
        frames = int(gate.shape[1])
        try:
            x_view = _reshape_hidden_states_for_frames(x, frames)
            x_view.mul_(gate.unsqueeze(2))
            return x
        except Exception:
            pass
    x.mul_(gate)
    return x


def apply_scale_shift_inplace(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    In-place `x = x * (1 + scale) + shift` without allocating `(1 + scale)`.
    """
    if x.ndim == 3 and scale.ndim == 3 and scale.shape[1] != x.shape[1]:
        frames = int(scale.shape[1])
        try:
            x_view = _reshape_hidden_states_for_frames(x, frames)
            scale_u = scale.unsqueeze(2)
            shift_u = shift.unsqueeze(2)
            x_view.addcmul_(x_view, scale_u)
            x_view.add_(shift_u)
            return x
        except Exception:
            pass
    x.addcmul_(x, scale)
    x.add_(shift)
    return x


def chunked_feed_forward_inplace(
    ff: torch.nn.Module,
    hidden_states: torch.Tensor,
    chunk_dim: int,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """
    Chunked feed-forward with an inference-only in-place fast path.

    When gradients are disabled and we are chunking along the sequence dimension for
    [B, S, D], update the output directly into the input buffer to avoid concat
    allocations. Falls back to the standard chunked implementation otherwise.
    """
    if chunk_size is None:
        return ff(hidden_states)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    dim_len = hidden_states.shape[chunk_dim]
    if dim_len <= chunk_size:
        return ff(hidden_states)

    can_inplace = (
        not torch.is_grad_enabled()
        and not hidden_states.requires_grad
        and hidden_states.ndim == 3
        and chunk_dim == 1
    )
    if can_inplace:
        x_flat = hidden_states
        for chunk in torch.split(x_flat, int(chunk_size), dim=1):
            chunk[...] = ff(chunk)
        return hidden_states

    outputs = []
    for start in range(0, dim_len, chunk_size):
        end = min(start + chunk_size, dim_len)
        hs_chunk = hidden_states.narrow(chunk_dim, start, end - start)
        outputs.append(ff(hs_chunk))
    return torch.cat(outputs, dim=chunk_dim)


def _rope_complex_multiply_inplace(
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    freqs_real: torch.Tensor,
    freqs_imag: torch.Tensor,
) -> None:
    x_real_orig = x_real.clone()
    x_real.mul_(freqs_real).addcmul_(x_imag, freqs_imag, value=-1.0)
    x_imag.mul_(freqs_real).addcmul_(x_real_orig, freqs_imag, value=1.0)


def apply_wan_rope_inplace(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
    freqs_may_be_cpu: bool = True,
) -> torch.Tensor:
    """
    In-place RoPE for WAN-style complex freqs.

    hidden_states: [B, H, T, D] with even D
    freqs: complex tensor broadcastable to [1, 1, T, D/2]
    """
    if hidden_states.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dim D must be even for RoPE, got D={hidden_states.shape[-1]}"
        )

    if not torch.is_complex(freqs):
        raise TypeError(f"Expected complex freqs, got dtype={freqs.dtype}")

    while freqs.dim() < 4:
        freqs = freqs.unsqueeze(0)

    _, _, t, d = hidden_states.shape
    if freqs.shape[-2] != t or freqs.shape[-1] != d // 2:
        raise ValueError(
            f"RoPE shape mismatch: T={t}, D/2={d//2}, freqs={tuple(freqs.shape)}"
        )

    if chunk_size is None:
        chunk_size = t
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        hs_chunk = hidden_states[:, :, start:end]
        x_pairs = hs_chunk.unflatten(3, (-1, 2))
        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]

        freqs_chunk = freqs[:, :, start:end]
        if freqs_may_be_cpu and freqs_chunk.device != hidden_states.device:
            freqs_chunk = freqs_chunk.to(hidden_states.device)
        freqs_real = freqs_chunk.real.to(dtype=hs_chunk.dtype)
        freqs_imag = freqs_chunk.imag.to(dtype=hs_chunk.dtype)
        _rope_complex_multiply_inplace(x_real, x_imag, freqs_real, freqs_imag)
    return hidden_states


def apply_cos_sin_rope_inplace(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    *,
    chunk_size: Optional[int] = None,
    sequence_dim: int = 1,
) -> torch.Tensor:
    """
    In-place RoPE for cos/sin freqs.

    x: [B, S, H, D] or [B, H, S, D]
    freqs_cis: (cos, sin) with shape [S, D/2] (half) or [S, D] (full, repeated-interleave)
    """
    cos, sin = freqs_cis
    seq_len = x.shape[sequence_dim]
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(
            f"Last dim D must be even for RoPE, got D={head_dim} for x={tuple(x.shape)}"
        )
    half_dim = head_dim // 2

    if chunk_size is None:
        chunk_size = seq_len
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        if sequence_dim == 1:
            x_chunk = x[:, start:end]
            cos_chunk = cos[start:end][None, :, None, :]
            sin_chunk = sin[start:end][None, :, None, :]
        else:
            x_chunk = x[:, :, start:end]
            cos_chunk = cos[start:end][None, None, :, :]
            sin_chunk = sin[start:end][None, None, :, :]

        cos_chunk = cos_chunk.to(device=x.device, dtype=x.dtype)
        sin_chunk = sin_chunk.to(device=x.device, dtype=x.dtype)

        # Support both common RoPE layouts:
        # - half layout: cos/sin last dim == D/2 (matches x_real/x_imag)
        # - full layout: cos/sin last dim == D (values repeated per pair), as used by
        #   some Diffusers models (e.g. HunyuanVideo15's `get_1d_rotary_pos_embed` path).
        if cos_chunk.shape[-1] == head_dim:
            cos_chunk = cos_chunk[..., ::2]
            sin_chunk = sin_chunk[..., ::2]
        elif cos_chunk.shape[-1] != half_dim:
            raise ValueError(
                "RoPE shape mismatch: expected cos/sin last dim to be "
                f"{half_dim} or {head_dim}, got cos={tuple(cos_chunk.shape)} "
                f"sin={tuple(sin_chunk.shape)} for x={tuple(x.shape)}"
            )

        x_pairs = x_chunk.unflatten(-1, (-1, 2))
        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]

        # Match common reference implementations that upcast to fp32 for the math,
        # then cast back (important for fp16/bf16 stability).
        if x.dtype in (torch.float16, torch.bfloat16):
            x_real_f = x_real.float()
            x_imag_f = x_imag.float()
            cos_f = cos_chunk.float()
            sin_f = sin_chunk.float()
            _rope_complex_multiply_inplace(x_real_f, x_imag_f, cos_f, sin_f)
            x_real.copy_(x_real_f.to(dtype=x.dtype))
            x_imag.copy_(x_imag_f.to(dtype=x.dtype))
        else:
            _rope_complex_multiply_inplace(x_real, x_imag, cos_chunk, sin_chunk)

    return x
