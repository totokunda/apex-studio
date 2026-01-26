import mlx.core as mx
from typing import Union, Tuple
import numpy as np


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, mx.array, int],
    theta: float = 10000.0,
    use_real: bool = False,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype: mx.Dtype = mx.float32,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    Precomputes the frequency tensor for rotary positional embeddings.

    Args:
        dim (int):
            Dimension of the frequency tensor.
        pos (Union[np.ndarray, mx.array, int]):
            Position indices for the frequency tensor. [S] or scalar.
        theta (float, optional):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional):
            If True, return real and imaginary parts separately. Otherwise, return complex numbers.
        linear_factor (float, optional):
            Scaling factor for context extrapolation. Defaults to 1.0.
        ntk_factor (float, optional):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (bool, optional):
            If True and use_real, real/imag parts are interleaved. Otherwise, they are concatenated.
        freqs_dtype (mx.Dtype, optional):
            The dtype of the frequency tensor. Defaults to mx.float32.

    Returns:
        Union[mx.array, Tuple[mx.array, mx.array]]:
            Precomputed frequency tensor. [S, D/2] if complex, or two [S, D] arrays if real.
    """
    assert dim % 2 == 0, "Dimension must be even."

    if isinstance(pos, int):
        pos = mx.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = mx.array(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (mx.arange(0, dim, 2, dtype=freqs_dtype) / dim)) / linear_factor
    )  # [D/2]

    freqs = mx.outer(pos, freqs)  # [S, D/2]

    if use_real and repeat_interleave_real:
        # Interleave real and imaginary parts
        freqs_cos = mx.cos(freqs)
        freqs_sin = mx.sin(freqs)

        # Simulate repeat_interleave(2, dim=1)
        # Expand, broadcast, and reshape to interleave the values
        freqs_cos = mx.broadcast_to(
            freqs_cos[:, :, None], (freqs_cos.shape[0], freqs_cos.shape[1], 2)
        ).reshape(freqs_cos.shape[0], -1)
        freqs_sin = mx.broadcast_to(
            freqs_sin[:, :, None], (freqs_sin.shape[0], freqs_sin.shape[1], 2)
        ).reshape(freqs_sin.shape[0], -1)

        return freqs_cos.astype(freqs_dtype), freqs_sin.astype(freqs_dtype)
    elif use_real:
        # Concatenate real and imaginary parts
        freqs_cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1)
        freqs_sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1)
        return freqs_cos.astype(freqs_dtype), freqs_sin.astype(freqs_dtype)
    else:
        # Return complex numbers
        return mx.exp(1j * freqs).astype(mx.complex64)


def view_as_complex(arr: mx.array) -> mx.array:
    """
    Converts a real-valued MLX array with a last dimension of size 2
    into a complex-valued array. This emulates PyTorch's ``torch.view_as_complex``.

    Notes on behavior parity:
    - PyTorch returns a view (no data copy) with strict stride/layout checks. MLX
      does not currently expose the same view semantics here, so this returns a
      computed array with complex dtype (no stride validation).
    - Dtype requirements follow PyTorch as closely as possible: the input must be
      a floating real tensor and its last dimension must be exactly 2.

    Args:
        arr (mx.array): The input array with a shape of (..., 2).

    Returns:
        mx.array: The complex-valued array.

    Raises:
        ValueError: If the last dimension of the input array is not 2 or if ``arr``
            is not a floating real tensor.
    """
    if arr.shape[-1] != 2:
        raise ValueError(
            f"The last dimension of the input array must be 2, but got shape {arr.shape}"
        )

    # Validate dtype roughly matching torch's constraints (float dtypes only)
    valid_real_dtypes = {
        getattr(mx, "float16", None),
        getattr(mx, "bfloat16", None),
        mx.float32,
        getattr(mx, "float64", None),
    }
    # Filter out Nones for older runtimes without those dtypes
    valid_real_dtypes = {dt for dt in valid_real_dtypes if dt is not None}

    if arr.dtype not in valid_real_dtypes:
        raise ValueError(
            f"view_as_complex expects a floating real tensor, but got dtype {arr.dtype}"
        )

    # Construct the complex array by slicing the last dimension
    # arr[..., 0] becomes the real part
    # arr[..., 1] becomes the imaginary part
    # Choose complex dtype similar to PyTorch mapping
    # float64 -> complex128, other supported floats -> complex64
    if arr.dtype == getattr(mx, "float64", None):
        out_complex_dtype = getattr(mx, "complex128", None) or mx.complex64
    else:
        out_complex_dtype = mx.complex64

    complex_arr = arr[..., 0] + 1j * arr[..., 1]
    if complex_arr.dtype != out_complex_dtype:
        complex_arr = complex_arr.astype(out_complex_dtype)
    return complex_arr


def view_as_real(arr: mx.array) -> mx.array:
    """
    Converts a complex-valued MLX array into a real-valued array,
    where the last dimension is 2 and represents the real and imaginary parts.
    This emulates PyTorch's ``torch.view_as_real``.

    Notes on behavior parity:
    - PyTorch returns a view (no data copy) with strict stride/layout checks. MLX
      does not currently expose the same view semantics here, so this returns a
      computed real-valued array with last-dim size 2.

    Args:
        arr (mx.array): The complex-valued input array.

    Returns:
        mx.array: The real-valued array with a new last dimension of size 2.

    Raises:
        ValueError: If ``arr`` is not a complex tensor.
    """
    # Validate complex dtype
    valid_complex_dtypes = {
        getattr(mx, "complex64", None),
        getattr(mx, "complex128", None),
    }
    valid_complex_dtypes = {dt for dt in valid_complex_dtypes if dt is not None}

    if arr.dtype not in valid_complex_dtypes:
        raise ValueError(
            f"view_as_real expects a complex tensor, but got dtype {arr.dtype}"
        )

    return mx.stack([mx.real(arr), mx.imag(arr)], axis=-1)
