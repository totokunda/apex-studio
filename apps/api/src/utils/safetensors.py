from __future__ import annotations

from typing import Any, Dict, Union
import json
import os
import torch
import safetensors

try:
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover - MLX is not available on Windows/Linux
    mx = None  # type: ignore


def _validate_safetensors_header(file_path: Union[str, os.PathLike]) -> None:
    """
    Best-effort safetensors integrity validation without importing torch storage.

    Why: on Windows, partially-downloaded/corrupt safetensors can sometimes trigger
    hard crashes (access violations) when mapped/read by native extensions.
    We validate the header and ensure all declared `data_offsets` fit in the file.
    """

    path = os.fspath(file_path)

    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise RuntimeError(f"Cannot stat weights file: {path}") from e

    # Need at least 8 bytes for the header length prefix.
    if size < 8:
        raise RuntimeError(
            f"Invalid safetensors file (too small): {path} ({size} bytes)"
        )

    with open(path, "rb") as fp:
        header_len_bytes = fp.read(8)
        if len(header_len_bytes) != 8:
            raise RuntimeError(
                f"Invalid safetensors file (missing header length): {path}"
            )

        header_len = int.from_bytes(header_len_bytes, byteorder="little", signed=False)
        # Heuristic cap to avoid trying to allocate absurd header sizes on corrupted files.
        if header_len <= 2 or header_len > 100_000_000:
            raise RuntimeError(
                f"Invalid safetensors header length ({header_len}) for file: {path}"
            )

        if 8 + header_len > size:
            raise RuntimeError(
                f"Truncated safetensors file (header exceeds file size): {path} "
                f"(header={header_len} bytes, file={size} bytes)"
            )

        header_bytes = fp.read(header_len)
        if len(header_bytes) != header_len:
            raise RuntimeError(f"Truncated safetensors header: {path}")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid safetensors header JSON: {path}") from e

    if not isinstance(header, dict):
        raise RuntimeError(f"Invalid safetensors header (not a dict): {path}")

    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise RuntimeError(
                f"Invalid safetensors entry metadata for {key!r}: {path}"
            )
        offsets = meta.get("data_offsets")
        if (
            not isinstance(offsets, (list, tuple))
            or len(offsets) != 2
            or not all(isinstance(x, int) for x in offsets)
        ):
            raise RuntimeError(
                f"Invalid/missing data_offsets for tensor {key!r} in: {path}"
            )
        start, end = offsets
        if start < 0 or end < 0 or end < start:
            raise RuntimeError(
                f"Invalid data_offsets for tensor {key!r} in: {path} ({start}, {end})"
            )
        # Offsets are relative to the start of the data section (after header).
        abs_end = 8 + header_len + end
        if abs_end > size:
            raise RuntimeError(
                f"Truncated/corrupt safetensors data for tensor {key!r} in: {path} "
                f"(declared end={abs_end}, file={size})"
            )


def is_safetensors_file(file_path: str, framework: str = "pt") -> bool:
    """
    Returns True if the file *appears* to be a safetensors file.

    Note: this avoids calling `safetensors.safe_open` because corrupt/truncated
    files can sometimes trigger hard crashes when mapped by native code.
    """
    _ = framework  # kept for backward compatibility; validation is framework-agnostic
    try:
        path = os.fspath(file_path)
        if not path.lower().endswith(".safetensors"):
            return False
        _validate_safetensors_header(path)
        return True
    except Exception:
        return False


def is_floating_point_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]


def load_safetensors(
    filename: Union[str, os.PathLike],
    device: Union[str, int] = "cpu",
    dtype: Any = None,
    framework: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result: Dict[str, torch.Tensor] = {}

    # Fail fast with a clear error instead of risking a hard crash later.
    _validate_safetensors_header(filename)

    # Cache local variables and flags outside the loop to minimize
    # repeated attribute lookups and isinstance checks in the hot path.
    framework_is_np = framework == "np"
    has_dtype = dtype is not None
    is_torch_dtype = isinstance(dtype, torch.dtype)
    is_mx_dtype = mx is not None and isinstance(dtype, mx.Dtype)
    to_mx_array = mx.array if mx is not None else None
    is_fp = is_floating_point_tensor

    if (framework_is_np or is_mx_dtype) and mx is None:
        raise RuntimeError(
            "Requested an MLX/numpy-style safetensors load, but MLX is not available "
            "on this platform/environment."
        )

    with safetensors.safe_open(filename, framework=framework, device=device) as f:
        # Fast path: no framework conversion and no dtype casting.
        get_key_func = getattr(f, "offset_keys", f.keys)
        if not framework_is_np and not has_dtype:
            for k in get_key_func():
                result[k] = f.get_tensor(k)
            return result

        # General path with optional framework and/or dtype conversion.
        for k in get_key_func():
            tensor = f.get_tensor(k)

            if framework_is_np:
                tensor = to_mx_array(tensor)  # type: ignore[misc]

            if has_dtype:
                if (
                    is_torch_dtype
                    and isinstance(tensor, torch.Tensor)
                    and is_fp(tensor)
                ):
                    if tensor.dtype != dtype:
                        tensor = tensor.to(dtype, copy=False, non_blocking=True)
                elif is_mx_dtype and mx is not None and isinstance(tensor, mx.array):
                    if tensor.dtype != dtype:
                        tensor = tensor.astype(dtype)

            result[k] = tensor
    return result
