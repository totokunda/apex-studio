from __future__ import annotations

import os
import re
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterator, Iterable, Tuple, Any, Optional

try:
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover - MLX is not available on Windows/Linux
    mx = None  # type: ignore
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from loguru import logger
from tqdm import tqdm

# ------------------------------
# Utilities
# ------------------------------

_SIZE_REGEX = re.compile(
    r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]i?B)?$", re.IGNORECASE
)


def _parse_size_to_bytes(size: int | str) -> int:
    if isinstance(size, int):
        return size
    m = _SIZE_REGEX.match(size.strip())
    if not m:
        raise ValueError(f"Invalid size string: {size}")
    num = float(m.group("num"))
    unit = (m.group("unit") or "B").upper()
    pow10 = {
        "B": 1,
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "TB": 10**12,
        "PB": 10**15,
    }
    pow2 = {
        "KIB": 2**10,
        "MIB": 2**20,
        "GIB": 2**30,
        "TIB": 2**40,
        "PIB": 2**50,
    }
    if unit in pow10:
        return int(num * pow10[unit])
    if unit in pow2:
        return int(num * pow2[unit])
    if unit == "B":
        return int(num)
    raise ValueError(f"Unknown unit in size: {size}")


def _is_weight_file(p: Path) -> bool:
    return p.suffix.lower() in {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}


def _dtype_to_mx(dtype: str | None) -> Optional[mx.Dtype]:
    if dtype is None or dtype == "auto":
        return None
    dtype = dtype.lower()
    mapping = {
        "float16": mx.float16,
        "fp16": mx.float16,
        "half": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _torch_or_np_to_mx(
    value: Any, target_dtype: Optional[mx.Dtype]
) -> Tuple[mx.array, int]:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        # Try zero-copy if it looks like a numpy array interface
        if hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.array(value)
    # check if has 5 dimensions, if so make (N,D,H,W,C)
    if arr.ndim == 5:
        arr = arr.transpose(0, 2, 3, 4, 1)  # for conv3d (N,C,D,H,W) -> (N,D,H,W,C)
    elif arr.ndim == 4:  # (N,C,H,W)
        arr = arr.transpose(0, 2, 3, 1)  # for conv2d (N,C,H,W) -> (N,H,W,C)

    mx_arr = (
        mx.array(arr, dtype=target_dtype) if target_dtype is not None else mx.array(arr)
    )
    # Estimate bytes using numpy buffer; mx may be lazy
    nbytes = int(arr.nbytes)
    return mx_arr, nbytes


def _iter_tensors_from_safetensors(file_path: Path) -> Iterator[Tuple[str, Any]]:
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            # Load one tensor at a time
            yield name, f.get_tensor(name)


def _iter_tensors_from_torch_file(file_path: Path) -> Iterator[Tuple[str, Any]]:
    # Best-effort memory-conscious load using mmap; may still materialize dict
    try:
        state = torch.load(
            str(file_path), map_location="cpu", mmap=True, weights_only=True
        )
    except TypeError:
        state = torch.load(str(file_path), map_location="cpu", mmap=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Unsupported torch checkpoint structure in {file_path}; expected a state dict."
        )
    for name, tensor in state.items():
        yield name, tensor


def _iter_all_tensors(paths: Iterable[Path]) -> Iterator[Tuple[str, Any]]:
    for p in paths:
        suffix = p.suffix.lower()
        if suffix == ".safetensors":
            yield from _iter_tensors_from_safetensors(p)
        elif suffix in {".pt", ".pth", ".bin", ".ckpt"}:
            yield from _iter_tensors_from_torch_file(p)
        else:
            continue


def _write_index_file(
    index_path: Path,
    weight_map: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "weight_map": weight_map,
        "metadata": metadata or {},
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_weights(path: Path | str):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".safetensors":
        return safetensors_load_file(path)
    else:
        return torch.load(path, map_location="cpu", weights_only=True)


def convert_weights_to_mlx(
    path: Path | str,
    output_path: Path | str,
    max_shard_size: int | str = "5GB",
    dtype: str | None = None,
    shard_prefix: str = "model",
    overwrite: bool = True,
    copy_non_weight_files: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Convert a directory or single torch/safetensors file to MLX safetensors shards.

    - Streams tensors one by one (lazy for .safetensors) to avoid loading everything.
    - Groups tensors into shards up to `max_shard_size` and saves via `mx.save_safetensors`.
    - Copies non-weight files from the source directory to the output directory if requested.

    Returns a small report dict with counts and paths.
    """
    if isinstance(path, str):
        path = Path(path)
    if isinstance(output_path, str):
        output_path = Path(output_path)

    shard_limit_bytes = _parse_size_to_bytes(max_shard_size)
    target_dtype = _dtype_to_mx(dtype)

    # Prepare output directory
    if output_path.exists():
        if not overwrite and any(output_path.iterdir()):
            raise FileExistsError(
                f"Output directory {output_path} exists and is not empty. Pass overwrite=True to proceed."
            )
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    # Discover input weight files
    if path.is_file():
        model_paths = [path]
        src_dir = path.parent
    elif path.is_dir():
        model_paths = [p for p in path.glob("*") if p.is_file() and _is_weight_file(p)]
        src_dir = path
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Optionally copy non-weight files
    if copy_non_weight_files and src_dir.is_dir():
        for item in src_dir.iterdir():
            dest = output_path / item.name
            if item.is_dir():
                # Copy entire subdirectories as-is
                if dest.exists() and not overwrite:
                    continue
                shutil.copytree(item, dest, dirs_exist_ok=True)
            elif item.is_file():
                if _is_weight_file(item):
                    # Skip original weight files; we are producing MLX shards
                    continue
                shutil.copy2(item, dest)

    # Streaming conversion and sharding
    shard_idx = 1
    tensors_in_shard: Dict[str, mx.array] = {}
    current_shard_bytes = 0
    total_tensors = 0
    total_bytes = 0
    weight_map: Dict[str, str] = {}
    written_files: list[str] = []

    def flush_shard() -> None:
        nonlocal tensors_in_shard, current_shard_bytes, shard_idx
        if not tensors_in_shard:
            return
        shard_name = f"{shard_prefix}-{shard_idx:05d}.safetensors"
        shard_path = output_path / shard_name
        if dry_run:
            logger.info(
                f"[DRY-RUN] Would save shard {shard_path} with {len(tensors_in_shard)} tensors, ~{current_shard_bytes} bytes"
            )
        else:
            mx.save_safetensors(str(shard_path), tensors_in_shard)
        written_files.append(shard_name)
        shard_idx += 1
        tensors_in_shard = {}
        current_shard_bytes = 0

    for name, tensor in tqdm(_iter_all_tensors(model_paths), desc="Converting weights"):
        # Convert and measure
        mx_tensor, nbytes = _torch_or_np_to_mx(tensor, target_dtype)

        # If this single tensor would exceed the shard limit and shard has content, flush first
        if current_shard_bytes > 0 and current_shard_bytes + nbytes > shard_limit_bytes:
            flush_shard()

        tensors_in_shard[name] = mx_tensor
        current_shard_bytes += nbytes
        total_tensors += 1
        total_bytes += nbytes

        # Record where this tensor will live
        current_shard_name = f"{shard_prefix}-{shard_idx:05d}.safetensors"
        weight_map[name] = current_shard_name

        # If the shard now exceeds limit, flush immediately for back-to-back giant tensors
        if current_shard_bytes >= shard_limit_bytes:
            flush_shard()

    # Final flush
    flush_shard()

    # Write index file
    index_name = f"{shard_prefix}.safetensors.index.json"
    index_path = output_path / index_name
    metadata = {
        "format": "mlx",
        "dtype": dtype or "auto",
        "total_tensors": total_tensors,
        "total_input_files": len(model_paths),
        "total_output_files": len(written_files),
        "approx_total_bytes": total_bytes,
    }
    if dry_run:
        logger.info(f"[DRY-RUN] Would write index file at {index_path}")
    else:
        _write_index_file(index_path, weight_map, metadata)

    return {
        "output_dir": str(output_path),
        "index_file": str(index_path),
        "written_files": written_files,
        "total_tensors": total_tensors,
        "approx_total_bytes": total_bytes,
    }
