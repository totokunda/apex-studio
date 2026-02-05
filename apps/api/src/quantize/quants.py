from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence
from math import log2, ceil
from numpy.typing import DTypeLike
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, QK_K
from gguf.lazy import LazyNumpyTensor
import numpy as np
from enum import Enum
import gguf
import os
import sys
import sysconfig
import warnings
from importlib import import_module
from pathlib import Path

_GGML_QUANTS_IMPORT_WARNED = False


def _load_ggml_quants():  # pragma: no cover
    global _GGML_QUANTS_IMPORT_WARNED
    try:
        return import_module(f"{__package__}._ggml_quants")  # type: ignore[no-any-return]
    except Exception as exc:
        if not _GGML_QUANTS_IMPORT_WARNED and os.environ.get(
            "APEX_GGML_QUANTS_VERBOSE", "0"
        ) in {"1", "true", "True"}:
            expected_suffix = sysconfig.get_config_var("EXT_SUFFIX")
            mod_dir = Path(__file__).resolve().parent
            built = sorted(mod_dir.glob("_ggml_quants*.so"))
            if built:
                hint = (
                    f"Found `{built[0].name}` but this Python expects extension suffix "
                    f"`{expected_suffix}`. Rebuild with: `{sys.executable} scripts/build_ggml_quants_ext.py`."
                )
            else:
                hint = (
                    f"Build the extension with: `{sys.executable} scripts/build_ggml_quants_ext.py`."
                )
            warnings.warn(
                f"ggml quants extension unavailable ({exc}). {hint}",
                RuntimeWarning,
                stacklevel=2,
            )
            _GGML_QUANTS_IMPORT_WARNED = True
        return None


_ggml_quants = _load_ggml_quants()


def _use_ggml_quants() -> bool:
    if _ggml_quants is None:
        return False
    # Allow opt-out for debugging / bisects.
    return os.environ.get("APEX_DISABLE_GGML_QUANTS", "0") not in {"1", "true", "True"}


class QuantConfig:
    ftype: gguf.LlamaFileType
    qtype: gguf.GGMLQuantizationType

    def __init__(self, ftype: gguf.LlamaFileType, qtype: gguf.GGMLQuantizationType):
        self.ftype = ftype
        self.qtype = qtype


class QuantType(Enum):
    AUTO = "AUTO"
    F32 = "F32"
    F16 = "F16"
    BF16 = "BF16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q5_1 = "Q5_1"
    Q5_0 = "Q5_0"
    Q4_K_L = "Q4_K_L"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q4_1 = "Q4_1"
    Q4_0 = "Q4_0"
    Q3_K = "Q3_K"
    Q3_K_L = "Q3_K_L"
    Q3_K_M = "Q3_K_M"
    Q3_K_S = "Q3_K_S"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"


qconfig_map: dict[str, QuantConfig] = {
    "AUTO": QuantConfig(gguf.LlamaFileType.GUESSED, gguf.GGMLQuantizationType.F16),
    "F32": QuantConfig(gguf.LlamaFileType.ALL_F32, gguf.GGMLQuantizationType.F32),
    "F16": QuantConfig(gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16),
    "BF16": QuantConfig(gguf.LlamaFileType.MOSTLY_BF16, gguf.GGMLQuantizationType.BF16),
    "Q8_0": QuantConfig(gguf.LlamaFileType.MOSTLY_Q8_0, gguf.GGMLQuantizationType.Q8_0),
    "Q6_K": QuantConfig(gguf.LlamaFileType.MOSTLY_Q6_K, gguf.GGMLQuantizationType.Q6_K),
    "Q5_K_M": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q5_K_M, gguf.GGMLQuantizationType.Q5_K
    ),
    "Q5_K_S": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q5_K_S, gguf.GGMLQuantizationType.Q5_K
    ),
    "Q5_1": QuantConfig(gguf.LlamaFileType.MOSTLY_Q5_1, gguf.GGMLQuantizationType.Q5_1),
    "Q5_0": QuantConfig(gguf.LlamaFileType.MOSTLY_Q5_0, gguf.GGMLQuantizationType.Q5_0),
    "Q4_K_M": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q4_K_M, gguf.GGMLQuantizationType.Q4_K
    ),
    "Q4_K_S": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q4_K_S, gguf.GGMLQuantizationType.Q4_K
    ),
    "Q4_1": QuantConfig(gguf.LlamaFileType.MOSTLY_Q4_1, gguf.GGMLQuantizationType.Q4_1),
    "Q4_0": QuantConfig(gguf.LlamaFileType.MOSTLY_Q4_0, gguf.GGMLQuantizationType.Q4_0),
    "Q3_K_L": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q3_K_L, gguf.GGMLQuantizationType.Q3_K
    ),
    "Q3_K_M": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q3_K_M, gguf.GGMLQuantizationType.Q3_K
    ),
    "Q3_K_S": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q3_K_S, gguf.GGMLQuantizationType.Q3_K
    ),
    "Q2_K": QuantConfig(gguf.LlamaFileType.MOSTLY_Q2_K, gguf.GGMLQuantizationType.Q2_K),
    "Q2_K_S": QuantConfig(
        gguf.LlamaFileType.MOSTLY_Q2_K_S, gguf.GGMLQuantizationType.Q2_K
    ),
}


# This is faster than np.vectorize and np.apply_along_axis because it works on more than one row at a time
def _apply_over_grouped_rows(
    func: Callable[[np.ndarray], np.ndarray],
    arr: np.ndarray,
    otype: DTypeLike,
    oshape: tuple[int, ...],
) -> np.ndarray:
    """
    Apply `func` over the last-dimension "rows" of `arr` efficiently.

    `func` is expected to map a 2D array of shape (n_rows, row_len) to another
    2D array (n_rows, out_row_len) with a fixed per-row output size.

    This replaces the previous `np.array_split` + list-comprehension +
    `np.concatenate` approach with a chunked in-place fill to avoid:
    - Python list allocation of intermediate results
    - extra temporary concatenation buffers
    - excessive Python overhead for large tensors
    """

    # Fast empty handling (keeps shape/dtype contract).
    if arr.size == 0:
        return np.empty(oshape, dtype=otype)

    rows = arr.reshape((-1, arr.shape[-1]))

    out = np.empty(oshape, dtype=otype)
    out_rows = out.reshape((-1, out.shape[-1]))
    assert out_rows.shape[0] == rows.shape[0]

    # Choose a chunk size that keeps the working set reasonable while
    # minimizing Python loop overhead.
    #
    # Target ~4MiB of output per chunk (empirically good on most CPUs).
    out_row_bytes = out_rows.shape[1] * np.dtype(otype).itemsize
    target_bytes = 4 * 1024 * 1024
    if out_row_bytes <= 0:
        chunk_rows = rows.shape[0]
    else:
        chunk_rows = max(1, target_bytes // out_row_bytes)
        # Clamp to keep `func` vectorized and amortize overhead.
        chunk_rows = int(min(max(chunk_rows, 64), 16384))

    for i in range(0, rows.shape[0], chunk_rows):
        j = min(i + chunk_rows, rows.shape[0])
        out_rows[i:j, :] = func(rows[i:j])

    return out


# round away from zero
# ref: https://stackoverflow.com/a/59143326/22827863
def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


class QuantError(Exception): ...


_type_traits: dict[GGMLQuantizationType, type[__Quant]] = {}


def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    elif qtype == GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.quantize(data)
    else:
        raise NotImplementedError(
            f"Quantization for {qtype.name} is not yet implemented"
        )


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.view(np.float32)
    elif qtype == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.dequantize(data)
    else:
        raise NotImplementedError(
            f"Dequantization for {qtype.name} is not yet implemented"
        )


class __Quant(ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int

    grid: np.ndarray[Any, np.dtype[np.float32]] | None = None
    grid_shape: tuple[int, int] = (0, 0)
    grid_map: tuple[int | float, ...] = ()
    grid_hex: bytes | None = None

    def __init__(self):
        return TypeError("Quant conversion classes can't have instances")

    def __init_subclass__(cls, qtype: GGMLQuantizationType) -> None:
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]
        cls.__quantize_lazy = LazyNumpyTensor._wrap_fn(
            cls.__quantize_array, meta_noop=(np.uint8, cls.__shape_to_bytes)
        )
        cls.__dequantize_lazy = LazyNumpyTensor._wrap_fn(
            cls.__dequantize_array, meta_noop=(np.float32, cls.__shape_from_bytes)
        )
        assert qtype not in _type_traits
        _type_traits[qtype] = cls

    @classmethod
    def init_grid(cls):
        if cls.grid is not None or cls.grid_hex is None:
            return

        bits_per_elem = ceil(log2(len(cls.grid_map)))
        assert bits_per_elem != 0, cls.qtype.name
        elems_per_byte = 8 // bits_per_elem

        grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
        # decode hexadecimal chars from grid
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array(
            [4, 0], dtype=np.uint8
        ).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        # unpack the grid values
        grid = grid.reshape((-1, 1)) >> np.array(
            [i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8
        ).reshape((1, elems_per_byte))
        grid = (grid & ((1 << bits_per_elem) - 1)).reshape((-1, 1))
        grid_map = np.array(cls.grid_map, dtype=np.float32).reshape((1, -1))
        grid = np.take_along_axis(grid_map, grid, axis=-1)
        cls.grid = grid.reshape((1, 1, *cls.grid_shape))

    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))

    @classmethod
    def dequantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))

    @classmethod
    def __shape_to_bytes(cls, shape: Sequence[int]):
        return gguf.quant_shape_to_byte_shape(shape, cls.qtype)

    @classmethod
    def __shape_from_bytes(cls, shape: Sequence[int]):
        return gguf.quant_shape_from_byte_shape(shape, cls.qtype)

    @classmethod
    def __quantize_array(cls, array: np.ndarray) -> np.ndarray:
        if _use_ggml_quants():
            try:
                return _ggml_quants.quantize(array, int(cls.qtype))
            except NotImplementedError:
                pass
        return _apply_over_grouped_rows(
            cls.quantize_rows,
            arr=array,
            otype=np.uint8,
            oshape=cls.__shape_to_bytes(array.shape),
        )

    @classmethod
    def __dequantize_array(cls, array: np.ndarray) -> np.ndarray:
        if _use_ggml_quants():
            try:
                return _ggml_quants.dequantize(array, int(cls.qtype))
            except NotImplementedError:
                pass
        cls.init_grid()
        return _apply_over_grouped_rows(
            cls.dequantize_rows,
            arr=array,
            otype=np.float32,
            oshape=cls.__shape_from_bytes(array.shape),
        )

    @classmethod
    def __quantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def __dequantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def can_quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> bool:
        return tensor.shape[-1] % cls.block_size == 0

    @classmethod
    def quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if not cls.can_quantize(tensor):
            return tensor
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__quantize_lazy(tensor)
        else:
            return cls.__quantize_array(tensor)

    @classmethod
    def dequantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__dequantize_lazy(tensor)
        else:
            return cls.__dequantize_array(tensor)


class BF16(__Quant, qtype=GGMLQuantizationType.BF16):
    @classmethod
    # same as ggml_compute_fp32_to_bf16 in ggml-impl.h
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n = blocks.view(np.uint32)
        # force nan to quiet
        n = np.where(
            (n & 0x7FFFFFFF) > 0x7F800000,
            (n & np.uint32(0xFFFF0000)) | np.uint32(64 << 16),
            n,
        )
        # round to nearest even
        n = (np.uint64(n) + (0x7FFF + ((n >> 16) & 1))) >> 16
        return n.astype(np.uint16).view(np.uint8)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)


class Q4_0(__Quant, qtype=GGMLQuantizationType.Q4_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -8
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        # FIXME: Q4_0's reference rounding is cursed and depends on FMA
        qs = (
            np.trunc(
                (np.float64(blocks) * np.float64(id)) + np.float64(8.5),
                dtype=np.float32,
            )
            .astype(np.uint8)
            .clip(0, 15)
        )

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        return d * qs.astype(np.float32)


class Q4_1(__Quant, qtype=GGMLQuantizationType.Q4_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)

        d = (max - min) / 15
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = (
            np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32)
            .astype(np.uint8)
            .clip(0, 15)
        )

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)

        return np.concatenate([d, m, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)

        return (d * qs) + m


class Q5_0(__Quant, qtype=GGMLQuantizationType.Q5_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -16
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        # FIXME: Q5_0's reference rounding is cursed and depends on FMA
        q = (
            np.trunc(
                (np.float64(blocks) * np.float64(id)) + np.float64(16.5),
                dtype=np.float32,
            )
            .astype(np.uint8)
            .clip(0, 31)
        )

        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

        qh = np.packbits(
            q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little"
        ).reshape(n_blocks, 4)

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array(
            [i for i in range(32)], dtype=np.uint32
        ).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)

        return d * qs.astype(np.float32)


class Q5_1(__Quant, qtype=GGMLQuantizationType.Q5_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)

        d = (max - min) / 31
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        q = (
            np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32)
            .astype(np.uint8)
            .clip(0, 31)
        )

        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

        qh = np.packbits(
            q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little"
        ).reshape(n_blocks, 4)

        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)

        return np.concatenate([d, m, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        m, rest = np.hsplit(rest, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array(
            [i for i in range(32)], dtype=np.uint32
        ).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.float32)

        return (d * qs) + m


class Q8_0(__Quant, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    # Implementation of Q8_0 with bit-exact same results as reference implementation in ggml-quants.c
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:

        d = abs(blocks).max(axis=1, keepdims=True) / 127
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)

        # (n_blocks, 2)
        d = d.astype(np.float16).view(np.uint8)
        # (n_blocks, block_size)
        qs = qs.astype(np.int8).view(np.uint8)

        return np.concatenate([d, qs], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)

        return x * d


class Q2_K(__Quant, qtype=GGMLQuantizationType.Q2_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        """
        Quantizes a numpy array of floats into Q2_K format.
        Vectorized implementation of the C++ reference code (quantize_row_q2_K_ref).
        """
        # blocks can come in any shape as long as the last dim is a multiple of QK_K
        if blocks.shape[-1] % QK_K != 0:
            raise ValueError(
                f"The last dimension of the input array must be a multiple of {QK_K}, "
                f"but got {blocks.shape[-1]}"
            )

        # Flatten to (n_blocks, QK_K)
        n_blocks = blocks.size // QK_K
        sub_blocks = blocks.reshape((n_blocks, QK_K // 16, 16)).astype(
            np.float32, copy=False
        )

        # --- Vectorized make_qkx2_quants logic (n=16, nmax=3, use_mad=true) ---
        nmax = 3
        rmin = -0.5
        rdelta = 0.1
        nstep = 15

        # Weights: w[i] = |x[i]|
        weights = np.abs(sub_blocks)
        sum_w = np.sum(weights, axis=-1, keepdims=True)
        sum_x = np.sum(weights * sub_blocks, axis=-1, keepdims=True)

        # Initial min / max per 16-element sub-block
        min_v = np.min(sub_blocks, axis=-1, keepdims=True)
        max_v = np.max(sub_blocks, axis=-1, keepdims=True)
        # As in make_qkx2_quants: clamp min to be <= 0
        min_v[min_v > 0] = 0.0

        max_minus_min = max_v - min_v

        # Handle flat sub-blocks
        is_flat = max_minus_min < 1e-8
        max_minus_min[is_flat] = 1.0  # avoid division by zero

        with np.errstate(divide="ignore", invalid="ignore"):
            iscale = nmax / max_minus_min
        scale = 1.0 / iscale
        scale[is_flat] = 0.0

        # Initial codes and error (use MAD as in ref: use_mad = true)
        l_current = (
            np_roundf(iscale * (sub_blocks - min_v)).clip(0, nmax).astype(np.uint8)
        )
        diff = scale * l_current + min_v - sub_blocks
        best_error = np.sum(weights * np.abs(diff), axis=-1)

        scale_best = scale.squeeze(-1)
        min_best = min_v.squeeze(-1)

        # Iterative search over r in [rmin, rmin + nstep * rdelta]
        for is_ in range(nstep + 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                current_iscale = (rmin + rdelta * is_ + nmax) / max_minus_min
            current_iscale[is_flat] = 0.0

            l_aux = (
                np_roundf(current_iscale * (sub_blocks - min_v))
                .clip(0, nmax)
                .astype(np.uint8)
            )

            w_l = weights * l_aux
            sum_l = np.sum(w_l, axis=-1, keepdims=True)
            sum_l2 = np.sum(w_l * l_aux, axis=-1, keepdims=True)
            sum_xl = np.sum(w_l * sub_blocks, axis=-1, keepdims=True)

            D = sum_w * sum_l2 - sum_l * sum_l
            valid_D_mask = D > 0

            # Least-squares solution for scale and min
            this_scale = np.divide(
                (sum_w * sum_xl - sum_x * sum_l),
                D,
                out=np.zeros_like(D),
                where=valid_D_mask,
            )
            this_min = np.divide(
                (sum_l2 * sum_x - sum_l * sum_xl),
                D,
                out=np.zeros_like(D),
                where=valid_D_mask,
            )

            # If candidate min > 0, force min = 0 and recompute scale
            min_gt_zero_mask = valid_D_mask & (this_min > 0)
            if np.any(min_gt_zero_mask):
                recalc_scale = np.divide(
                    sum_xl,
                    sum_l2,
                    out=np.zeros_like(sum_xl),
                    where=sum_l2 > 0,
                )
                this_scale = np.where(min_gt_zero_mask, recalc_scale, this_scale)
                this_min = np.where(min_gt_zero_mask, 0.0, this_min)

            # Current error using MAD
            diff = this_scale * l_aux + this_min - sub_blocks
            current_error = np.sum(weights * np.abs(diff), axis=-1)

            improvement_mask = valid_D_mask.squeeze(-1) & (current_error < best_error)
            if np.any(improvement_mask):
                best_error[improvement_mask] = current_error[improvement_mask]
                scale_best[improvement_mask] = this_scale.squeeze(-1)[improvement_mask]
                min_best[improvement_mask] = this_min.squeeze(-1)[improvement_mask]

        scales_all = scale_best  # (n_blocks, QK_K // 16)
        mins_all = -min_best  # store positive mins as in ref

        # --- Block-level d and dmin + packing 4-bit scales/mins ---
        q4scale = 15.0

        max_scale_per_block = np.max(scales_all, axis=1, keepdims=True)
        max_min_per_block = np.max(mins_all, axis=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            inv_scale = np.where(
                max_scale_per_block == 0.0, 0.0, q4scale / max_scale_per_block
            )
            inv_min = np.where(
                max_min_per_block == 0.0, 0.0, q4scale / max_min_per_block
            )

        ls = np.clip(np_roundf(scales_all * inv_scale), 0, 15).astype(np.uint8)
        lm = np.clip(np_roundf(mins_all * inv_min), 0, 15).astype(np.uint8)

        # One byte per 16-element sub-block: low 4 bits = ls, high 4 bits = lm
        scales_packed = (ls & np.uint8(0x0F)) | ((lm & np.uint8(0x0F)) << 4)

        # Block-level d and dmin (stored as f16)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_val = np.where(
                max_scale_per_block == 0.0,
                0.0,
                max_scale_per_block / q4scale,
            )
            dmin_val = np.where(
                max_min_per_block == 0.0, 0.0, max_min_per_block / q4scale
            )

        d = d_val.reshape(n_blocks, 1).astype(np.float16).view(np.uint8)
        dmin = dmin_val.reshape(n_blocks, 1).astype(np.float16).view(np.uint8)

        # --- Final re-quantization to 2 bits using block-level d/dmin + packed scales/mins ---
        d_eff = (d_val * ls.astype(np.float32)).reshape(n_blocks, -1, 1)
        m_eff = (dmin_val * lm.astype(np.float32)).reshape(n_blocks, -1, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            L_float = np.divide(
                sub_blocks + m_eff,
                d_eff,
                out=np.zeros_like(sub_blocks),
                where=d_eff != 0,
            )

        L = np.clip(np_roundf(L_float), 0, 3).astype(np.uint8)

        # Pack 2-bit codes into bytes to match dequantize_blocks layout
        # Shape: (n_blocks, QK_K // 16, 16) -> (n_blocks, groups, 4, 32)
        L_grouped = L.reshape(n_blocks, -1, 4, 32)
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
        qs_bytes = np.sum(
            (L_grouped & np.uint8(0x03)) << shifts, axis=2, dtype=np.uint8
        ).reshape(n_blocks, -1)

        # Assemble final block: [scales (QK_K//16), qs (QK_K//4), d (2), dmin (2)]
        return np.concatenate([scales_packed, qs_bytes, d, dmin], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        scales, rest = np.hsplit(blocks, [QK_K // 16])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        d, dmin = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        # (n_blocks, 16, 1)
        dl = (d * (scales & 0xF).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
        ml = (dmin * (scales >> 4).astype(np.float32)).reshape(
            (n_blocks, QK_K // 16, 1)
        )

        shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))

        qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)

        qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)

        qs = dl * qs - ml

        return qs.reshape((n_blocks, -1))


class Q3_K(__Quant, qtype=GGMLQuantizationType.Q3_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        """
        Quantizes a numpy array of floats into Q3_K format.
        Vectorized implementation of the C++ reference code.
        """
        n_blocks = blocks.shape[0]
        sub_blocks = blocks.reshape((n_blocks, 16, 16))

        # --- Vectorized make_qx_quants logic for per-sub-block scales ---
        nmax_data = 4  # Quantization range for data: [-4, 3]

        flat_sub_blocks = sub_blocks.reshape(-1, 16)
        weights_data = flat_sub_blocks * flat_sub_blocks  # rmse_type=1 uses w=x*x

        # Find max absolute values for each sub-block
        abs_sub_blocks = np.abs(flat_sub_blocks)
        max_indices = np.argmax(abs_sub_blocks, axis=-1, keepdims=True)
        max_vals = np.take_along_axis(flat_sub_blocks, max_indices, axis=-1)

        # Iteratively find the best scale for each sub-block
        with np.errstate(divide="ignore", invalid="ignore"):
            initial_iscale = np.where(max_vals == 0, 0, -nmax_data / max_vals)

        # Initial calculation (is=0)
        l = np_roundf(flat_sub_blocks * initial_iscale).clip(-nmax_data, nmax_data - 1)
        sumlx = np.sum(weights_data * flat_sub_blocks * l, axis=-1)
        suml2 = np.sum(weights_data * l * l, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            current_scales = np.divide(
                sumlx, suml2, out=np.zeros_like(sumlx), where=suml2 != 0
            )

        best_scores = current_scales * sumlx
        best_scales = current_scales.copy()

        # Iterative search over potential iscale adjustments
        for is_ in range(-9, 10):
            if is_ == 0:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                iscale_try = -(nmax_data + 0.1 * is_) / max_vals
                iscale_try[max_vals == 0] = 0

            l_try = np_roundf(flat_sub_blocks * iscale_try).clip(
                -nmax_data, nmax_data - 1
            )
            sumlx_try = np.sum(weights_data * flat_sub_blocks * l_try, axis=-1)
            suml2_try = np.sum(weights_data * l_try * l_try, axis=-1)

            improvement_mask = (suml2_try > 0) & (
                sumlx_try * sumlx_try * suml2 > best_scores * suml2_try
            )
            if np.any(improvement_mask):
                with np.errstate(divide="ignore", invalid="ignore"):
                    scales_try = np.divide(
                        sumlx_try,
                        suml2_try,
                        out=np.zeros_like(sumlx_try),
                        where=suml2_try != 0,
                    )
                best_scores[improvement_mask] = (scales_try * sumlx_try)[
                    improvement_mask
                ]
                best_scales[improvement_mask] = scales_try[improvement_mask]
                # Update suml2 for the next comparison in the loop
                suml2[improvement_mask] = suml2_try[improvement_mask]

        scales = best_scales.reshape(n_blocks, 16)

        # --- Vectorized logic to quantize the scales themselves ---
        nmax_scales = 32  # Quantization range for scales: [-32, 31]
        abs_scales = np.abs(scales)
        max_scale_indices = np.argmax(abs_scales, axis=-1, keepdims=True)
        max_scale_vals = np.take_along_axis(scales, max_scale_indices, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            iscale_s = np.where(max_scale_vals == 0, 0, -nmax_scales / max_scale_vals)

        l_s = np_roundf(scales * iscale_s).clip(-nmax_scales, nmax_scales - 1)
        d_val = np.divide(
            np.sum(scales * l_s, axis=-1, keepdims=True),
            np.sum(l_s * l_s, axis=-1, keepdims=True),
            out=np.zeros((n_blocks, 1)),
            where=np.sum(l_s * l_s, axis=-1, keepdims=True) != 0,
        )

        # Pack the 6-bit quantized scales into 12 bytes
        l = (l_s + 32).astype(np.uint8)
        scales_packed = np.zeros((n_blocks, 12), dtype=np.uint8)
        l_low = l & 0x0F
        l_high = (l >> 4) & 0x03
        scales_packed[:, 0:8] = l_low[:, 0:8] | (l_low[:, 8:16] << 4)
        l_high_reshaped = l_high.reshape(n_blocks, 4, 4).transpose(0, 2, 1)
        packed_high_bits = (
            l_high_reshaped[:, :, 0]
            | (l_high_reshaped[:, :, 1] << 2)
            | (l_high_reshaped[:, :, 2] << 4)
            | (l_high_reshaped[:, :, 3] << 6)
        )
        scales_packed[:, 8:12] = packed_high_bits
        d = d_val.astype(np.float16).view(np.uint8)

        # --- Re-quantize data with final scales and pack ---
        sc_dequant = (l.astype(np.int8) - 32).astype(np.float32)
        d_eff = (d_val * sc_dequant).reshape(n_blocks, 16, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            l_data_float = np.divide(
                sub_blocks, d_eff, out=np.zeros_like(sub_blocks), where=d_eff != 0
            )

        l_data = (np.clip(np_roundf(l_data_float), -4, 3) + 4).astype(np.uint8)
        l_data = l_data.reshape(n_blocks, 256)

        # hmask stores the 3rd bit
        hmask_values = (l_data > 3).reshape(n_blocks, 8, 32).transpose(0, 2, 1)
        hmask = np.packbits(hmask_values, axis=-1, bitorder="little").reshape(
            n_blocks, -1
        )

        # qs stores the lower 2 bits
        l_data[l_data > 3] -= 4
        l_data_low = (l_data & 0x03).reshape(n_blocks, 2, 4, 32)
        qs_parts = (
            l_data_low[:, :, 0, :]
            | (l_data_low[:, :, 1, :] << 2)
            | (l_data_low[:, :, 2, :] << 4)
            | (l_data_low[:, :, 3, :] << 6)
        )
        qs = qs_parts.reshape(n_blocks, 64)

        return np.concatenate([hmask, qs, scales_packed, d], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        hmask, rest = np.hsplit(blocks, [QK_K // 8])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [12])

        d = d.view(np.float16).astype(np.float32)

        # The scales are packed at 6-bit each in this pattern:
        #  0: IIIIAAAA
        #  1: JJJJBBBB
        #  2: KKKKCCCC
        #  3: LLLLDDDD
        #  4: MMMMEEEE
        #  5: NNNNFFFF
        #  6: OOOOGGGG
        #  7: PPPPHHHH
        #  8: MMIIEEAA
        #  9: NNJJFFBB
        # 10: OOKKGGCC
        # 11: PPLLHHDD
        lscales, hscales = np.hsplit(scales, [8])
        lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = (lscales & np.uint8(0x0F)) | (
            (hscales & np.uint8(0x03)) << np.uint8(4)
        )
        scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)

        dl = (d * scales).reshape((n_blocks, 16, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        ).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
        qh = qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1)
        qh = qh ^ np.uint8(1)  # strangely, the offset is zero when the bitmask is 1
        q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(
            np.float32
        )

        return (dl * q).reshape((n_blocks, QK_K))


class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):
    K_SCALE_SIZE = 12
    QK_K = QK_K  # Block size

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        """
        Quantizes a numpy array of floats into Q4_K format.
        Vectorized implementation inspired by the C++ reference code.
        """
        if blocks.shape[-1] % cls.QK_K != 0:
            raise ValueError(
                f"The last dimension of the input array must be a multiple of {cls.QK_K}, but got {blocks.shape[-1]}"
            )

        n_blocks = blocks.size // cls.QK_K
        sub_blocks = blocks.reshape((n_blocks, 8, 32))

        # --- Vectorized make_qkx2_quants logic ---
        nmax = 15
        rmin = -1.0
        rdelta = 0.1
        nstep = 20

        # Calculate weights for all sub-blocks
        sum_x2 = np.sum(sub_blocks * sub_blocks, axis=-1, keepdims=True)
        # Use np.maximum to avoid sqrt of negative number due to float precision
        av_x = np.sqrt(np.maximum(0, sum_x2 / 32.0))
        weights = av_x + np.abs(sub_blocks)
        sum_w = np.sum(weights, axis=-1, keepdims=True)
        sum_x = np.sum(weights * sub_blocks, axis=-1, keepdims=True)

        # Initial guess for scales and mins
        min_v = np.min(sub_blocks, axis=-1, keepdims=True)
        max_v = np.max(sub_blocks, axis=-1, keepdims=True)
        min_v[min_v > 0] = 0.0

        max_minus_min = max_v - min_v

        # Handle cases where all values in a sub-block are the same
        is_flat = max_minus_min < 1e-8
        max_minus_min[is_flat] = 1.0  # Avoid division by zero

        with np.errstate(divide="ignore"):
            iscale = nmax / max_minus_min
        scale = 1.0 / iscale
        scale[is_flat] = 0.0

        l_current = (
            np_roundf(iscale * (sub_blocks - min_v)).clip(0, nmax).astype(np.uint8)
        )
        diff = scale * l_current + min_v - sub_blocks
        best_mse = np.sum(weights * (diff * diff), axis=-1)

        scale_best = scale.squeeze(-1)
        min_best = min_v.squeeze(-1)

        # Iterative search loop over all sub-blocks at once
        for is_ in range(nstep + 1):
            with np.errstate(divide="ignore"):
                current_iscale = (rmin + rdelta * is_ + nmax) / max_minus_min
            current_iscale[is_flat] = 0.0

            l_aux = (
                np_roundf(current_iscale * (sub_blocks - min_v))
                .clip(0, nmax)
                .astype(np.uint8)
            )

            w_l = weights * l_aux
            sum_l = np.sum(w_l, axis=-1, keepdims=True)
            sum_l2 = np.sum(w_l * l_aux, axis=-1, keepdims=True)
            sum_xl = np.sum(w_l * sub_blocks, axis=-1, keepdims=True)

            D = sum_w * sum_l2 - sum_l * sum_l

            valid_D_mask = D > 0
            # Use np.where for safe division, filling invalid entries with 0
            this_scale = np.divide(
                (sum_w * sum_xl - sum_x * sum_l),
                D,
                out=np.zeros_like(D),
                where=valid_D_mask,
            )
            this_min = np.divide(
                (sum_l2 * sum_x - sum_l * sum_xl),
                D,
                out=np.zeros_like(D),
                where=valid_D_mask,
            )

            # Handle case where candidate min > 0
            min_gt_zero_mask = valid_D_mask & (this_min > 0)
            if np.any(min_gt_zero_mask):
                recalc_scale = np.divide(
                    sum_xl, sum_l2, out=np.zeros_like(sum_xl), where=sum_l2 > 0
                )
                this_scale = np.where(min_gt_zero_mask, recalc_scale, this_scale)
                this_min = np.where(min_gt_zero_mask, 0.0, this_min)

            # Calculate current MSE
            diff = this_scale * l_aux + this_min - sub_blocks
            current_mse = np.sum(weights * (diff * diff), axis=-1)

            # Update best values where MSE has improved
            improvement_mask = valid_D_mask.squeeze(-1) & (current_mse < best_mse)
            if np.any(improvement_mask):
                best_mse[improvement_mask] = current_mse[improvement_mask]
                scale_best[improvement_mask] = this_scale.squeeze(-1)[improvement_mask]
                min_best[improvement_mask] = this_min.squeeze(-1)[improvement_mask]

        scales_all = scale_best
        mins_all = -min_best
        # --- End of vectorized search ---

        # Find block-level d and dmin
        max_scale_per_block = np.max(scales_all, axis=1, keepdims=True)
        max_min_per_block = np.max(mins_all, axis=1, keepdims=True)

        # Quantize and pack scales and mins
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_scale = np.where(
                max_scale_per_block == 0, 0, 63.0 / max_scale_per_block
            )
            inv_min = np.where(max_min_per_block == 0, 0, 63.0 / max_min_per_block)

        ls = np.clip(np_roundf(scales_all * inv_scale), 0, 63).astype(np.uint8)
        lm = np.clip(np_roundf(mins_all * inv_min), 0, 63).astype(np.uint8)

        scales_packed = np.zeros((n_blocks, cls.K_SCALE_SIZE), dtype=np.uint8)
        scales_packed[:, 0:4] = ls[:, 0:4] & 0x3F
        scales_packed[:, 4:8] = lm[:, 0:4] & 0x3F
        scales_packed[:, 8:12] = (ls[:, 4:8] & 0x0F) | ((lm[:, 4:8] & 0x0F) << 4)
        scales_packed[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
        scales_packed[:, 4:8] |= (lm[:, 4:8] >> 4) << 6

        # Store block-level d and dmin
        with np.errstate(divide="ignore", invalid="ignore"):
            d_val = np.where(max_scale_per_block == 0, 0, max_scale_per_block / 63.0)
            dmin_val = np.where(max_min_per_block == 0, 0, max_min_per_block / 63.0)

        d = d_val.reshape(n_blocks, 1).astype(np.float16).view(np.uint8)
        dmin = dmin_val.reshape(n_blocks, 1).astype(np.float16).view(np.uint8)

        # Re-quantize the actual data
        d_eff = (d_val * ls.astype(np.float32)).reshape(n_blocks, 8, 1)
        m_eff = (dmin_val * lm.astype(np.float32)).reshape(n_blocks, 8, 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            L_float = np.divide(
                sub_blocks + m_eff,
                d_eff,
                out=np.zeros_like(sub_blocks),
                where=d_eff != 0,
            )

        L = np.clip(np_roundf(L_float), 0, 15).astype(np.uint8)

        # Pack the 4-bit quantized data
        L_reshaped = L.reshape((n_blocks, cls.QK_K // 64, 2, 32))
        L_low = L_reshaped[:, :, 0, :].reshape(n_blocks, -1)
        L_high = L_reshaped[:, :, 1, :].reshape(n_blocks, -1)
        qs = L_low | (L_high << 4)

        # Assemble and return the final block
        return np.concatenate([d, dmin, scales_packed, qs], axis=1)

    @staticmethod
    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_blocks = scales.shape[0]
        s = scales.view(np.uint8).reshape(n_blocks, Q4_K.K_SCALE_SIZE)

        sc = np.zeros((n_blocks, 8), dtype=np.uint8)
        m = np.zeros((n_blocks, 8), dtype=np.uint8)

        sc[:, 0:4] = s[:, 0:4] & 0x3F
        m[:, 0:4] = s[:, 4:8] & 0x3F

        sc[:, 4:8] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
        m[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)

        return sc, m

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [cls.K_SCALE_SIZE])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = cls.get_scale_min(scales)

        d_eff = (d * sc.astype(np.float32)).reshape((n_blocks, 8, 1))
        dm_eff = (dmin * m.astype(np.float32)).reshape((n_blocks, 8, 1))

        # Unpack 4-bit values and arrange back into sub-blocks
        qs_reshaped = qs.reshape(n_blocks, QK_K // 64, 32)
        qs_unpacked = np.empty((n_blocks, 8, 32), dtype=np.float32)
        qs_unpacked[:, [0, 2, 4, 6], :] = qs_reshaped & 0x0F
        qs_unpacked[:, [1, 3, 5, 7], :] = qs_reshaped >> 4

        return (d_eff * qs_unpacked - dm_eff).reshape((n_blocks, QK_K))


class Q5_K(__Quant, qtype=GGMLQuantizationType.Q5_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        """
        Quantizes a numpy array of floats into Q5_K format.
        Vectorized implementation of the C++ reference code.
        """
        if blocks.shape[-1] % QK_K != 0:
            raise ValueError(
                f"The last dimension of the input array must be a multiple of {QK_K}, but got {blocks.shape[-1]}"
            )

        n_blocks = blocks.size // QK_K
        sub_blocks = blocks.reshape((n_blocks, 8, 32))

        # --- Vectorized make_qkx3_quants logic for 5 bits ---
        nmax = 31
        nstep = 36
        rmin = -0.9
        rdelta = 0.05

        # Calculate weights for all sub-blocks
        sum_x2 = np.sum(sub_blocks * sub_blocks, axis=-1, keepdims=True)
        av_x = np.sqrt(np.maximum(0, 2 * sum_x2 / QK_K))  # sigma calculation from C++
        weights = av_x + np.abs(sub_blocks)
        sum_w = np.sum(weights, axis=-1, keepdims=True)
        sum_x = np.sum(weights * sub_blocks, axis=-1, keepdims=True)

        min_v = np.min(sub_blocks, axis=-1, keepdims=True)
        max_v = np.max(sub_blocks, axis=-1, keepdims=True)
        min_v[min_v > 0] = 0.0

        max_minus_min = max_v - min_v
        is_flat = max_minus_min < 1e-8
        max_minus_min[is_flat] = 1.0

        # Initial mse for comparison
        with np.errstate(divide="ignore"):
            iscale_initial = nmax / max_minus_min
        scale_initial = 1.0 / iscale_initial
        scale_initial[is_flat] = 0.0
        l_initial = (
            np_roundf(iscale_initial * (sub_blocks - min_v))
            .clip(0, nmax)
            .astype(np.uint8)
        )
        diff = scale_initial * l_initial + min_v - sub_blocks
        best_mse = np.sum(weights * (diff * diff), axis=-1)

        scale_best = scale_initial.squeeze(-1)
        min_best = min_v.squeeze(-1)

        # Iterative search
        for is_ in range(nstep + 1):
            with np.errstate(divide="ignore"):
                current_iscale = (rmin + rdelta * is_ + nmax) / max_minus_min
            current_iscale[is_flat] = 0.0

            l_aux = (
                np_roundf(current_iscale * (sub_blocks - min_v))
                .clip(0, nmax)
                .astype(np.uint8)
            )
            w_l = weights * l_aux
            sum_l = np.sum(w_l, axis=-1, keepdims=True)
            sum_l2 = np.sum(w_l * l_aux, axis=-1, keepdims=True)
            sum_xl = np.sum(w_l * sub_blocks, axis=-1, keepdims=True)

            D = sum_w * sum_l2 - sum_l * sum_l
            valid_D_mask = D > 0
            this_scale = np.divide(
                (sum_w * sum_xl - sum_x * sum_l),
                D,
                out=np.zeros_like(D),
                where=valid_D_mask,
            )
            this_min = np.divide(
                (sum_l2 * sum_x - sum_l * sum_xl),
                D,
                out=np.zeros_like(D),
                where=valid_D_mask,
            )

            min_gt_zero_mask = valid_D_mask & (this_min > 0)
            if np.any(min_gt_zero_mask):
                recalc_scale = np.divide(
                    sum_xl, sum_l2, out=np.zeros_like(sum_xl), where=sum_l2 > 0
                )
                this_scale = np.where(min_gt_zero_mask, recalc_scale, this_scale)
                this_min = np.where(min_gt_zero_mask, 0.0, this_min)

            diff = this_scale * l_aux + this_min - sub_blocks
            current_mse = np.sum(weights * (diff * diff), axis=-1)
            improvement_mask = valid_D_mask.squeeze(-1) & (current_mse < best_mse)
            if np.any(improvement_mask):
                best_mse[improvement_mask] = current_mse[improvement_mask]
                scale_best[improvement_mask] = this_scale.squeeze(-1)[improvement_mask]
                min_best[improvement_mask] = this_min.squeeze(-1)[improvement_mask]

        scales_all = scale_best
        mins_all = -min_best

        # --- Quantize and pack scales/mins (identical to Q4_K) ---
        max_scale_per_block = np.max(scales_all, axis=1, keepdims=True)
        max_min_per_block = np.max(mins_all, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_scale = np.where(
                max_scale_per_block == 0, 0, 63.0 / max_scale_per_block
            )
            inv_min = np.where(max_min_per_block == 0, 0, 63.0 / max_min_per_block)
        ls = np.clip(np_roundf(scales_all * inv_scale), 0, 63).astype(np.uint8)
        lm = np.clip(np_roundf(mins_all * inv_min), 0, 63).astype(np.uint8)

        scales_packed = np.zeros((n_blocks, Q4_K.K_SCALE_SIZE), dtype=np.uint8)
        scales_packed[:, 0:4] = ls[:, 0:4] & 0x3F
        scales_packed[:, 4:8] = lm[:, 0:4] & 0x3F
        scales_packed[:, 8:12] = (ls[:, 4:8] & 0x0F) | ((lm[:, 4:8] & 0x0F) << 4)
        scales_packed[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
        scales_packed[:, 4:8] |= (lm[:, 4:8] >> 4) << 6

        # --- Store block-level d and dmin (identical to Q4_K) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            d_val = np.where(max_scale_per_block == 0, 0, max_scale_per_block / 63.0)
            dmin_val = np.where(max_min_per_block == 0, 0, max_min_per_block / 63.0)
        d = d_val.reshape(n_blocks, 1).astype(np.float16).view(np.uint8)
        dmin = dmin_val.reshape(n_blocks, 1).astype(np.float16).view(np.uint8)

        # --- Re-quantize the actual data to 5 bits ---
        d_eff = (d_val * ls.astype(np.float32)).reshape(n_blocks, 8, 1)
        m_eff = (dmin_val * lm.astype(np.float32)).reshape(n_blocks, 8, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            L_float = np.divide(
                sub_blocks + m_eff,
                d_eff,
                out=np.zeros_like(sub_blocks),
                where=d_eff != 0,
            )
        L = np.clip(np_roundf(L_float), 0, 31).astype(np.uint8)

        # --- Pack the 5-bit quantized data into qh and qs ---
        # qh (high bits)
        h = (L > 15).astype(np.uint8)
        h_reshaped = h.reshape(n_blocks, 8, 32).transpose(0, 2, 1)
        bit_shifts = 2 ** np.arange(8, dtype=np.uint8).reshape(1, 1, 8)
        qh = np.sum(h_reshaped * bit_shifts, axis=-1).astype(np.uint8)

        # qs (low bits)
        L[L > 15] -= 16
        l_reshaped = L.reshape(n_blocks, 8, 32)
        part1 = l_reshaped[:, ::2, :].reshape(n_blocks, -1)
        part2 = l_reshaped[:, 1::2, :].reshape(n_blocks, -1)
        qs = part1 | (part2 << 4)

        return np.concatenate([d, dmin, scales_packed, qh, qs], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [Q4_K.K_SCALE_SIZE])
        qh, qs = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d_eff = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm_eff = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        # Unpack high bits (qh)
        bit_shifts = 2 ** np.arange(8, dtype=np.uint8).reshape(1, 1, 8)
        qh_unpacked = (qh[:, :, np.newaxis] & bit_shifts) != 0
        qh_unpacked = qh_unpacked.transpose(0, 2, 1).reshape(n_blocks, -1, 32)

        # Unpack low bits (qs)
        ql_unpacked = np.empty((n_blocks, 8, 32), dtype=np.uint8)
        qs_reshaped = qs.reshape(n_blocks, 4, 32)
        ql_unpacked[:, ::2, :] = qs_reshaped & 0x0F
        ql_unpacked[:, 1::2, :] = qs_reshaped >> 4

        # Combine high and low bits and dequantize
        q = (ql_unpacked + (qh_unpacked * 16)).astype(np.float32)
        return (d_eff * q - dm_eff).reshape((n_blocks, QK_K))


class Q6_K(__Quant, qtype=GGMLQuantizationType.Q6_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        """
        Quantizes a numpy array of floats into Q6_K format.
        Vectorized implementation of the C++ reference code.
        """
        n_blocks = blocks.shape[0]
        # Reshape for sub-block processing
        sub_blocks = blocks.reshape(n_blocks * 16, 16)

        # --- Vectorized `make_qx_quants` for all sub-blocks to find initial scales ---
        nmax_data = 32  # For Q6_K, data range is [-32, 31]

        # Weights are x*x for the reference implementation
        weights_data = sub_blocks * sub_blocks

        # Find max absolute values for each sub-block to determine the initial scale
        abs_sub_blocks = np.abs(sub_blocks)
        max_indices = np.argmax(abs_sub_blocks, axis=-1, keepdims=True)
        max_vals = np.take_along_axis(sub_blocks, max_indices, axis=-1)

        is_zero_mask = np.abs(max_vals) < 1e-15

        with np.errstate(divide="ignore", invalid="ignore"):
            initial_iscale = np.where(is_zero_mask, 0, -nmax_data / max_vals)

        # Use np.round for round-half-to-even, matching C's nearest_int
        l = np.round(sub_blocks * initial_iscale).clip(-nmax_data, nmax_data - 1)
        sumlx = np.sum(weights_data * sub_blocks * l, axis=-1)
        suml2 = np.sum(weights_data * l * l, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            scales_cand = np.divide(
                sumlx, suml2, out=np.zeros_like(sumlx), where=suml2 != 0
            )
        best_scores = scales_cand * sumlx
        best_l = l.copy()

        # Iterative search over potential iscale adjustments
        for is_ in range(-9, 10):
            if is_ == 0:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                iscale_try = np.where(
                    is_zero_mask, 0, -(nmax_data + 0.1 * is_) / max_vals
                )

            l_try = np.round(sub_blocks * iscale_try).clip(-nmax_data, nmax_data - 1)
            sumlx_try = np.sum(weights_data * sub_blocks * l_try, axis=-1)
            suml2_try = np.sum(weights_data * l_try * l_try, axis=-1)

            improvement_mask = (suml2_try > 0) & (
                sumlx_try * sumlx_try * suml2 > best_scores * suml2_try
            )
            if np.any(improvement_mask):
                with np.errstate(divide="ignore", invalid="ignore"):
                    new_best_scores = np.divide(
                        sumlx_try * sumlx_try, suml2_try, where=suml2_try > 0
                    )
                best_scores[improvement_mask] = new_best_scores[improvement_mask]
                best_l[improvement_mask] = l_try[improvement_mask]
                suml2[improvement_mask] = suml2_try[improvement_mask]

        # Recompute final best scales from the best quants (best_l)
        sumlx_final = np.sum(weights_data * sub_blocks * best_l, axis=-1)
        suml2_final = np.sum(weights_data * best_l * best_l, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scales = np.divide(
                sumlx_final,
                suml2_final,
                out=np.zeros_like(sumlx_final),
                where=suml2_final != 0,
            )

        scales[np.all(sub_blocks == 0, axis=-1)] = 0.0
        scales = scales.reshape(n_blocks, 16)

        # --- Quantize the scales themselves ---
        abs_scales = np.abs(scales)
        max_abs_scale_indices = np.argmax(abs_scales, axis=-1, keepdims=True)
        max_scale_vals = np.take_along_axis(scales, max_abs_scale_indices, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            is_zero_mask = np.abs(max_scale_vals) < 1e-15
            iscale_s = np.where(is_zero_mask, 0, -128.0 / max_scale_vals)
            d_val = np.where(is_zero_mask, 0, max_scale_vals / -128.0)

        quantized_scales = np.round(scales * iscale_s).clip(-128, 127).astype(np.int8)
        d = d_val.astype(np.float16).view(np.uint8)

        # --- Re-quantize original data with final scales ---
        d_sub = d_val * quantized_scales.astype(np.float32)
        d_sub_reshaped = d_sub.reshape(n_blocks, 16, 1)

        sub_blocks_reshaped = blocks.reshape(n_blocks, 16, 16)
        with np.errstate(divide="ignore", invalid="ignore"):
            l_float = np.divide(
                sub_blocks_reshaped,
                d_sub_reshaped,
                out=np.zeros_like(sub_blocks_reshaped),
                where=d_sub_reshaped != 0,
            )

        l_final = np.round(l_float).clip(-32, 31).astype(np.int8)
        L = (l_final + 32).astype(np.uint8).reshape(n_blocks, 256)

        # --- Pack the 6-bit quantized data ---
        L_reshaped = L.reshape(n_blocks, 2, 4, 32)
        L_low = L_reshaped & 0xF
        L_high = L_reshaped >> 4

        # Pack lower 4 bits into ql
        ql = np.empty((n_blocks, 128), dtype=np.uint8)
        ql[:, 0:32] = L_low[:, 0, 0, :] | (L_low[:, 0, 2, :] << 4)
        ql[:, 32:64] = L_low[:, 0, 1, :] | (L_low[:, 0, 3, :] << 4)
        ql[:, 64:96] = L_low[:, 1, 0, :] | (L_low[:, 1, 2, :] << 4)
        ql[:, 96:128] = L_low[:, 1, 1, :] | (L_low[:, 1, 3, :] << 4)

        # Pack higher 2 bits into qh
        qh_packed = (
            L_high[:, :, 0, :]
            | (L_high[:, :, 1, :] << 2)
            | (L_high[:, :, 2, :] << 4)
            | (L_high[:, :, 3, :] << 6)
        )
        qh = qh_packed.reshape(n_blocks, -1)

        # Final assembly: view scales as uint8 before concatenating
        return np.concatenate([ql, qh, quantized_scales.view(np.uint8), d], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])

        scales = scales.view(np.int8).astype(np.float32)
        d = d.view(np.float16).astype(np.float32)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

        ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
        q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)

        return (d * q).reshape((n_blocks, QK_K))


class TQ1_0(__Quant, qtype=GGMLQuantizationType.TQ1_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs0, qs1, qh = (
            qs[..., : (32 * 5)],
            qs[..., (32 * 5) : (48 * 5)],
            qs[..., (48 * 5) :],
        )
        qs0 = qs0.reshape((n_blocks, -1, 5, 32)) * np.array(
            [81, 27, 9, 3, 1], dtype=np.uint8
        ).reshape((1, 1, 5, 1))
        qs0 = np.sum(qs0, axis=-2).reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 5, 16)) * np.array(
            [81, 27, 9, 3, 1], dtype=np.uint8
        ).reshape((1, 1, 5, 1))
        qs1 = np.sum(qs1, axis=-2).reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 4, 4)) * np.array(
            [81, 27, 9, 3], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qh = np.sum(qh, axis=-2).reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = (qs.astype(np.uint16) * 256 + (243 - 1)) // 243

        qs = qs.astype(np.uint8)
        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, rest = np.hsplit(blocks, [(QK_K - 4 * QK_K // 64) // 5])
        qh, d = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)

        qs0, qs1 = qs[..., :32], qs[..., 32:]
        qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * np.array(
            [1, 3, 9, 27, 81], dtype=np.uint8
        ).reshape((1, 1, 5, 1))
        qs0 = qs0.reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * np.array(
            [1, 3, 9, 27, 81], dtype=np.uint8
        ).reshape((1, 1, 5, 1))
        qs1 = qs1.reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 1, 4)) * np.array(
            [1, 3, 9, 27], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qh = qh.reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = ((qs.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)

        return d * qs.astype(np.float32)


class TQ2_0(__Quant, qtype=GGMLQuantizationType.TQ2_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs = qs.reshape((n_blocks, -1, 4, 32)) << np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qs = qs[..., 0, :] | qs[..., 1, :] | qs[..., 2, :] | qs[..., 3, :]
        qs = qs.reshape((n_blocks, -1))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, d = np.hsplit(blocks, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 1, 4, 1))
        qs = (qs & 0x03).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)

        return d * qs.astype(np.float32)


class IQ2_XXS(__Quant, qtype=GGMLQuantizationType.IQ2_XXS):
    ksigns: bytes = (
        b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
        b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
        b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
        b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
        b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
        b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
        b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
        b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
    )

    # iq2xxs_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (256, 8)
    grid_map = (0x08, 0x19, 0x2B)
    grid_hex = (
        b"00000200050008000a00110014002000220028002a0041004400500058006100"
        b"6400800082008a00a20001010401100115014001840198010002020222028202"
        b"010404041004210424044004420448046004810484049004a404000502050805"
        b"200546056905800591050906100640068406a406000805080808140828084108"
        b"440850085208880804094009020a140a01100410101021104010601084109010"
        b"951000110811201150115a118011241245120014081420142514491480141815"
        b"6215001616160118041810184018811800190519a019511a002002200a204420"
        b"6120802082202921482100220222012404241024402456240025412564259026"
        b"082820289428442a014004401040184021402440404048405640604081408440"
        b"9040004120416141804185410142104248425642684200440844204480449944"
        b"124524450046014804481048404845480049584961498249454a904a00500850"
        b"1150195020508050885004514251a4519152905492540a550156545600581158"
        b"195864584059085a046010604060686000615561186260620064056410651265"
        b"84654268008002800a8041808280048118814081118201840484108415844084"
        b"608400854685948509864086608602880489118a0490109024904090a1901691"
        b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
    )

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.view(np.uint32).reshape(n_blocks, -1, 2)

        db = (
            d
            * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32))
            * np.float32(0.25)
        )
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = qs[..., 1].reshape((n_blocks, -1, 1)) >> np.array(
            [0, 7, 14, 21], dtype=np.uint32
        ).reshape((1, 1, 4))
        ksigns = np.frombuffer(cls.ksigns, dtype=np.uint8).reshape((1, 1, 1, 128))
        signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        ).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(
            cls.grid,
            qs[..., 0].copy().view(np.uint8).reshape((n_blocks, -1, 1, 1)),
            axis=-2,
        )
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ2_XS(__Quant, qtype=GGMLQuantizationType.IQ2_XS):
    # iq2xs_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (512, 8)
    grid_map = (0x08, 0x19, 0x2B)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"49005000520055005800610064008000820085008800910094009900a0000101"
        b"04010601090110011201150118011a0121012401400142014501480151015401"
        b"6001680181018401900100020202050208021102140220024102440250025502"
        b"80028a0201040404060409041004120415041804210424044004420445044804"
        b"5104540456046004810484049004000502050505080511051405200541054405"
        b"500561058005010604061006260640064206840600080208050808080a081108"
        b"14082008250841084408500858088008a008aa08010904091009400981098909"
        b"000a200a280a960aa00a01100410061009101010121015101810211024104010"
        b"4210451048105110541060106a10811084109010001102110511081111111411"
        b"2011411144115011801194119611011204120612101240126012001402140514"
        b"0814111414142014411444144914501464148014011504151015401500161416"
        b"49160118041810181218401854188618001905196619511aa91a002002200520"
        b"08200a201120142020204120442050208020a020012104211021402148216521"
        b"002222228022a82201240424102429244024002541255225992501261a26a626"
        b"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
        b"0640094010401240154018402140244040404240454048404a40514054406040"
        b"6540814084409040004102410541084111411441204141414441504180418541"
        b"a241014204421042124229424042004402440544084411441444194420444144"
        b"4444504480449444014504451045244540459a4500460a464446504601480448"
        b"1048404845485448624800491149444950496949044a00500250055008501150"
        b"145020502850415044505050805001510451105115514051425100524452aa52"
        b"0154045410542154405460548154a154005508558055885521566856a1560058"
        b"14584158505899581a5940594259855a0160046010604060546062608660a960"
        b"006124624a62926200641664106540654565a46501686a682569066a546a626a"
        b"00800280058008801180148020802a8041804480508080808280a880aa800181"
        b"0481068110814081518159810082208280828282a082a8820184048410841284"
        b"158440846084898400854485a58518866a860088088825885a8880888288a888"
        b"0689228a808a888a968aa88a0190049010904090569084900091229164915692"
        b"89920094059444945094589429959095929541965198a6984999159a609a00a0"
        b"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
        b"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
    )

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [2 * QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        qs = qs.view(np.uint16)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape(1, 1, 128)
        signs = np.take_along_axis(signs, (qs >> 9).reshape((n_blocks, -1, 1)), axis=-1)
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        ).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(
            cls.grid, (qs & np.uint16(511)).reshape((n_blocks, -1, 1, 1)), axis=-2
        )
        grid = grid.reshape((n_blocks, -1, 2, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ2_S(__Quant, qtype=GGMLQuantizationType.IQ2_S):
    # iq2s_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (1024, 8)
    grid_map = (0x08, 0x19, 0x2B)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"490050005200550058006100640066006900800082008500880091009400a000"
        b"a500aa0001010401060109011001120115011801210124014001420145014801"
        b"510154015601590160016501680181018401900192019501a101a40100020202"
        b"050208021102140220022a02410244024602490250025502800285028a029402"
        b"a202010404040604090410041204150418042104240426042904400442044504"
        b"48044a0451045404560459046004620465048104840486048904900495049804"
        b"a104a40400050205050508050a05110514051605190520052505280541054405"
        b"46054905500552055505580561056405800582058505880591059405a0050106"
        b"0406060609061006150640064506480651065406600681068406900600080208"
        b"050808081108140816081908200825082a084108440846084908500852085508"
        b"580861086408800885089408aa08010904091009120915091809210940094509"
        b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
        b"0610091010101210151018102110241026104010421045104810511054105610"
        b"59106010621065106810811084108610901095109810a110a410001102110511"
        b"08110a1111111411161119112011221125112811411144114611491150115211"
        b"5511581161116411801182118511881191119411011204120912101215122112"
        b"2412401245125112541281128412901200140214051408141114141416141914"
        b"2014251428144114441446144914501452145514581461146414801482148514"
        b"881491149414a014011504150615091510151215151518152115241540154215"
        b"4515481551155415601581158415901500160516081611161416201641164416"
        b"50168016aa160118041806180918101815181818211840184218451848185118"
        b"541860188118841800190219051908191119141920194119441950196919a219"
        b"041a101a401a561a00200220052008201120142016201920202025202a204120"
        b"4420502052205520642080208a209420aa200121042110211221152121214021"
        b"4221452151215421602181218421902100220a22222228222a22442250228822"
        b"8a22a82201240424062409241024152418242124242440244224452448245124"
        b"5424602481248424902400250525082511251425202541254425502566258025"
        b"0126042610264026592600280528112814284128442850288a28aa2801290429"
        b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
        b"21402440264040404240454048404a4051405440564059406040624065408140"
        b"8440904095409840a140a4400041024105410841114114411641194120412241"
        b"2541414144414641494150415241554158416141644180418241854188419141"
        b"9441a04101420442104212421542184224424042454248425142544260428142"
        b"844200440244054408440a441144144416441944204422442544284441444444"
        b"46444944504452445544584461446444804482448544884491449444a0440145"
        b"0445064509451045124515451845214524454045424545454845514554456045"
        b"6a4581458445904500460246054608461146144620464146444650468046a546"
        b"0148044809481048124815481848214824484048424845484848514854486048"
        b"84489048004902490549084911491449204941494449504980499649014a044a"
        b"104a404a00500250055008501150145016501950205022502550285041504450"
        b"4650495050505250555058506150645080508250855088509150945001510451"
        b"0651095110511251155118512151245140514251455148515151545160518151"
        b"8451905100520552085211521452205241524452505269528052015404540654"
        b"0954105412541554185421542454405442544554485451545454605481548454"
        b"9054005502550555085511551455205541554455505580550156045610562656"
        b"405600580258055808581158145820584158445850585a588058015904591059"
        b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
        b"4860516054606060846090600061026105610861116114612061416144615061"
        b"806199610462106240625662a162006405640864116414642064416444645064"
        b"806401650465106540654a656865926500669466016804681068656898680069"
        b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
        b"5580588061808080858091809480018104810981108112811581188121812481"
        b"408142814581488151815481818184819081a981008205820a82118214824182"
        b"4482508201840484068409841084128415841884218440844284458448845184"
        b"5484608481848484908400850285058508851185148520854185448550858085"
        b"8a85018604861086298640860088058811881488418844885088a28801890489"
        b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
        b"4290459048905190549060908190849090900091059111911491419144915091"
        b"5a910192049210924092a6920094029405940894119414942094419444945094"
        b"8094969401950495109540959895a19500964696649601980498109826984098"
        b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
        b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
        b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
    )

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 8])
        signs, rest = np.hsplit(rest, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 32])

        d = d.view(np.float16).astype(np.float32)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        # unpack the sign bits
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        ).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 2, 4, 6], dtype=np.uint8
        ).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 0x03).astype(np.uint16) << 8).reshape(
            (n_blocks, -1)
        )

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ3_XXS(__Quant, qtype=GGMLQuantizationType.IQ3_XXS):
    grid_shape = (256, 4)
    grid_map = (0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3E)
    grid_hex = (
        b"0000020004001100130017002000220031004200730075000101030110011201"
        b"2101250130013201410154017001000202020402110220022202310233023702"
        b"5102570275020103070310031203250370031304370444045704730475040105"
        b"0705320552053506640610071407160743076107011003101010121021102310"
        b"3010321034104710501000110211111120112211011203121012121221123012"
        b"7212001302132013311346136613011405145014201524154615711505162217"
        b"4017002002201120132020202220262031204220012103210521102112212121"
        b"3021632167217021002202221122172220222222372240225522012310231423"
        b"7023742335245324032527254125742501270327162745270130103012302130"
        b"2330503065307230003102312031313144314631013203321032253252327232"
        b"1133333330344734723400350635223555351436363663363337603704401740"
        b"3540374053405740744120423742404260426642074345430444514464442545"
        b"4345704505471047124730471250415070500051065126515551145232527252"
        b"0253535310542354275472540255315550562457425724604460466064602161"
        b"6161176264623063366344640565526533660367216703700570077010703270"
        b"5270267140711272457252720073157333736073217441740075027524753076"
    )

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32)
        scales = scales.view(np.uint32)

        db = d * (np.float32(0.5) + (scales >> 28).astype(np.float32)) * np.float32(0.5)
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = scales.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 7, 14, 21], dtype=np.uint32
        ).reshape((1, 1, 4))
        ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape((1, 1, 1, 128))
        signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        ).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ3_S(__Quant, qtype=GGMLQuantizationType.IQ3_S):
    grid_shape = (512, 4)
    grid_map = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F)
    grid_hex = (
        b"0000010002000500070010001100120014001600200021002500330040004200"
        b"4500470051005300600062007100740077000001010102010401100111011501"
        b"2001230127013101350144016101650172010002010205020702100213021602"
        b"2102250230023402420245024702510253027002730203031103150320032203"
        b"3103330336034403500352036703710375030004130417042104240432044004"
        b"4304510470040205040520052205260533054105450547056605730506061106"
        b"1306310652067106000702070407200722072607330750075407001001100210"
        b"0410101011101310151017102010221031103410361054105610611072100011"
        b"0111031106111011141121113011331141115011521170117611001212121512"
        b"1712201224123212401243125512601272120113041307131013131321132713"
        b"3013341341136213701303140514121414143114331442144614501454140115"
        b"1015131521153015321551152016241627164416461601170317101712172117"
        b"3517411762177017002001200320052007201020122014201620212023202720"
        b"3020322041204320452050205220672070207320752000210221102113211721"
        b"2221252131213421422151210122042207222122232230223722412253225722"
        b"7122742200230223052311232223242331233323422350236623012407242024"
        b"2324322435244124722475240425112522253725402553257025002602260726"
        b"2126552661260527112726273027432750270230113013301530173022303130"
        b"3330353042304430473051306330713001310331053114312131233140316031"
        b"7231763100321232203232323432503201331033143321332333273330334133"
        b"4333473355337333033411341634223431345234603464340135103512352535"
        b"3235443556357335163641360137033720372237353700400440124020402440"
        b"2740324041405040704002410741114113412241304135414341514155410142"
        b"0342104215422142334240425742624270420443114313432043224331433543"
        b"0044024424443744404471440545074521456245134634466046104715473047"
        b"4347514702501050145022504050445047505250665074500151035105511251"
        b"2151325172510052115223523052365253520253075310532753445351536553"
        b"7353015404542054325446541255265551555355425602570457225711601360"
        b"1560316033606060006120612761646112623462426255626262706200631463"
        b"2163406325644364626400650365346560650566406611671367007004700770"
        b"2070227036704070547062700271117124714371457101720472107216722172"
        b"3072517202733273357353730174057413742074507422754275027631760077"
    )

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        qh, rest = np.hsplit(rest, [QK_K // 32])
        signs, scales = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (1 + 2 * scales)
        db = db.reshape((n_blocks, -1, 1, 1))

        # unpack the sign bits
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        ).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array(
            [i for i in range(8)], dtype=np.uint8
        )
        qh = (qh & 0x01).astype(np.uint16).reshape((n_blocks, -1))
        qs = qs.astype(np.uint16) | (qh << 8)

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ1_S(__Quant, qtype=GGMLQuantizationType.IQ1_S):
    # iq1s_grid, with each byte packed into 2 bits
    # -1, 0, 1 <=> 0, 1, 2
    grid_shape = (2048, 8)
    grid_map = (-1, 0, 1)
    grid_hex = (
        b"00000200050008000a00110015002000220028002a0045005100540056006500"
        b"8000820088008a009500a000a200a800aa000401050111011401160119011a01"
        b"2501410146014901520155015a0161016401660168018501910194019601a501"
        b"0002020208020a0215022002220228022a024502510259026402690280028202"
        b"88028a02910295029902a002a202a802aa021104140416042504410449045504"
        b"5a046404650491049904a5040105040505050605150518051a05290540054505"
        b"4a0550055105540555055605590560056205650568056a058105910595059805"
        b"9a05a105a405a505a605a9051406190641064406500652065506580660066106"
        b"6606690685069106940699060008020808080a0815082008220828082a084508"
        b"5108560865088008820888088a089508a008a208a808aa080509110914091909"
        b"2409250941095009510955096109640969099109940996099909a509000a020a"
        b"080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a"
        b"8a0a950aa00aa20aa80aaa0a1010111014101910241025104110441050105510"
        b"58106110641065106910911094109610a110a510011104110611091110111211"
        b"1511181121112411291145114a11501151115211541155115611591160116511"
        b"841192119511a111a41111121412161225124012461249125212551258125a12"
        b"641266128512911294129612a512011406140914141415141814191421142614"
        b"41144514461448144a1451145414551456145914621465146814841489149014"
        b"94149514981499149a14a114a414a514a914021505150a151115141515151615"
        b"191520152215251528152a154115441545154615511552155415551556155915"
        b"5a1561156415651566156915801582158415851588158a159015911594159515"
        b"961599159a15a015a215a51501160416051606161516161618161a1621162616"
        b"401642164416451648164a165116551656165816591661166416651668166916"
        b"6a1686168a1692169516a416a916111816182518411844184618491850185518"
        b"58185a1860186118641866186918851891189418a5181019121915191a192119"
        b"25194219441945194819511954195519561959195a19601965196a1989199119"
        b"921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a"
        b"581a611a661a691a851a911a961a9a1a0020022008200a201520202022202520"
        b"28202a20452051205920612065208020822088208a209520a020a220a520a820"
        b"aa2005211121142119212521422144214921552158215a216121642165216621"
        b"8521902196219921a521012208220a22112215222022222228222a2245225122"
        b"562259226522812288228a2291229522a022a222a822aa220524142416241924"
        b"252444244524462449245224552458245a2466248524912494249924a124a524"
        b"0925152521252925402545254825512554255525592562256525682589259025"
        b"9425952598259a25a125a425a625a92505261026122619262526412649265526"
        b"6026612669268426862690269a260028022808280a2815282028222828282a28"
        b"45285128542865288028822888288a28a028a228a828aa280929112914291929"
        b"2529462949295229552961296429662969298529902996299929a429a529002a"
        b"022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a"
        b"952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440"
        b"664094409940a140a6400041014104410641094112411541164118411a412141"
        b"26412941454148414a41514154415541564159415a41654168416a4181418441"
        b"8641904192419541a041a141a241054211421442164225424142524255425a42"
        b"6442694289429442a5420144154419442944454448444a445144544455445644"
        b"61446244654468446a44814486448944904492449544a044a144a94401450245"
        b"05450a4511451445154516451945204525452a45414544454545464549455045"
        b"5145544555455645584559456145644565456645694582458445854588459145"
        b"94459545964599459a45a545a845aa450146054609461446154618461a462146"
        b"2446294640464246454648465046514652465546564659466246654668468146"
        b"85468a4694469546a146a446a6460548114815481a4825484248494850485548"
        b"5848614864486648694885489148944896489948a5480149054906490a491049"
        b"144915491849214924492649404945494a495149524954495549564959496049"
        b"6249654966496a49864989499249954996499849a149a449a649a949164a444a"
        b"464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550"
        b"1a50215024502950405045504850515054505550565059506550685086508950"
        b"95509850a050a150a650a9500551085109510a51115114511551165118511951"
        b"20512551265128512a5141514451455146514951505151515251545155515651"
        b"585159515a51615164516551665169518251855191519451955196519951a051"
        b"a551aa5101520652125215521a5221522452425245524a525152545255525652"
        b"595262526552855290529252955299529a52a452045405541154145415541654"
        b"185419542154255428542a54415444544554465449544a545054515454545554"
        b"5654585459545a54615462546454655466546954805488548a54915494549554"
        b"96549954a154a454a554aa540155025504550555065509551055115512551455"
        b"1555165519551a55215524552555265529554055415542554455455546554855"
        b"4955505551555255545555555655585559555a55605561556455655566556855"
        b"69556a5581558455855589558a559055915594559555965598559955a155a455"
        b"a555a655a9550056015602560456065608560956115614561556185619562056"
        b"2156225624562556265628562956415645564656485649564a56505651565256"
        b"545655565656585659565a566156645665566956825685568656885689568a56"
        b"915695569a56a256a556a656a856a95604580558065809581058155818582158"
        b"2a58455848584a58515854585558565858585958605862586458655882588958"
        b"9058925895589858a158a9580159025905590a59115914591559165919592559"
        b"41594459455946594959505951595259545955595659585959595a5961596459"
        b"655966596959815985598959915994599559965998599959a559045a085a155a"
        b"1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a"
        b"6a5a815a8a5a925a955a965a985a9a5aa15a0560146016601960256044605060"
        b"5560566058605a60616064606660696081609660a56001610461066109611261"
        b"15612161226126612961456149615161556156615961656166616a6184618a61"
        b"92619561a161a661a96111621662196240624162466255625662586260628562"
        b"91629662a56211641264156416641a6421642664296440644264456448644a64"
        b"516454645564566459645a646064626465648464856489649064926494649564"
        b"966498649a64a164a464a964056508650a651165156516651965446545654665"
        b"496550655165546555655665596561656465656566656965866589658a659165"
        b"9565966599659a65a265a565a665a86502660966156620662666286629664066"
        b"456648664a66516654665566566658665a666066656668668066826685668a66"
        b"9466966698669966a066a466a666aa661668196825684168526855685a686168"
        b"6968856891689868a66801690469106915692169246926692969406941694569"
        b"4669486951695469556956695969606965696a69826984698a699569a169a469"
        b"a569a969116a166a186a416a446a496a506a556a586a5a6a646a656a696a866a"
        b"946a986a9a6aa66a0080028008800a802080228028802a804580508051805480"
        b"5680598065808080828088808a809580a080a280a880aa800581118114811681"
        b"1981258141814481498150815281558156815881598164816681698185818981"
        b"948196819981a5810082028208820a8215822082228228822a82518254825982"
        b"65828082828288828a829582a082a282a882aa82148419844184448451845584"
        b"5a846184648469849484998401850985128515851a8526852985408541854585"
        b"4885518554855585568559855a856585668568856a8581858485868589859085"
        b"928595859885a68511861686198625864186448649864a865086558659865a86"
        b"618666866a86858691869a86a4860088028808880a8815882088228828882a88"
        b"41884588518854885988658869888088828888888a889588a088a288a888aa88"
        b"05890689118914891689258941894489468949895089528955895a8961896489"
        b"858996899989a589008a028a088a0a8a158a208a228a288a2a8a458a518a548a"
        b"568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590"
        b"419046904990559058905a9069906a9085909190949096909990a59001910491"
        b"069109911091159118911a912191249126912991409145915091519154915591"
        b"569159916291659184918691929195919891a191a491a691a991059211921492"
        b"19922592449246924992509252925592589266926992859294929692a9920194"
        b"04940694109415941894269440944a9451945494559456945894599460946194"
        b"62946594849486949294949495949894a194a9940095059508950a9510951195"
        b"14951595169519952195259529952a9541954495459546954995509551955295"
        b"549555955695589559955a956195649565956695699581958595889591959295"
        b"94959595969599959a95a095a295a595a895aa95019604961096159619962096"
        b"2696299645964896499651965296559656965996659668968296849689968a96"
        b"929694969596a496a696a9960598169819982598419846985098529855985698"
        b"5a98649865988598919896989998a59804990699099910991299159918991a99"
        b"209921992499269940994299459948994a995199549955995699599962996599"
        b"66996a99819984999099929995999a99a199a699059a159a259a449a469a499a"
        b"509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0"
        b"28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa0"
        b"05a109a111a114a116a119a11aa146a149a151a155a158a15aa161a164a185a1"
        b"90a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2"
        b"65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a4"
        b"55a458a45aa461a465a466a468a469a485a406a509a510a512a515a518a526a5"
        b"29a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5"
        b"95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a6"
        b"62a686a690a695a696a699a6a1a6a4a6a6a600a802a808a80aa820a822a828a8"
        b"2aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9"
        b"19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa"
        b"20aa22aa28aa2aaa51aa54aa56aa80aa82aa88aa8aaa95aaa0aaa2aaa8aaaaaa"
    )

    delta = np.float32(0.125)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, qh = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint16)

        dl = d * (2 * ((qh >> 12) & 7) + 1)
        dl = dl.reshape((n_blocks, -1, 1, 1))
        delta = np.where((qh & np.uint16(0x8000)) == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 1, 1))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 3, 6, 9], dtype=np.uint16
        ).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 7) << 8).reshape((n_blocks, -1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (dl * (grid + delta)).reshape((n_blocks, -1))


class IQ1_M(__Quant, qtype=GGMLQuantizationType.IQ1_M):
    grid_shape = IQ1_S.grid_shape
    grid_map = IQ1_S.grid_map
    grid_hex = IQ1_S.grid_hex

    delta = IQ1_S.delta

    # Okay *this* type is weird. It's the only one which stores the f16 scales in multiple parts.
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, rest = np.hsplit(blocks, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 16])

        # The f16 scale is packed across multiple bytes
        scales = scales.view(np.uint16)
        d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array(
            [12, 8, 4, 0], dtype=np.uint16
        ).reshape((1, 4))
        d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

        scales = scales.reshape(n_blocks, -1, 1) >> np.array(
            [0, 3, 6, 9], dtype=np.uint16
        ).reshape((1, 1, 4))
        scales = (scales & 0x07).reshape((n_blocks, -1))
        dl = d * (2 * scales + 1)
        dl = dl.reshape((n_blocks, -1, 2, 1, 1))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape(
            (1, 1, 2)
        )
        qs = qs.astype(np.uint16) | ((qh & 0x07).astype(np.uint16) << 8).reshape(
            (n_blocks, -1)
        )

        delta = np.where(qh & 0x08 == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 2, 2, 1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 2, 8))

        return (dl * (grid + delta)).reshape((n_blocks, -1))


class IQ4_NL(__Quant, qtype=GGMLQuantizationType.IQ4_NL):
    kvalues = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))

        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = (
            np.take_along_axis(kvalues, qs, axis=-1)
            .astype(np.float32)
            .reshape((n_blocks, -1))
        )

        return d * qs


class IQ4_XS(__Quant, qtype=GGMLQuantizationType.IQ4_XS):
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        scales_h, rest = np.hsplit(rest, [2])
        scales_l, qs = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)
        scales_h = scales_h.view(np.uint16)

        scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2))
        scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array(
            [2 * i for i in range(QK_K // 32)], dtype=np.uint16
        ).reshape((1, -1, 1))
        scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(0x0F)
        scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8) & np.uint8(0x03)

        scales = (scales_l | (scales_h << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array(
            [0, 4], dtype=np.uint8
        ).reshape((1, 1, 2, 1))
        qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(0x0F)

        kvalues = np.array(IQ4_NL.kvalues, dtype=np.int8).reshape((1, 1, 1, -1))
        qs = (
            np.take_along_axis(kvalues, qs, axis=-1)
            .astype(np.float32)
            .reshape((n_blocks, -1, 32))
        )

        return (dl * qs).reshape((n_blocks, -1))
