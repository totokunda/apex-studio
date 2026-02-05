import ctypes
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import gguf

from gguf.constants import GGMLQuantizationType

from src.quantize import quants as qmod


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_ext():
    try:
        import src.quantize._ggml_quants as ext  # type: ignore

        return ext
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "scripts/build_ggml_quants_ext.py"], cwd=_repo_root()
            )
            importlib.invalidate_caches()
            import src.quantize._ggml_quants as ext  # type: ignore

            return ext
        except Exception as e:
            pytest.skip(f"ggml quants extension could not be built/imported: {e}")


def _ggml_lib():
    lib_path = _repo_root() / "llama-b7902" / "libggml-base.so"
    if not lib_path.exists():
        pytest.skip(f"Missing ggml library at {lib_path}")

    lib = ctypes.CDLL(os.fspath(lib_path))
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    return lib


@pytest.mark.parametrize(
    "qtype",
    [
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q5_1,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    ],
)
def test_cpp_quantize_matches_ggml(qtype: GGMLQuantizationType):
    ext = _ensure_ext()
    lib = _ggml_lib()

    qcls = qmod._type_traits[qtype]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(size=(17, qcls.block_size * 4), dtype=np.float32)

    got = ext.quantize(x, int(qtype))
    expected = np.empty(gguf.quant_shape_to_byte_shape(x.shape, qtype), dtype=np.uint8)
    lib.ggml_quantize_chunk(
        int(qtype),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        expected.ctypes.data,
        0,
        x.shape[0],
        x.shape[1],
        None,
    )

    assert got.dtype == np.uint8
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(
    "qtype",
    [
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q5_1,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    ],
)
def test_cpp_dequantize_matches_python(qtype: GGMLQuantizationType):
    ext = _ensure_ext()
    qcls = qmod._type_traits[qtype]

    rng = np.random.default_rng(1)
    x = rng.standard_normal(size=(11, qcls.block_size * 4), dtype=np.float32)

    q = ext.quantize(x, int(qtype))
    got = ext.dequantize(q, int(qtype))
    expected = qcls.dequantize_rows(q)

    assert got.dtype == np.float32
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_cpp_quantize_parallel_matches_ggml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APEX_GGML_QUANTS_THREADS", "2")

    ext = _ensure_ext()
    lib = _ggml_lib()

    qtype = GGMLQuantizationType.Q4_K
    qcls = qmod._type_traits[qtype]
    rng = np.random.default_rng(2)
    x = rng.standard_normal(size=(17, qcls.block_size * 4), dtype=np.float32)

    got = ext.quantize(x, int(qtype))
    expected = np.empty(gguf.quant_shape_to_byte_shape(x.shape, qtype), dtype=np.uint8)
    lib.ggml_quantize_chunk(
        int(qtype),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        expected.ctypes.data,
        0,
        x.shape[0],
        x.shape[1],
        None,
    )

    assert np.array_equal(got, expected)


def test_cpp_dequantize_parallel_matches_python(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APEX_GGML_QUANTS_THREADS", "2")

    ext = _ensure_ext()

    qtype = GGMLQuantizationType.Q4_K
    qcls = qmod._type_traits[qtype]
    rng = np.random.default_rng(3)
    x = rng.standard_normal(size=(11, qcls.block_size * 4), dtype=np.float32)

    q = ext.quantize(x, int(qtype))
    got = ext.dequantize(q, int(qtype))
    expected = qcls.dequantize_rows(q)

    assert np.array_equal(got, expected)
