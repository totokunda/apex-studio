import numpy as np
import pytest
import gguf

from gguf.constants import GGMLQuantizationType
from gguf.lazy import LazyNumpyTensor

from src.quantize import quants as qmod


def _slow_apply_over_grouped_rows(func, arr: np.ndarray, otype, oshape: tuple[int, ...]):
    """
    Reference implementation: this matches the *previous* behavior of
    `src.quantize.quants._apply_over_grouped_rows` (pre-optimization).
    """
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    n_groups = (rows.shape[0] // 16) or 1
    np.concatenate(
        [func(group).ravel() for group in np.array_split(rows, n_groups)],
        axis=0,
        out=out,
    )
    return out.reshape(oshape)


@pytest.mark.parametrize(
    "qtype",
    [
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q5_1,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q6_K,
    ],
)
def test_apply_over_grouped_rows_matches_previous(
    qtype: GGMLQuantizationType, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("APEX_DISABLE_GGML_QUANTS", "1")

    qcls = qmod._type_traits[qtype]

    # Stress chunking behavior by using a non-multiple of common chunk sizes.
    n_rows = 257
    row_len = qcls.block_size * 4

    rng = np.random.default_rng(0)
    x = rng.standard_normal(size=(n_rows, row_len), dtype=np.float32)

    # ---- quantize (eager ndarray) -----------------------------------------
    qshape = gguf.quant_shape_to_byte_shape(x.shape, qtype)
    expected_q = _slow_apply_over_grouped_rows(qcls.quantize_rows, x, np.uint8, qshape)
    got_q = qcls._Quant__quantize_array(x)
    assert got_q.dtype == np.uint8
    assert got_q.shape == qshape
    assert np.array_equal(got_q, expected_q)

    # ---- quantize (LazyNumpyTensor) ---------------------------------------
    lazy_x = LazyNumpyTensor.from_eager(x)
    lazy_q = qcls.quantize(lazy_x)
    eager_q = LazyNumpyTensor.to_eager(lazy_q)
    assert np.array_equal(eager_q, got_q)

    # ---- dequantize (eager ndarray) ---------------------------------------
    dqshape = gguf.quant_shape_from_byte_shape(qshape, qtype)
    expected_dq = _slow_apply_over_grouped_rows(
        qcls.dequantize_rows, got_q, np.float32, dqshape
    )
    got_dq = qcls._Quant__dequantize_array(got_q)
    assert got_dq.dtype == np.float32
    assert got_dq.shape == dqshape
    assert np.array_equal(got_dq, expected_dq)

    # ---- dequantize (LazyNumpyTensor) -------------------------------------
    lazy_q2 = LazyNumpyTensor.from_eager(got_q)
    lazy_dq = qcls.dequantize(lazy_q2)
    eager_dq = LazyNumpyTensor.to_eager(lazy_dq)
    assert np.array_equal(eager_dq, got_dq)
