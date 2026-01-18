import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_dir() -> Path:
    return _repo_root() / "runs" / "ltx2-19b-text-to-image-to-video-1.0.0.v1"


def _load_run_inputs() -> tuple[Dict[str, Any], Dict[str, Any]]:
    run_dir = _run_dir()
    path = run_dir / "model_inputs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run inputs: {path}")

    data = json.loads(path.read_text())
    engine_kwargs = data["engine_kwargs"]
    inputs = data["inputs"]

    for input_key, input_value in list(inputs.items()):
        if isinstance(input_value, str) and input_value.startswith("assets"):
            inputs[input_key] = str(run_dir / input_value)

    return engine_kwargs, inputs


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_mem_snapshot() -> Dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        dev = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(dev)
        return {
            "free_gb": float(free) / 1e9,
            "total_gb": float(total) / 1e9,
            "alloc_gb": float(torch.cuda.memory_allocated(dev)) / 1e9,
            "reserved_gb": float(torch.cuda.memory_reserved(dev)) / 1e9,
            "max_alloc_gb": float(torch.cuda.max_memory_allocated(dev)) / 1e9,
            "max_reserved_gb": float(torch.cuda.max_memory_reserved(dev)) / 1e9,
        }
    except Exception:
        return {}


def _cuda_reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _sweep_cases() -> List[Tuple[int, int, int]]:
    """
    Increasing (height, width, duration_frames) cases.

    - Default runs a small smoke case only (fast-ish).
    - Set `APEX_TEST_LTX2_SWEEP=1` to run the full increasing sweep.
    """
    if _env_flag("APEX_TEST_LTX2_SWEEP", default=False):
        return [
            (512, 768, 25),
            (640, 1024, 49),
            (720, 1280, 73),
            (896, 1344, 97),
            (1088, 1440, 121),
        ]
    return [(512, 768, 25)]

def run_ltx2_sweep() -> None:
    """
    Script-friendly runner (does not require pytest).

    Environment:
    - `APEX_TEST_LTX2_SWEEP=1` to enable full sweep
    - `APEX_MEM_DEBUG=1` to print memory-manager logs
    """
    if not _cuda_available():
        print("SKIP: CUDA not available; LTX2 19B sweep requires a GPU environment.")
        return

    # Allow running as a plain script without needing `python -m ...`.
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    run_dir = _run_dir()
    if not run_dir.exists():
        print(f"SKIP: Run directory missing: {run_dir}")
        return

    from src.engine.registry import UniversalEngine

    engine_kwargs, base_inputs = _load_run_inputs()

    base_inputs = dict(base_inputs)
    base_inputs["num_inference_steps"] = int(base_inputs.get("num_inference_steps", 1) or 1)
    base_inputs["upsample"] = False

    os.environ.setdefault("APEX_MEM_DEBUG", "1")

    engine = UniversalEngine(**engine_kwargs)

    for (h, w, frames) in _sweep_cases():
        inputs = dict(base_inputs)
        inputs["height"] = int(h)
        inputs["width"] = int(w)
        inputs["duration"] = int(frames)

        _cuda_reset_peak()
        before = _cuda_mem_snapshot()
        t0 = time.time()
        out = engine.run(**inputs)
        elapsed = time.time() - t0
        after = _cuda_mem_snapshot()

        print(
            "\n[sweep] "
            f"h={h} w={w} frames={frames} "
            f"elapsed={elapsed:.2f}s "
            f"mem_before={before} "
            f"mem_after={after} "
            f"out_type={type(out)}"
        )

        if out is None:
            raise RuntimeError("Engine returned None")


if pytest is not None:
    @pytest.mark.slow
    def test_ltx2_memory_manager_sweep_increasing_resolution_duration() -> None:
        run_ltx2_sweep()


if __name__ == "__main__":
    run_ltx2_sweep()
