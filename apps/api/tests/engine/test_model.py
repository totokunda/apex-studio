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
    # Add `apps/api/` to sys.path so `import src...` works when running as script.
    return Path(__file__).resolve().parent


def _run_dir() -> Path:
    return Path("api/runs/ltx2-19b-text-to-image-to-video-1.0.0.v1")


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
            (720, 1280, 481),
            (1440, 1920, 161),
            (1920, 2560, 201),
            (1920, 2560, 241),
            (2560, 3440, 281),
        ]
    return [(1088, 1920, 481)]


def _parse_profiles_env() -> List[str]:
    """
    Profiles to test, in order.

    - `APEX_TEST_BUDGET_PROFILES=all` -> all 5 profiles
    - `APEX_TEST_BUDGET_PROFILES=VerylowRAM_LowVRAM,LowRAM_LowVRAM`
    - `APEX_TEST_BUDGET_PROFILES=5,4,2`
    """
    raw = (os.environ.get("APEX_TEST_BUDGET_PROFILES") or "").strip()
    if not raw:
        return ["VerylowRAM_LowVRAM"]
    if raw.lower() == "all":
        return ["HighRAM_HighVRAM", "HighRAM_LowVRAM", "LowRAM_HighVRAM", "LowRAM_LowVRAM", "VerylowRAM_LowVRAM"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def _budget_profile_config(profile: str, *, model_id: str = "transformer") -> Dict[str, Any]:
    """
    Mirror mmgp profile defaults, mapped into Apex's memory_management config.

    Notes:
    - `budget_mb=0` means "no budgeting" (full module can stay resident) in this test harness.
    - `pin_cpu_memory` maps to reserved/pinned RAM behavior.
    """
    p = str(profile or "").strip()
    # accept numeric
    if p.isdigit():
        p = {
            "1": "HighRAM_HighVRAM",
            "2": "HighRAM_LowVRAM",
            "3": "LowRAM_HighVRAM",
            "4": "LowRAM_LowVRAM",
            "5": "VerylowRAM_LowVRAM",
        }.get(p, "VerylowRAM_LowVRAM")

    # mmgp-like defaults
    budget_mb: Any = 0
    pin_cpu_memory = False
    if p == "HighRAM_HighVRAM":
        pin_cpu_memory = True
        budget_mb = 0
    elif p == "HighRAM_LowVRAM":
        pin_cpu_memory = True
        budget_mb = 3000
    elif p == "LowRAM_HighVRAM":
        pin_cpu_memory = (model_id == "transformer")
        budget_mb = 0
    elif p == "LowRAM_LowVRAM":
        pin_cpu_memory = (model_id == "transformer")
        budget_mb = 3000
    else:  # VerylowRAM_LowVRAM
        pin_cpu_memory = False
        budget_mb = 100 if model_id == "transformer" else 3000

    # Allow env overrides for quick tuning
    env_budget = os.environ.get("APEX_TEST_BUDGET_MB")
    if env_budget is not None and env_budget.strip() != "":
        budget_mb = env_budget.strip()

    async_transfers = _env_flag("APEX_TEST_ASYNC_TRANSFERS", default=False)
    prefetch = _env_flag("APEX_TEST_PREFETCH", default=False)
    offload_after_forward = _env_flag("APEX_TEST_OFFLOAD_AFTER_FORWARD", default=True)
    vram_safety_coefficient = float(os.environ.get("APEX_TEST_VRAM_SAFETY_COEFFICIENT", "0.8") or "0.8")

    return {
        "offload_mode": "budget",
        "budget_mb": budget_mb,
        "async_transfers": async_transfers,
        "prefetch": prefetch,
        "pin_cpu_memory": pin_cpu_memory,
        "vram_safety_coefficient": vram_safety_coefficient,
        "offload_after_forward": offload_after_forward,
    }


def _inject_budget_offload(engine_kwargs: Dict[str, Any], *, profile: str) -> Dict[str, Any]:
    engine_kwargs = dict(engine_kwargs)
    memory_management = dict(engine_kwargs.get("memory_management") or {})
    
    # Fallback so any component not explicitly listed below still uses budget offloading.
    # (BaseEngine resolves `all` when no name/type-specific entry exists.)
    memory_management.setdefault(
        "all", {**_budget_profile_config(profile, model_id="all"), **(memory_management.get("all") or {})}
    )

    component_keys = (
        "transformer",
        "text_encoder",
        "vae",
        "video_vae",
        "audio_vae",
        "transformer_vae",
        "latent_upsampler",
        "connectors",
        "vocoder",
    )
    for key in component_keys:
        memory_management[key] = {
            **_budget_profile_config(profile, model_id=key),
            **(memory_management.get(key) or {}),
        }
    if "transformer" in memory_management:
        memory_management["transformer"].setdefault(
            "block_modules", ["transformer_blocks"]
        )
    if "text_encoder" in memory_management:
        memory_management["text_encoder"].setdefault(
            "block_modules", ["model.language_model"]
        )
    engine_kwargs["memory_management"] = memory_management
    return engine_kwargs

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

    os.environ.setdefault("APEX_MEM_DEBUG", "1")
    profiles = _parse_profiles_env()

    for prof in profiles:
        engine_kwargs, base_inputs = _load_run_inputs()
        engine_kwargs = _inject_budget_offload(engine_kwargs, profile=prof)

        base_inputs = dict(base_inputs)
        base_inputs["num_inference_steps"] = 1
        base_inputs["rope_on_cpu"] = True

        print(f"\n=== Budget offload profile: {prof} ===")
        engine = UniversalEngine(**engine_kwargs)

        for (h, w, frames) in _sweep_cases():
            inputs = dict(base_inputs)
            inputs["height"] = int(h)
            inputs["width"] = int(w)
            inputs["duration"] = int(frames)
            print(inputs["height"], inputs["width"], inputs["duration"])

            _cuda_reset_peak()
            before = _cuda_mem_snapshot()
            t0 = time.time()
            out = engine.run(**inputs)
            elapsed = time.time() - t0
            after = _cuda_mem_snapshot()
            from src.utils.save_audio_video import save_video_ltx2
            save_video_ltx2(out[0], out[1], f"result_{h}_{w}_{frames}")

            print(
                "\n[sweep] "
                f"profile={prof} "
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
