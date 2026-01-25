from fastapi import APIRouter
import psutil
from typing import Any, Dict, List, Optional, Literal
from functools import partial
import anyio
from fastapi import HTTPException
from pydantic import BaseModel

# Reuse helpers to detect device type and query GPU memory info
from .ray_resources import (
    _on_mps,
    _gpu_mem_info_torch,
    _gpu_mem_info_nvml,
    _gpu_mem_info_nvidia_smi,
)
import subprocess
import re

router = APIRouter(prefix="/system", tags=["system"])


async def _run_blocking(func, *args, **kwargs):
    """Run blocking (sync) work in a worker thread so we don't block the event loop."""
    return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))


def _collect_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Return aggregate and per-adapter GPU VRAM usage when available.

    Structure:
      {
        "device_type": "cuda",
        "count": int,
        "adapters": [{"index": int, "total": int, "free": int, "used": int, "percent": float}],
        "total": int,
        "used": int,
        "percent": float
      }
    Returns None if no discrete GPU is available.
    """

    # Prefer torch (fast, no extra deps), fall back to NVML or nvidia-smi
    infos: Optional[List[Dict[str, int]]] = (
        _gpu_mem_info_torch() or _gpu_mem_info_nvml() or _gpu_mem_info_nvidia_smi()
    )
    if not infos:
        return None

    adapters: List[Dict[str, Any]] = []
    total_total = 0
    total_used = 0
    for info in infos:
        total = int(info.get("total", 0))
        free = int(info.get("free", 0))
        used = max(total - free, 0)
        percent = (used / total * 100.0) if total > 0 else 0.0
        adapters.append(
            {
                "index": int(info.get("index", 0)),
                "total": total,
                "free": free,
                "used": used,
                "percent": percent,
            }
        )
        total_total += total
        total_used += used

    percent_total = (total_used / total_total * 100.0) if total_total > 0 else 0.0
    return {
        "device_type": "cuda",
        "count": len(adapters),
        "adapters": adapters,
        "total": total_total,
        "used": total_used,
        "percent": percent_total,
    }


@router.get("/memory")
async def get_system_memory() -> Dict[str, Any]:
    return await _run_blocking(_get_system_memory_sync)


class FreeMemoryRequest(BaseModel):
    active: Optional[str] = None
    target: Literal["cpu", "disk"] = "disk"


@router.post("/free-memory")
async def free_memory(request: FreeMemoryRequest) -> Dict[str, Any]:
    """
    Best-effort free GPU memory by offloading tracked modules.

    Query params:
      - active: comma-separated component names to keep resident (e.g. "transformer,vae")
      - target: "cpu" or "disk" offload destination (default: disk)
    """
    active_set = set()
    if request.active:
        active_set = {p.strip() for p in request.active.split(",") if p.strip()}

    try:
        # Run the free operation in the Ray worker process that holds the warm pool.
        # This avoids creating a new engine (which requires a yaml_path).
        from .ray_app import get_ray_app  # lazy import (avoid ray.init at import-time)
        from .ray_resources import get_best_gpu, get_ray_resources
        from .ray_tasks import (
            get_engine_runner_actor,
            kill_engine_runner_actor,
            free_unused_modules_in_warm_pool,
        )

        ray = get_ray_app()
        device_index, device_type = get_best_gpu()
        resources = get_ray_resources(device_index, device_type, load_profile="light")
        runner = None
        # Route free-memory through the long-lived EngineRunner process whenever we
        # have an accelerator-backed worker (CUDA or Apple MPS). This ensures we
        # operate on the *correct* per-process warm pool and can actually release
        # accelerator caches in that worker.
        if device_type in {"cuda", "mps"} and device_index is not None:
            runner = get_engine_runner_actor(
                device_index=device_index, device_type=device_type, resources=resources
            )
            ref = runner.free_unused_modules_in_warm_pool.remote(
                active=sorted(active_set), target=request.target
            )
        else:
            # CPU-only fallback (no GPU actor needed).
            ref = free_unused_modules_in_warm_pool.options(**resources).remote(
                active=sorted(active_set), target=request.target
            )
        worker_result = await _run_blocking(ray.get, ref)

        # If the user asked for a full "free memory" and nothing is running, we can
        # terminate the runner actor process to release the accelerator context/caches
        # in that process (most relevant for CUDA; still helpful for MPS/unified memory).
        actor_killed = False
        try:
            skipped = (
                (worker_result or {}).get("skipped_in_use", [])
                if isinstance(worker_result, dict)
                else []
            )
            nothing_in_use = not bool(skipped)
            if (
                runner is not None
                and request.target == "disk"
                and not active_set
                and nothing_in_use
            ):
                actor_killed = bool(
                    kill_engine_runner_actor(
                        device_index=device_index, device_type=device_type
                    )
                )
        except Exception:
            actor_killed = False
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to free memory: {e}")

    # Best-effort: cancel any active processor jobs and stop mask tracking streams.
    # This helps ensure `/system/free-memory` also frees resources used by preprocessors,
    # postprocessors, and masks (which should not persist/warm).
    cancelled_processor_jobs: List[str] = []
    cancelled_mask_tracking_ids: List[str] = []
    mask_cleanup: Optional[Dict[str, Any]] = None
    try:
        # Cancel running/queued pre/postprocessor jobs tracked in the unified job store.
        from .job_store import job_store

        def _cancel_processors_sync() -> List[str]:
            cancelled: List[str] = []
            try:
                all_ids = list(job_store.all_job_ids())
            except Exception:
                all_ids = []
            for jid in all_ids:
                try:
                    info = job_store.get(jid) or {}
                    jtype = str(info.get("type") or "").strip().lower()
                    if jtype not in {"preprocessor", "postprocessor"}:
                        continue
                    st = (job_store.status(jid) or {}).get("status", "")
                    if str(st).strip().lower() in {"running", "queued"}:
                        job_store.cancel(jid)
                        cancelled.append(str(jid))
                except Exception:
                    continue
            return cancelled

        cancelled_processor_jobs = await _run_blocking(_cancel_processors_sync)
    except Exception:
        cancelled_processor_jobs = []

    try:
        # Cooperative cancellation for mask tracking streams running in this API process.
        from .mask import ACTIVE_TRACKING, CANCEL_TRACKING

        try:
            active_ids = list(ACTIVE_TRACKING)
        except Exception:
            active_ids = []
        for mid in active_ids:
            try:
                CANCEL_TRACKING.add(mid)
                cancelled_mask_tracking_ids.append(str(mid))
            except Exception:
                pass
    except Exception:
        cancelled_mask_tracking_ids = []

    try:
        # Clear any in-process SAM2 predictor singletons/caches (best-effort).
        from src.mask.mask import free_mask_memory

        mask_cleanup = free_mask_memory(hard=True)
    except Exception:
        mask_cleanup = None

    # Backward-friendly: keep `offloaded` as a flat module_id -> location mapping.
    # Also include per-engine details for debugging.
    aggregated: Dict[str, Any] = {}
    by_engine = (
        (worker_result or {}).get("offloaded", {})
        if isinstance(worker_result, dict)
        else {}
    )
    if isinstance(by_engine, dict):
        for _key, mapping in by_engine.items():
            if isinstance(mapping, dict):
                aggregated.update(mapping)

    return {
        "offloaded": aggregated,
        "by_engine": by_engine,
        "errors": (
            (worker_result or {}).get("errors", {})
            if isinstance(worker_result, dict)
            else {}
        ),
        "skipped_in_use": (
            (worker_result or {}).get("skipped_in_use", [])
            if isinstance(worker_result, dict)
            else []
        ),
        "pool": (
            (worker_result or {}).get("pool", {})
            if isinstance(worker_result, dict)
            else {}
        ),
        "target": request.target,
        "actor_killed": actor_killed,
        "cancelled_processor_jobs": cancelled_processor_jobs,
        "cancelled_mask_tracking_ids": cancelled_mask_tracking_ids,
        "mask_cleanup": mask_cleanup,
    }


def _get_system_memory_sync() -> Dict[str, Any]:
    """Report current memory usage for system RAM and GPU VRAM.

    On Apple Silicon (unified memory), only a single unified memory metric is returned
    via the "unified" key and GPU will be None.
    """

    vm = psutil.virtual_memory()
    # Use a consistent definition: used = total - available; percent = used / total
    total_ram = int(vm.total)
    available_ram = int(vm.available)
    used_ram = max(total_ram - available_ram, 0)
    percent_ram = (used_ram / total_ram * 100.0) if total_ram > 0 else 0.0
    system_ram = {
        "total": total_ram,
        "available": available_ram,
        "used": used_ram,
        "percent": percent_ram,
    }

    # Unified memory (Apple Silicon / MPS)
    if _on_mps():
        # macOS Activity Monitor style details for easier comparison
        details: Dict[str, Any] = {}
        try:
            vm_named = psutil.virtual_memory()
            swap = psutil.swap_memory()
            page_bytes = 4096  # macOS page size (commonly 4096); keeps this simple

            # Attempt to read compressed pages from vm_stat output
            compressed_bytes = None
            try:
                out = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
                m = re.search(r"Pages occupied by compressor:\s+([0-9\.]+)", out)
                if m:
                    pages = int(m.group(1).replace(".", ""))
                    compressed_bytes = pages * page_bytes
            except Exception:
                compressed_bytes = None

            details = {
                "active": int(getattr(vm_named, "active", 0)),
                "inactive": int(getattr(vm_named, "inactive", 0)),
                "wired": int(getattr(vm_named, "wired", 0)),
                "free": int(getattr(vm_named, "free", 0)),
                "cached_files": int(getattr(vm_named, "inactive", 0)),  # approximation
                "compressed": int(compressed_bytes or 0),
                "swap_used": int(getattr(swap, "used", 0)),
            }
        except Exception:
            pass

        return {
            "unified": {
                "total": system_ram["total"],
                "used": system_ram["used"],
                "available": system_ram["available"],
                "percent": system_ram["percent"],
                "details": details,
            },
            "cpu": None,
            "gpu": None,
            "device_type": "mps",
        }

    gpu = _collect_gpu_memory_info()
    return {
        "unified": None,
        "cpu": system_ram,
        "gpu": gpu,
        "device_type": ("cuda" if gpu else "cpu"),
    }
