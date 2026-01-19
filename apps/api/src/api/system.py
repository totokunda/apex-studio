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
        from .ray_tasks import free_unused_modules_in_warm_pool

        ray = get_ray_app()
        device_index, device_type = get_best_gpu()
        resources = get_ray_resources(device_index, device_type, load_profile="light")
        ref = free_unused_modules_in_warm_pool.options(**resources).remote(
            active=sorted(active_set), target=request.target
        )
        worker_result = await _run_blocking(ray.get, ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to free memory: {e}")

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
