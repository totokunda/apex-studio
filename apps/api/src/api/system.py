from fastapi import APIRouter
import psutil
from typing import Any, Dict, List, Optional
from functools import partial
import anyio

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
