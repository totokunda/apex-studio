"""
Optional environment-gated shims that run at Python startup.

Python auto-imports `sitecustomize` (if present on `sys.path`) after `site`.
We use this to spoof system RAM for heuristics that rely on psutil.
"""

from __future__ import annotations

import os
from typing import Any


def _maybe_fake_ram() -> None:
    fake_gb = os.environ.get("APEX_FAKE_RAM_GB")
    if not fake_gb:
        return

    try:
        gb = float(fake_gb)
    except Exception:
        return

    if gb <= 0:
        return

    try:
        import psutil  # type: ignore
    except Exception:
        return

    desired_total = int(gb * 1024**3)
    real_virtual_memory = psutil.virtual_memory

    def _patched_virtual_memory() -> Any:
        vm = real_virtual_memory()
        total = desired_total

        # Cap available/free to <= total so derived math remains sane.
        available = int(min(getattr(vm, "available", 0), total))
        free = int(min(getattr(vm, "free", available), available))

        used = max(total - available, 0)
        percent = (used / total * 100.0) if total > 0 else 0.0

        # psutil returns an svmem namedtuple (platform-dependent fields).
        svmem_type = type(vm)
        data = {name: getattr(vm, name) for name in getattr(vm, "_fields", ())}

        # Overwrite the key fields that our codebase reads.
        data["total"] = total
        if "available" in data:
            data["available"] = available
        if "free" in data:
            data["free"] = free
        if "used" in data:
            data["used"] = used
        if "percent" in data:
            data["percent"] = percent

        try:
            return svmem_type(**data)
        except Exception:
            # Very defensive fallback: return the original object if reconstruction fails.
            return vm

    psutil.virtual_memory = _patched_virtual_memory  # type: ignore[attr-defined]


_maybe_fake_ram()


