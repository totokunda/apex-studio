"""
Tiny debugging helper to pause execution and print memory usage.

Typical usage:

    from src.utils.step_mem import step_mem

    step_mem("before forward", reset_peak=True)
    out = model(x)
    step_mem("after forward")
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import IO, Any, Dict, Optional

import psutil


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    n_f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n_f) < 1024.0:
            return f"{n_f:,.2f} {unit}"
        n_f /= 1024.0
    return f"{n_f:,.2f} PiB"


def _safe_get_callsite(skip: int = 2) -> str:
    """
    Best-effort callsite: "file.py:123 in func".
    skip=2 generally points to the caller of step_mem().
    """
    try:
        frame = inspect.stack()[skip]
        return f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.function}"
    except Exception:
        return "unknown"


def _collect_cpu_mem() -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    rss = None
    try:
        rss = int(proc.memory_info().rss)
    except Exception:
        rss = None

    vm_total = vm_avail = vm_used = vm_percent = None
    try:
        vm = psutil.virtual_memory()
        vm_total = int(vm.total)
        vm_avail = int(vm.available)
        vm_used = max(vm_total - vm_avail, 0) if vm_total is not None and vm_avail is not None else None
        vm_percent = (vm_used / vm_total * 100.0) if (vm_total and vm_used is not None) else None
    except Exception:
        pass

    return {
        "proc_rss": rss,
        "system_total": vm_total,
        "system_available": vm_avail,
        "system_used": vm_used,
        "system_used_percent": vm_percent,
    }


def _collect_cuda_mem(device: Optional[int] = None, sync: bool = True) -> Optional[Dict[str, Any]]:
    """
    Return CUDA memory stats when torch+cuda is available, otherwise None.

    Includes:
      - torch allocator stats (allocated/reserved + peaks)
      - driver-reported free/total (torch.cuda.mem_get_info when available)
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return None

    dev = torch.cuda.current_device() if device is None else int(device)
    try:
        if sync:
            torch.cuda.synchronize(dev)
    except Exception:
        pass

    free_b = total_b = None
    try:
        free_b, total_b = torch.cuda.mem_get_info(dev)
        free_b = int(free_b)
        total_b = int(total_b)
    except Exception:
        free_b = total_b = None

    allocated = reserved = max_allocated = max_reserved = None
    try:
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        max_allocated = int(torch.cuda.max_memory_allocated(dev))
        max_reserved = int(torch.cuda.max_memory_reserved(dev))
    except Exception:
        pass

    name = None
    try:
        name = torch.cuda.get_device_name(dev)
    except Exception:
        name = None

    return {
        "device": dev,
        "name": name,
        "free": free_b,
        "total": total_b,
        "used": (total_b - free_b) if (total_b is not None and free_b is not None) else None,
        "torch_allocated": allocated,
        "torch_reserved": reserved,
        "torch_max_allocated": max_allocated,
        "torch_max_reserved": max_reserved,
    }


def step_mem(
    label: Optional[str] = None,
    *,
    device: Optional[int] = None,
    pause: bool = True,
    reset_peak: bool = False,
    sync_cuda: bool = True,
    stream: Optional[IO[str]] = None,
) -> None:
    """
    Print process/system RAM + CUDA VRAM stats, then (optionally) pause until Enter.

    - Set env `APEX_STEP_MEM_DISABLE=1` to disable completely.
    - If stdin is not a TTY (non-interactive), it will NOT block, but still prints.
    - `reset_peak=True` resets PyTorch CUDA peak stats for the selected device.
    """
    if os.environ.get("APEX_STEP_MEM_DISABLE", "").strip() not in ("", "0", "false", "False", "FALSE"):
        return

    out = stream or sys.stderr
    callsite = _safe_get_callsite()
    hdr = f"[step_mem] {label} @ {callsite}" if label else f"[step_mem] @ {callsite}"
    print(hdr, file=out, flush=True)

    cpu = _collect_cpu_mem()
    print(
        "  CPU:"
        f" proc_rss={_fmt_bytes(cpu.get('proc_rss'))}"
        f" | system_used={_fmt_bytes(cpu.get('system_used'))}/{_fmt_bytes(cpu.get('system_total'))}"
        + (
            f" ({cpu.get('system_used_percent'):.1f}%)"
            if isinstance(cpu.get("system_used_percent"), (int, float))
            else ""
        ),
        file=out,
        flush=True,
    )

    cuda = _collect_cuda_mem(device=device, sync=sync_cuda)
    if cuda is None:
        print("  CUDA: n/a", file=out, flush=True)
    else:
        dev_name = (cuda.get("name") or "").strip()
        dev_str = f" dev={cuda.get('device')}" + (f" {dev_name}" if dev_name else "")
        # Optional peak reset happens after we print current/peak (so you can see it once),
        # but still before you continue execution.
        print(
            "  CUDA:"
            f"{dev_str}"
            f" | free/total={_fmt_bytes(cuda.get('free'))}/{_fmt_bytes(cuda.get('total'))}"
            f" | used={_fmt_bytes(cuda.get('used'))}"
            f" | torch_alloc={_fmt_bytes(cuda.get('torch_allocated'))}"
            f" | torch_resv={_fmt_bytes(cuda.get('torch_reserved'))}"
            f" | peak_alloc={_fmt_bytes(cuda.get('torch_max_allocated'))}"
            f" | peak_resv={_fmt_bytes(cuda.get('torch_max_reserved'))}",
            file=out,
            flush=True,
        )

        if reset_peak:
            try:
                import torch  # type: ignore

                dev = torch.cuda.current_device() if device is None else int(device)
                torch.cuda.reset_peak_memory_stats(dev)
                print("  CUDA: reset_peak_memory_stats()", file=out, flush=True)
            except Exception as e:
                print(f"  CUDA: failed to reset peak stats ({e})", file=out, flush=True)

    # Pause last so you can read the output.
    should_pause = bool(pause) and sys.stdin is not None and sys.stdin.isatty()
    if should_pause:
        try:
            _ = input("  Press Enter to continue... ")
        except EOFError:
            # Non-interactive runner or stdin closed; don't block.
            pass

