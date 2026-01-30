"""
Tiny debugging helper to pause execution and print memory usage.

Typical usage:

    from src.utils.step_mem import step_mem

    step_mem("before forward", reset_peak=True)
    out = model(x)
    step_mem("after forward")
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
from typing import IO, Any, Dict, List, Optional, Tuple

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
        vm_used = (
            max(vm_total - vm_avail, 0)
            if vm_total is not None and vm_avail is not None
            else None
        )
        vm_percent = (
            (vm_used / vm_total * 100.0) if (vm_total and vm_used is not None) else None
        )
    except Exception:
        pass

    return {
        "proc_rss": rss,
        "system_total": vm_total,
        "system_available": vm_avail,
        "system_used": vm_used,
        "system_used_percent": vm_percent,
    }


def _collect_cuda_mem(
    device: Optional[int] = None, sync: bool = True
) -> Optional[Dict[str, Any]]:
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
        "used": (
            (total_b - free_b) if (total_b is not None and free_b is not None) else None
        ),
        "torch_allocated": allocated,
        "torch_reserved": reserved,
        "torch_max_allocated": max_allocated,
        "torch_max_reserved": max_reserved,
    }


def _device_matches_filter(device_str: str, device_filter: Optional[str]) -> bool:
    if not device_filter:
        return True
    filt = device_filter.strip().lower()
    dev = device_str.strip().lower()
    if filt in ("cuda", "gpu"):
        return dev.startswith("cuda")
    if filt in ("cpu",):
        return dev == "cpu"
    return dev == filt


def _safe_storage_info(t: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (storage_ptr, storage_nbytes) best-effort.
    """
    try:
        # PyTorch 2.x preferred API
        s = t.untyped_storage()
        ptr = int(s.data_ptr())
        nbytes = int(s.nbytes())
        return ptr, nbytes
    except Exception:
        pass
    try:
        # Older API; may warn in newer versions
        s = t.storage()
        ptr = int(s.data_ptr())
        nbytes = int(s.nbytes())
        return ptr, nbytes
    except Exception:
        return None, None


def _collect_live_tensors(
    *,
    device_filter: Optional[str] = None,
    min_tensor_bytes: int = 0,
    top_k: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Best-effort inventory of live torch Tensors visible to Python's GC.

    Notes:
    - This can be slow (walks gc.get_objects()) and may be noisy.
    - It will not show freed tensors, and may miss tensors not tracked by gc.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None

    rows: List[Dict[str, Any]] = []
    totals_by_device: Dict[str, int] = {}
    storages_by_device: Dict[str, Dict[int, int]] = {}

    for obj in gc.get_objects():
        t = None
        try:
            if torch.is_tensor(obj):
                t = obj
            elif hasattr(obj, "data") and torch.is_tensor(getattr(obj, "data")):
                t = getattr(obj, "data")
            else:
                continue

            device_str = str(t.device)
            if not _device_matches_filter(device_str, device_filter):
                continue

            # tensor bytes (view size), not necessarily unique storage
            tensor_bytes = int(t.numel()) * int(t.element_size())
            if tensor_bytes < int(min_tensor_bytes):
                continue

            storage_ptr, storage_nbytes = _safe_storage_info(t)

            totals_by_device[device_str] = (
                totals_by_device.get(device_str, 0) + tensor_bytes
            )
            if storage_ptr is not None and storage_nbytes is not None:
                dev_storages = storages_by_device.setdefault(device_str, {})
                # keep max nbytes for this ptr (should be stable; defensive for races)
                prev = dev_storages.get(storage_ptr)
                if prev is None or int(storage_nbytes) > int(prev):
                    dev_storages[storage_ptr] = int(storage_nbytes)

            grad_fn_name = None
            try:
                grad_fn = getattr(t, "grad_fn", None)
                grad_fn_name = type(grad_fn).__name__ if grad_fn is not None else None
            except Exception:
                grad_fn_name = None

            rows.append(
                {
                    "tensor_bytes": tensor_bytes,
                    "storage_bytes": (
                        int(storage_nbytes) if storage_nbytes is not None else None
                    ),
                    "shape": (
                        tuple(int(x) for x in t.shape)
                        if getattr(t, "shape", None) is not None
                        else None
                    ),
                    "dtype": str(getattr(t, "dtype", None)),
                    "device": device_str,
                    "requires_grad": bool(getattr(t, "requires_grad", False)),
                    "is_leaf": bool(getattr(t, "is_leaf", False)),
                    "grad_fn": grad_fn_name,
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda r: int(r.get("tensor_bytes") or 0), reverse=True)
    if int(top_k) > 0:
        rows = rows[: int(top_k)]

    unique_storage_bytes_by_device: Dict[str, int] = {}
    for dev, ptr_map in storages_by_device.items():
        unique_storage_bytes_by_device[dev] = int(sum(int(v) for v in ptr_map.values()))

    return {
        "rows": rows,
        "totals_by_device": totals_by_device,
        "unique_storage_bytes_by_device": unique_storage_bytes_by_device,
        "n_rows": len(rows),
    }


def step_mem(
    label: Optional[str] = None,
    *,
    device: Optional[int] = None,
    pause: bool = True,
    reset_peak: bool = False,
    sync_cuda: bool = True,
    log_tensors: bool = False,
    tensors_top_k: int = 30,
    tensors_min_bytes: int = 0,
    tensors_device: Optional[str] = "cuda",
    stream: Optional[IO[str]] = None,
    cuda_only: bool = True,
) -> None:
    """
    Print process/system RAM + CUDA VRAM stats, then (optionally) pause until Enter.

    - Set env `APEX_STEP_MEM_DISABLE=1` to disable completely.
    - Set env `APEX_STEP_MEM_PAUSE=1` to force pausing (when stdin is a TTY).
    - If stdin is not a TTY (non-interactive), it will NOT block, but still prints.
    - `reset_peak=True` resets PyTorch CUDA peak stats for the selected device.
    - `log_tensors=True` walks Python GC to list live torch tensors (can be slow/noisy).
    """
    if os.environ.get("APEX_STEP_MEM_DISABLE", "").strip() not in (
        "",
        "0",
        "false",
        "False",
        "FALSE",
    ):
        return

    out = stream or sys.stderr
    callsite = _safe_get_callsite()
    hdr = f"[step_mem] {label} @ {callsite}" if label else f"[step_mem] @ {callsite}"
    print(hdr, file=out, flush=True)

    if not cuda_only:
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

    if log_tensors:
        inv = _collect_live_tensors(
            device_filter=tensors_device,
            min_tensor_bytes=int(tensors_min_bytes),
            top_k=int(tensors_top_k),
        )
        if inv is None:
            print("  TENSORS: n/a (torch not available)", file=out, flush=True)
        else:
            totals_by_device = inv.get("totals_by_device") or {}
            unique_storage_by_device = inv.get("unique_storage_bytes_by_device") or {}
            if not totals_by_device:
                suffix = f" (filter={tensors_device})" if tensors_device else ""
                print(f"  TENSORS: none found{suffix}", file=out, flush=True)
            else:
                suffix = f" (filter={tensors_device})" if tensors_device else ""
                print(f"  TENSORS{suffix}:", file=out, flush=True)
                for dev in sorted(
                    set(
                        list(totals_by_device.keys())
                        + list(unique_storage_by_device.keys())
                    )
                ):
                    total = totals_by_device.get(dev)
                    uniq = unique_storage_by_device.get(dev)
                    print(
                        f"    {dev}: tensor_bytes={_fmt_bytes(total)} | unique_storage={_fmt_bytes(uniq)}",
                        file=out,
                        flush=True,
                    )
                rows = inv.get("rows") or []
                if rows:
                    print(
                        f"    top_k={len(rows)} (by tensor_bytes):",
                        file=out,
                        flush=True,
                    )
                    for i, r in enumerate(rows, start=1):
                        print(
                            "      "
                            f"{i:02d} tensor={_fmt_bytes(r.get('tensor_bytes'))}"
                            f" storage={_fmt_bytes(r.get('storage_bytes'))}"
                            f" device={r.get('device')}"
                            f" dtype={r.get('dtype')}"
                            f" shape={r.get('shape')}"
                            f" req_grad={r.get('requires_grad')}"
                            f" leaf={r.get('is_leaf')}"
                            + (
                                f" grad_fn={r.get('grad_fn')}"
                                if r.get("grad_fn")
                                else ""
                            ),
                            file=out,
                            flush=True,
                        )

    # Pause last so you can read the output.
    pause_env = os.environ.get("APEX_STEP_MEM_PAUSE", "0").strip() not in (
        "",
        "0",
        "false",
        "False",
        "FALSE",
    )
    should_pause = (
        (bool(pause) or bool(pause_env))
        and sys.stdin is not None
        and sys.stdin.isatty()
    )
    if should_pause:
        try:
            _ = input("  Press Enter to continue... ")
        except EOFError:
            # Non-interactive runner or stdin closed; don't block.
            pass
