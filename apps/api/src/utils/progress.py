from __future__ import annotations

import asyncio
import inspect
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class _ProgressEvent:
    callback: Callable[[float, str], Any]
    p: float
    msg: str
    loop: Optional[asyncio.AbstractEventLoop]


_PROGRESS_QUEUE: "queue.Queue[_ProgressEvent]" = queue.Queue()
_PROGRESS_WORKER_STARTED = False
_PROGRESS_WORKER_LOCK = threading.Lock()


def _ensure_progress_worker_started() -> None:
    global _PROGRESS_WORKER_STARTED
    if _PROGRESS_WORKER_STARTED:
        return
    with _PROGRESS_WORKER_LOCK:
        if _PROGRESS_WORKER_STARTED:
            return

        t = threading.Thread(
            target=_progress_worker_loop,
            name="apex-progress",
            daemon=True,
        )
        t.start()
        _PROGRESS_WORKER_STARTED = True


def _progress_worker_loop() -> None:
    while True:
        ev = _PROGRESS_QUEUE.get()
        try:
            cb = ev.callback
            if ev.loop is not None and not ev.loop.is_closed():
                # Execute on the captured event loop thread to keep callback thread-affinity,
                # and block this worker until completion to preserve strict ordering.
                if inspect.iscoroutinefunction(cb):
                    fut = asyncio.run_coroutine_threadsafe(cb(ev.p, ev.msg), ev.loop)  # type: ignore[misc]
                    fut.result()
                else:
                    done = threading.Event()

                    def _run_sync() -> None:
                        try:
                            cb(ev.p, ev.msg)
                        finally:
                            done.set()

                    ev.loop.call_soon_threadsafe(_run_sync)
                    done.wait()
            else:
                # No event loop context captured: run directly in this worker thread.
                if inspect.iscoroutinefunction(cb):
                    asyncio.run(cb(ev.p, ev.msg))  # type: ignore[misc]
                else:
                    cb(ev.p, ev.msg)
        except Exception:
            # Swallow callback errors to avoid interrupting pipelines
            pass
        finally:
            _PROGRESS_QUEUE.task_done()


def safe_emit_progress(
    progress_callback: Optional[Callable[[float, str], None]],
    p: float,
    message: Optional[str] = None,
) -> None:
    if progress_callback is None:
        return
    try:
        p = float(max(0.0, min(1.0, p)))
    except Exception:
        p = 0.0

    msg = message if message is not None else ""

    # Fire-and-forget + ordered:
    # Enqueue to a single worker that processes events FIFO, so callback invocation order
    # matches the order of safe_emit_progress() calls, without blocking model execution.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        _ensure_progress_worker_started()
        _PROGRESS_QUEUE.put(
            _ProgressEvent(callback=progress_callback, p=p, msg=msg, loop=loop)  # type: ignore[arg-type]
        )
    except Exception:
        # Swallow callback errors to avoid interrupting pipelines
        pass


def make_mapped_progress(
    progress_callback: Optional[Callable[[float, str], None]],
    start: float = 0.0,
    end: float = 1.0,
) -> Callable[[float, Optional[str]], None]:
    start = float(max(0.0, min(1.0, start)))
    end = float(max(0.0, min(1.0, end)))

    def _mapped(local_progress: float, message: Optional[str] = None) -> None:
        span = max(0.0, end - start)
        try:
            lp = float(max(0.0, min(1.0, local_progress)))
        except Exception:
            lp = 0.0
        overall = start + span * lp
        safe_emit_progress(
            progress_callback,
            overall,
            message if message is not None else f"Denoising {int(lp * 100)}%",
        )

    return _mapped
