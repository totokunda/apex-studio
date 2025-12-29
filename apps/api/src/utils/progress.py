from typing import Callable, Optional


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
    try:
        progress_callback(p, message if message is not None else "")
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
