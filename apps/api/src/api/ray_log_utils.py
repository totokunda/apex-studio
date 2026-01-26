from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional


_WORKER_ID_RE = re.compile(r"Worker ID:\s*([0-9a-f]{40})", re.IGNORECASE)
_WORKER_PID_RE = re.compile(r"Worker PID:\s*(\d+)")


def extract_worker_id_and_pid(message: str) -> tuple[Optional[str], Optional[int]]:
    """
    Best-effort extraction of Worker ID / PID from a Ray error string.
    """
    wid = None
    pid = None
    m = _WORKER_ID_RE.search(message or "")
    if m:
        wid = m.group(1)
    m = _WORKER_PID_RE.search(message or "")
    if m:
        try:
            pid = int(m.group(1))
        except Exception:
            pid = None
    return wid, pid


def _tail_text_file(path: Path, *, max_lines: int = 120, max_bytes: int = 512_000) -> str:
    """
    Read up to the last `max_bytes` bytes of a text file and return the last `max_lines`.
    """
    try:
        size = path.stat().st_size
    except Exception:
        size = None

    data: bytes
    try:
        with path.open("rb") as f:
            if size is not None and size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
    except Exception as e:
        return f"<failed to read {path}: {e}>"

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def find_ray_temp_root() -> Path:
    """
    Resolve Ray's local temp root.

    On Windows this typically looks like:
      %TEMP%\\<N>\\ray
    On Linux/macOS:
      /tmp/ray (or $RAY_TMPDIR/ray)
    """
    base = os.getenv("RAY_TMPDIR") or tempfile.gettempdir()
    return Path(base) / "ray"


def tail_worker_logs_from_ray_sessions(
    *,
    worker_id: str,
    worker_pid: Optional[int] = None,
    max_lines: int = 120,
    max_sessions: int = 15,
) -> Optional[str]:
    """
    Best-effort: locate the matching `worker-<id>-*-<pid>.err/.out` logs under recent
    Ray `session_*` folders and return a short tail for diagnostics.
    """
    if not worker_id:
        return None

    ray_root = find_ray_temp_root()
    if not ray_root.exists():
        return None

    sessions = sorted(
        (p for p in ray_root.iterdir() if p.is_dir() and p.name.startswith("session_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[: max(1, int(max_sessions))]

    # Prefer exact PID matches if we have one.
    pid_suffix = f"-{int(worker_pid)}" if worker_pid is not None else None

    candidates: list[Path] = []
    for sess in sessions:
        logs_dir = sess / "logs"
        if not logs_dir.exists():
            continue

        # Common patterns:
        # - worker-<worker_id>-ffffffff-<pid>.err
        # - python-core-worker-<worker_id>_<pid>.log (often empty on native crash)
        glob_pat = f"worker-{worker_id}-*.err"
        for p in logs_dir.glob(glob_pat):
            if pid_suffix is None or p.name.endswith(pid_suffix + ".err"):
                candidates.append(p)

    if not candidates:
        return None

    # Most recent match first
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    err_path = candidates[0]
    out_path = err_path.with_suffix(".out")

    parts: list[str] = [f"[ray worker log] {err_path}"]
    parts.append(_tail_text_file(err_path, max_lines=max_lines))
    if out_path.exists():
        parts.append(f"\n[ray worker stdout] {out_path}")
        parts.append(_tail_text_file(out_path, max_lines=max_lines))

    return "\n".join(parts)

