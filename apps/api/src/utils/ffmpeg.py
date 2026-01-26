"""
FFmpeg executable resolution for the Python API.

Electron can export an absolute FFmpeg path via env:
  - APEX_FFMPEG_PATH
  - (fallback) FFMPEG_PATH

If unset or invalid, we default to plain `ffmpeg` and rely on PATH.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from shutil import which
from typing import Optional, Sequence, Tuple, Union

import subprocess


def get_ffmpeg_path(default: str = "ffmpeg") -> str:
    """
    Return the ffmpeg executable to use.
    Prefers absolute (or existing) paths provided via env, otherwise falls back
    to `default` (typically `ffmpeg`).
    """

    candidate = os.environ.get("APEX_FFMPEG_PATH") or os.environ.get("FFMPEG_PATH")
    if candidate:
        try:
            if Path(candidate).exists():
                return candidate
        except OSError:
            pass
        # Allow env to point to a command name that is on PATH.
        if which(candidate):
            return candidate
    return default


def get_ffmpeg_timeout_seconds(
    *,
    env_var: str = "APEX_FFMPEG_TIMEOUT_SECONDS",
    default: float = 120.0,
    min_s: float = 5.0,
    max_s: float = 60.0 * 10.0,
) -> float:
    """
    Read a global FFmpeg timeout from env.

    This is meant as a safety net so ffmpeg can't wedge a server or long-lived worker.
    """
    try:
        raw = os.environ.get(env_var, None)
        timeout_s = float(raw) if raw not in (None, "") else float(default)
    except Exception:
        timeout_s = float(default)
    return float(max(min_s, min(max_s, timeout_s)))


def _infer_default_ffmpeg_log_path() -> Path:
    base_dir = os.environ.get("APEX_FFMPEG_LOG_DIR", "") or ""
    try:
        d = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
    except Exception:
        d = Path(tempfile.gettempdir())
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort.
        pass
    ts = int(time.time() * 1000)
    return d / f"apex_ffmpeg_{ts}_{os.getpid()}.log"


def _uses_stdin_pipe(cmd: Sequence[str]) -> bool:
    # Detect ffmpeg commands that use stdin as an input ("-i pipe:0" or "-i -").
    try:
        args = [str(x) for x in cmd]
    except Exception:
        return False
    for i, a in enumerate(args[:-1]):
        if a == "-i":
            nxt = args[i + 1]
            if nxt in {"pipe:0", "-"}:
                return True
    return False


def run_ffmpeg(
    cmd: Sequence[str],
    *,
    timeout_s: Optional[float] = None,
    log_path: Union[str, Path, None] = None,
    stdin_bytes: Optional[bytes] = None,
    capture_stdout: bool = False,
    check: bool = False,
) -> Tuple[int, Path, Optional[bytes]]:
    """
    Run an ffmpeg command safely:

    - Enforces a timeout (default via `APEX_FFMPEG_TIMEOUT_SECONDS`)
    - Avoids stdin interaction for file-based commands
    - Logs stdout/stderr to a file to avoid buffering issues

    Returns (returncode, log_path, stdout_bytes).
    `stdout_bytes` is only set when `capture_stdout=True`.
    """
    timeout_val = (
        float(timeout_s) if timeout_s is not None else get_ffmpeg_timeout_seconds()
    )

    lp = Path(log_path) if log_path is not None else _infer_default_ffmpeg_log_path()
    stdout_bytes: Optional[bytes] = None

    # If ffmpeg reads from stdin (pipe:0 / "-"), we must not detach stdin.
    uses_stdin = _uses_stdin_pipe(cmd) or stdin_bytes is not None

    cmd_list = [str(x) for x in cmd]
    if (not uses_stdin) and ("-nostdin" not in cmd_list):
        # Insert immediately after the executable for readability.
        if len(cmd_list) >= 1:
            cmd_list = [cmd_list[0], "-nostdin", *cmd_list[1:]]

    try:
        with lp.open("w", encoding="utf-8") as log:
            try:
                log.write(f"[apex] timeout_s={timeout_val:.1f}\n")
                log.write("[apex] cmd:\n")
                log.write(" ".join(cmd_list) + "\n\n")
                log.flush()
            except Exception:
                pass
            try:
                proc = subprocess.run(
                    cmd_list,
                    input=stdin_bytes,
                    stdin=None if uses_stdin else subprocess.DEVNULL,
                    stdout=subprocess.PIPE if capture_stdout else log,
                    stderr=log,
                    check=bool(check),
                    timeout=timeout_val,
                )
            except subprocess.TimeoutExpired:
                try:
                    log.write(f"\n[apex] ffmpeg timed out after {timeout_val:.1f}s\n")
                except Exception:
                    pass
                # 124 is the conventional timeout exit code.
                return 124, lp, None
    except Exception:
        # If we can't even write the log file or spawn, bubble up.
        raise

    if capture_stdout:
        try:
            stdout_bytes = proc.stdout  # type: ignore[assignment]
        except Exception:
            stdout_bytes = None

    return int(proc.returncode), lp, stdout_bytes
