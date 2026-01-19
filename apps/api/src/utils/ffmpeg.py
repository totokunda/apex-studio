"""
FFmpeg executable resolution for the Python API.

Electron can export an absolute FFmpeg path via env:
  - APEX_FFMPEG_PATH
  - (fallback) FFMPEG_PATH

If unset or invalid, we default to plain `ffmpeg` and rely on PATH.
"""

from __future__ import annotations

import os
from pathlib import Path
from shutil import which


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
