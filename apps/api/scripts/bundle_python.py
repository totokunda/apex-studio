#!/usr/bin/env python3
"""
Entrypoint for bundling the Python API.

The implementation lives under `scripts/bundling/bundle_python.py`.

Example:
    python scripts/bundle_python.py --platform auto --gpu auto --output ./dist --tar-zst-level 12
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _resolve_python_runner() -> str:
    """
    Resolve the Python interpreter to exec into.

    On Windows, some entrypoint wrappers can result in `sys.executable` pointing to an
    entrypoint executable rather than `python.exe`. For bundling we want the actual
    interpreter so the process behaves like a normal Python invocation.
    """
    exe = Path(sys.executable).resolve()
    name = exe.name.lower()
    if name.startswith("python"):
        return str(exe)
    if sys.platform == "win32":
        candidate = exe.with_name("python.exe")
        if candidate.exists():
            return str(candidate)
    base = getattr(sys, "_base_executable", None)
    if base:
        base_path = Path(str(base)).resolve()
        if base_path.exists():
            return str(base_path)
    return str(exe)


def main() -> None:
    target = Path(__file__).resolve().parent / "bundling" / "bundle_python.py"
    py = _resolve_python_runner()
    # IMPORTANT (Windows):
    # `os.execv` is not a true exec on Windows; it can behave like "spawn then exit",
    # which makes callers (like `subprocess.run(...)`) think we're done while the
    # real bundler continues in the background.
    #
    # Use a normal subprocess invocation and propagate the exit code so parents
    # reliably wait for bundling to finish.
    res = subprocess.run([py, str(target), *sys.argv[1:]])
    raise SystemExit(res.returncode)


if __name__ == "__main__":
    main()
