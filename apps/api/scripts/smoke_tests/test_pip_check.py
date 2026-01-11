from __future__ import annotations

import subprocess
import sys

from .common import log, fail, SmokeContext


def run(ctx: SmokeContext) -> None:
    log("[smoke] pip check")
    res = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ctx.bundle_root),
    )
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        fail(f"`pip check` failed (exit {res.returncode}). stderr:\n{(res.stderr or '').strip()}")


