from __future__ import annotations

import platform
import re
import subprocess
import sys

from .common import log, fail, SmokeContext

_MEDIAPIPE_UNSUPPORTED_RE = re.compile(
    r"^mediapipe\s+\S+\s+is not supported on this platform$"
)


def _can_ignore_mediapipe_unsupported() -> bool:
    """
    We have a known false-positive on macOS arm64 where the installed mediapipe wheel can
    be tagged x86_64 (and thus fails `pip check`), but the module (including native bindings)
    works correctly at runtime in our bundle environment.
    """
    if platform.system() != "Darwin":
        return False
    if platform.machine() != "arm64":
        return False
    try:
        import mediapipe  # noqa: F401

        # Ensure native bindings load too (not just top-level import).
        from mediapipe.python import _framework_bindings  # noqa: F401

        return True
    except Exception:
        return False


def _filter_pip_check_stderr(stderr: str) -> tuple[str, list[str]]:
    lines = [ln.rstrip("\n") for ln in (stderr or "").splitlines()]
    ignored: list[str] = []
    kept: list[str] = []
    can_ignore = _can_ignore_mediapipe_unsupported()
    for ln in lines:
        if can_ignore and _MEDIAPIPE_UNSUPPORTED_RE.match(ln.strip()):
            ignored.append(ln)
        else:
            kept.append(ln)
    return ("\n".join([l for l in kept if l.strip()]), ignored)


def run(ctx: SmokeContext) -> None:
    log("[smoke] pip check")
    log(f"cwd: {str(ctx.bundle_root)}")
    log(f"sys.executable: {sys.executable}")
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
        filtered, ignored = _filter_pip_check_stderr(res.stderr or "")
        if not filtered.strip():
            if ignored:
                log("[smoke] NOTE: ignored known pip-check false-positive(s):")
                for ln in ignored:
                    log(f"[smoke]   {ln}")
            return
        fail(f"`pip check` failed (exit {res.returncode}). stderr:\n{filtered.strip()}")
