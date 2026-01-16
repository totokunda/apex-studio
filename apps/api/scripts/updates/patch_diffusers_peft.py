#!/usr/bin/env python3
"""
Patch diffusers' PEFT loader to match expected upstream behavior.

We patch the installed file in-place by locating:
  diffusers/loaders/peft.py

Patch behavior:
  - Remove the try/except KeyError fallback around:
      scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]
  - This makes missing mappings fail loudly (matches upstream intent), rather than
    silently falling back to an identity lambda.

Toggles:
  - Set APEX_PATCH_DIFFUSERS_PEFT=0 to disable.
  - Bundle compatibility: APEX_BUNDLE_PATCH_DIFFUSERS_PEFT=0 also disables.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _enabled() -> bool:
    v = os.environ.get("APEX_PATCH_DIFFUSERS_PEFT")
    if v is None:
        v = os.environ.get("APEX_BUNDLE_PATCH_DIFFUSERS_PEFT", "1")
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _find_peft_path() -> Path | None:
    try:
        import diffusers  # type: ignore
    except Exception:
        return None
    base = Path(diffusers.__file__).resolve().parent
    return (base / "loaders" / "peft.py").resolve()


def _patch_src(src: str) -> tuple[str, int]:
    # Match the try/except block exactly (indentation-aware).
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)try:\n"
        r"(?P=indent)[ \t]*scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING\[self\.__class__\.__name__\]\n"
        r"(?P=indent)[ \t]*except KeyError:\n"
        r"(?P=indent)[ \t]*scale_expansion_fn = lambda model, weights: weights[ \t]*$"
    )

    def _repl(m: re.Match) -> str:
        indent = m.group("indent")
        return f"{indent}scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]"

    return pattern.subn(_repl, src, count=1)


def main() -> int:
    if not _enabled():
        print("Skipping diffusers peft.py patch (APEX_PATCH_DIFFUSERS_PEFT=0)")
        return 0

    p = _find_peft_path()
    if p is None:
        # diffusers may be an optional dep in some installs
        print("diffusers not found; skipping")
        return 0
    if not p.exists():
        raise SystemExit(f"diffusers peft.py not found at expected path: {p}")

    before = p.read_text(encoding="utf-8")
    after, n = _patch_src(before)
    if n != 1:
        # If the file already has the desired one-liner, treat as success/idempotent.
        if "_SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]" in before and "except KeyError" not in before:
            print(f"diffusers peft.py already patched: {p}")
            return 0
        raise SystemExit(f"Failed to apply diffusers peft.py patch (pattern not found). File: {p}")

    p.write_text(after, encoding="utf-8")
    print(f"Patched diffusers peft.py: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


