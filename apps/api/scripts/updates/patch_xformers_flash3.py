#!/usr/bin/env python3
"""
Patch xformers' FA3 integration to avoid import-time crashes due to import ordering.

We patch the installed file in-place by locating:
  xformers/ops/fmha/flash3.py

The patch is equivalent to `patches/xformers-ops-fmha-flash3.patch` (kept for auditability).

The patched logic:
  - Prefer xformers-bundled flash_attn_3 extension if present
  - If that import fails at runtime, fall back to pip-installed flash_attn_3
  - Never raise at import time; log and continue with _C_flashattention3=None

Toggles:
  - Set APEX_PATCH_XFORMERS_FLASH3=0 to disable.
  - Bundle compatibility: APEX_BUNDLE_PATCH_XFORMERS_FLASH3=0 also disables.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _find_flash3_path() -> Path | None:
    spec = importlib.util.find_spec("xformers.ops.fmha.flash3")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _patch_contents(src: str) -> str:
    # Replace the entire "how do we locate FA3" block between:
    #   _C_flashattention3 = None
    # and:
    #   def _heuristic_kvsplit
    #
    # This is more robust than keying off one exact if/elif ordering because
    # xformers has changed this block across versions.
    start_needle = "_C_flashattention3 = None"
    end_needle = "\n\n\ndef _heuristic_kvsplit"

    blk_start = src.find(start_needle)
    if blk_start < 0:
        raise RuntimeError("Could not locate _C_flashattention3 assignment")
    blk_end = src.find(end_needle, blk_start)
    if blk_end < 0:
        raise RuntimeError("Could not locate end of flash_attn_3 resolution block")

    replacement = '''_C_flashattention3 = None

if importlib.util.find_spec("...flash_attn_3._C", package=__package__):
    try:
        from ..._cpp_lib import _build_metadata
        from ...flash_attn_3 import _C  # type: ignore[attr-defined]  # noqa: F401

        if _build_metadata is not None:
            FLASH_VERSION = _build_metadata.flash_version.lstrip("v")
        FLASH3_HAS_DETERMINISTIC_MODE = True
        _C_flashattention3 = torch.ops.flash_attn_3
    except Exception:
        logger.warning(
            "Failed to import xformers-bundled flash_attn_3; will try pip flash_attn_3 if available",
            exc_info=True,
        )

if (
    _C_flashattention3 is None
    and importlib.util.find_spec("flash_attn_3")
    and importlib.util.find_spec("flash_attn_3._C")
):
    try:
        import flash_attn_3._C  # type: ignore[attr-defined]  # noqa: F401

        incompat_reason = _flash_attention3_incompatible_reason()
        if incompat_reason is None:
            _C_flashattention3 = torch.ops.flash_attn_3
            FLASH_VERSION = "pip_pkg"
            FLASH3_HAS_PAGED_ATTENTION = True
            FLASH3_HAS_FLOAT8 = True
        else:
            logger.warning(f"Flash-Attention 3 package can't be used: {incompat_reason}")
    except Exception:
        logger.warning("Failed to import pip flash_attn_3", exc_info=True)
'''

    # splice: keep everything before the assignment line, then our replacement, then the rest
    # starting at def _heuristic_kvsplit.
    # Move blk_start to the beginning of the line containing the assignment to preserve formatting.
    line_start = src.rfind("\n", 0, blk_start)
    if line_start < 0:
        line_start = 0
    else:
        line_start += 1

    return src[:line_start] + replacement + src[blk_end:]


def main() -> int:
    v = os.environ.get("APEX_PATCH_XFORMERS_FLASH3")
    if v is None:
        v = os.environ.get("APEX_BUNDLE_PATCH_XFORMERS_FLASH3", "1")
    if str(v).strip().lower() in ("0", "false", "no", "off"):
        print("Skipping xformers flash3 patch (APEX_PATCH_XFORMERS_FLASH3=0)")
        return 0

    p = _find_flash3_path()
    if p is None:
        # xformers not installed (optional dependency)
        print("xformers.ops.fmha.flash3 not found; skipping")
        return 0

    before = p.read_text(encoding="utf-8")
    after = _patch_contents(before)
    if after == before:
        print(f"xformers flash3 already patched: {p}")
        return 0

    p.write_text(after, encoding="utf-8")
    print(f"Patched xformers flash3: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


