#!/usr/bin/env python3
"""
Patch xformers' FA3 integration to avoid import-time crashes due to import ordering.

We patch the installed file in-place by locating:
  xformers/ops/fmha/flash3.py

The patch is equivalent to `patches/xformers-ops-fmha-flash3.patch` (kept for auditability).

The patched logic:
  - On Windows, prefer pip-installed flash_attn_3 (if compatible) to avoid crashes
    caused by import ordering / extension resolution differences.
  - Fall back to xformers-bundled flash_attn_3 extension if present.

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
    try:
        spec = importlib.util.find_spec("xformers.ops.fmha.flash3")
    except ModuleNotFoundError:
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _patch_contents(src: str) -> str:
    # Replace the entire "how do we locate FA3" block between:
    #   FLASH3_HAS_* flags / _C_flashattention3 = None
    # and:
    #   def _heuristic_kvsplit
    #
    # This is more robust than keying off one exact if/elif ordering because
    # xformers has changed this block across versions.
    end_needle = "\n\n\ndef _heuristic_kvsplit"

    blk_start = -1
    for start_needle in (
        "FLASH3_HAS_PAGED_ATTENTION",
        "FLASH3_HAS_FLOAT8",
        "FLASH3_HAS_DETERMINISTIC_MODE",
        "_C_flashattention3 = None",
    ):
        blk_start = src.find(start_needle)
        if blk_start >= 0:
            break
    if blk_start < 0:
        raise RuntimeError("Could not locate flash_attn_3 resolution block start")
    blk_end = src.find(end_needle, blk_start)
    if blk_end < 0:
        raise RuntimeError("Could not locate end of flash_attn_3 resolution block")

    # Keep this in sync with the desired Windows behavior.
    # NOTE: This intentionally prefers pip-installed flash_attn_3 first.
    replacement = '''FLASH3_HAS_PAGED_ATTENTION = True
FLASH3_HAS_FLOAT8 = False
FLASH3_HAS_DETERMINISTIC_MODE = False
_C_flashattention3 = None
if importlib.util.find_spec("flash_attn_3") and importlib.util.find_spec(
    "flash_attn_3._C"
):
    import flash_attn_3._C  # type: ignore[attr-defined]  # noqa: F401

    incompat_reason = _flash_attention3_incompatible_reason()
    if incompat_reason is None:
        _C_flashattention3 = torch.ops.flash_attn_3
        FLASH_VERSION = "pip_pkg"
        FLASH3_HAS_PAGED_ATTENTION = True
        FLASH3_HAS_FLOAT8 = True
    else:
        logger.warning(f"Flash-Attention 3 package can't be used: {incompat_reason}")

elif importlib.util.find_spec("...flash_attn_3._C", package=__package__):
    from ..._cpp_lib import _build_metadata
    from ...flash_attn_3 import _C  # type: ignore[attr-defined]  # noqa: F401

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version.lstrip("v")
    FLASH3_HAS_DETERMINISTIC_MODE = True
    _C_flashattention3 = torch.ops.flash_attn_3
'''

    new_block = '''FLASH3_HAS_PAGED_ATTENTION = True
FLASH3_HAS_FLOAT8 = False
FLASH3_HAS_DETERMINISTIC_MODE = False
_C_flashattention3 = None
if importlib.util.find_spec("flash_attn_3") and importlib.util.find_spec(
    "flash_attn_3._C"
):
    import flash_attn_3._C  # type: ignore[attr-defined]  # noqa: F401

    incompat_reason = _flash_attention3_incompatible_reason()
    if incompat_reason is None:
        _C_flashattention3 = torch.ops.flash_attn_3
        FLASH_VERSION = "pip_pkg"
        FLASH3_HAS_PAGED_ATTENTION = True
        FLASH3_HAS_FLOAT8 = True
    else:
        logger.warning(f"Flash-Attention 3 package can't be used: {incompat_reason}")

elif importlib.util.find_spec("...flash_attn_3._C", package=__package__):
    from ..._cpp_lib import _build_metadata
    from ...flash_attn_3 import _C  # type: ignore[attr-defined]  # noqa: F401

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version.lstrip("v")
    FLASH3_HAS_DETERMINISTIC_MODE = True
    _C_flashattention3 = torch.ops.flash_attn_3
'''

    if old_block not in src:
        raise RuntimeError(
            "Could not locate expected upstream FA3 block in xformers/ops/fmha/flash3.py; "
            "xformers may have changed the code. Please update patch_xformers_flash3.py accordingly."
        )
    return src.replace(old_block, new_block, 1)


def _has_desired_windows_block(src: str) -> bool:
    """True if the FA3 resolution block matches our expected Windows ordering."""
    pip_idx = src.find('find_spec("flash_attn_3")')
    in_tree_idx = src.find('elif importlib.util.find_spec("...flash_attn_3._C", package=__package__)')
    return (
        pip_idx >= 0
        and in_tree_idx >= 0
        and pip_idx < in_tree_idx
        and "FLASH3_HAS_FLOAT8 = False" in src
        and "FLASH3_HAS_DETERMINISTIC_MODE = False" in src
        and 'FLASH_VERSION = "pip_pkg"' in src
    )


def main() -> int:
    v = os.environ.get("APEX_PATCH_XFORMERS_FLASH3")
    if v is None:
        v = os.environ.get("APEX_BUNDLE_PATCH_XFORMERS_FLASH3", "1")
    if str(v).strip().lower() in ("0", "false", "no", "off"):
        print("Skipping xformers flash3 patch (APEX_PATCH_XFORMERS_FLASH3=0)")
        return 0

    # This patch is currently only needed/validated on Windows.
    if os.name != "nt":
        print("Non-Windows platform detected; skipping xformers flash3 patch")
        return 0

    p = _find_flash3_path()
    if p is None:
        # xformers not installed (optional dependency)
        print("xformers.ops.fmha.flash3 not found; skipping")
        return 0

    before = p.read_text(encoding="utf-8")
    
    if _has_desired_windows_block(before):
        print(f"xformers flash3 already patched for Windows (verified content): {p}")
        return 0

    after = _patch_contents(before)
    if after == before:
        print(f"xformers flash3 content already matches patch target: {p}")
        return 0

    p.write_text(after, encoding="utf-8")
    print(f"Patched xformers flash3: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


