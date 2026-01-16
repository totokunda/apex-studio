#!/usr/bin/env python3
"""
Patch xformers' FA3 integration to avoid import-time crashes due to import ordering.

We patch the installed file in-place by locating:
  xformers/ops/fmha/flash3.py

The patched logic (simple block swap):
  - Prefer pip-installed flash_attn_3 first (avoids TORCH_LIBRARY namespace double-registration)
  - Fall back to xformers-bundled flash_attn_3 extension if pip package is unavailable/unusable

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
    # Blocked, deterministic patch: replace the exact upstream block (as observed in our env)
    # with the desired ordering.
    old_block = '''FLASH3_HAS_PAGED_ATTENTION = True
FLASH3_HAS_FLOAT8 = False
FLASH3_HAS_DETERMINISTIC_MODE = False
_C_flashattention3 = None
if importlib.util.find_spec("...flash_attn_3._C", package=__package__):
    from ..._cpp_lib import _build_metadata
    from ...flash_attn_3 import _C  # type: ignore[attr-defined]  # noqa: F401

    if _build_metadata is not None:
        FLASH_VERSION = _build_metadata.flash_version.lstrip("v")
    FLASH3_HAS_DETERMINISTIC_MODE = True
    _C_flashattention3 = torch.ops.flash_attn_3

elif importlib.util.find_spec("flash_attn_3") and importlib.util.find_spec(
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


def _is_patched(src: str) -> bool:
    """Check if the file already has the patched logic."""
    pip_if = 'if importlib.util.find_spec("flash_attn_3") and importlib.util.find_spec('
    bundled_elif = 'elif importlib.util.find_spec("...flash_attn_3._C", package=__package__):'
    if pip_if not in src or bundled_elif not in src:
        return False
    return src.find(pip_if) < src.find(bundled_elif)


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
    
    if _is_patched(before):
        print(f"xformers flash3 already patched (verified content): {p}")
        return 0

    after = _patch_contents(before)
    if after == before:
        # Should be covered by _is_patched, but just in case
        print(f"xformers flash3 content matches patch target: {p}")
        return 0

    p.write_text(after, encoding="utf-8")
    print(f"Patched xformers flash3: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


