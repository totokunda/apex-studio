#!/usr/bin/env python3
"""
Patch diffusers' PEFT loader to ensure robustness against missing model mappings.

We patch the installed file in-place by locating:
  diffusers/loaders/peft.py

Patch behavior:
  - Add a try/except KeyError fallback around:
      scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]
  - This prevents crashes when using a model architecture not yet explicitly supported
    in the diffusers mapping, falling back to an identity function.

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


def _is_patched(src: str) -> bool:
    """Check if the file already has the try/except KeyError fallback."""
    # We look for the 'except KeyError' block near the assignment.
    return "except KeyError:" in src and "_SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]" in src


def _patch_src(src: str) -> tuple[str, bool]:
    # We want to replace the bare assignment with a try/except block.
    # The assignment looks like:
    # scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]
    
    target_line = "_SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]"
    
    # Regex to find the line and capture indentation
    # We only match if it's NOT preceded by 'try:'.
    # But relying on lookbehind for variable whitespace is hard.
    # Instead, we'll find the line, check if it's already in a try block, and if not, replace it.
    
    lines = src.splitlines()
    new_lines = []
    patched_any = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if target_line in line:
            # Found the assignment. Check context.
            is_inside_try = False
            # Look backwards for 'try:' at the same or lower indentation level?
            # Actually, 'try:' would be at the SAME indentation level as the current line if this line was inside it?
            # No, if inside, 'try:' is outer.
            # But here we assume the file structure is flat (function body).
            
            # Simple check: check previous non-empty line
            prev_idx = i - 1
            while prev_idx >= 0:
                prev_line = lines[prev_idx].strip()
                if prev_line and not prev_line.startswith("#"):
                    if prev_line == "try:":
                        is_inside_try = True
                    break
                prev_idx -= 1
            
            if not is_inside_try:
                # Apply patch
                indent = line[:len(line) - len(line.lstrip())]
                
                # Construct the block
                # try:
                #     scale_expansion_fn = ...
                # except KeyError:
                #     scale_expansion_fn = lambda model, weights: weights
                
                new_lines.append(f"{indent}try:")
                new_lines.append(f"{indent}    {line.strip()}")
                new_lines.append(f"{indent}except KeyError:")
                new_lines.append(f"{indent}    scale_expansion_fn = lambda model, weights: weights")
                
                patched_any = True
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
        
    return "\n".join(new_lines), patched_any


def main() -> int:
    if not _enabled():
        print("Skipping diffusers peft.py patch (APEX_PATCH_DIFFUSERS_PEFT=0)")
        return 0

    p = _find_peft_path()
    if p is None:
        print("diffusers not found; skipping")
        return 0
    if not p.exists():
        raise SystemExit(f"diffusers peft.py not found at expected path: {p}")

    before = p.read_text(encoding="utf-8")
    
    if _is_patched(before):
        print(f"diffusers peft.py already patched (fallback present): {p}")
        return 0

    after, applied = _patch_src(before)

    if not applied:
        # If we didn't apply the patch but _is_patched returned False, 
        # it means we couldn't find the target line to wrap.
        raise SystemExit(
            f"Failed to apply diffusers peft.py patch (target line not found).\n"
            f"File: {p}\n"
            f"Please verify the content of the file."
        )

    p.write_text(after, encoding="utf-8")
    print(f"Patched diffusers peft.py (added fallback): {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
