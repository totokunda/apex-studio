#!/usr/bin/env python3
"""
Patch diffusers to be more tolerant of older/CPU-only torch builds.

Problem
-------
Some diffusers versions reference newer PyTorch APIs at import time. On older
torch builds (notably our Intel macOS stack pinned to torch==2.2.2), those APIs
may not exist, causing import-time crashes such as:
  AttributeError: module 'torch' has no attribute 'xpu'
  AttributeError: module 'torch.distributed' has no attribute 'device_mesh'

We patch the installed file in-place by locating:
  diffusers/utils/torch_utils.py
  diffusers/models/_modeling_parallel.py

Patch behavior
--------------
- Inject a stub `torch.xpu` implementation when missing, so import-time tables
  referencing `torch.xpu.*` do not crash.
- Add `from __future__ import annotations` to `_modeling_parallel.py` so type
  annotations referencing newer torch symbols don't get evaluated at import time.

Toggles
-------
- Set APEX_PATCH_DIFFUSERS_TORCH_XPU=0 to disable.
- Bundle compatibility: APEX_BUNDLE_PATCH_DIFFUSERS_TORCH_XPU=0 also disables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _enabled() -> bool:
    v = os.environ.get("APEX_PATCH_DIFFUSERS_TORCH_XPU")
    if v is None:
        v = os.environ.get("APEX_BUNDLE_PATCH_DIFFUSERS_TORCH_XPU", "1")
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _find_diffusers_root() -> Path | None:
    """
    Find the installed `diffusers` package directory without importing it.

    Importing diffusers can fail *because* of the issue we're patching, so we must
    locate it by scanning sys.path.
    """
    for entry in sys.path:
        if not entry:
            continue
        try:
            base = Path(entry).resolve()
        except Exception:
            continue
        pkg = base / "diffusers"
        if (pkg / "__init__.py").exists():
            return pkg
    return None


def _is_patched(src: str) -> bool:
    return "APEX_PATCH_DIFFUSERS_TORCH_XPU_STUB" in src


def _patch_src(src: str) -> tuple[str, bool]:
    # Robust approach: provide a stub `torch.xpu` object if torch doesn't have it.
    # This avoids needing to patch every individual `torch.xpu.*` reference.

    marker = "APEX_PATCH_DIFFUSERS_TORCH_XPU_STUB"
    lines = src.splitlines()

    insert_at: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "import torch":
            insert_at = i + 1
            break

    if insert_at is None:
        return src, False

    # If we've already inserted the stub block, do nothing.
    if any(marker in l for l in lines[max(0, insert_at - 5) : min(len(lines), insert_at + 40)]):
        return src, False

    stub_block = [
        "",
        f"# {marker}: provide `torch.xpu` for older torch builds",
        "if not hasattr(torch, \"xpu\"):",
        "    class _ApexTorchXpuStub:",
        "        \"\"\"Best-effort stub for torch.xpu on builds without XPU support.\"\"\"",
        "        def __getattr__(self, name: str):  # noqa: ANN001",
        "            # Return callables by default to satisfy diffusers import-time tables.",
        "            if name == \"device_count\":",
        "                return lambda *args, **kwargs: 0",
        "            if name == \"is_available\":",
        "                return lambda *args, **kwargs: False",
        "            return lambda *args, **kwargs: None",
        "    torch.xpu = _ApexTorchXpuStub()  # type: ignore[attr-defined]",
        "",
    ]

    out_lines = lines[:insert_at] + stub_block + lines[insert_at:]
    return "\n".join(out_lines), True


def _patch_future_annotations(src: str, marker: str) -> tuple[str, bool]:
    if "from __future__ import annotations" in src:
        return src, False

    lines = src.splitlines()
    i = 0

    # Shebang
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1

    # Encoding cookie
    if i < len(lines) and ("coding:" in lines[i] or "coding=" in lines[i]):
        i += 1

    # Module docstring (must keep future import after it)
    j = i
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    if j < len(lines):
        s = lines[j].lstrip()
        if s.startswith('"""') or s.startswith("'''"):
            q = s[:3]
            # Single-line docstring
            if s.count(q) >= 2 and len(s) > 3:
                j += 1
            else:
                j += 1
                while j < len(lines) and q not in lines[j]:
                    j += 1
                if j < len(lines):
                    j += 1
            i = j
        else:
            i = j

    insert = f"from __future__ import annotations  # {marker}"
    out = lines[:i] + [insert, ""] + lines[i:]
    return "\n".join(out), True


def main() -> int:
    if not _enabled():
        print("Skipping diffusers torch.xpu patch (APEX_PATCH_DIFFUSERS_TORCH_XPU=0)")
        return 0

    root = _find_diffusers_root()
    if root is None:
        print("diffusers package path not found; skipping")
        return 0

    patched_any = False

    # 1) torch.xpu stub in torch_utils.py
    torch_utils = (root / "utils" / "torch_utils.py").resolve()
    if torch_utils.exists():
        before = torch_utils.read_text(encoding="utf-8")
        if _is_patched(before):
            print(f"diffusers torch_utils.py already patched: {torch_utils}")
        else:
            after, applied = _patch_src(before)
            if applied:
                torch_utils.write_text(after, encoding="utf-8")
                print(f"Patched diffusers torch_utils.py (guarded torch.xpu): {torch_utils}")
                patched_any = True
            else:
                print(f"diffusers torch_utils.py did not look patchable; skipping: {torch_utils}")
    else:
        print(f"diffusers torch_utils.py not found; skipping: {torch_utils}")

    # 2) Avoid import-time evaluation of annotations referencing newer torch symbols.
    modeling_parallel = (root / "models" / "_modeling_parallel.py").resolve()
    if modeling_parallel.exists():
        before = modeling_parallel.read_text(encoding="utf-8")
        after, applied = _patch_future_annotations(
            before, marker="APEX_PATCH_DIFFUSERS_TORCH_DEVICE_MESH"
        )
        if applied:
            modeling_parallel.write_text(after, encoding="utf-8")
            print(
                "Patched diffusers _modeling_parallel.py (postponed annotations): "
                f"{modeling_parallel}"
            )
            patched_any = True
    else:
        # Not all diffusers versions ship this file.
        pass

    if not patched_any:
        print("No diffusers torch compatibility patches were applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

