#!/usr/bin/env python3
"""
Apply Apex-maintained third-party patches to the *current* Python environment.

This is used by:
  - local development installs (after pip/uv installs)
  - the bundling pipeline (to patch dependencies inside the bundled venv)

Patches:
  - diffusers: scripts/updates/patch_diffusers_peft.py
  - xformers:  scripts/updates/patch_xformers_flash3.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(script_path: Path) -> None:
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> int:
    print(f"Applying patches using: {sys.executable}")
    project_root = Path(__file__).resolve().parent.parent.parent  # apps/api
    updates_dir = project_root / "scripts" / "updates"

    _run(updates_dir / "patch_diffusers_peft.py")
    _run(updates_dir / "patch_xformers_flash3.py")
    print("Patches applied successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


