#!/usr/bin/env python3
"""
Compatibility entrypoint.

The implementation lives under `scripts/updates/apply_code_update.py`.
Keep this file so bundle/runtime callers can continue to reference `scripts/apply_code_update.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "updates" / "apply_code_update.py"
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    main()
