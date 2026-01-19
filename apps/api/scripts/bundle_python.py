#!/usr/bin/env python3
"""
Entrypoint for bundling the Python API.

The implementation lives under `scripts/bundling/bundle_python.py`.

Example:
    python scripts/bundle_python.py --platform auto --gpu auto --output ./dist --tar-zst --tar-zst-level 12 --sign
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "bundling" / "bundle_python.py"
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    main()
