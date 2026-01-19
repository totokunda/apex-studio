#!/usr/bin/env python3
"""
Compatibility entrypoint.

The implementation lives under `scripts/bundling/bundle_python.py`.
Keep this file so callers can continue to run: `python scripts/bundle_python.py ...`
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
