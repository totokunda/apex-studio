"""
Compatibility entrypoint.

The implementation lives under `scripts/setup/setup.py`.
Keep this file so existing tooling can continue to reference `scripts/setup.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "setup" / "setup.py"
    os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    main()
