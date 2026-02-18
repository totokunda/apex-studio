from __future__ import annotations

import sys
from pathlib import Path


# Pytest 9 can run with importlib-based collection/imports, which doesn't always
# include the repository root on `sys.path`. Our package is a top-level `src/`
# module, so ensure the project root is importable for tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

