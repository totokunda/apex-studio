from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from loguru import logger

def log(msg: str) -> None:
    logger.debug("{}", msg)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


@dataclass(frozen=True)
class SmokeContext:
    bundle_root: Path
    gpu_type: str
    strict_gpu: bool


def resolve_bundle_root(arg: str | None) -> Path:
    """
    Determine the bundle root dir (the directory that contains `src/` and `manifest/`).
    """
    if arg:
        return Path(arg).resolve()
    env = (os.environ.get("APEX_BUNDLE_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    # Last resort: current working directory
    return Path.cwd().resolve()


def ensure_bundle_on_syspath(bundle_root: Path) -> None:
    """
    Ensure imports like `import src...` work by putting bundle_root on sys.path.
    """
    bundle_root = Path(bundle_root).resolve()
    s = str(bundle_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def import_module_or_fail(name: str) -> object:
    try:
        return importlib.import_module(name)
    except Exception as e:
        fail(f"Failed to import {name}: {e}")


def try_import(name: str) -> tuple[bool, Optional[object], Optional[Exception]]:
    try:
        m = importlib.import_module(name)
        return True, m, None
    except Exception as e:
        return False, None, e


def iter_manifest_files(manifest_dir: Path) -> Iterable[Path]:
    if not manifest_dir.exists():
        return []
    files = list(manifest_dir.rglob("*.yml")) + list(manifest_dir.rglob("*.yaml"))
    return sorted(files)
