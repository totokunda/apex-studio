"""
RIFE postprocessor package.

Keep this module lightweight: importing RIFE postprocessor can pull in torch.
We expose a lightweight downloader for setup/install flows.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["RifePostprocessor", "download_rife_assets"]

_SYMBOL_TO_MODULE = {
    "RifePostprocessor": "rife",
    "download_rife_assets": "download",
}

if TYPE_CHECKING:
    from .download import download_rife_assets as download_rife_assets
    from .rife import RifePostprocessor as RifePostprocessor


def __getattr__(name: str):
    mod_name = _SYMBOL_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(name)
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    val = getattr(mod, name)
    globals()[name] = val
    return val

