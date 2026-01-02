"""
Mixins package.

Important: Keep this module lightweight.

This package is imported very early during API startup, and eagerly importing
heavy dependencies (e.g. diffusers/torch) here can make server boot slow or fail.
We therefore lazily load mixin symbols on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["DownloadMixin", "LoaderMixin", "OffloadMixin", "ToMixin", "CompileMixin"]

_SYMBOL_TO_MODULE = {
    "DownloadMixin": "download_mixin",
    "LoaderMixin": "loader_mixin",
    "OffloadMixin": "offload_mixin",
    "ToMixin": "to_mixin",
    "CompileMixin": "compile_mixin",
}

if TYPE_CHECKING:
    from .compile_mixin import CompileMixin as CompileMixin
    from .download_mixin import DownloadMixin as DownloadMixin
    from .loader_mixin import LoaderMixin as LoaderMixin
    from .offload_mixin import OffloadMixin as OffloadMixin
    from .to_mixin import ToMixin as ToMixin


def __getattr__(name: str):
    mod_name = _SYMBOL_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(name)
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    val = getattr(mod, name)
    globals()[name] = val
    return val
