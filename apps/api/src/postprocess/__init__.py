"""
Postprocessors package.

Keep this module lightweight: importing `src.postprocess` should not eagerly import
heavy optional dependencies (e.g. transformers/sklearn/scipy/torch).

We therefore lazily resolve postprocessor classes on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "CosmosGuardrailPostprocessor",
    "RifePostprocessor",
    "BasePostprocessor",
    "PostprocessorCategory",
    "postprocessor_registry",
]

_SYMBOL_TO_MODULE = {
    # Keep these as module names relative to this package.
    "CosmosGuardrailPostprocessor": "cosmos",
    "RifePostprocessor": "rife",
    # Base symbols are also lazily imported to avoid importing torch at package import time.
    "BasePostprocessor": "base",
    "PostprocessorCategory": "base",
    "postprocessor_registry": "base",
}

if TYPE_CHECKING:
    from .base import (
        BasePostprocessor as BasePostprocessor,
        PostprocessorCategory as PostprocessorCategory,
        postprocessor_registry as postprocessor_registry,
    )
    from .cosmos import CosmosGuardrailPostprocessor as CosmosGuardrailPostprocessor
    from .rife import RifePostprocessor as RifePostprocessor


def __getattr__(name: str):
    mod_name = _SYMBOL_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(name)
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    val = getattr(mod, name)
    globals()[name] = val
    return val
