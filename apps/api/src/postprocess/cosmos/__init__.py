"""
Cosmos postprocessor package.

Keep this module lightweight: guardrail implementation can pull in heavy deps.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = ["CosmosGuardrailPostprocessor"]

if TYPE_CHECKING:
    from .guardrail import CosmosGuardrailPostprocessor as CosmosGuardrailPostprocessor


def __getattr__(name: str):
    if name != "CosmosGuardrailPostprocessor":
        raise AttributeError(name)
    mod = importlib.import_module(f"{__name__}.guardrail")
    val = getattr(mod, name)
    globals()[name] = val
    return val

