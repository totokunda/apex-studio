"""
Engine package public API.

Engines are now discovered automatically based on the directory / filename
convention implemented in ``src.engine.registry.EngineRegistry``. New engines
do not need to be imported or re-exported here; they are picked up via
autodiscovery.
"""

from typing import TYPE_CHECKING

from .registry import (
    EngineRegistry,
    UniversalEngine,
    get_engine_registry,
    create_engine,
    list_available_engines,
)

if TYPE_CHECKING:
    # Avoid import-time circular dependencies by only importing for type checking.
    from .base_engine import BaseEngine

__all__ = [
    "EngineRegistry",
    "UniversalEngine",
    "get_engine_registry",
    "create_engine",
    "list_available_engines",
    "BaseEngine",
]


def __getattr__(name: str):
    # Lazily expose BaseEngine without importing it at module import time.
    if name == "BaseEngine":
        from .base_engine import BaseEngine  # noqa: PLC0415

        return BaseEngine
    raise AttributeError(name)
