"""
Engine package public API.

Engines are now discovered automatically based on the directory / filename
convention implemented in ``src.engine.registry.EngineRegistry``. New engines
do not need to be imported or re-exported here; they are picked up via
autodiscovery.
"""

from .base_engine import BaseEngine
from .registry import (
    EngineRegistry,
    UniversalEngine,
    get_engine_registry,
    create_engine,
    list_available_engines,
)

__all__ = [
    "EngineRegistry",
    "UniversalEngine",
    "get_engine_registry",
    "create_engine",
    "list_available_engines",
    "BaseEngine",
]
