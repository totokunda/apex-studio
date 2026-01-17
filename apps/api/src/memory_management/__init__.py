"""
Lightweight memory management utilities.

Currently this module exposes `MemoryConfig`, which is used to configure how
models are offloaded via diffusers' native group offloading APIs.
"""

from .config import MemoryConfig
from .group_offloading import apply_group_offloading
from .weight_manager import GlobalWeightManager, get_global_weight_manager

__all__ = [
    "MemoryConfig",
    "apply_group_offloading",
    "GlobalWeightManager",
    "get_global_weight_manager",
]

__version__ = "2.0.0"
