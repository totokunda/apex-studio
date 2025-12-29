"""
Lightweight memory management utilities.

Currently this module exposes `MemoryConfig`, which is used to configure how
models are offloaded via diffusers' native group offloading APIs.
"""

from .config import MemoryConfig

__all__ = [
    "MemoryConfig",
]

__version__ = "2.0.0"
