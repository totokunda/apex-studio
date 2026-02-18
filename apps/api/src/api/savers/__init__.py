"""
Helpers for saving/muxing artifacts produced by engines, preprocessors, and postprocessors.

These utilities exist to keep task orchestration code (e.g. `ray_tasks.py`) focused on
control-flow and error handling, while centralizing all filesystem/media I/O behavior.
"""

from __future__ import annotations

