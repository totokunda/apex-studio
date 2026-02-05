from __future__ import annotations

from pathlib import Path
from typing import Optional


def make_preview_url(
    *,
    result_path: str,
    results_base: Path,
    route_prefix: str,
) -> Optional[str]:
    """
    Convert a filesystem path under `results_base` into a `/files/...` URL.

    Example:
      result_path=/cache/engine_results/abc/result.mp4
      results_base=/cache/engine_results
      route_prefix=/files/engine_results
      -> /files/engine_results/abc/result.mp4
    """
    try:
        relative_path = Path(result_path).relative_to(results_base)
        return f"{route_prefix}/{relative_path}"
    except Exception:
        return None

