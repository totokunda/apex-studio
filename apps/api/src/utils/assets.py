from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def candidate_asset_roots() -> list[Path]:
    """
    Return a list of candidate directories that may contain the engine `assets/` folder.

    This supports both:
    - **dev checkout**: `apps/api/assets/...`
    - **installed app bundle**: `sys.prefix/../assets/...` (assets next to embedded python prefix)
    """
    roots: list[Path] = []

    env_assets = os.environ.get("APEX_ASSETS_DIR") or os.environ.get("APEX_ENGINE_ASSETS_DIR")
    if env_assets:
        roots.append(Path(env_assets))

    # Dev checkout layout: apps/api/src/utils/assets.py -> parents[2] == apps/api/src, parents[3] == apps/api
    try:
        here = Path(__file__).resolve()
        roots.append(here.parents[3] / "assets")
    except Exception:
        pass

    # Installed bundle layout: assets live next to the embedded python prefix directory.
    try:
        prefix = Path(sys.prefix).resolve()
        roots.append(prefix / "assets")
        roots.append(prefix.parent / "assets")
    except Exception:
        pass

    # Last-resort: cwd-based lookup (useful in some dev/CI invocations)
    try:
        cwd = Path.cwd().resolve()
        roots.append(cwd / "assets")
        roots.append(cwd.parent / "assets")
    except Exception:
        pass

    return _dedupe_paths(roots)


def get_asset_path(*relative_parts: str, must_exist: bool = True) -> str:
    """
    Resolve a path under the engine `assets/` directory.

    Example:
        get_asset_path("magi", "special_tokens.npz")
    """
    rel = Path(*relative_parts)
    searched: list[str] = []
    for root in candidate_asset_roots():
        p = root / rel
        searched.append(str(p))
        if p.exists():
            return str(p)
    if must_exist:
        raise FileNotFoundError(
            f"Asset not found: {rel}. Searched: " + ", ".join(searched[:8])
        )
    # Best-effort fallback (may not exist)
    return str((candidate_asset_roots()[0] / rel) if candidate_asset_roots() else rel)


