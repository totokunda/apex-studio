"""
Config Registry - Resolves config_id references to local config file paths.

Instead of downloading config files from HuggingFace at runtime, model configs
are stored locally in the ``configs/`` directory and referenced by a short
``config_id`` in manifest YAML files.

Convention
----------
* ``config_id`` format: ``ModelFamily/component``
    - Example: ``Wan2.2-I2V/vae``
* Resolved local path: ``<CONFIGS_DIR>/ModelFamily/component/<filename>``
    - Example: ``configs/Wan2.2-I2V/vae/config.json``

The registry auto-discovers config files in the ``configs/`` directory, so
adding a new model config is as simple as placing the file in the right
directory and referencing the ``config_id`` in the manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Base directory for bundled configs, relative to this module.
# Layout: configs/<config_id>/<filename>
# ---------------------------------------------------------------------------
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def get_configs_dir() -> Path:
    """Return the absolute path to the configs directory."""
    return _CONFIGS_DIR


def resolve_config_path(config_id: str, filename: str = "config.json") -> str:
    """Resolve a *config_id* to the absolute path of the local config file.

    Parameters
    ----------
    config_id:
        Short identifier, e.g. ``"Wan2.2-I2V/vae"`` or
        ``"apple/DFN5B-CLIP-ViT-H-14-384"``.
    filename:
        Name of the config file inside the config_id directory.
        Defaults to ``"config.json"``.

    Returns
    -------
    str
        Absolute filesystem path to the config file.

    Raises
    ------
    FileNotFoundError
        If no config file is found for the given *config_id*.
    """
    config_dir = _CONFIGS_DIR / config_id

    # First try the explicit filename
    candidate = config_dir / filename
    if candidate.is_file():
        return str(candidate)

    # Auto-discover: look for any JSON/YAML config file in the directory
    if config_dir.is_dir():
        for ext in ("*.json", "*.yml", "*.yaml"):
            files = list(config_dir.glob(ext))
            if files:
                return str(files[0])

    raise FileNotFoundError(
        f"Config not found for config_id='{config_id}' "
        f"(looked in {config_dir})"
    )


def resolve_config_dir(config_id: str) -> str:
    """Resolve a *config_id* to the directory path (for preprocessor dirs etc).

    Parameters
    ----------
    config_id:
        Short identifier pointing to a directory of config files,
        e.g. ``"Wan2.1-I2V-480P/image_processor"``.

    Returns
    -------
    str
        Absolute filesystem path to the config directory.

    Raises
    ------
    FileNotFoundError
        If no directory is found for the given *config_id*.
    """
    config_dir = _CONFIGS_DIR / config_id
    if config_dir.is_dir():
        return str(config_dir)

    raise FileNotFoundError(
        f"Config directory not found for config_id='{config_id}' "
        f"(looked for {config_dir})"
    )


def load_config(config_id: str, filename: str = "config.json") -> Dict[str, Any]:
    """Load and return the parsed config dict for a *config_id*.

    Parameters
    ----------
    config_id:
        Short identifier, e.g. ``"Wan2.2-I2V/vae"``.
    filename:
        Name of the config file. Defaults to ``"config.json"``.

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    path = resolve_config_path(config_id, filename)

    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith((".yml", ".yaml")):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Try JSON first, then YAML
        with open(path, "r") as f:
            content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return yaml.safe_load(content)


def config_exists(config_id: str, filename: str = "config.json") -> bool:
    """Check whether a config exists for the given *config_id*."""
    try:
        resolve_config_path(config_id, filename)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# HF-path ↔ config_id conversion helpers (used by migration tooling)
# ---------------------------------------------------------------------------

_TOTOKU_PREFIX = "totoku/apex-models/"


def hf_path_to_config_id(hf_path: str) -> str:
    """Convert a HuggingFace model path to a config_id.

    Examples
    --------
    >>> hf_path_to_config_id("totoku/apex-models/Wan2.2-I2V/vae/config.json")
    'Wan2.2-I2V/vae'
    >>> hf_path_to_config_id("apple/DFN5B-CLIP-ViT-H-14-384/config.json")
    'apple/DFN5B-CLIP-ViT-H-14-384'
    >>> hf_path_to_config_id("totoku/apex-models/Wan2.1-I2V-480P/image_processor")
    'Wan2.1-I2V-480P/image_processor'
    """
    from urllib.parse import urlparse

    # Handle direct URLs
    if hf_path.startswith("http"):
        parsed = urlparse(hf_path)
        parts = parsed.path.strip("/").split("/")
        # Expected: [namespace, repo, "resolve", "main", subpath..., filename]
        namespace = parts[0]
        repo = parts[1]
        subpath_parts = parts[4:]  # skip "resolve" and branch
        if subpath_parts and subpath_parts[-1].endswith(".json"):
            subpath_parts = subpath_parts[:-1]
        return f"{namespace}/{repo}/{'/'.join(subpath_parts)}"

    # Handle totoku/apex-models paths
    if hf_path.startswith(_TOTOKU_PREFIX):
        remainder = hf_path[len(_TOTOKU_PREFIX):]
        if remainder.endswith(".json"):
            remainder = str(Path(remainder).parent)
        return remainder

    # Handle other HF paths (apple/..., nvidia/...)
    if "/" in hf_path:
        if hf_path.endswith(".json"):
            return str(Path(hf_path).parent)
        return hf_path

    return hf_path
