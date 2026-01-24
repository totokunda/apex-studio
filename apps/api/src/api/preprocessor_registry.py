"""
Enhanced preprocessor registry with detailed parameter information
"""

from typing import Dict, Any, List
import os
from pathlib import Path
from functools import lru_cache
from src.utils.yaml import load_yaml as load_yaml_file

PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / "manifest" / "preprocessor"
PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / "manifest" / "preprocessor"


@lru_cache(maxsize=None)
def _available_preprocessor_names() -> List[str]:
    if not PREPROCESSOR_PATH.exists():
        return []
    names: List[str] = []
    for entry in PREPROCESSOR_PATH.iterdir():
        if (
            entry.is_file()
            and entry.suffix in {".yml", ".yaml"}
            and not entry.name.startswith("shared")
        ):
            names.append(entry.stem)
    return sorted(names)


@lru_cache(maxsize=None)
def _load_preprocessor_yaml(preprocessor_name: str) -> Dict[str, Any]:
    file_path_yml = PREPROCESSOR_PATH / f"{preprocessor_name}.yml"
    file_path_yaml = PREPROCESSOR_PATH / f"{preprocessor_name}.yaml"
    file_path = (
        file_path_yml
        if file_path_yml.exists()
        else (file_path_yaml if file_path_yaml.exists() else None)
    )
    if file_path is None:
        available = _available_preprocessor_names()
        raise ValueError(
            f"Preprocessor {preprocessor_name} not found. Available: {available}"
        )
    data = load_yaml_file(file_path)
    if not isinstance(data, dict):
        data = {}
    data.setdefault("name", preprocessor_name)
    data.setdefault("category", "")
    data.setdefault("description", "")
    data.setdefault("module", "")
    data.setdefault("class", "")
    data.setdefault("supports_video", True)
    data.setdefault("supports_image", True)
    data.setdefault("parameters", [])
    return data


detect_resolution_parameter = {
    "name": "detect_resolution",
    "display_name": "Detection Resolution",
    "type": "category",
    "default": 512,
    "options": [
        {"name": "Standard", "value": 512},
        {"name": "High Definition", "value": 1024},
        {"name": "Current Image", "value": 0},
    ],
    "description": "The resolution used for detection and inference. Higher resolutions provide more detail but require more processing time and memory.",
}

upscale_method_parameter = {
    "name": "upscale_method",
    "display_name": "Upscale Method",
    "type": "category",
    "default": "INTER_CUBIC",
    "options": [
        {"name": "Nearest Neighbor", "value": "INTER_NEAREST"},
        {"name": "Linear", "value": "INTER_LINEAR"},
        {"name": "Cubic", "value": "INTER_CUBIC"},
        {"name": "Lanczos", "value": "INTER_LANCZOS4"},
    ],
    "description": "The interpolation method used when resizing images. Bicubic and Lanczos provide smoother results, while Nearest Neighbor preserves sharp edges.",
}


def get_preprocessor_info(preprocessor_name: str) -> Dict[str, Any]:
    """
    Get preprocessor module and class info.

    Args:
        preprocessor_name: Name of the preprocessor

    Returns:
        Dictionary with preprocessor information
    """
    return _load_preprocessor_yaml(preprocessor_name)


def list_preprocessors(check_downloaded: bool = False) -> List[Dict[str, Any]]:
    """
    List all available preprocessors with their metadata.

    Args:
        check_downloaded: If True, check download status for each preprocessor (slower)

    Returns:
        List of preprocessor information dictionaries
    """
    result: List[Dict[str, Any]] = []
    for name in _available_preprocessor_names():
        info = _load_preprocessor_yaml(name)
        files = info.get("files", [])
        # Resolve absolute paths if files exist under DEFAULT_PREPROCESSOR_SAVE_PATH
        try:
            from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH

            base = Path(DEFAULT_PREPROCESSOR_SAVE_PATH)
            resolved_files: List[Dict[str, Any]] = []
            for f in files:
                rel_path = f.get("path", "")
                abs_path = base / rel_path
                if abs_path.exists():
                    # prefer absolute path when downloaded
                    resolved_files.append(
                        {"path": abs_path.__str__(), "size_bytes": f.get("size_bytes")}
                    )
                else:
                    resolved_files.append(f)
        except Exception:
            resolved_files = files
        preprocessor_info = {
            "id": name,
            "name": info.get("name", name),
            "category": info.get("category", ""),
            "description": info.get("description", ""),
            "supports_video": bool(info.get("supports_video", True)),
            "supports_image": bool(info.get("supports_image", True)),
            "parameters": info.get("parameters", []),
            "files": resolved_files,
        }
        if check_downloaded:
            preprocessor_info["is_downloaded"] = check_preprocessor_downloaded(name)
        result.append(preprocessor_info)
    return sorted(result, key=lambda x: x["name"])


def initialize_download_tracking():
    """
    Initialize the download tracking file with preprocessors that don't require downloads.
    This should be called on app startup.
    """
    from src.preprocess.base_preprocessor import BasePreprocessor

    # Preprocessors that don't require downloads
    NO_DOWNLOAD_REQUIRED = [
        "binary",
        "canny",
        "color",
        "pyracanny",
        "recolor",
        "scribble",
        "scribble_xdog",
        "shuffle",
        "tile",
        "tile_gf",
        "tile_simple",
    ]

    for preprocessor_name in NO_DOWNLOAD_REQUIRED:
        BasePreprocessor._mark_as_downloaded(preprocessor_name)


def check_preprocessor_downloaded(preprocessor_name: str) -> bool:
    """
    Check if a preprocessor's model files are downloaded.

    Args:
        preprocessor_name: Name of the preprocessor

    Returns:
        True if downloaded/ready, False otherwise
    """
    # Preprocessors that don't require downloads
    NO_DOWNLOAD_REQUIRED = {
        "binary",
        "canny",
        "color",
        "pyracanny",
        "recolor",
        "scribble",
        "scribble_xdog",
        "shuffle",
        "tile",
        "tile_gf",
        "tile_simple",
    }

    if preprocessor_name in NO_DOWNLOAD_REQUIRED:
        return True

    # Check the downloaded preprocessors tracking file
    from src.preprocess.base_preprocessor import BasePreprocessor

    return BasePreprocessor._is_downloaded(preprocessor_name)


def get_preprocessor_details(preprocessor_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific preprocessor.

    Args:
        preprocessor_name: Name of the preprocessor

    Returns:
        Dictionary with detailed preprocessor information including parameters
    """
    info = _load_preprocessor_yaml(preprocessor_name)
    is_downloaded = check_preprocessor_downloaded(preprocessor_name)
    files = info.get("files", [])
    # Resolve absolute paths if files exist under DEFAULT_PREPROCESSOR_SAVE_PATH
    try:
        from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH

        base = Path(DEFAULT_PREPROCESSOR_SAVE_PATH)
        resolved_files: List[Dict[str, Any]] = []
        for f in files:
            rel_path = f.get("path", "")
            abs_path = base / rel_path
            if abs_path.exists():
                # prefer absolute path when downloaded
                resolved_files.append(
                    {
                        "path": abs_path.__str__(),
                        "size_bytes": f.get("size_bytes"),
                        "name": f.get("name"),
                    }
                )
            else:
                resolved_files.append(f)
    except Exception:
        resolved_files = files

    return {
        "id": preprocessor_name,
        "name": info.get("name", preprocessor_name),
        "category": info.get("category", ""),
        "description": info.get("description", ""),
        "module": info.get("module", ""),
        "class": info.get("class", ""),
        "supports_video": bool(info.get("supports_video", True)),
        "supports_image": bool(info.get("supports_image", True)),
        "parameters": info.get("parameters", []),
        "is_downloaded": is_downloaded,
        "files": resolved_files,
    }
