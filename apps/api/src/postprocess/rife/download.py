from __future__ import annotations

import os
import shutil
import sys
import zipfile
from typing import Callable, Optional

from src.utils.defaults import DEFAULT_POSTPROCESSOR_SAVE_PATH
from src.mixins.download_mixin import DownloadMixin


class _RifeDownloader(DownloadMixin):
    """
    Minimal helper to reuse DownloadMixin's URL handling and download implementation.
    Intentionally does NOT import torch or any model code.
    """

    pass


def download_rife_assets(
    *,
    save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
    model_url: str = "https://drive.google.com/uc?id=1zlKblGuKNatulJNFf5jdB-emp9AqGK05",
    progress_callback: Optional[
        Callable[[int, Optional[int], Optional[str]], None]
    ] = None,
) -> str:
    """
    Download and extract the RIFE assets into `${save_path}/rife/`.

    Returns the local path to the extracted `train_log` directory.

    This is designed for setup/install flows and avoids importing torch.
    """
    dl = _RifeDownloader()
    save_rife_path = os.path.join(save_path, "rife")

    # Already installed
    if os.path.exists(os.path.join(save_rife_path, "train_log")):
        # Ensure import path for train_log
        if save_rife_path not in sys.path:
            sys.path.insert(0, save_rife_path)
        _ensure_bundled_model_folder(save_rife_path)
        _ensure_engine_root_on_syspath()
        return os.path.join(save_rife_path, "train_log")

    os.makedirs(save_rife_path, exist_ok=True)

    if not dl._is_url(model_url):
        # Local path: mimic rife.py behavior (accept train_log directory or parent)
        abs_path = os.path.abspath(model_url)
        parent_dir = os.path.dirname(abs_path)
        if os.path.basename(abs_path) == "train_log":
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            # Also ensure our local 'model' folder parent is on sys.path for 'model.*' imports
            rife_dir = os.path.dirname(__file__)
            if rife_dir not in sys.path:
                sys.path.insert(0, rife_dir)
            _ensure_engine_root_on_syspath()
            return abs_path

        # Otherwise treat as a directory containing train_log
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
        _ensure_engine_root_on_syspath()
        return os.path.join(abs_path, "train_log")

    # Download archive to save_path (DownloadMixin manages naming)
    archive_path = dl._download_from_url(
        model_url, save_path=save_path, progress_callback=progress_callback
    )

    # Extract full contents so Python modules under train_log are available
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(save_rife_path)

    # Remove the downloaded archive to save space
    try:
        if os.path.isfile(archive_path):
            os.remove(archive_path)
    except Exception:
        pass

    _ensure_bundled_model_folder(save_rife_path)

    # Ensure import path points to directory that contains train_log/
    if save_rife_path not in sys.path:
        sys.path.insert(0, save_rife_path)

    _ensure_engine_root_on_syspath()

    return os.path.join(save_rife_path, "train_log")


def _ensure_bundled_model_folder(save_rife_path: str) -> None:
    """
    Copy bundled `model/` folder next to extracted train_log to satisfy imports.
    """
    try:
        src_model_dir = os.path.join(os.path.dirname(__file__), "model")
        dst_model_dir = os.path.join(save_rife_path, "model")
        if os.path.isdir(src_model_dir) and not os.path.exists(dst_model_dir):
            shutil.copytree(src_model_dir, dst_model_dir)
    except Exception:
        pass


def _ensure_engine_root_on_syspath() -> None:
    """
    Ensure engine root (parent of 'src') is available for 'src.*' imports.
    """
    try:
        engine_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        if engine_root not in sys.path:
            sys.path.insert(0, engine_root)
    except Exception:
        pass
