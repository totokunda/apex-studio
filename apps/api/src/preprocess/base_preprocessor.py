"""
Base preprocessor class with websocket support for progress updates
"""

import uuid
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict
from pathlib import Path
import json
import numpy as np
from src.mixins import LoaderMixin
from src.types import (
    InputImage,
    InputVideo,
    OutputImage,
    OutputVideo,
    InputMedia,
    OutputMedia,
)
from PIL import Image
from tqdm import tqdm
from src.utils.defaults import DEFAULT_CACHE_PATH, DEFAULT_PREPROCESSOR_SAVE_PATH
import os
import threading
import torch

# Thread-safe lock for registry updates
_registry_lock = threading.Lock()

# Path to downloaded preprocessors tracking file
DOWNLOADED_PREPROCESSORS_FILE = (
    Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "downloaded_preprocessors.json"
)


class ProgressCallback:
    """Wrapper for websocket progress updates"""

    def __init__(self, job_id: str, websocket_manager: Optional[Any] = None):
        self.job_id = job_id
        self.websocket_manager = websocket_manager

    async def update(
        self, progress: float, message: str, metadata: Optional[Dict] = None
    ):
        """Send progress update through websocket"""
        if self.websocket_manager:
            await self.websocket_manager.send_update(
                self.job_id,
                {"progress": progress, "message": message, "metadata": metadata or {}},
            )


class BasePreprocessor(LoaderMixin, ABC):
    """
    Base class for auxiliary preprocessors with websocket support
    """

    def __init__(self):
        self.cache_path = Path(DEFAULT_CACHE_PATH) / "preprocessor_results"
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _mark_as_downloaded(cls, preprocessor_name: str):
        """Mark a preprocessor as downloaded in the tracking file"""
        with _registry_lock:
            try:
                DOWNLOADED_PREPROCESSORS_FILE.parent.mkdir(parents=True, exist_ok=True)

                # Read existing data
                if DOWNLOADED_PREPROCESSORS_FILE.exists():
                    with open(DOWNLOADED_PREPROCESSORS_FILE, "r") as f:
                        downloaded = json.load(f)
                else:
                    downloaded = {}

                # Update with current preprocessor
                downloaded[preprocessor_name] = {
                    "downloaded": True,
                    "timestamp": str(uuid.uuid4()),  # Simple timestamp placeholder
                }

                # Write back
                with open(DOWNLOADED_PREPROCESSORS_FILE, "w") as f:
                    json.dump(downloaded, f, indent=2)

            except Exception as e:
                # Don't fail if we can't write to tracking file
                pass

    @classmethod
    def _is_downloaded(cls, preprocessor_name: str) -> bool:
        """Check if a preprocessor is marked as downloaded"""
        try:
            if not DOWNLOADED_PREPROCESSORS_FILE.exists():
                return False

            with open(DOWNLOADED_PREPROCESSORS_FILE, "r") as f:
                downloaded = json.load(f)

            return downloaded.get(preprocessor_name, {}).get("downloaded", False)
        except Exception:
            return False

    @classmethod
    def _unmark_as_downloaded(cls, preprocessor_name: str):
        """Unmark a preprocessor as downloaded in the tracking file."""
        with _registry_lock:
            try:
                if not DOWNLOADED_PREPROCESSORS_FILE.exists():
                    return
                with open(DOWNLOADED_PREPROCESSORS_FILE, "r") as f:
                    downloaded = json.load(f)
                if preprocessor_name in downloaded:
                    # Either remove or set flag to False
                    downloaded[preprocessor_name]["downloaded"] = False
                with open(DOWNLOADED_PREPROCESSORS_FILE, "w") as f:
                    json.dump(downloaded, f, indent=2)
            except Exception:
                # Best-effort only
                pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the preprocessor model"""
        pass

    def process_image(self, input_image: InputImage, **kwargs) -> OutputImage:
        target_size = self._get_image_size(input_image)
        processed = self.process(input_image, **kwargs)
        if kwargs.get("ensure_image_size", True):
            return self._ensure_image_size(processed, target_size)
        else:
            return processed

    def process_video(self, input_video: InputVideo, **kwargs) -> OutputVideo:
        """Process video frames iteratively, yielding results to avoid memory overload"""
        # Check if input is already an iterator/generator
        if hasattr(input_video, "__iter__") and not isinstance(
            input_video, (list, str)
        ):
            frames = input_video
            total_frames = kwargs.get("total_frames", None)
        else:
            frames = self._load_video(input_video)
            total_frames = len(frames)

        progress_callback = kwargs.get("progress_callback", None)
        frame_idx = 0

        for frame in tqdm(frames, desc="Processing frames", total=total_frames):
            if progress_callback is not None:
                progress_callback(
                    frame_idx + 1, total_frames if total_frames else frame_idx + 1
                )

            target_size = (
                frame.size
                if isinstance(frame, Image.Image)
                else self._get_image_size(frame)
            )
            anno_frame = self.process(frame, **kwargs)
            if kwargs.get("ensure_image_size", True):
                anno_frame = self._ensure_image_size(anno_frame, target_size)
            yield anno_frame
            frame_idx += 1

        # Send final frame completion
        if progress_callback is not None:
            progress_callback(frame_idx, frame_idx)

    def __call__(self, input_media: InputMedia, **kwargs) -> OutputMedia:
        if (
            isinstance(input_media, list)
            or hasattr(input_media, "__iter__")
            or (
                isinstance(input_media, str)
                and self.get_media_type(input_media) == "video"
            )
        ):
            return self.process_video(input_media, **kwargs)
        elif isinstance(input_media, Image.Image) or (
            isinstance(input_media, str) and self.get_media_type(input_media) == "image"
        ):
            return self.process_image(input_media, **kwargs)
        else:
            raise ValueError(f"Invalid media type: {type(input_media)}")

    @abstractmethod
    def process(self, input_media: InputMedia, **kwargs) -> OutputMedia:
        """Process the media"""
        pass

    def _get_image_size(self, img: InputImage) -> tuple[int, int]:
        if isinstance(img, Image.Image):
            return img.size
        if isinstance(img, str):
            try:
                with Image.open(img) as im:
                    return im.size
            except Exception:
                raise ValueError(f"Cannot determine size for image path: {img}")
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return (img.shape[1], img.shape[0])
            if img.ndim == 3:
                # Assume HWC for numpy arrays by default
                return (img.shape[1], img.shape[0])
        if isinstance(img, torch.Tensor):
            tensor = img.detach().cpu()
            if tensor.ndim == 2:
                return (int(tensor.shape[1]), int(tensor.shape[0]))
            if tensor.ndim == 3:
                # Heuristic: CHW if first dim is channels (1/3/4) and others are larger
                if (
                    int(tensor.shape[0]) in (1, 3, 4)
                    and int(tensor.shape[1]) > 4
                    and int(tensor.shape[2]) > 4
                ):
                    return (int(tensor.shape[2]), int(tensor.shape[1]))
                else:
                    # Assume HWC
                    return (int(tensor.shape[1]), int(tensor.shape[0]))
        raise ValueError(f"Unsupported image type for size inference: {type(img)}")

    def _ensure_image_size(
        self, output_img: OutputImage, target_size: tuple[int, int]
    ) -> OutputImage:
        # Normalize to PIL Image first
        pil_img: Image.Image
        if isinstance(output_img, Image.Image):
            pil_img = output_img
        elif isinstance(output_img, np.ndarray):
            arr = output_img
            if arr.dtype != np.uint8:
                # Many preprocessors output float in [0,1]
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                pil_img = Image.fromarray(arr)
            elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                if arr.shape[2] == 1:
                    pil_img = Image.fromarray(arr.squeeze(2))
                else:
                    pil_img = Image.fromarray(arr)
            else:
                # Fallback: try to interpret as RGB
                pil_img = Image.fromarray(arr)
        elif isinstance(output_img, torch.Tensor):
            tensor = output_img.detach().cpu()
            if tensor.ndim == 3 and int(tensor.shape[0]) in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            np_img = tensor.numpy()
            if np_img.dtype != np.uint8:
                if np_img.mean() <= 1.0:
                    np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    np_img = np_img.astype(np.uint8)
            if np_img.ndim == 2:
                pil_img = Image.fromarray(np_img)
            else:
                if np_img.shape[2] == 1:
                    pil_img = Image.fromarray(np_img.squeeze(2))
                else:
                    pil_img = Image.fromarray(np_img)
        else:
            # Unknown type, return as-is
            return output_img

        if pil_img.size != target_size:
            # Choose resampling: use NEAREST for single-channel maps, BILINEAR otherwise
            mode = pil_img.mode
            resample = Image.NEAREST if mode in ("1", "P", "L") else Image.BILINEAR
            pil_img = pil_img.resize(target_size, resample=resample)
        return pil_img
