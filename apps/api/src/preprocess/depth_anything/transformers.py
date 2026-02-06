"""
Modern DepthAnything implementation using HuggingFace transformers.
Replaces legacy torch.hub.load DINOv2 backbone with transformers pipeline.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path

from src.preprocess.util import HWC3, resize_image_with_pad
from src.utils.defaults import get_torch_device
from src.mixins import ToMixin
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH


class DepthAnythingDetector(ToMixin, BasePreprocessor):
    """DepthAnything depth estimation using HuggingFace transformers."""

    def __init__(self, model_name="LiheYoung/depth-anything-large-hf", cache_dir=None):
        """Initialize DepthAnything with specified model."""
        super().__init__()
        if cache_dir is None:
            cache_dir = Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "depth_anything"

        # Scoped monkey-patch: route Hugging Face download progress through our callback
        from src.preprocess.util import DOWNLOAD_PROGRESS_CALLBACK
        import os as _os
        import tqdm as _tqdm_mod

        try:
            from tqdm import auto as _tqdm_auto
        except Exception:
            _tqdm_auto = None
        try:
            import huggingface_hub.file_download as _hf_fd
        except Exception:
            _hf_fd = None
        try:
            import huggingface_hub.utils.tqdm as _hf_utils_tqdm_mod
        except Exception:
            _hf_utils_tqdm_mod = None

        _orig_tqdm = getattr(_tqdm_mod, "tqdm", None)
        _orig_tqdm_auto = getattr(_tqdm_auto, "tqdm", None) if _tqdm_auto else None
        _orig_hf_fd_tqdm = getattr(_hf_fd, "tqdm", None) if _hf_fd else None
        _orig_hf_utils_tqdm = (
            getattr(_hf_utils_tqdm_mod, "tqdm", None) if _hf_utils_tqdm_mod else None
        )

        class _ProgressTqdm(_tqdm_mod.tqdm):
            def __init__(self, *args, **kwargs):
                # HF Hub passes `name=` to group progress bars; tqdm doesn't accept it.
                kwargs.pop("name", None)
                super().__init__(*args, **kwargs)
                self._filename = kwargs.get("desc") or model_name

            def update(self, n=1):
                result = super().update(n)
                print("update", self._filename, self.n, self.total)
                try:
                    cb = DOWNLOAD_PROGRESS_CALLBACK
                    if cb and self.total:
                        cb(self._filename, self.n, self.total)
                except Exception:
                    pass
                return result

        try:
            try:
                _tqdm_mod.tqdm = _ProgressTqdm
                if _tqdm_auto:
                    _tqdm_auto.tqdm = _ProgressTqdm
                if _hf_fd and _orig_hf_fd_tqdm is not None:
                    _hf_fd.tqdm = _ProgressTqdm
                if _hf_utils_tqdm_mod and _orig_hf_utils_tqdm is not None:
                    _hf_utils_tqdm_mod.tqdm = _ProgressTqdm
            except Exception:
                pass

            # Ensure HF progress bars are enabled during download
            _prev_disable = _os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
            # Prefetch model repo with custom tqdm
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(cache_dir),
                    tqdm_class=_ProgressTqdm,
                )
            except Exception:
                pass
            from transformers import pipeline

            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                model_kwargs={"cache_dir": str(cache_dir)},
                device=get_torch_device(),
            )
        finally:
            # Restore env var
            try:
                if _prev_disable is None:
                    _os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _prev_disable
            except Exception:
                pass
            # Restore tqdm references
            try:
                if _hf_utils_tqdm_mod and _orig_hf_utils_tqdm is not None:
                    _hf_utils_tqdm_mod.tqdm = _orig_hf_utils_tqdm
            except Exception:
                pass
            try:
                if _hf_fd and _orig_hf_fd_tqdm is not None:
                    _hf_fd.tqdm = _orig_hf_fd_tqdm
            except Exception:
                pass
            try:
                if _tqdm_auto and _orig_tqdm_auto is not None:
                    _tqdm_auto.tqdm = _orig_tqdm_auto
            except Exception:
                pass
            try:
                if _orig_tqdm is not None:
                    _tqdm_mod.tqdm = _orig_tqdm
            except Exception:
                pass

        self.device = get_torch_device()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=None, filename="depth_anything_vitl14.pth"
    ):
        """Create DepthAnything from pretrained model, mapping legacy names to HuggingFace models."""

        # Map legacy checkpoint names to modern HuggingFace models
        model_mapping = {
            "depth_anything_vitl14.pth": "LiheYoung/depth-anything-large-hf",
            "depth_anything_vitb14.pth": "LiheYoung/depth-anything-base-hf",
            "depth_anything_vits14.pth": "LiheYoung/depth-anything-small-hf",
        }

        model_name = model_mapping.get(filename, "LiheYoung/depth-anything-large-hf")
        cache_dir = Path(DEFAULT_PREPROCESSOR_SAVE_PATH) / "depth_anything"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cls(model_name=model_name, cache_dir=cache_dir)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        """Perform depth estimation on input image."""
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.asarray(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image

        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]

            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)

            # Normalize depth values to 0-255 range
            depth_min = depth_array.min()
            depth_max = depth_array.max()
            if depth_max > depth_min:
                depth_array = (
                    (depth_array - depth_min) / (depth_max - depth_min) * 255.0
                )
            else:
                depth_array = np.zeros_like(depth_array)

            depth_image = depth_array.astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        detected_map = Image.fromarray(detected_map)

        return detected_map
