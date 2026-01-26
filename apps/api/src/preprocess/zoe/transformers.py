"""
ZoeDepth implementation using HuggingFace transformers.
Uses official Intel models for depth estimation.
"""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline, AutoImageProcessor, ZoeDepthForDepthEstimation
from src.preprocess.util import resize_image_with_pad, HWC3
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device


class ZoeDetector(ToMixin, BasePreprocessor):
    """ZoeDepth depth estimation using HuggingFace transformers."""

    def __init__(self, model_name="Intel/zoedepth-nyu-kitti"):
        """Initialize ZoeDepth with specified model."""
        super().__init__()
        self.device = get_torch_device()
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
                super().__init__(*args, **kwargs)
                self._filename = kwargs.get("desc") or model_name

            def update(self, n=1):
                result = super().update(n)
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
                    tqdm_class=_ProgressTqdm,
                )
            except Exception:
                pass
            from transformers import pipeline

            self.pipe = pipeline(
                task="depth-estimation", model=model_name, device=self.device
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="Intel/zoedepth-nyu-kitti",
        filename=None,
        **kwargs,
    ):
        """Create ZoeDetector from pretrained model."""
        return cls(model_name=pretrained_model_or_path)

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
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        pil_image = Image.fromarray(input_image)

        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]

            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)

            vmin = np.percentile(depth_array, 2)
            vmax = np.percentile(depth_array, 85)

            depth_array = depth_array - vmin
            depth_array = depth_array / (vmax - vmin)
            depth_array = 1.0 - depth_array
            depth_image = (depth_array * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))

        detected_map = Image.fromarray(detected_map)

        return detected_map


class ZoeDepthAnythingDetector(ToMixin, BasePreprocessor):
    """ZoeDepthAnything implementation using HuggingFace transformers."""

    def __init__(self, model_name="Intel/zoedepth-nyu-kitti"):
        """Initialize ZoeDepthAnything detector."""
        super().__init__()
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
            import huggingface_hub.utils._tqdm as _hf_utils_tqdm_mod
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
                super().__init__(*args, **kwargs)
                self._filename = kwargs.get("desc") or model_name

            def update(self, n=1):
                result = super().update(n)
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
            from transformers import pipeline

            self.pipe = pipeline(task="depth-estimation", model=model_name)
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
        self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="Intel/zoedepth-nyu-kitti",
        filename=None,
        **kwargs,
    ):
        """Create from pretrained model."""
        return cls(model_name=pretrained_model_or_path)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        """Perform depth estimation."""
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        pil_image = Image.fromarray(input_image)

        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]

            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)

            vmin = np.percentile(depth_array, 2)
            vmax = np.percentile(depth_array, 85)

            depth_array = depth_array - vmin
            depth_array = depth_array / (vmax - vmin)
            depth_array = 1.0 - depth_array
            depth_image = (depth_array * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))

        detected_map = Image.fromarray(detected_map)

        return detected_map
