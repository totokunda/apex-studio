# src/postprocessors/rife_postprocessor.py

from __future__ import annotations

import os
import sys
import math
import zipfile
import shutil
from typing import Any, List, Optional, Tuple, Union, Callable

import numpy as np
from src.types import InputVideo
import torch
from torch.nn import functional as F
from loguru import logger
from src.utils.defaults import DEFAULT_POSTPROCESSOR_SAVE_PATH, DEFAULT_DEVICE

from src.postprocess.base import (
    BasePostprocessor,
    PostprocessorCategory,
    postprocessor_registry,
)
from PIL import Image
from collections import deque
from tqdm import tqdm
from src.utils.defaults import get_torch_device


def _load_rife_model(model_dir: str, device: torch.device, logger=None):
    """
    Import and load the RIFE model dynamically, matching the fallback chain used in the RIFE scripts.
    """
    # Ensure the parent directory of train_log is on sys.path so we can import like inference script
    try:
        train_log_dir = os.path.abspath(model_dir)
        parent_dir = os.path.dirname(train_log_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        # Import Model exactly as in Practical-RIFE inference script
        from train_log.RIFE_HDv3 import Model  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import RIFE Model from {model_dir}: {e}")

    model = Model()
    if not hasattr(model, "version"):
        model.version = 0
    # Mirror inference script load behavior (use -1 to select latest)
    try:
        model.load_model(model_dir, -1)
    except TypeError:
        # Older implementations may not accept the second argument
        model.load_model(model_dir)
    model.eval()
    model.flownet.to(device)
    if logger:
        logger.info(f"Loaded RIFE model from: {model_dir}")
    return model


@postprocessor_registry("video.rife")
class RifePostprocessor(BasePostprocessor):
    """
    RIFE frame interpolation postprocessor.

    Usage:
        pp = RifePostprocessor(engine, target_fps=60)  # or exp=1 for 2x
        video_out = pp(video_in)  # returns tensor (1, C, F_out, H, W) in [-1, 1]
    """

    def __init__(
        self,
        engine=None,
        # Targeting / control
        target_fps: Optional[float] = None,
        exp: Optional[int] = None,  # if set, overrides target_fps
        scale: float = 1.0,  # RIFE's 'scale' (try 0.5 for UHD)
        # SSIM gating (optional, mirrors RIFE script behavior)
        ssim_static_thresh: float = 0.996,  # treat near-identical frames as static
        ssim_hardcut_thresh: float = 0.20,  # when < thresh, fill with duplicates
        # Weights / code locations
        device: torch.device = DEFAULT_DEVICE,
        save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
        model_dir: str = "https://drive.google.com/uc?id=1zlKblGuKNatulJNFf5jdB-emp9AqGK05",
        **kwargs: Any,
    ):
        super().__init__(engine, PostprocessorCategory.FRAME_INTERPOLATION, **kwargs)
        self.save_path = save_path
        self.scale = scale
        self.target_fps = target_fps
        self.exp = exp
        self.ssim_static_thresh = ssim_static_thresh
        self.ssim_hardcut_thresh = ssim_hardcut_thresh
        model_dir = self.download_rife(model_dir, save_path=save_path)

        # Build model
        self.device = device
        self.model = _load_rife_model(model_dir, self.device, logger=logger)

    def download_rife(self, model_dir: str, save_path: str):
        if self._is_url(model_dir):
            # check if the save_path exists
            save_rife_path = os.path.join(save_path, "rife")
            if os.path.exists(os.path.join(save_rife_path, "train_log")):
                # Ensure import path for train_log
                if save_rife_path not in sys.path:
                    sys.path.insert(0, save_rife_path)
                # Ensure Practical-RIFE 'model' folder is available alongside train_log
                try:
                    src_model_dir = os.path.join(os.path.dirname(__file__), "model")
                    dst_model_dir = os.path.join(save_rife_path, "model")
                    if os.path.isdir(src_model_dir) and not os.path.exists(
                        dst_model_dir
                    ):
                        shutil.copytree(src_model_dir, dst_model_dir)
                except Exception:
                    pass
                # Ensure engine root (parent of 'src') is available for 'src.utils.*' imports
                try:
                    engine_root = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "..", "..")
                    )
                    if engine_root not in sys.path:
                        sys.path.insert(0, engine_root)
                except Exception:
                    pass
                return os.path.join(save_rife_path, "train_log")
            os.makedirs(save_rife_path, exist_ok=True)
            path = self._download_from_url(model_dir, save_path=save_path)
            # Extract full contents so Python modules under train_log are available
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(save_rife_path)
            # Remove the downloaded archive to save space
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
            # Copy bundled 'model' folder next to extracted train_log to satisfy imports
            try:
                src_model_dir = os.path.join(os.path.dirname(__file__), "model")
                dst_model_dir = os.path.join(save_rife_path, "model")
                if os.path.isdir(src_model_dir) and not os.path.exists(dst_model_dir):
                    shutil.copytree(src_model_dir, dst_model_dir)
            except Exception:
                pass
            # Ensure import path points to directory that contains train_log/
            if save_rife_path not in sys.path:
                sys.path.insert(0, save_rife_path)
            # Ensure engine root (parent of 'src') is available for 'src.utils.*' imports
            try:
                engine_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..")
                )
                if engine_root not in sys.path:
                    sys.path.insert(0, engine_root)
            except Exception:
                pass
            return os.path.join(save_rife_path, "train_log")
        else:
            # Local path: if it is the train_log directory, add its parent to sys.path
            abs_path = os.path.abspath(model_dir)
            parent_dir = os.path.dirname(abs_path)
            if os.path.basename(abs_path) == "train_log":
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                # Also ensure our local 'model' folder parent is on sys.path for 'model.*' imports
                rife_dir = os.path.dirname(__file__)
                if rife_dir not in sys.path:
                    sys.path.insert(0, rife_dir)
                # Ensure engine root (parent of 'src') is available for 'src.utils.*' imports
                try:
                    engine_root = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "..", "..")
                    )
                    if engine_root not in sys.path:
                        sys.path.insert(0, engine_root)
                except Exception:
                    pass
            else:
                # If a parent contains train_log, prefer adding that parent
                tl_candidate = os.path.join(abs_path, "train_log")
                if os.path.isdir(tl_candidate) and abs_path not in sys.path:
                    sys.path.insert(0, abs_path)
                    # Ensure engine root (parent of 'src') is available for 'src.utils.*' imports
                    try:
                        engine_root = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "..", "..", "..")
                        )
                        if engine_root not in sys.path:
                            sys.path.insert(0, engine_root)
                    except Exception:
                        pass
                    return tl_candidate
                # Fallback: expose local 'model' folder parent to sys.path
                rife_dir = os.path.dirname(__file__)
                if rife_dir not in sys.path:
                    sys.path.insert(0, rife_dir)
                # Ensure engine root (parent of 'src') is available for 'src.utils.*' imports
                try:
                    engine_root = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "..", "..")
                    )
                    if engine_root not in sys.path:
                        sys.path.insert(0, engine_root)
                except Exception:
                    pass
            return model_dir

    @torch.no_grad()
    def __call__(self, video: InputVideo, **kwargs) -> List[Image.Image]:
        """
        RIFE interpolation that mirrors the provided reference script:
          - multi expansion (default 2, or 2**exp when exp != 1)
          - padding to multiples of max(128, int(128/scale))
          - version-aware inference: v>=3.9 uses t ∈ (0,1), else recursive midpoint
          - SSIM gates for static frames (>0.996) and hard cuts (<0.2)

        Returns:
            (1, C, F_out, H, W) in [-1, 1]
        """

        # ---- 1) Load frames & original FPS via LoaderMixin ----
        frames, orig_fps = self._load_video(video, return_fps=True)
        if not frames:
            return torch.empty(1, 3, 0, 0, 0)

        self.target_fps = kwargs.get("target_fps", self.target_fps)
        self.exp = kwargs.get("exp", self.exp)
        self.scale = kwargs.get("scale", self.scale)
        self.ssim_static_thresh = kwargs.get(
            "ssim_static_thresh", self.ssim_static_thresh
        )
        self.ssim_hardcut_thresh = kwargs.get(
            "ssim_hardcut_thresh", self.ssim_hardcut_thresh
        )

        # ---- 2) Determine multi (like argparse logic) ----
        # Priority: explicit kwargs.multi -> exp (2**exp if exp != 1) -> target_fps vs orig_fps -> default 2
        multi = (
            int(kwargs.get("multi", 0)) if kwargs.get("multi", None) is not None else 0
        )
        if multi < 2:
            if self.exp is not None:
                multi = (2 ** int(self.exp)) if int(self.exp) != 1 else 2
            elif self.target_fps and orig_fps and orig_fps > 0:
                # Allow any integer multi (script multiplies fps by multi)
                multi = max(2, int(round(float(self.target_fps) / float(orig_fps))))
            else:
                multi = 2
        n_inserts = multi - 1

        # Optional progress callback (idx, total, message)
        progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = (
            kwargs.get("progress_callback")
        )

        # ---- 3) Prepare numpy frames like inference script ----
        # Ensure we use the model's actual device for all tensors

        dev = get_torch_device()
        frames_np: List[np.ndarray] = []
        for im in frames:
            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 2)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            frames_np.append(arr)

        tot_frame = len(frames_np)
        h, w, _ = frames_np[0].shape

        # ---- 4) Padding logic (like script) ----
        tmp = max(128, int(128 / self.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)

        def pad_image(img: torch.Tensor) -> torch.Tensor:
            return F.pad(img, padding)

        # ---- 5) Version-aware make_inference (exact to script) ----
        version = float(getattr(self.model, "version", 0))
        if not hasattr(self.model, "version"):
            setattr(self.model, "version", version)

        def make_inference(I0: torch.Tensor, I1: torch.Tensor, n: int):
            if n <= 0:
                return []
            if version >= 3.9:
                res = []
                for i in range(n):
                    res.append(
                        self.model.inference(
                            I0, I1, (i + 1) * 1.0 / (n + 1), self.scale
                        )
                    )
                return res
            else:
                middle = self.model.inference(I0, I1, self.scale)
                if n == 1:
                    return [middle]
                first_half = make_inference(I0, middle, n // 2)
                second_half = make_inference(middle, I1, n // 2)
                if n % 2:
                    return [*first_half, middle, *second_half]
                else:
                    return [*first_half, *second_half]

        # ---- 6) Main while loop replicated from inference_video.py ----
        from src.postprocess.rife.ssim import ssim_matlab as _ssim

        # progress setup
        total_steps = max(0, tot_frame - 1)
        processed_steps = 0
        if progress_callback and total_steps > 0:
            try:
                progress_callback(0, total_steps, "Starting interpolation")
            except Exception:
                pass

        pbar = tqdm(total=tot_frame, desc="RIFE Interpolation")
        lastframe = frames_np[0]
        videogen = frames_np[1:]
        write_out: List[np.ndarray] = []

        I1 = (
            torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
            .to(dev, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1)
        temp = None
        idx = 0

        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = videogen[idx] if idx < len(videogen) else None
                idx += 1 if frame is not None else 0
            if frame is None:
                break
            I0 = I1
            I1 = (
                torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                .to(dev, non_blocking=True)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            I1 = pad_image(I1)
            I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = float(_ssim(I0_small[:, :3], I1_small[:, :3]))

            break_flag = False
            if ssim > 0.996:
                # read a new frame
                next_frame = videogen[idx] if idx < len(videogen) else None
                idx += 1 if next_frame is not None else 0
                if next_frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = next_frame
                I1 = (
                    torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                    .to(dev, non_blocking=True)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                I1 = pad_image(I1)
                I1 = self.model.inference(I0, I1, scale=self.scale)
                I1_small = F.interpolate(
                    I1, (32, 32), mode="bilinear", align_corners=False
                )
                ssim = float(_ssim(I0_small[:, :3], I1_small[:, :3]))
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

            if ssim < 0.2:
                output = []
                for _ in range(multi - 1):
                    output.append(I0)
            else:
                output = make_inference(I0, I1, multi - 1)

            # write frames (non-montage path)
            write_out.append(lastframe)
            for mid in output:
                mid_np = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
                write_out.append(mid_np[:h, :w])
            pbar.update(1)
            processed_steps += 1
            if progress_callback and total_steps > 0:
                try:
                    progress_callback(processed_steps, total_steps, "Interpolating")
                except Exception:
                    pass
            lastframe = frame
            if break_flag:
                break

        # finalize
        write_out.append(lastframe)
        pbar.close()
        if progress_callback and total_steps > 0:
            try:
                progress_callback(total_steps, total_steps, "Finalizing")
            except Exception:
                pass

        # Convert to PIL Images
        out_images = [Image.fromarray(f.astype(np.uint8)) for f in write_out]
        return out_images
