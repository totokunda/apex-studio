"""
Hello, welcome on board,
"""

from __future__ import print_function

import os
import cv2
import numpy as np

import torch

from .ted import TED  # TEED architecture
from einops import rearrange
from src.preprocess.util import (
    safe_step,
    custom_hf_download,
    BDS_MODEL_NAME,
    resize_image_with_pad,
    HWC3,
)
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from PIL import Image


class TEDDetector(ToMixin, BasePreprocessor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=BDS_MODEL_NAME,
        filename="7_model.pth",
        subfolder="Annotators",
    ):
        model_path = custom_hf_download(
            pretrained_model_or_path, filename, subfolder=subfolder
        )
        model = TED()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        model.to("cpu")  # Explicitly move to CPU
        return cls(model)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        safe_steps=2,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        H, W, _ = input_image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(input_image.copy()).float().to(self.device)
            image_teed = rearrange(image_teed, "h w c -> 1 c h w")
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(edge))
        detected_map = Image.fromarray(detected_map[..., :3])

        return detected_map
