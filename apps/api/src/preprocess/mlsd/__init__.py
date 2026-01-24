import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from src.preprocess.util import (
    HWC3,
    resize_image_with_pad,
    custom_hf_download,
    HF_MODEL_NAME,
)
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines


class MLSDdetector(ToMixin, BasePreprocessor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=HF_MODEL_NAME, filename="mlsd_large_512_fp32.pth"
    ):
        subfolder = (
            "annotator/ckpts"
            if pretrained_model_or_path == "lllyasviel/ControlNet"
            else ""
        )
        model_path = custom_hf_download(
            pretrained_model_or_path, filename, subfolder=subfolder
        )
        model = MobileV2_MLSD_Large()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False), strict=True
        )
        model.eval()
        model.to("cpu")

        return cls(model)

    def process(
        self,
        input_image: InputImage,
        thr_v=0.1,
        thr_d=0.1,
        detect_resolution=512,
        upscale_method="INTER_AREA",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detected_map, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )
        img = detected_map
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(
                    img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d
                )
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(
                        img_output,
                        (x_start, y_start),
                        (x_end, y_end),
                        [255, 255, 255],
                        1,
                    )
        except Exception as e:
            pass

        detected_map = remove_pad(HWC3(img_output[:, :, 0]))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
