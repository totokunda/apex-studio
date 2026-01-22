# MangaLineExtraction_PyTorch
# https://github.com/ljsabc/MangaLineExtraction_PyTorch

# NOTE: This preprocessor is designed to work with lineart_anime ControlNet so the result will be white lines on black canvas

import torch
import numpy as np
import os
import cv2
from einops import rearrange
from .model_torch import res_skip
from PIL import Image
import warnings

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


class LineartMangaDetector(ToMixin, BasePreprocessor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=HF_MODEL_NAME, filename="erika.pth"
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        net = res_skip()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        for key in list(ckpt.keys()):
            if "module." in key:
                ckpt[key.replace("module.", "")] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        net.eval()
        net.to("cpu")
        return cls(net)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detected_map, remove_pad = resize_image_with_pad(
            input_image,
            256 * int(np.ceil(float(detect_resolution) / 256.0)),
            upscale_method,
        )

        img = cv2.cvtColor(detected_map, cv2.COLOR_RGB2GRAY)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(self.device)
            image_feed = rearrange(image_feed, "h w -> 1 1 h w")

            line = self.model(image_feed)
            line = line.cpu().numpy()[0, 0, :, :]
            line[line > 255] = 255
            line[line < 0] = 0

            line = line.astype(np.uint8)

        detected_map = HWC3(line)
        detected_map = remove_pad(255 - detected_map)
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
