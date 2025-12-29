import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from src.preprocess.util import (
    HWC3,
    nms,
    resize_image_with_pad,
    safe_step,
    custom_hf_download,
    HF_MODEL_NAME,
)
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from .model import pidinet


class PidiNetDetector(ToMixin, BasePreprocessor):
    def __init__(self, netNetwork):
        super().__init__()
        self.netNetwork = netNetwork
        self.device = get_torch_device()
        self.to_device(self.netNetwork, device=self.device)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=HF_MODEL_NAME, filename="table5_pidinet.pth"
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        netNetwork = pidinet()
        netNetwork.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in torch.load(model_path)["state_dict"].items()
            }
        )
        netNetwork.eval()
        netNetwork.to("cpu")  # Explicitly move to CPU

        return cls(netNetwork)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        safe=False,
        scribble=False,
        apply_filter=False,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detected_map, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        detected_map = detected_map[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(detected_map).float().to(self.device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, "h w c -> 1 c h w")
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_filter:
                edge = edge > 0.5
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        detected_map = Image.fromarray(detected_map)

        return detected_map
