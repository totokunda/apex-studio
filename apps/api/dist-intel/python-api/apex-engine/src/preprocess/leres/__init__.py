import os

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
from .leres.depthmap import estimateboost, estimateleres
from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import strip_prefix_if_present
from .pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
from .pix2pix.options.test_options import TestOptions


class LeresDetector(ToMixin, BasePreprocessor):
    def __init__(self, model, pix2pixmodel):
        super().__init__()
        self.model = model
        self.pix2pixmodel = pix2pixmodel
        self.device = get_torch_device()
        self.to_device(self.model, self.pix2pixmodel, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=HF_MODEL_NAME,
        filename="res101.pth",
        pix2pix_filename="latest_net_G.pth",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )

        model = RelDepthModel(backbone="resnext101")
        model.load_state_dict(
            strip_prefix_if_present(checkpoint["depth_model"], "module."), strict=True
        )
        model.eval()
        model.to("cpu")  # Explicitly move to CPU
        del checkpoint

        pix2pix_model_path = custom_hf_download(
            pretrained_model_or_path, pix2pix_filename
        )

        opt = TestOptions().parse()
        opt.gpu_ids = []  # Force CPU mode for Ray workers
        pix2pixmodel = Pix2Pix4DepthModel(opt)
        pix2pixmodel.save_dir = os.path.dirname(pix2pix_model_path)
        pix2pixmodel.load_networks("latest")
        pix2pixmodel.eval()

        return cls(model, pix2pixmodel)

    def process(
        self,
        input_image: InputImage,
        thr_a=0,
        thr_b=0,
        boost=False,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detected_map, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        with torch.no_grad():
            if boost:
                depth = estimateboost(
                    detected_map,
                    self.model,
                    0,
                    self.pix2pixmodel,
                    max(detected_map.shape[1], detected_map.shape[0]),
                )
            else:
                depth = estimateleres(
                    detected_map,
                    self.model,
                    detected_map.shape[1],
                    detected_map.shape[0],
                )

            numbytes = 2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2 ** (8 * numbytes)) - 1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape)

            # single channel, 16 bit image
            depth_image = out.astype("uint16")

            # convert to uint8
            depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0 / 65535.0))

            # remove near
            if thr_a != 0:
                thr_a = (thr_a / 100) * 255
                depth_image = cv2.threshold(depth_image, thr_a, 255, cv2.THRESH_TOZERO)[
                    1
                ]

            # invert image
            depth_image = cv2.bitwise_not(depth_image)

            # remove bg
            if thr_b != 0:
                thr_b = (thr_b / 100) * 255
                depth_image = cv2.threshold(depth_image, thr_b, 255, cv2.THRESH_TOZERO)[
                    1
                ]

        detected_map = HWC3(remove_pad(depth_image))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
