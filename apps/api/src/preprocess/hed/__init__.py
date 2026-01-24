# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

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
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(
            input_channel=3, output_channel=64, layer_number=2
        )
        self.block2 = DoubleConvBlock(
            input_channel=64, output_channel=128, layer_number=2
        )
        self.block3 = DoubleConvBlock(
            input_channel=128, output_channel=256, layer_number=3
        )
        self.block4 = DoubleConvBlock(
            input_channel=256, output_channel=512, layer_number=3
        )
        self.block5 = DoubleConvBlock(
            input_channel=512, output_channel=512, layer_number=3
        )

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector(ToMixin, BasePreprocessor):
    def __init__(self, netNetwork):
        super().__init__()
        self.netNetwork = netNetwork
        self.device = get_torch_device()
        self.to_device(self.netNetwork, device=self.device)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=HF_MODEL_NAME, filename="ControlNetHED.pth"
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        netNetwork = ControlNetHED_Apache2()
        netNetwork.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False)
        )
        netNetwork.float().eval()
        netNetwork.to("cpu")  # Explicitly move to CPU

        return cls(netNetwork)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        safe=False,
        scribble=False,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        assert input_image.ndim == 3
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float().to(self.device)
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edges = self.netNetwork(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
