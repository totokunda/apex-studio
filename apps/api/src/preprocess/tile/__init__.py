import warnings
import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import get_upscale_method, HWC3
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from .guided_filter import FastGuidedFilter


class TileDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """TileDetector doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        pyrUp_iters=3,
        upscale_method="INTER_AREA",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        H, W, _ = input_image.shape
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        detected_map = cv2.resize(
            input_image,
            (W // (2**pyrUp_iters), H // (2**pyrUp_iters)),
            interpolation=get_upscale_method(upscale_method),
        )
        detected_map = HWC3(detected_map)

        for _ in range(pyrUp_iters):
            detected_map = cv2.pyrUp(detected_map)

        detected_map = Image.fromarray(detected_map)

        return detected_map


# Source: https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/blob/main/TTP_tile_preprocessor_v5.py


def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image


def apply_guided_filter(image_np, radius, eps, scale):
    filter = FastGuidedFilter(image_np, radius, eps, scale)
    return filter.filter(image_np)


class TTPlanet_Tile_Detector_GF(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """TTPlanet_Tile_Detector_GF doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        scale_factor=2.0,
        blur_strength=2.0,
        radius=5,
        eps=0.2,
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        img_np = input_image[:, :, ::-1]  # RGB to BGR

        # Apply Gaussian blur
        img_np = apply_gaussian_blur(
            img_np, ksize=int(blur_strength), sigmaX=blur_strength / 2
        )

        # Apply Guided Filter
        img_np = apply_guided_filter(img_np, radius, eps, scale_factor)

        # Resize image
        height, width = img_np.shape[:2]
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        resized_down = cv2.resize(
            img_np, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        resized_img = cv2.resize(
            resized_down, (width, height), interpolation=cv2.INTER_CUBIC
        )
        detected_map = HWC3(resized_img[:, :, ::-1])  # BGR to RGB

        detected_map = Image.fromarray(detected_map)

        return detected_map


class TTPLanet_Tile_Detector_Simple(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """TTPLanet_Tile_Detector_Simple doesn't require pretrained models"""
        return cls()

    def process(
        self, input_image: InputImage, scale_factor=2.0, blur_strength=2.0, **kwargs
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        img_np = input_image[:, :, ::-1]  # RGB to BGR

        # Resize image first if you want blur to apply after resizing
        height, width = img_np.shape[:2]
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        resized_down = cv2.resize(
            img_np, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        resized_img = cv2.resize(
            resized_down, (width, height), interpolation=cv2.INTER_LANCZOS4
        )

        # Apply Gaussian blur after resizing
        img_np = apply_gaussian_blur(
            resized_img, ksize=int(blur_strength), sigmaX=blur_strength / 2
        )
        detected_map = HWC3(img_np[:, :, ::-1])  # BGR to RGB

        detected_map = Image.fromarray(detected_map)

        return detected_map
