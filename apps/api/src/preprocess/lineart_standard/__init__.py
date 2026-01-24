import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import resize_image_with_pad, HWC3
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor


class LineartStandardDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """LineartStandard detector doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        guassian_sigma=6.0,
        intensity_threshold=8,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)

        detected_map = HWC3(remove_pad(detected_map))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)
        return detected_map
