import warnings
import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import HWC3, resize_image_with_pad
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor


# Not to be confused with "scribble" from HED. That is "fake scribble" which is more accurate and less picky than this.
class ScribbleDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """ScribbleDetector doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        upscale_method="INTER_AREA",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        detected_map = np.zeros_like(input_image, dtype=np.uint8)
        detected_map[np.min(input_image, axis=2) < 127] = 255
        detected_map = 255 - detected_map

        detected_map = remove_pad(detected_map)

        detected_map = Image.fromarray(detected_map)

        return detected_map


class ScribbleXDogDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """ScribbleXDogDetector doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        thr_a=32,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        g1 = cv2.GaussianBlur(input_image.astype(np.float32), (0, 0), 0.5)
        g2 = cv2.GaussianBlur(input_image.astype(np.float32), (0, 0), 5.0)
        dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
        result = np.zeros_like(input_image, dtype=np.uint8)
        result[2 * (255 - dog) > thr_a] = 255
        # result = 255 - result

        detected_map = HWC3(remove_pad(result))

        detected_map = Image.fromarray(detected_map)

        return detected_map
