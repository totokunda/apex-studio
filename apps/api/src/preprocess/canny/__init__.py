import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import resize_image_with_pad, common_input_validate, HWC3
from src.preprocess.base_preprocessor import BasePreprocessor


class CannyDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        return cls()

    def process(
        self,
        input_image=None,
        low_threshold=100,
        high_threshold=200,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ):
        input_image, output_type = common_input_validate(input_image, None, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )
        detected_map = cv2.Canny(detected_map, low_threshold, high_threshold)
        detected_map = HWC3(remove_pad(detected_map))

        detected_map = Image.fromarray(detected_map)

        return detected_map
