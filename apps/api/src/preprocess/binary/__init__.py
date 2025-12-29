import warnings
import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import HWC3, resize_image_with_pad
from src.types import OutputImage, InputImage
from src.preprocess.base_preprocessor import BasePreprocessor


class BinaryDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """Binary detector doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        bin_threshold=0,
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

        img_gray = cv2.cvtColor(detected_map, cv2.COLOR_RGB2GRAY)
        if bin_threshold == 0 or bin_threshold == 255:
            # Otsu's threshold
            otsu_threshold, img_bin = cv2.threshold(
                img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            print("Otsu threshold:", otsu_threshold)
        else:
            _, img_bin = cv2.threshold(
                img_gray, bin_threshold, 255, cv2.THRESH_BINARY_INV
            )

        detected_map = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
        detected_map = HWC3(remove_pad(255 - detected_map))

        detected_map = Image.fromarray(detected_map)

        return detected_map
