import warnings
from typing import Union

import cv2
import numpy as np
from PIL import Image

from src.preprocess.util import HWC3, resize_image_with_pad
from .mediapipe_face_common import generate_annotation
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor


class MediapipeFaceDetector(BasePreprocessor):
    @classmethod
    def from_pretrained(cls):
        """Mediapipe face detector doesn't require pretrained models"""
        return cls()

    def process(
        self,
        input_image: InputImage,
        max_faces: int = 1,
        min_confidence: float = 0.5,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:

        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        detected_map, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )
        detected_map = generate_annotation(detected_map, max_faces, min_confidence)
        detected_map = remove_pad(HWC3(detected_map))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
