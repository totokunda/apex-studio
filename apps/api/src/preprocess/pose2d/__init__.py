import os
os.environ["APEX_USE_RUST_DOWNLOAD"] = "0"
from .pose2d import Pose2d
from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.defaults import get_torch_device
from src.preprocess.util import custom_hf_download, POSE2D_MODEL_NAME
from src.types import InputImage, OutputImage
import numpy as np
import cv2
from src.preprocess.pose2d.utils import get_face_bboxes, resize_by_area
from PIL import Image
import math
from src.preprocess.pose2d.pose2d_utils import AAPoseMeta
from src.preprocess.pose2d.human_visualization import draw_aapose_by_meta_new


class Pose2dDetector(BasePreprocessor):
    def __init__(
        self, checkpoint, detector_checkpoint=None, device=get_torch_device(), **kwargs
    ):
        self.device = device
        super(Pose2dDetector, self).__init__()
        self.pose2d = Pose2d(checkpoint, detector_checkpoint, device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=POSE2D_MODEL_NAME,
        filepath="vitpose_h_wholebody.onnx",
        det_pretrained_model_or_path=POSE2D_MODEL_NAME,
        det_filepath="yolov10m.onnx",
    ):

        checkpoint = custom_hf_download(
            pretrained_model_or_path, filepath, subfolder="process_checkpoint/pose2d"
        )
        det_checkpoint = custom_hf_download(
            det_pretrained_model_or_path,
            det_filepath,
            subfolder="process_checkpoint/det",
        )
        return cls(checkpoint, det_checkpoint)

    def process(
        self,
        input_image: InputImage,
        upscale_method="INTER_CUBIC",
        mode="pose",
        target_size=(480, 832),
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = resize_by_area(input_image, math.prod(target_size), divisor=16)
        meta = self.pose2d(input_image)
        if mode == "face":
            face_bbox_for_image = get_face_bboxes(
                meta["keypoints_face"][:, :2],
                scale=1.3,
                image_shape=(input_image.shape[0], input_image.shape[1]),
            )
            x1, x2, y1, y2 = face_bbox_for_image
            face_image = input_image[y1:y2, x1:x2]
            face_image = cv2.resize(face_image, (512, 512))
            face_image = Image.fromarray(face_image)
            return face_image
        elif mode == "pose":
            aa_pose_meta = AAPoseMeta.from_humanapi_meta(meta)
            canvas = np.zeros_like(input_image)
            conditioning_image = draw_aapose_by_meta_new(canvas, aa_pose_meta)
            conditioning_image = Image.fromarray(conditioning_image)
            return conditioning_image
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __call__(self, *args, **kwargs) -> OutputImage:
        """
        Process the input image and return the output image.
        """
        mode = kwargs.get("mode", "pose")
        return super().__call__(*args, ensure_image_size=mode == "pose", **kwargs)
