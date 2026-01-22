import torchvision  # Fix issue Unknown builtin op: torchvision::nms
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

from src.preprocess.util import (
    HWC3,
    resize_image_with_pad,
    common_input_validate,
    custom_hf_download,
    DENSEPOSE_MODEL_NAME,
)
from .densepose import (
    DensePoseMaskedColormapResultsVisualizer,
    _extract_i_from_iuvarr,
    densepose_chart_predictor_output_to_result_with_confidences,
)
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor

N_PART_LABELS = 24


class DenseposeDetector(ToMixin, BasePreprocessor):
    def __init__(self, model):
        super().__init__()
        self.dense_pose_estimation = model
        self.device = get_torch_device()
        self.result_visualizer = DensePoseMaskedColormapResultsVisualizer(
            alpha=1,
            data_extractor=_extract_i_from_iuvarr,
            segm_extractor=_extract_i_from_iuvarr,
            val_scale=255.0 / N_PART_LABELS,
        )
        if torch.cuda.is_available():
            self.to_device(self.dense_pose_estimation, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=DENSEPOSE_MODEL_NAME,
        filename="densepose_r50_fpn_dl.torchscript",
    ):
        torchscript_model_path = custom_hf_download(pretrained_model_or_path, filename)
        densepose = torch.jit.load(torchscript_model_path, map_location="cpu")
        densepose.eval()
        densepose.to("cpu")  # Explicitly move to CPU
        return cls(densepose)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        cmap="viridis",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.asarray(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )
        H, W = input_image.shape[:2]

        hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
        hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])

        with torch.no_grad():
            input_image_tensor = rearrange(
                torch.from_numpy(input_image).to(self.device), "h w c -> c h w"
            )

            pred_boxes, corase_segm, fine_segm, u, v = self.dense_pose_estimation(
                input_image_tensor
            )

            extractor = densepose_chart_predictor_output_to_result_with_confidences
            densepose_results = [
                extractor(
                    pred_boxes[i : i + 1],
                    corase_segm[i : i + 1],
                    fine_segm[i : i + 1],
                    u[i : i + 1],
                    v[i : i + 1],
                )
                for i in range(len(pred_boxes))
            ]

            if cmap == "viridis":
                self.result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
                hint_image = self.result_visualizer.visualize(
                    hint_image_canvas, densepose_results
                )
                hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
                hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
                hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
                hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
            else:
                self.result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
                hint_image = self.result_visualizer.visualize(
                    hint_image_canvas, densepose_results
                )
                hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

        detected_map = remove_pad(HWC3(hint_image))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
