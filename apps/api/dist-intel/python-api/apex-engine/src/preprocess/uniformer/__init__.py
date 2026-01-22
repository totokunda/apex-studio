import os
from .inference import init_segmentor, inference_segmentor, show_result_pyplot
import warnings
import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import (
    HWC3,
    resize_image_with_pad,
    custom_hf_download,
    HF_MODEL_NAME,
)
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
import torch

from src.preprocess.custom_mmpkg.custom_mmseg.core.evaluation import get_palette

config_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "upernet_global_small.py"
)


class UniformerSegmentor(ToMixin, BasePreprocessor):
    def __init__(self, netNetwork):
        super().__init__()
        self.model = netNetwork
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=HF_MODEL_NAME, filename="upernet_global_small.pth"
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        netNetwork = init_segmentor(config_file, model_path, device="cpu")
        netNetwork.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in torch.load(model_path)["state_dict"].items()
            }
        )
        netNetwork.eval()
        netNetwork.to("cpu")  # Explicitly move to CPU

        return cls(netNetwork)

    def _inference(self, img):
        if next(self.model.parameters()).device.type == "mps":
            # adaptive_avg_pool2d can fail on MPS, workaround with CPU
            import torch.nn.functional

            orig_adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d

            def cpu_if_exception(input, *args, **kwargs):
                try:
                    return orig_adaptive_avg_pool2d(input, *args, **kwargs)
                except:
                    return orig_adaptive_avg_pool2d(input.cpu(), *args, **kwargs).to(
                        input.device
                    )

            try:
                torch.nn.functional.adaptive_avg_pool2d = cpu_if_exception
                result = inference_segmentor(self.model, img)
            finally:
                torch.nn.functional.adaptive_avg_pool2d = orig_adaptive_avg_pool2d
        else:
            result = inference_segmentor(self.model, img)

        res_img = show_result_pyplot(
            self.model, img, result, get_palette("ade"), opacity=1
        )
        return res_img

    def process(
        self,
        input_image: InputImage,
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

        detected_map = self._inference(input_image)
        detected_map = remove_pad(HWC3(detected_map))

        detected_map = Image.fromarray(detected_map)

        return detected_map
