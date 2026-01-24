from src.preprocess.diffusion_edge.model import DiffusionEdge, prepare_args
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from src.preprocess.util import (
    HWC3,
    resize_image_with_pad,
    custom_hf_download,
    DIFFUSION_EDGE_MODEL_NAME,
)
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor


class DiffusionEdgeDetector(ToMixin, BasePreprocessor):
    def __init__(self, model: DiffusionEdge):
        super().__init__()
        self.model = model
        self.device = get_torch_device()
        self.to_device(self.model.model, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=DIFFUSION_EDGE_MODEL_NAME,
        filename="diffusion_edge_indoor.pt",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DiffusionEdge(prepare_args(model_path))
        return cls(model)

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        patch_batch_size=8,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.asarray(input_image, dtype=np.uint8)

        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method
        )

        with torch.no_grad():
            input_image = rearrange(torch.from_numpy(input_image), "h w c -> 1 c h w")
            input_image = input_image.float() / 255.0
            line = self.model(input_image, patch_batch_size)
            line = rearrange(line, "1 c h w -> h w c")

        detected_map = line.cpu().numpy().__mul__(255.0).astype(np.uint8)
        detected_map = remove_pad(HWC3(detected_map))
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
