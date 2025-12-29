import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from pathlib import Path

from .models.dsine_arch import DSINE
from .utils.utils import get_intrins_from_fov
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


# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location="cpu")["model"]

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k_ = k.replace("module.", "")
            load_dict[k_] = v
        else:
            load_dict[k] = v

    # Load compatible weights only
    model_state = model.state_dict()
    compatible_dict = {}
    skipped_keys = []

    for k, v in load_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                skipped_keys.append(
                    f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}"
                )
        else:
            skipped_keys.append(f"{k}: not found in model")

    print(
        f"Loading checkpoint: {len(compatible_dict)} compatible, {len(skipped_keys)} skipped"
    )
    if skipped_keys:
        print("Skipped keys with shape mismatches:")
        for key in skipped_keys[:5]:  # Show first 5 mismatches
            print(f"  {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")

    model.load_state_dict(compatible_dict, strict=False)
    return model


def get_pad(orig_H, orig_W):
    if orig_W % 64 == 0:
        l = 0
        r = 0
    else:
        new_W = 64 * ((orig_W // 64) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 64 == 0:
        t = 0
        b = 0
    else:
        new_H = 64 * ((orig_H // 64) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b


class DsineDetector(ToMixin, BasePreprocessor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.device = get_torch_device()
        self.to_device(self.model, device=self.device)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_or_path=DIFFUSION_EDGE_MODEL_NAME, filename="dsine.pt"
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DSINE()
        model = load_checkpoint(model_path, model)
        model.eval()
        model.to("cpu")  # Explicitly move to CPU

        return cls(model)

    def process(
        self,
        input_image: InputImage,
        fov=60.0,
        iterations=5,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        **kwargs,
    ) -> OutputImage:
        self.model.num_iter = iterations
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        orig_H, orig_W = input_image.shape[:2]
        l, r, t, b = get_pad(orig_H, orig_W)
        input_image, remove_pad = resize_image_with_pad(
            input_image, detect_resolution, upscale_method, mode="constant"
        )
        with torch.no_grad():
            input_image = torch.from_numpy(input_image).float().to(self.device)
            input_image = input_image / 255.0
            input_image = rearrange(input_image, "h w c -> 1 c h w")
            input_image = self.norm(input_image)

            intrins = get_intrins_from_fov(
                new_fov=fov, H=orig_H, W=orig_W, device=self.device
            ).unsqueeze(0)
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t

            normal = self.model(input_image, intrins)
            normal = normal[-1][0]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal, "c h w -> h w c").cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(normal_image)
        detected_map = detected_map.astype(np.uint8)
        detected_map = Image.fromarray(detected_map)

        return detected_map
