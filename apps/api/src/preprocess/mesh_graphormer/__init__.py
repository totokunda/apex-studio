import cv2
import numpy as np
from PIL import Image
from src.preprocess.util import (
    resize_image_with_pad,
    HWC3,
    custom_hf_download,
    MESH_GRAPHORMER_MODEL_NAME,
)
from src.preprocess.mesh_graphormer.pipeline import MeshGraphormerMediapipe, args
from src.types import InputImage, OutputImage
from src.preprocess.base_preprocessor import BasePreprocessor
from src.utils.defaults import get_torch_device
import random, torch


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class MeshGraphormerDetector(BasePreprocessor):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=MESH_GRAPHORMER_MODEL_NAME,
        filename="graphormer_hand_state_dict.bin",
        hrnet_filename="hrnetv2_w64_imagenet_pretrained.pth",
        detect_thr=0.6,
        presence_thr=0.6,
    ):
        args.resume_checkpoint = custom_hf_download(pretrained_model_or_path, filename)
        args.hrnet_checkpoint = custom_hf_download(
            pretrained_model_or_path, hrnet_filename
        )
        args.device = get_torch_device()
        pipeline = MeshGraphormerMediapipe(
            args, detect_thr=detect_thr, presence_thr=presence_thr
        )
        return cls(pipeline)

    def process(
        self,
        input_image: InputImage,
        mask_bbox_padding=30,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        seed=88,
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        set_seed(seed, 0)
        depth_map, mask, info = self.pipeline.get_depth(input_image, mask_bbox_padding)
        if depth_map is None:
            depth_map = np.zeros_like(input_image)
            mask = np.zeros_like(input_image)

        # The hand is small - ensure both depth and mask are HWC3 format
        depth_map, mask = HWC3(depth_map), HWC3(mask)

        # Resize both depth_map and mask to the same resolution
        depth_map, remove_pad = resize_image_with_pad(
            depth_map, detect_resolution, upscale_method
        )
        mask, remove_pad_mask = resize_image_with_pad(
            mask, detect_resolution, upscale_method
        )

        # Remove padding from both to get back to original aspect ratio
        depth_map = remove_pad(depth_map)
        mask = remove_pad_mask(mask)

        # Ensure both have the same dimensions (in case of any rounding differences)
        if depth_map.shape != mask.shape:
            # Resize mask to exactly match depth_map dimensions
            mask = cv2.resize(
                mask,
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Overlay mask on depth map - convert mask to grayscale and blend with depth
        if mask.shape[-1] == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask

        # Normalize mask to 0-1 range for alpha blending
        mask_alpha = mask_gray.astype(np.float32) / 255.0

        # Create a 3-channel alpha mask
        mask_alpha_3c = np.stack([mask_alpha] * 3, axis=-1)

        # Blend: depth * alpha + background * (1 - alpha)
        # Where mask is present (white), show depth; where mask is absent (black), fade to background
        blended = (depth_map.astype(np.float32) * mask_alpha_3c).astype(np.uint8)

        blended = Image.fromarray(blended)

        return blended
