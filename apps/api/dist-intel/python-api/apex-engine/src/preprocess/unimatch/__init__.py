import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from src.preprocess.util import (
    resize_image_with_pad,
    custom_hf_download,
    UNIMATCH_MODEL_NAME,
    HWC3,
)
from src.types import InputImage, OutputImage, InputVideo, OutputVideo
from src.preprocess.base_preprocessor import BasePreprocessor
from src.mixins import ToMixin
from src.utils.defaults import get_torch_device
from .utils.flow_viz import save_vis_flow_tofile, flow_to_image
from .unimatch.unimatch import UniMatch
import torch.nn.functional as F
from argparse import Namespace
from tqdm import tqdm


def inference_flow(
    model,
    image1,  # np array of HWC
    image2,
    padding_factor=8,
    inference_size=None,
    attn_type="swin",
    attn_splits_list=None,
    corr_radius_list=None,
    prop_radius_list=None,
    num_reg_refine=1,
    pred_bidir_flow=False,
    pred_bwd_flow=False,
    fwd_bwd_consistency_check=False,
    device="cpu",
    **kwargs,
):
    fixed_inference_size = inference_size
    transpose_img = False
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    nearest_size = [
        int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor,
    ]
    # resize to nearest size or specified size
    inference_size = (
        nearest_size if fixed_inference_size is None else fixed_inference_size
    )
    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(
            image1, size=inference_size, mode="bilinear", align_corners=True
        )
        image2 = F.interpolate(
            image2, size=inference_size, mode="bilinear", align_corners=True
        )
    if pred_bwd_flow:
        image1, image2 = image2, image1

    results_dict = model(
        image1,
        image2,
        attn_type=attn_type,
        attn_splits_list=attn_splits_list,
        corr_radius_list=corr_radius_list,
        prop_radius_list=prop_radius_list,
        num_reg_refine=num_reg_refine,
        task="flow",
        pred_bidir_flow=pred_bidir_flow,
    )
    flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]

    # resize back
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(
            flow_pr, size=ori_size, mode="bilinear", align_corners=True
        )
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

    vis_image = flow_to_image(flow)

    # also predict backward flow
    if pred_bidir_flow:
        assert flow_pr.size(0) == 2  # [2, H, W, 2]
        flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        vis_image = flow_to_image(flow_bwd)
        flow = flow_bwd
    return flow, vis_image


MODEL_CONFIGS = {
    "gmflow-scale1": Namespace(
        num_scales=1,
        upsample_factor=8,
        attn_type="swin",
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        attn_splits_list=[2],
        corr_radius_list=[-1],
        prop_radius_list=[-1],
        reg_refine=False,
        num_reg_refine=1,
    ),
    "gmflow-scale2": Namespace(
        num_scales=2,
        upsample_factor=4,
        padding_factor=32,
        attn_type="swin",
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        reg_refine=False,
        num_reg_refine=1,
    ),
    "gmflow-scale2-regrefine6": Namespace(
        num_scales=2,
        upsample_factor=4,
        padding_factor=32,
        attn_type="swin",
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        reg_refine=True,
        num_reg_refine=6,
    ),
}


class UnimatchDetector(ToMixin, BasePreprocessor):
    def __init__(self, unimatch, config_args):
        super().__init__()
        self.unimatch = unimatch
        self.config_args = config_args
        self.device = get_torch_device()
        self.to_device(self.unimatch, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=UNIMATCH_MODEL_NAME,
        filename="gmflow-scale2-regrefine6-mixdata.pth",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        config_args = None
        for key in list(MODEL_CONFIGS.keys())[::-1]:
            if key in filename:
                config_args = MODEL_CONFIGS[key]
                break
        assert config_args, f"Couldn't find hardcoded Unimatch config for {filename}"

        model = UniMatch(
            feature_channels=config_args.feature_channels,
            num_scales=config_args.num_scales,
            upsample_factor=config_args.upsample_factor,
            num_head=config_args.num_head,
            ffn_dim_expansion=config_args.ffn_dim_expansion,
            num_transformer_layers=config_args.num_transformer_layers,
            reg_refine=config_args.reg_refine,
            task="flow",
        )

        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd["model"])
        model.to("cpu")  # Explicitly move to CPU
        return cls(model, config_args)

    def process_image(self, input_image: InputImage, **kwargs) -> OutputImage:
        raise NotImplementedError("UnimatchDetector does not support image processing")

    def process_video(self, input_video: InputVideo, **kwargs) -> OutputVideo:
        # we must pass two images to unimatch
        from tqdm import tqdm

        # Check if input is a generator/iterator
        if hasattr(input_video, "__iter__") and not isinstance(
            input_video, (list, str)
        ):
            frames_iter = iter(input_video)
            total_frames = kwargs.get("total_frames", None)
        else:
            frames = self._load_video(input_video)
            frames_iter = iter(frames)
            total_frames = len(frames)

        progress_callback = kwargs.get("progress_callback", None)
        frame_idx = 0

        # Get first frame
        try:
            prev_frame = next(frames_iter)
            frame_idx += 1
        except StopIteration:
            raise ValueError("Video must have at least 2 frames for optical flow")

        prev_anno_frame = None

        # Process each pair of consecutive frames using a sliding window
        for current_frame in tqdm(
            frames_iter,
            desc="Processing frames",
            total=total_frames - 1 if total_frames else None,
        ):
            # Update progress
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

            # Process the frame pair
            anno_frame = self.process(prev_frame, image2=current_frame, **kwargs)
            prev_anno_frame = anno_frame
            yield anno_frame

            # Slide the window
            prev_frame = current_frame
            frame_idx += 1

        # For the last frame, duplicate the previous result
        if progress_callback is not None:
            progress_callback(frame_idx, total_frames if total_frames else frame_idx)

        if prev_anno_frame is not None:
            yield prev_anno_frame

        # Send final frame completion
        if progress_callback is not None:
            progress_callback(frame_idx, frame_idx)

    def process(
        self,
        input_image: InputImage,
        image2=None,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        return_flow=False,
        pred_bwd_flow=False,
        pred_bidir_flow=False,
        **kwargs,
    ) -> OutputImage:
        # Note: Unimatch needs two images, so we expect image2 to be passed in kwargs or as parameter
        image1 = self._load_image(input_image)

        if not isinstance(image1, np.ndarray):
            image1 = np.asarray(image1, dtype=np.uint8)

        if image2 is None:
            raise ValueError(
                "UnimatchDetector requires a second image (image2 parameter)"
            )

        if not isinstance(image2, np.ndarray):
            image2 = self._load_image(image2)
            image2 = np.asarray(image2, dtype=np.uint8)

        image1, remove_pad1 = resize_image_with_pad(
            image1, detect_resolution, upscale_method
        )
        image2, remove_pad2 = resize_image_with_pad(
            image2, detect_resolution, upscale_method
        )

        assert (
            image1.shape == image2.shape
        ), f"[Unimatch] image1 and image2 must have the same size, got {image1.shape} and {image2.shape}"

        with torch.no_grad():
            flow, vis_image = inference_flow(
                self.unimatch,
                image1,
                image2,
                device=self.device,
                pred_bwd_flow=pred_bwd_flow,
                pred_bidir_flow=pred_bidir_flow,
                **vars(self.config_args),
            )

        vis_image = remove_pad1(HWC3(vis_image))
        vis_image = Image.fromarray(vis_image)

        # Return both flow and vis_image as a tuple (special case for this preprocessor)
        if return_flow:
            return flow, vis_image
        else:
            return vis_image
