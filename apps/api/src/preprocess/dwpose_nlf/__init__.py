from src.preprocess.dwpose import DwposeDetector, Wholebody
from src.types import InputImage, OutputImage
from typing import List, Tuple
import torch
import numpy as np
import math
import random
import copy
from src.mixins.download_mixin import DownloadMixin
from src.preprocess.dwpose_nlf.render_helpers import (
    collect_smpl_poses,
    draw_pose_to_canvas_np,
    scale_image_hw_keep_size,
    shift_dwpose_according_to_nlf,
    render_whole,
    get_single_pose_cylinder_specs,
)
import taichi as ti
from src.preprocess.util import (
    HWC3,
    resize_image_with_pad,
    custom_hf_download,
    DWPOSE_MODEL_NAME,
    DOWNLOAD_PROGRESS_CALLBACK,
    annotator_ckpts_path,
)
from PIL import Image
from src.utils.defaults import get_torch_device
import os
from tqdm import tqdm
from typing import Dict, Any, Optional


global_cached_dwpose = Wholebody()


def _parse_background_color(val):
    if val is None:
        return (0, 0, 0)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("black", "0", "zero"):
            return (0, 0, 0)
        if v in ("white", "1"):
            return (255, 255, 255)
        if v in ("gray", "grey"):
            return (127, 127, 127)
        raise ValueError(
            f"Unknown background '{val}'. Use 'black'/'white' or an (r,g,b) tuple."
        )
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return (int(val[0]), int(val[1]), int(val[2]))
    raise ValueError(f"Invalid background_color: {val!r}")


def _parse_background(background, background_color):
    """
    Returns:
      ("source", None) to composite over the input frame
      ("color", (r,g,b)) to composite over a solid color
    """
    if background_color is not None:
        return ("color", _parse_background_color(background_color))
    if background is None:
        return ("source", None)
    if isinstance(background, str) and background.strip().lower() in (
        "source",
        "input",
        "original",
    ):
        return ("source", None)
    return ("color", _parse_background_color(background))


def _to_rgb_uint8(
    x: np.ndarray,
    background_color=(0, 0, 0),
    background_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert (H,W,1|3|4) image to RGB uint8.
    For RGBA, alpha-composite over background_rgb if provided; otherwise background_color.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.dtype != np.uint8:
        x = x.astype(np.uint8)
    H, W, C = x.shape
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        if background_rgb is not None:
            if not isinstance(background_rgb, np.ndarray):
                background_rgb = np.array(background_rgb)
            if background_rgb.ndim == 2:
                background_rgb = background_rgb[:, :, None]
            if background_rgb.shape[2] == 4:
                background_rgb = background_rgb[:, :, :3]
            if background_rgb.dtype != np.uint8:
                background_rgb = background_rgb.astype(np.uint8)
            bg = background_rgb.astype(np.float32)
        else:
            bg = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
        y = color * alpha + bg * (1.0 - alpha)
        return y.clip(0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported channel count: {C}")


class DwposeNlfDetector(DwposeDetector):
    def __init__(self, dw_pose_estimation, nlf_model, device=None):
        super().__init__(dw_pose_estimation)
        self.nlf_model = nlf_model
        if device is None:
            self.device = get_torch_device()
        else:
            self.device = device
        ti.init(arch=ti.gpu if self.device.type == "cuda" else ti.cpu)

        self.nlf_model.to(self.device).eval()

    @staticmethod
    def download_nlf_model(url, nlf_model_filename):
        def _progress_cb(downloaded: int, total: int, _label: str = None):
            if DOWNLOAD_PROGRESS_CALLBACK:
                try:
                    # total can be None; use 0 to keep signature stable
                    DOWNLOAD_PROGRESS_CALLBACK(
                        int(downloaded or 0), int(total or 0), nlf_model_filename
                    )
                except Exception:
                    pass

        dl = DownloadMixin()
        local_dir = os.path.join(annotator_ckpts_path, "nlf")
        os.makedirs(local_dir, exist_ok=True)
        downloaded_path = dl.download_from_url(
            url=url,
            save_path=local_dir,
            progress_callback=_progress_cb,
        )
        return downloaded_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path=DWPOSE_MODEL_NAME,
        pretrained_det_model_or_path=None,
        pretrained_nlf_model_or_path=None,
        det_filename=None,
        nlf_model_filename=None,
        pose_filename=None,
        torchscript_device="cpu",
    ):
        global global_cached_dwpose
        pretrained_det_model_or_path = (
            pretrained_det_model_or_path or pretrained_model_or_path
        )

        pose_filename = pose_filename or "dw-ll_ucoco_384.onnx"
        nlf_model_filename = nlf_model_filename or "nlf_l_multi_0.3.2.torchscript"

        det_filename = det_filename or "yolox_l.onnx"
        det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)

        pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)

        if pretrained_nlf_model_or_path is None:
            pretrained_nlf_model_or_path = "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript"

        nlf_model_path = cls.download_nlf_model(
            pretrained_nlf_model_or_path, nlf_model_filename
        )

        if (
            global_cached_dwpose.det is None
            or global_cached_dwpose.det_filename != det_filename
        ):
            t = Wholebody(det_model_path, None, torchscript_device=torchscript_device)
            t.pose = global_cached_dwpose.pose
            t.pose_filename = global_cached_dwpose.pose
            global_cached_dwpose = t

        if (
            global_cached_dwpose.pose is None
            or global_cached_dwpose.pose_filename != pose_filename
        ):
            t = Wholebody(None, pose_model_path, torchscript_device=torchscript_device)
            t.det = global_cached_dwpose.det
            t.det_filename = global_cached_dwpose.det_filename
            global_cached_dwpose = t

        # load nlf model with torch.jit
        nlf_model = torch.jit.load(nlf_model_path, map_location=get_torch_device().type)
        return cls(global_cached_dwpose, nlf_model)

    def render_nlf_as_images(
        self,
        data,
        poses,
        reshape_pool=None,
        intrinsic_matrix=None,
        draw_2d=True,
        aug_2d=False,
        aug_cam=False,
    ):
        """return a list of images"""
        height, width = data[0]["video_height"], data[0]["video_width"]
        video_length = len(data)

        base_colors_255_dict = {
            # Warm Colors for Right Side (R.) - Red, Orange, Yellow
            "Red": [255, 0, 0],
            "Orange": [255, 85, 0],
            "Golden Orange": [255, 170, 0],
            "Yellow": [255, 240, 0],
            "Yellow-Green": [180, 255, 0],
            # Cool Colors for Left Side (L.) - Green, Blue, Purple
            "Bright Green": [0, 255, 0],
            "Light Green-Blue": [0, 255, 85],
            "Aqua": [0, 255, 170],
            "Cyan": [0, 255, 255],
            "Sky Blue": [0, 170, 255],
            "Medium Blue": [0, 85, 255],
            "Pure Blue": [0, 0, 255],
            "Purple-Blue": [85, 0, 255],
            "Medium Purple": [170, 0, 255],
            # Neutral/Central Colors (e.g., for Neck, Nose, Eyes, Ears)
            "Grey": [150, 150, 150],
            "Pink-Magenta": [255, 0, 170],
            "Dark Pink": [255, 0, 85],
            "Violet": [100, 0, 255],
            "Dark Violet": [50, 0, 255],
        }

        ordered_colors_255 = [
            base_colors_255_dict["Red"],  # Neck -> R. Shoulder (Red)
            base_colors_255_dict["Cyan"],  # Neck -> L. Shoulder (Cyan)
            base_colors_255_dict["Orange"],  # R. Shoulder -> R. Elbow (Orange)
            base_colors_255_dict[
                "Golden Orange"
            ],  # R. Elbow -> R. Wrist (Golden Orange)
            base_colors_255_dict["Sky Blue"],  # L. Shoulder -> L. Elbow (Sky Blue)
            base_colors_255_dict["Medium Blue"],  # L. Elbow -> L. Wrist (Medium Blue)
            base_colors_255_dict["Yellow-Green"],  # Neck -> R. Hip ( Yellow-Green)
            base_colors_255_dict[
                "Bright Green"
            ],  # R. Hip -> R. Knee (Bright Green - transitioning warm to cool spectrum)
            base_colors_255_dict[
                "Light Green-Blue"
            ],  # R. Knee -> R. Ankle (Light Green-Blue - transitioning)
            base_colors_255_dict["Pure Blue"],  # Neck -> L. Hip (Pure Blue)
            base_colors_255_dict["Purple-Blue"],  # L. Hip -> L. Knee (Purple-Blue)
            base_colors_255_dict[
                "Medium Purple"
            ],  # L. Knee -> L. Ankle (Medium Purple)
            base_colors_255_dict["Grey"],  # Neck -> Nose (Grey)
            base_colors_255_dict["Pink-Magenta"],  # Nose -> R. Eye (Pink/Magenta)
            base_colors_255_dict["Dark Violet"],  # R. Eye -> R. Ear (Dark Pink)
            base_colors_255_dict["Pink-Magenta"],  # Nose -> L. Eye (Violet)
            base_colors_255_dict["Dark Violet"],  # L. Eye -> L. Ear (Dark Violet)
        ]

        limb_seq = [
            [1, 2],  # 0 Neck -> R. Shoulder
            [1, 5],  # 1 Neck -> L. Shoulder
            [2, 3],  # 2 R. Shoulder -> R. Elbow
            [3, 4],  # 3 R. Elbow -> R. Wrist
            [5, 6],  # 4 L. Shoulder -> L. Elbow
            [6, 7],  # 5 L. Elbow -> L. Wrist
            [1, 8],  # 6 Neck -> R. Hip
            [8, 9],  # 7 R. Hip -> R. Knee
            [9, 10],  # 8 R. Knee -> R. Ankle
            [1, 11],  # 9 Neck -> L. Hip
            [11, 12],  # 10 L. Hip -> L. Knee
            [12, 13],  # 11 L. Knee -> L. Ankle
            [1, 0],  # 12 Neck -> Nose
            [0, 14],  # 13 Nose -> R. Eye
            [14, 16],  # 14 R. Eye -> R. Ear
            [0, 15],  # 15 Nose -> L. Eye
            [15, 17],  # 16 L. Eye -> L. Ear
        ]

        draw_seq = [
            0,
            2,
            3,  # Neck -> R. Shoulder -> R. Elbow -> R. Wrist
            1,
            4,
            5,  # Neck -> L. Shoulder -> L. Elbow -> L. Wrist
            6,
            7,
            8,  # Neck -> R. Hip -> R. Knee -> R. Ankle
            9,
            10,
            11,  # Neck -> L. Hip -> L. Knee -> L. Ankle
            12,  # Neck -> Nose
            13,
            14,  # Nose -> R. Eye -> R. Ear
            15,
            16,  # Nose -> L. Eye -> L. Ear
        ]  # 从近心端往外扩展

        colors = [
            [c / 300 + 0.15 for c in color_rgb] + [0.8]
            for color_rgb in ordered_colors_255
        ]

        if poses is not None:
            # 重新收集poses
            smpl_poses = collect_smpl_poses(data)
            aligned_poses = copy.deepcopy(poses)
            if reshape_pool is not None:
                for i in range(video_length):
                    persons_joints_list = smpl_poses[i]
                    poses_list = aligned_poses[i]
                    # 对里面每一个人，取关节并进行变形；并且修改2d；如果3d不存在，把2d的手/脸也去掉
                    for person_idx, person_joints in enumerate(persons_joints_list):
                        candidate = poses_list["bodies"]["candidate"][person_idx]
                        subset = poses_list["bodies"]["subset"][person_idx]
                        face = poses_list["faces"][person_idx]
                        right_hand = poses_list["hands"][2 * person_idx]
                        left_hand = poses_list["hands"][2 * person_idx + 1]
                        reshape_pool.apply_random_reshapes(
                            person_joints,
                            candidate,
                            left_hand,
                            right_hand,
                            face,
                            subset,
                        )
        else:
            smpl_poses = [
                item["nlfpose"] for item in data
            ]  # 主要为了兼容多人评测集；搭配process_video_nlf_original

        if intrinsic_matrix is None:
            intrinsic_matrix = self._intrinsic_matrix_from_field_of_view(
                (height, width)
            )
        focal_x = intrinsic_matrix[0, 0]
        focal_y = intrinsic_matrix[1, 1]
        princpt = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])  # 主点 (cx, cy)
        if aug_cam and random.random() < 0.3:
            w_shift_factor = random.uniform(-0.04, 0.04)
            h_shift_factor = random.uniform(-0.04, 0.04)
            princpt = (
                princpt[0] - w_shift_factor * width,
                princpt[1] - h_shift_factor * height,
            )  # princpt变化和点的变化相反
            new_intrinsic_matrix = copy.deepcopy(intrinsic_matrix)
            new_intrinsic_matrix[0, 2] = princpt[0]
            new_intrinsic_matrix[1, 2] = princpt[1]
            shift_dwpose_according_to_nlf(
                smpl_poses,
                aligned_poses,
                intrinsic_matrix,
                new_intrinsic_matrix,
                height,
                width,
            )

        # 串行获取每一帧的cylinder_specs
        cylinder_specs_list = []
        for i in range(video_length):
            cylinder_specs = get_single_pose_cylinder_specs(
                (i, smpl_poses[i], None, None, None, None, colors, limb_seq, draw_seq)
            )
            cylinder_specs_list.append(cylinder_specs)

        frames_np_rgba = render_whole(
            cylinder_specs_list,
            H=height,
            W=width,
            fx=focal_x,
            fy=focal_y,
            cx=princpt[0],
            cy=princpt[1],
        )
        if poses is not None and draw_2d:
            canvas_2d = draw_pose_to_canvas_np(
                aligned_poses,
                pool=None,
                H=height,
                W=width,
                reshape_scale=0,
                show_feet_flag=False,
                show_body_flag=False,
                show_cheek_flag=True,
                dw_hand=True,
            )
            # 覆盖 + rescale
            scale_h = random.uniform(0.85, 1.15)
            scale_w = random.uniform(0.85, 1.15)
            rescale_flag = random.random() < 0.4 if reshape_pool is not None else False
            for i in range(len(frames_np_rgba)):
                frame_img = frames_np_rgba[i]
                canvas_img = canvas_2d[i]
                mask = canvas_img != 0
                frame_img[:, :, :3][mask] = canvas_img[mask]
                # Ensure alpha is opaque wherever 2D pixels were drawn; otherwise RGBA->RGB
                # conversion can blend them away (most noticeable on face/hands).
                if frame_img.shape[2] == 4:
                    opaque = mask.any(axis=2)
                    frame_img[:, :, 3][opaque] = 255
                frames_np_rgba[i] = frame_img
                if aug_2d:
                    if rescale_flag:
                        frames_np_rgba[i] = scale_image_hw_keep_size(
                            frames_np_rgba[i], scale_h, scale_w
                        )
                    if reshape_pool is not None:
                        # 4%的概率完全消除某些帧
                        if random.random() < 0.04:
                            frames_np_rgba[i][:, :, 0:3] = 0
        else:
            scale_h = random.uniform(0.85, 1.15)
            scale_w = random.uniform(0.85, 1.15)
            rescale_flag = random.random() < 0.4 if reshape_pool is not None else False
            for i in range(len(frames_np_rgba)):
                if aug_2d:
                    if rescale_flag:
                        frames_np_rgba[i] = scale_image_hw_keep_size(
                            frames_np_rgba[i], scale_h, scale_w
                        )
                    if reshape_pool is not None:
                        # 4%的概率完全消除某些帧
                        if random.random() < 0.04:
                            frames_np_rgba[i][:, :, 0:3] = 0

        return frames_np_rgba

    def _get_multi_result_from_est(self, candidate, score_result, det_result, H, W):
        nums, keys, locs = candidate.shape  # n 所有身体关键点数量，坐标
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        subset_score = score_result[:, :24]  # 按照24个骨骼关键点来区分可见位置
        face_score = score_result[:, 24:92]
        hand_score = score_result[:, 92:113]
        hand_score = np.vstack([hand_score, score_result[:, 113:]])

        body_candidate = candidate[:, :24].copy()  # body(n, 24, 2)
        for i in range(len(subset_score)):  # n 个
            for j in range(len(subset_score[i])):
                if subset_score[i][j] > 0.3:
                    subset_score[i][
                        j
                    ] = j  # 标注序号，这样后续用的时候可以快速查出可用点
                else:
                    subset_score[i][j] = -1  # 躯干中去除掉不可见的骨骼

        un_visible = score_result < 0.3
        candidate[un_visible] = -1  # 全部关键点中去掉不可见骨骼

        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]  # hands(2*n, 21, 2)
        hands = np.vstack([hands, candidate[:, 113:]])

        bodies = dict(candidate=body_candidate, subset=subset_score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)
        score = dict(
            body_score=subset_score, hand_score=hand_score, face_score=face_score
        )

        new_det_result = []
        for bbox in det_result:
            x1, y1, x2, y2 = bbox
            new_x1 = x1 / W
            new_y1 = y1 / H
            new_x2 = x2 / W
            new_y2 = y2 / H
            new_bbox = [new_x1, new_y1, new_x2, new_y2]
            new_det_result.append(new_bbox)

        return pose, score, new_det_result

    def detect_poses(self, input_image: np.ndarray):
        candidate, subset, det_result = self.dw_pose_estimation(
            input_image.copy(), return_keypoints_scores=True
        )
        return self._get_multi_result_from_est(
            candidate, subset, det_result, input_image.shape[0], input_image.shape[1]
        )

    def _intrinsic_matrix_from_field_of_view(
        self, imshape, fov_degrees: float = 55
    ):  # nlf default fov_degrees 55
        imshape = np.array(imshape)
        fov_radians = fov_degrees * np.array(np.pi / 180)
        larger_side = np.max(imshape)
        focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
        # intrinsic_matrix 3*3
        return np.array(
            [
                [focal_length, 0, imshape[1] / 2],
                [0, focal_length, imshape[0] / 2],
                [0, 0, 1],
            ]
        )

    def process(
        self,
        input_image: InputImage,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        background="black",
        **kwargs,
    ) -> OutputImage:
        input_image = self._load_image(input_image)
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        # Prefer explicit `background=` argument; fall back to kwargs only if caller didn't pass it.
        bg_arg = (
            background if background is not None else kwargs.get("background", None)
        )
        bg_mode, bg_color = _parse_background(
            bg_arg, kwargs.get("background_color", None)
        )
        input_image, _ = resize_image_with_pad(
            input_image, detect_resolution, upscale_method, skip_hwc3=True
        )
        poses, score, det_result = self.process_dwpose(input_image)
        nlf_poses = self.process_nlf(input_image, det_result)
        target_H, target_W = input_image.shape[:2]
        ori_camera_pose = self._intrinsic_matrix_from_field_of_view(
            [target_H, target_W]
        )

        frames_np_rgba = self.render_nlf_as_images(
            [nlf_poses], [poses], reshape_pool=None, intrinsic_matrix=ori_camera_pose
        )[0]
        image, remove_pad = resize_image_with_pad(
            frames_np_rgba, detect_resolution, upscale_method, skip_hwc3=True
        )
        if bg_mode == "source":
            bg_img, bg_remove_pad = resize_image_with_pad(
                input_image, detect_resolution, upscale_method, skip_hwc3=True
            )
            detected_map = _to_rgb_uint8(
                remove_pad(image), background_rgb=bg_remove_pad(bg_img)
            )
        else:
            detected_map = _to_rgb_uint8(remove_pad(image), background_color=bg_color)
        detected_map = Image.fromarray(detected_map)
        return detected_map

    def process_video(
        self,
        input_video,
        detect_resolution=512,
        upscale_method="INTER_CUBIC",
        background="black",
        **kwargs,
    ):
        """
        Efficient iterator-based video processing for DWPose+NLF.

        Keeps the yield/iterator contract from BasePreprocessor.process_video(), but batches
        NLF inference across frames/bboxes (same approach as SCAIL process_pose.py / process_video_nlf).
        """
        # Check if input is already an iterator/generator
        if hasattr(input_video, "__iter__") and not isinstance(
            input_video, (list, str)
        ):
            frames = input_video
            total_frames = kwargs.get("total_frames", None)
        else:
            frames = self._load_video(input_video, **kwargs)
            total_frames = len(frames)

        progress_callback = kwargs.get("progress_callback", None)
        ensure_image_size = kwargs.get("ensure_image_size", True)
        # Prefer explicit `background=` argument; fall back to kwargs only if caller didn't pass it.
        bg_arg = (
            background if background is not None else kwargs.get("background", None)
        )
        bg_mode, bg_color = _parse_background(
            bg_arg, kwargs.get("background_color", None)
        )

        # NLF batching params
        batch_size = int(kwargs.get("batch_size", 64))
        render_batch_size = int(kwargs.get("render_batch_size", 8))

        frame_records: Dict[int, Dict[str, Any]] = {}
        completed_frames: Dict[int, bool] = {}

        # For streaming yield in-order
        next_frame_to_enqueue = 0
        next_frame_to_yield = 0

        # NLF buffer state
        buffer: Optional[torch.Tensor] = None
        buffer_count = 0
        sample_map = []  # [(frame_idx, bbox_idx), ...] for each buffer row
        affected_frames_in_buffer = set()

        # Render queue (in-order frames ready to render)
        render_queue = []

        def _mark_completed(frame_idx: int):
            completed_frames[frame_idx] = True

        def _try_enqueue_ready_frames():
            nonlocal next_frame_to_enqueue
            while completed_frames.get(next_frame_to_enqueue, False):
                render_queue.append(next_frame_to_enqueue)
                next_frame_to_enqueue += 1

        def _flush_nlf_buffer():
            nonlocal buffer_count, buffer, sample_map, affected_frames_in_buffer
            if buffer_count <= 0:
                return
            frame_batch = buffer[:buffer_count].permute(0, 3, 1, 2)
            pred = self.nlf_model.detect_smpl_batched(frame_batch)
            if isinstance(pred, dict) and "joints3d_nonparam" in pred:
                out_list = pred["joints3d_nonparam"]
            else:
                out_list = [None] * buffer_count

            # Assign outputs back to their frames
            for i, (fidx, bidx) in enumerate(sample_map[:buffer_count]):
                out = out_list[i]
                # Keep the same shape semantics as process_pose.py: nlfpose[bbox_idx][detect_idx]
                frame_records[fidx]["data"]["nlfpose"][bidx] = (
                    out if out is not None else []
                )
                frame_records[fidx]["pending"] -= 1

            # Mark frames completed when all bboxes are resolved
            for fidx in list(affected_frames_in_buffer):
                if frame_records[fidx]["pending"] <= 0:
                    _mark_completed(fidx)

            # Reset buffer
            buffer.zero_()
            buffer_count = 0
            sample_map = []
            affected_frames_in_buffer = set()

        def _flush_render_queue(force: bool = False):
            nonlocal next_frame_to_yield
            if not render_queue:
                return
            if (not force) and len(render_queue) < render_batch_size:
                return

            batch = (
                render_queue[:render_batch_size] if not force else list(render_queue)
            )
            del render_queue[: len(batch)]

            data_batch = [frame_records[i]["data"] for i in batch]
            poses_batch = [frame_records[i]["poses"] for i in batch]
            H = int(data_batch[0]["video_height"])
            W = int(data_batch[0]["video_width"])
            intrinsic_matrix = self._intrinsic_matrix_from_field_of_view((H, W))

            frames_np_rgba = self.render_nlf_as_images(
                data_batch,
                poses_batch,
                reshape_pool=None,
                intrinsic_matrix=intrinsic_matrix,
                draw_2d=True,
                aug_2d=False,
                aug_cam=False,
            )

            for out_idx, frame_img_rgba in enumerate(frames_np_rgba):
                fidx = batch[out_idx]
                image, remove_pad = resize_image_with_pad(
                    frame_img_rgba, detect_resolution, upscale_method, skip_hwc3=True
                )
                if bg_mode == "source":
                    bg_frame = frame_records[fidx].get("bg", None)
                    if bg_frame is None:
                        detected_map = _to_rgb_uint8(
                            remove_pad(image), background_color=(0, 0, 0)
                        )
                    else:
                        bg_img, bg_remove_pad = resize_image_with_pad(
                            bg_frame, detect_resolution, upscale_method, skip_hwc3=True
                        )
                        detected_map = _to_rgb_uint8(
                            remove_pad(image), background_rgb=bg_remove_pad(bg_img)
                        )
                else:
                    detected_map = _to_rgb_uint8(
                        remove_pad(image), background_color=bg_color
                    )
                detected_map = Image.fromarray(detected_map)
                if ensure_image_size:
                    detected_map = self._ensure_image_size(
                        detected_map, frame_records[fidx]["target_size"]
                    )

                if progress_callback is not None:
                    # Progress by yielded frames (matches the generator contract)
                    progress_callback(
                        next_frame_to_yield + 1,
                        total_frames if total_frames else next_frame_to_yield + 1,
                    )
                yield detected_map
                next_frame_to_yield += 1

        frame_idx = 0
        with torch.inference_mode():
            for frame in tqdm(frames, desc="Processing frames", total=total_frames):
                target_size = (
                    frame.size
                    if isinstance(frame, Image.Image)
                    else self._get_image_size(frame)
                )

                frame_np = frame
                if not isinstance(frame_np, np.ndarray):
                    frame_np = np.array(frame_np, dtype=np.uint8)

                frame_np, _ = resize_image_with_pad(
                    frame_np, detect_resolution, upscale_method, skip_hwc3=True
                )
                H, W = frame_np.shape[:2]

                poses, score, det_result = self.process_dwpose(frame_np)

                # Initialize record
                frame_records[frame_idx] = {
                    "target_size": target_size,
                    "poses": poses,
                    "bg": frame_np,
                    "data": {
                        "video_height": H,
                        "video_width": W,
                        "bboxes": det_result,
                        "nlfpose": [None for _ in range(len(det_result))],
                    },
                    "pending": len(det_result),
                }

                if len(det_result) == 0:
                    # No bboxes: mark complete; render will produce blank (render_whole handles empty)
                    frame_records[frame_idx]["data"]["nlfpose"] = []
                    frame_records[frame_idx]["pending"] = 0
                    _mark_completed(frame_idx)
                else:
                    # Allocate NLF buffer once we know H/W and dtype
                    if buffer is None or buffer.shape[1] != H or buffer.shape[2] != W:
                        buffer = torch.zeros(
                            (batch_size, H, W, 3),
                            dtype=torch.uint8,
                            device=self.device,
                        )

                    frame_t = torch.from_numpy(frame_np).to(
                        self.device, non_blocking=True
                    )

                    for bbox_idx, bbox in enumerate(det_result):
                        x1, y1, x2, y2 = bbox
                        x1_px = max(0, math.floor(x1 * W - W * 0.025))
                        y1_px = max(0, math.floor(y1 * H - H * 0.05))
                        x2_px = min(W, math.ceil(x2 * W + W * 0.025))
                        y2_px = min(H, math.ceil(y2 * H + H * 0.05))

                        cropped_region = frame_t[y1_px:y2_px, x1_px:x2_px, :]
                        buffer[buffer_count, y1_px:y2_px, x1_px:x2_px, :] = (
                            cropped_region
                        )

                        sample_map.append((frame_idx, bbox_idx))
                        affected_frames_in_buffer.add(frame_idx)
                        buffer_count += 1

                        if buffer_count == batch_size:
                            _flush_nlf_buffer()
                            _try_enqueue_ready_frames()
                            # Render queue flush (yield what we can)
                            yield from _flush_render_queue(force=False)

                # After ingesting this frame, try to enqueue any completed frames and render in batches
                _try_enqueue_ready_frames()
                yield from _flush_render_queue(force=False)

                frame_idx += 1

            # Final flushes
            _flush_nlf_buffer()
            _try_enqueue_ready_frames()
            yield from _flush_render_queue(force=True)

        if progress_callback is not None:
            progress_callback(next_frame_to_yield, next_frame_to_yield)

    @torch.inference_mode()
    def process_dwpose(
        self,
        input_image: InputImage,
    ):
        poses = self.detect_poses(input_image)
        return poses

    def recollect_nlf(self, item):
        new_item = item.copy()
        if len(item["bboxes"]) > 0:
            new_item["bboxes"] = item["bboxes"][:1]
            new_item["nlfpose"] = item["nlfpose"][:1]
        return new_item

    def recollect_dwposes(self, pose):
        new_pose = pose.copy()
        for i in range(1):
            bodies = pose["bodies"]
            faces = pose["faces"][i : i + 1]
            hands = pose["hands"][2 * i : 2 * i + 2]
            candidate = bodies["candidate"][
                i : i + 1
            ]  # candidate是所有点的坐标和置信度
            subset = bodies["subset"][i : i + 1]  # subset是认为的有效点
            new_pose = {
                "bodies": {"candidate": candidate, "subset": subset},
                "faces": faces,
                "hands": hands,
            }
        return new_pose

    @torch.inference_mode()
    def process_nlf(
        self,
        input_image: np.ndarray,
        bbox_list: List[Tuple[int, int, int, int]],
        **kwargs,
    ):
        height, width = input_image.shape[:2]
        input_image = torch.from_numpy(input_image)
        result_list = []
        batch_size = kwargs.get("batch_size", 64)
        buffer = torch.zeros(
            (batch_size, height, width, 3), dtype=input_image.dtype, device=self.device
        )
        buffer_count = 0
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox
            x1_px = max(0, math.floor(x1 * width - width * 0.025))
            y1_px = max(0, math.floor(y1 * height - height * 0.05))
            x2_px = min(width, math.ceil(x2 * width + width * 0.025))
            y2_px = min(height, math.ceil(y2 * height + height * 0.05))
            cropped_region = input_image[y1_px:y2_px, x1_px:x2_px, :]
            buffer[buffer_count, y1_px:y2_px, x1_px:x2_px, :] = cropped_region
            buffer_count += 1
            if buffer_count == batch_size:
                frame_batch = buffer.permute(0, 3, 1, 2)
                pred = self.nlf_model.detect_smpl_batched(frame_batch)
                if "joints3d_nonparam" in pred:
                    result_list.extend(pred["joints3d_nonparam"])
                else:
                    result_list.extend([None] * buffer_count)
                buffer.zero_()
                buffer_count = 0

        if buffer_count > 0:
            frame_batch = buffer[:buffer_count].permute(0, 3, 1, 2)
            pred = self.nlf_model.detect_smpl_batched(frame_batch)
            if "joints3d_nonparam" in pred:
                result_list.extend(pred["joints3d_nonparam"])
            else:
                result_list.extend([None] * buffer_count)
        return self.recollect_nlf(
            {
                "video_height": height,
                "video_width": width,
                "bboxes": bbox_list,
                "nlfpose": result_list,
            }
        )
