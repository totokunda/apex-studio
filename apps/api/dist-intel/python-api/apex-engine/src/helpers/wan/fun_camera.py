import numpy as np
from typing import List, Union
import torch
from packaging import version as pver
from einops import rearrange
from src.helpers.helpers import helpers


class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""

    def __init__(self, entry: List[float]):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [
        target_cam_c2w,
    ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


@helpers("wan.fun_camera")
class WanFunCamera:

    def read_camera_poses(self, path: str):
        with open(path, "r") as f:
            lines = f.readlines()
        # remove empty lines
        lines = [line for line in lines if line.strip()]
        poses = []
        for line in lines:
            entry = list(map(float, line.strip().split(" ")))
            poses.append(Camera(entry))
        return poses

    def __call__(
        self,
        poses: Union[List[Camera], str, List[float]],
        H: int,
        W: int,
        device: torch.device,
        original_H: int = 720,
        original_W: int = 1280,
    ):
        if isinstance(poses, str):
            poses = self.read_camera_poses(poses)
        elif isinstance(poses, list) and isinstance(poses[0], float):
            poses = [Camera(pose) for pose in poses]
        elif isinstance(poses, list) and isinstance(poses[0], Camera):
            pass
        else:
            raise ValueError(f"Invalid poses type: {type(poses)}")

        sample_wh_ratio = W / H
        pose_wh_ratio = (
            original_W / original_H
        )  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = H * pose_wh_ratio
            for cam_param in poses:
                cam_param.fx = resized_ori_w * cam_param.fx / W
        else:
            resized_ori_h = W / pose_wh_ratio
            for cam_param in poses:
                cam_param.fy = resized_ori_h * cam_param.fy / H

        intrinsic = np.asarray(
            [
                [cam_param.fx * W, cam_param.fy * H, cam_param.cx * W, cam_param.cy * H]
                for cam_param in poses
            ],
            dtype=np.float32,
        )

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(poses)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = (
            ray_condition(K, c2ws, H, W, device=device)[0]
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding.permute([3, 0, 1, 2]).unsqueeze(0)

    def __str__(self):
        return f"CameraPreprocessor(model_path={self.model_path}, preprocessor_path={self.preprocessor_path}, model_config_path={self.model_config_path}, model_config={self.model_config}, save_path={self.save_path}, config_save_path={self.config_save_path}, processor_class={self.processor_class}, model_class={self.model_class}, dtype={self.dtype})"

    def __repr__(self):
        return self.__str__()
