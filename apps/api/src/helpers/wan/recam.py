import json
import numpy as np
import torch
from einops import rearrange
from src.helpers.helpers import helpers


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


@helpers("wan.recam")
class WanRecam:
    def __init__(self, cam_type: int = 1):
        self.cam_type = cam_type

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split("] [")
        matrix = []
        for row in rows:
            row = row.replace("[", "").replace("]", "")
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
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

    def __call__(
        self, camera_extrinsics_path: str, num_frames: int, cam_type: int = None
    ):
        if cam_type is not None:
            self.cam_type = cam_type
        with open(camera_extrinsics_path, "r") as file:
            cam_data = json.load(file)

        cam_idx = list(range(num_frames))[::4]
        traj = [
            self.parse_matrix(cam_data[f"frame{idx}"][f"cam{int(self.cam_type):02d}"])
            for idx in cam_idx
        ]
        traj = np.stack(traj).transpose(0, 2, 1)
        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.0
            c2w[:3, 3] /= 100
            c2ws.append(c2w)
        tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
        relative_poses = []
        for i in range(len(tgt_cam_params)):
            relative_pose = self.get_relative_pose(
                [tgt_cam_params[0], tgt_cam_params[i]]
            )
            relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
        pose_embedding = torch.stack(relative_poses, dim=0)  # 21x3x4
        pose_embedding = rearrange(pose_embedding, "b c d -> b (c d)")
        return pose_embedding.to(torch.bfloat16)
