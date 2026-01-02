import numpy as np
import cv2
import src.preprocess.dwpose_nlf.draw_util as util
import torch


def _project_points_pinhole_xyz_to_uv(
    xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        xyz: (N, 3) float array in camera coordinates (X, Y, Z).
    Returns:
        uv: (N, 2) float array in pixel coords (u, v)
        z:  (N,) float array depth Z
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N,3), got {xyz.shape}")
    z = xyz[:, 2].astype(np.float32, copy=False)
    eps = 1e-6
    valid = z > eps
    uv = np.full((xyz.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        x = xyz[valid, 0].astype(np.float32, copy=False)
        y = xyz[valid, 1].astype(np.float32, copy=False)
        zv = z[valid]
        uv[valid, 0] = fx * (x / zv) + cx
        uv[valid, 1] = fy * (y / zv) + cy
    return uv, z


def process_data_to_COCO_format(joints):
    """Args:
        joints: numpy array of shape (24, 2) or (24, 3)
    Returns:
        new_joints: numpy array of shape (17, 2) or (17, 3)
    """
    if joints.ndim != 2:
        raise ValueError(f"Expected shape (24,2) or (24,3), got {joints.shape}")

    dim = joints.shape[1]  # 2D or 3D

    mapping = {
        15: 0,  # head
        12: 1,  # neck
        17: 2,  # left shoulder
        16: 5,  # right shoulder
        19: 3,  # left elbow
        18: 6,  # right elbow
        21: 4,  # left hand
        20: 7,  # right hand
        2: 8,  # left pelvis
        1: 11,  # right pelvis
        5: 9,  # left knee
        4: 12,  # right knee
        8: 10,  # left feet
        7: 13,  # right feet
    }

    new_joints = np.zeros((18, dim), dtype=joints.dtype)
    for src, dst in mapping.items():
        new_joints[dst] = joints[src]

    return new_joints


def get_single_pose_cylinder_specs(args):
    """渲染单个pose的辅助函数，用于并行处理"""
    idx, pose, focal, princpt, height, width, colors, limb_seq, draw_seq = args
    cylinder_specs = []

    for joints3d in pose:  # 多人
        joints3d = joints3d.cpu().numpy()
        joints3d = process_data_to_COCO_format(joints3d)
        for line_idx in draw_seq:
            line = limb_seq[line_idx]
            start, end = line[0], line[1]
            if np.sum(joints3d[start]) == 0 or np.sum(joints3d[end]) == 0:
                continue
            else:
                cylinder_specs.append(
                    (joints3d[start], joints3d[end], colors[line_idx])
                )
    return cylinder_specs


def draw_pose(
    pose,
    H,
    W,
    show_feet=False,
    show_body=True,
    show_hand=True,
    show_face=True,
    show_cheek=False,
    dw_bgr=False,
    dw_hand=False,
    aug_body_draw=False,
    optimized_face=False,
):
    final_canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for i in range(len(pose["bodies"]["candidate"])):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        bodies = pose["bodies"]
        faces = pose["faces"][i : i + 1]
        hands = pose["hands"][2 * i : 2 * i + 2]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i : i + 1]  # subset是认为的有效点

        if show_body:
            if len(subset[0]) <= 18 or show_feet == False:
                if aug_body_draw:
                    raise NotImplementedError("aug_body_draw is not implemented yet")
                else:
                    canvas = util.draw_bodypose(canvas, candidate, subset)
            else:
                canvas = util.draw_bodypose_with_feet(canvas, candidate, subset)
            if dw_bgr:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if show_cheek:
            assert (
                show_body == False
            ), "show_cheek and show_body cannot be True at the same time"
            canvas = util.draw_bodypose_augmentation(
                canvas,
                candidate,
                subset,
                drop_aug=True,
                shift_aug=False,
                all_cheek_aug=True,
            )
        if show_hand:
            if not dw_hand:
                canvas = util.draw_handpose_lr(canvas, hands)
            else:
                canvas = util.draw_handpose(canvas, hands)
        if show_face:
            canvas = util.draw_facepose(canvas, faces, optimized_face=optimized_face)
        final_canvas = final_canvas + canvas
    return final_canvas


def draw_pose_to_canvas_np(
    poses,
    pool,
    H,
    W,
    reshape_scale,
    show_feet_flag=False,
    show_body_flag=True,
    show_hand_flag=True,
    show_face_flag=True,
    show_cheek_flag=False,
    dw_bgr=False,
    dw_hand=False,
    aug_body_draw=False,
):
    canvas_np_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        canvas = draw_pose(
            pose,
            H,
            W,
            show_feet_flag,
            show_body_flag,
            show_hand_flag,
            show_face_flag,
            show_cheek_flag,
            dw_bgr,
            dw_hand,
            aug_body_draw,
            optimized_face=True,
        )
        canvas_np_lst.append(canvas)
    return canvas_np_lst


def collect_smpl_poses(data):
    uncollected_smpl_poses = [item["nlfpose"] for item in data]
    smpl_poses = [[] for _ in range(len(uncollected_smpl_poses))]
    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(
            len(uncollected_smpl_poses[frame_idx])
        ):  # 每个人（每个bbox）只给出一个pose
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:  # 有返回的骨骼
                smpl_poses[frame_idx].append(
                    uncollected_smpl_poses[frame_idx][person_idx][0]
                )
            else:
                smpl_poses[frame_idx].append(
                    torch.zeros((24, 3), dtype=torch.float32)
                )  # 没有检测到人，就放一个全0的

    return smpl_poses


def scale_image_hw_keep_size(img, scale_h, scale_w):
    """分别按 scale_h, scale_w 缩放图像，保持输出尺寸不变。"""
    H, W = img.shape[:2]
    new_H, new_W = int(H * scale_h), int(W * scale_w)
    scaled = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    result = np.zeros_like(img)

    # 计算在目标图上的放置范围
    # --- Y方向 ---
    if new_H >= H:
        y_start_src = (new_H - H) // 2
        y_end_src = y_start_src + H
        y_start_dst = 0
        y_end_dst = H
    else:
        y_start_src = 0
        y_end_src = new_H
        y_start_dst = (H - new_H) // 2
        y_end_dst = y_start_dst + new_H

    # --- X方向 ---
    if new_W >= W:
        x_start_src = (new_W - W) // 2
        x_end_src = x_start_src + W
        x_start_dst = 0
        x_end_dst = W
    else:
        x_start_src = 0
        x_end_src = new_W
        x_start_dst = (W - new_W) // 2
        x_end_dst = x_start_dst + new_W

    # 将 scaled 映射到 result
    result[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = scaled[
        y_start_src:y_end_src, x_start_src:x_end_src
    ]

    return result


def flatten_specs(specs_list):
    """把 specs_list 拉平为 numpy 数组 + 索引表"""
    starts, ends, colors = [], [], []
    frame_offset, frame_count = [], []
    offset = 0
    for specs in specs_list:
        frame_offset.append(offset)
        frame_count.append(len(specs))
        for s, e, c in specs:
            starts.append(s)
            ends.append(e)
            colors.append(c)
        offset += len(specs)
    return (
        np.array(starts, dtype=np.float32),
        np.array(ends, dtype=np.float32),
        np.array(colors, dtype=np.float32),
        np.array(frame_offset, dtype=np.int32),
        np.array(frame_count, dtype=np.int32),
    )


def render_whole(specs_list, H=480, W=640, fx=500, fy=500, cx=240, cy=320, radius=21.5):
    """
    Fast Taichi-free renderer.

    Instead of ray-marching SDF cylinders per pixel, we:
    - project each 3D limb segment into 2D with the provided intrinsics
    - draw thick anti-aliased lines (OpenCV C++ fast path)
    - alpha composite in depth order (far -> near) for a decent occlusion approximation

    This is typically *much* faster on CPU than pure-Python rasterization and avoids any
    extra external dependency beyond the already-imported `cv2`.
    """
    n_frames = len(specs_list)
    # Handle empty specs (e.g., no detected people in the whole batch)
    if n_frames == 0:
        return []

    # Find global depth range across the whole batch (to mimic the original depth fade).
    z_values = []
    for specs in specs_list:
        for s, e, _c in specs:
            s = np.asarray(s)
            e = np.asarray(e)
            if s.shape[0] >= 3:
                z_values.append(float(s[2]))
            if e.shape[0] >= 3:
                z_values.append(float(e[2]))
    if len(z_values) == 0:
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(n_frames)]

    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    znear = 0.1
    depth_near = max(z_min, znear)
    depth_far = min(z_max + 6000.0, 20000.0)
    if depth_far <= depth_near + 1e-6:
        depth_far = depth_near + 1.0

    max_thickness = max(2, int(round(0.06 * max(H, W))))

    frames_np_rgba: list[np.ndarray] = []

    # Reusable mask to avoid allocations in the inner loop.
    mask = np.zeros((H, W), dtype=np.uint8)

    for specs in specs_list:
        out_rgb = np.zeros((H, W, 3), dtype=np.float32)  # 0..1
        out_a = np.zeros((H, W), dtype=np.float32)  # 0..1

        if len(specs) == 0:
            frames_np_rgba.append(np.zeros((H, W, 4), dtype=np.uint8))
            continue

        # Vectorize projection + sort by depth (far -> near).
        starts = np.asarray([s for (s, _e, _c) in specs], dtype=np.float32)
        ends = np.asarray([e for (_s, e, _c) in specs], dtype=np.float32)
        cols = np.asarray([c for (_s, _e, c) in specs], dtype=np.float32)
        if cols.ndim == 1:
            cols = cols[None, :]
        if cols.shape[1] == 3:
            alpha_col = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols = np.concatenate([cols, alpha_col], axis=1)
        elif cols.shape[1] < 4:
            # Pad up to RGBA
            pad = np.zeros((cols.shape[0], 4 - cols.shape[1]), dtype=np.float32)
            cols = np.concatenate([cols, pad], axis=1)

        uv0, z0 = _project_points_pinhole_xyz_to_uv(starts, fx, fy, cx, cy)
        uv1, z1 = _project_points_pinhole_xyz_to_uv(ends, fx, fy, cx, cy)

        z_mean = 0.5 * (z0 + z1)
        # Skip segments behind the camera / too close (matches the original znear behavior).
        valid = (z0 > znear) & (z1 > znear) & np.isfinite(uv0).all(axis=1) & np.isfinite(
            uv1
        ).all(axis=1)
        if not np.any(valid):
            frames_np_rgba.append(np.zeros((H, W, 4), dtype=np.uint8))
            continue

        order = np.argsort(z_mean[valid])[::-1]  # far -> near
        valid_idx = np.nonzero(valid)[0][order]

        for i in valid_idx:
            u0, v0 = float(uv0[i, 0]), float(uv0[i, 1])
            u1, v1 = float(uv1[i, 0]), float(uv1[i, 1])
            zm = float(z_mean[i])
            if zm <= znear:
                continue

            # Thickness (world-space radius -> pixels). Clamp to avoid extreme sizes.
            r_px = int(round(radius * (0.5 * (fx + fy)) / max(zm, znear)))
            thickness = int(np.clip(r_px, 1, max_thickness))

            # Depth fade similar to the Taichi version
            depth_factor = 1.0 - (zm - depth_near) / (depth_far - znear)
            depth_factor = float(np.clip(depth_factor, 0.0, 1.0))

            rgb = cols[i, :3].astype(np.float32, copy=False)
            a_src = float(np.clip(cols[i, 3], 0.0, 1.0))
            rgb = np.clip(rgb * depth_factor, 0.0, 1.0)

            # Draw mask for this segment
            mask[:] = 0
            cv2.line(
                mask,
                (int(round(u0)), int(round(v0))),
                (int(round(u1)), int(round(v1))),
                color=255,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue

            y0b, y1b = int(ys.min()), int(ys.max()) + 1
            x0b, x1b = int(xs.min()), int(xs.max()) + 1

            m = (mask[y0b:y1b, x0b:x1b].astype(np.float32) / 255.0)[..., None]  # (h,w,1)
            a = (a_src * m).astype(np.float32)  # (h,w,1)

            # "over" compositing on the ROI
            roi_rgb = out_rgb[y0b:y1b, x0b:x1b, :]
            roi_a = out_a[y0b:y1b, x0b:x1b][..., None]

            roi_rgb[:] = roi_rgb * (1.0 - a) + rgb[None, None, :] * a
            roi_a[:] = roi_a * (1.0 - a) + a

            out_rgb[y0b:y1b, x0b:x1b, :] = roi_rgb
            out_a[y0b:y1b, x0b:x1b] = roi_a[..., 0]

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = (np.clip(out_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgba[:, :, 3] = (np.clip(out_a, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames_np_rgba.append(rgba)

    return frames_np_rgba


def p3d_single_p2d(points, intrinsic_matrix):
    X, Y, Z = points[0], points[1], points[2]
    u = (intrinsic_matrix[0, 0] * X / Z) + intrinsic_matrix[0, 2]
    v = (intrinsic_matrix[1, 1] * Y / Z) + intrinsic_matrix[1, 2]
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    return np.array([u_np, v_np])


def shift_dwpose_according_to_nlf(
    smpl_poses, aligned_poses, ori_intrinstics, modified_intrinstics, height, width
):
    ########## warning: 会改变body； shift 之后 body是不准的 ##########
    for i in range(len(smpl_poses)):
        persons_joints_list = smpl_poses[i]
        poses_list = aligned_poses[i]
        # 对里面每一个人，取关节并进行变形；并且修改2d；如果3d不存在，把2d的手/脸也去掉
        for person_idx, person_joints in enumerate(persons_joints_list):
            face = poses_list["faces"][person_idx]
            right_hand = poses_list["hands"][2 * person_idx]
            left_hand = poses_list["hands"][2 * person_idx + 1]
            candidate = poses_list["bodies"]["candidate"][person_idx]
            # 注意，这里不是coco format
            person_joint_15_2d_shift = (
                p3d_single_p2d(person_joints[15], modified_intrinstics)
                - p3d_single_p2d(person_joints[15], ori_intrinstics)
                if person_joints[15, 2] > 0.01
                else np.array([0.0, 0.0])
            )  # face
            person_joint_20_2d_shift = (
                p3d_single_p2d(person_joints[20], modified_intrinstics)
                - p3d_single_p2d(person_joints[20], ori_intrinstics)
                if person_joints[20, 2] > 0.01
                else np.array([0.0, 0.0])
            )  # right hand
            person_joint_21_2d_shift = (
                p3d_single_p2d(person_joints[21], modified_intrinstics)
                - p3d_single_p2d(person_joints[21], ori_intrinstics)
                if person_joints[21, 2] > 0.01
                else np.array([0.0, 0.0])
            )  # left hand

            face[:, 0] += person_joint_15_2d_shift[0] / width
            face[:, 1] += person_joint_15_2d_shift[1] / height
            right_hand[:, 0] += person_joint_20_2d_shift[0] / width
            right_hand[:, 1] += person_joint_20_2d_shift[1] / height
            left_hand[:, 0] += person_joint_21_2d_shift[0] / width
            left_hand[:, 1] += person_joint_21_2d_shift[1] / height
            candidate[:, 0] += person_joint_15_2d_shift[0] / width
            candidate[:, 1] += person_joint_15_2d_shift[1] / height
