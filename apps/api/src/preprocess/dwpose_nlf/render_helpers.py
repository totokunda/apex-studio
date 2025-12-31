import numpy as np
import cv2
import src.preprocess.dwpose_nlf.draw_util as util
import torch
import taichi as ti


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
    img = ti.Vector.field(4, dtype=ti.f32, shape=(H, W))
    starts, ends, colors, frame_offset, frame_count = flatten_specs(specs_list)
    total_cyl = len(starts)
    n_frames = len(specs_list)
    # Handle empty specs (e.g., no detected people in the whole batch)
    if total_cyl == 0:
        return [np.zeros((H, W, 4), dtype=np.uint8) for _ in range(n_frames)]
    z_min = min(starts[:, 2].min(), ends[:, 2].min())
    z_max = max(starts[:, 2].max(), ends[:, 2].max())

    # ========= 相机内参 =========
    znear = 0.1
    zfar = max(min(z_max, 25000), 10000)
    C = ti.Vector([0.0, 0.0, 0.0])  # 相机中心
    light_dir = ti.Vector([0.0, 0.0, 1.0])

    c_start = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_end = ti.Vector.field(3, dtype=ti.f32, shape=total_cyl)
    c_rgba = ti.Vector.field(4, dtype=ti.f32, shape=total_cyl)
    n_cyl = ti.field(dtype=ti.i32, shape=())  # 实际数量
    f_offset = ti.field(dtype=ti.i32, shape=n_frames)
    f_count = ti.field(dtype=ti.i32, shape=n_frames)
    frame_id = ti.field(dtype=ti.i32, shape=())  # 当前帧号
    z_min_field = ti.field(dtype=ti.f32, shape=())
    z_max_field = ti.field(dtype=ti.f32, shape=())

    z_min_field[None] = z_min
    z_max_field[None] = z_max

    # # ====== 拷贝数据一次 ======
    c_start.from_numpy(starts)
    c_end.from_numpy(ends)
    c_rgba.from_numpy(colors)
    f_offset.from_numpy(frame_offset)
    f_count.from_numpy(frame_count)

    @ti.func
    def sd_cylinder(p, a, b, r):
        pa = p - a
        ba = b - a
        h = ba.norm()
        eps = 1e-8
        res = 0.0
        if h < eps:
            res = pa.norm() - r
        else:
            ba_n = ba / h
            proj = pa.dot(ba_n)
            proj_clamped = min(max(proj, 0.0), h)
            res = (pa - proj_clamped * ba_n).norm() - r
        return res

    @ti.func
    def scene_sdf(p):
        best_d = 1e6
        best_col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        fid = frame_id[None]  # 从 field 里读出来，变成一个普通 int
        off = f_offset[fid]
        cnt = f_count[fid]
        for i in range(cnt):  # 只遍历实际数量
            a = c_start[off + i]
            b = c_end[off + i]
            r = radius
            col = c_rgba[off + i]
            d = sd_cylinder(p, a, b, r)
            if d < best_d:
                best_d = d
                best_col = col
        return best_d, best_col

    @ti.func
    def get_normal(p):
        e = 1e-3
        dx = (
            scene_sdf(p + ti.Vector([e, 0.0, 0.0]))[0]
            - scene_sdf(p - ti.Vector([e, 0.0, 0.0]))[0]
        )
        dy = (
            scene_sdf(p + ti.Vector([0.0, e, 0.0]))[0]
            - scene_sdf(p - ti.Vector([0.0, e, 0.0]))[0]
        )
        dz = (
            scene_sdf(p + ti.Vector([0.0, 0.0, e]))[0]
            - scene_sdf(p - ti.Vector([0.0, 0.0, e]))[0]
        )
        n = ti.Vector([dx, dy, dz])
        return n.normalized()

    @ti.func
    def pixel_to_ray(xi, yi):
        u = (xi - cx) / fx
        v = (yi - cy) / fy
        dir_cam = ti.Vector([u, v, 1.0]).normalized()
        Rcw = ti.Matrix.identity(ti.f32, 3)
        rd_world = Rcw @ dir_cam
        ro_world = C
        return ro_world, rd_world

    @ti.kernel
    def render():
        depth_near, depth_far = ti.max(z_min_field[None], 0.1), ti.min(
            z_max_field[None] + 6000, 20000
        )  # 能渲染出来的点，最大12000
        for y, x in img:
            ro, rd = pixel_to_ray(x, y)
            t = znear
            col_out = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for _ in range(300):
                p = ro + rd * t
                d, col = scene_sdf(p)
                if d < 1e-3:
                    #     n = get_normal(p)
                    #     diff = max(n.dot(-light_dir), 0.0)
                    #     lit = 0.3 + 0.7 * diff
                    #     col_out = ti.Vector([col.x * lit, col.y * lit, col.z * lit, col.w])
                    #     break

                    n = get_normal(p)
                    diff = max(n.dot(-light_dir), 0.0)

                    # === Blinn-Phong 镜面反射 ===
                    view_dir = -rd.normalized()
                    half_dir = (view_dir + -light_dir).normalized()
                    spec = (
                        max(n.dot(half_dir), 0.0) ** 32
                    )  # shininess=32，越小越散，越大越锐

                    depth_factor = 1.0 - (p.z - depth_near) / (depth_far - znear)
                    depth_factor = ti.max(0.0, ti.min(1.0, depth_factor))

                    # 原来的 diffuse/ambient 光照
                    diffuse_term = 0.3 + 0.7 * diff
                    base = col.xyz * diffuse_term * depth_factor

                    # 镜面高光（叠加到原有结果上）
                    highlight = ti.Vector([1.0, 1.0, 1.0]) * (0.5 * spec) * depth_factor

                    col_out = ti.Vector(
                        [
                            base.x + highlight.x,
                            base.y + highlight.y,
                            base.z + highlight.z,
                            col.w,
                        ]
                    )
                    break

                if t > zfar:
                    break
                t += max(d, 1e-4)
            img[y, x] = col_out

    frames_np_rgba = []
    for f in range(len(specs_list)):
        # start_time = time.time()
        frame_id[None] = f
        render()
        arr = np.clip(img.to_numpy(), 0, 1)
        # end_time = time.time()
        # print(f"Frame {f} time: {end_time - start_time} seconds")
        arr8 = (arr * 255).astype(np.uint8)
        frames_np_rgba.append(arr8)

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
