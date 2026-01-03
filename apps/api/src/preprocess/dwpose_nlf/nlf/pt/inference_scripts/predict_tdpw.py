import vtk  # noqa: F401

"""separator"""
import argparse
import functools
import glob
import itertools
import multiprocessing
import os.path as osp

import cameralib
import cv2
import more_itertools
import numpy as np
import poseviz
import rlemasklib
import scipy.optimize
import simplepyutils as spu
import torch
import torchvision  # noqa: F401
from posepile.joint_info import JointInfo
from simplepyutils import FLAGS, logger

from nlf.paths import DATA_ROOT, PROJDIR
from nlf.rendering import Renderer


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--default-fov', type=float, default=55)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--real-intrinsics', action=spu.argparse.BoolAction)
    parser.add_argument('--gtassoc', action=spu.argparse.BoolAction)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=64)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--testset-only', action=spu.argparse.BoolAction)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--clahe', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)


def main():
    initialize()
    logger.info('Loading model...')
    model = torch.jit.load(FLAGS.model_path)
    logger.info('Model loaded.')

    ji3d = get_joint_info(model, 'smpl_24')
    faces = np.load(f'{PROJDIR}/smpl_faces.npy')

    cano_verts = np.load(f'{PROJDIR}/canonical_verts/smpl.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints/smpl.npy')
    cano_all = torch.cat([torch.as_tensor(cano_verts), torch.as_tensor(cano_joints)], dim=0).to(
        dtype=torch.float32, device='cuda'
    )

    predict_fn = functools.partial(
        model.detect_poses_batched,
        default_fov_degrees=FLAGS.default_fov,
        detector_threshold=0.2,
        num_aug=FLAGS.num_aug,
        detector_nms_iou_threshold=0.7,
        internal_batch_size=FLAGS.internal_batch_size,
        antialias_factor=FLAGS.antialias_factor,
        suppress_implausible_poses=False,
        detector_flip_aug=True,
        weights=model.get_weights_for_canonical_points(cano_all),
    )

    seq_filepaths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    if FLAGS.testset_only:
        seq_filepaths = [p for p in seq_filepaths if spu.split_path(p)[-2] == 'test']
    seq_names = [osp.basename(p).split('.')[0] for p in seq_filepaths]

    ji2d = JointInfo(
        'nose,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,reye,leye,lear,rear',
        'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,lear-leye-nose-reye-rear',
    )
    viz = (
        poseviz.PoseViz(
            ji3d.names, ji3d.stick_figure_edges, body_model_faces=faces, resolution=(1920, 1080)
        )
        if FLAGS.viz
        else None
    )

    for pbar, (seq_name, seq_filepath) in spu.zip_progressbar(zip(seq_names, seq_filepaths)):
        pbar.set_description(seq_name)
        if FLAGS.viz:
            viz.reinit_camera_view()
            if FLAGS.out_video_dir:
                viz.new_sequence_output(f'{FLAGS.out_video_dir}/{seq_name}.mp4', fps=25)

        already_done_files = glob.glob(f'{FLAGS.output_path}/*/*.pkl')
        if any(seq_name in p for p in already_done_files):
            logger.info(f'{seq_name} has been processed already.')
            continue
        logger.info(f'Predicting {seq_name}...')
        frame_paths = spu.sorted_recursive_glob(
            f'{DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'
        )
        n_frames = len(frame_paths)
        poses2d_true = get_poses_3dpw(seq_name)
        frames_cpu_gpu = precuda(iter_frame_batches(frame_paths, FLAGS.batch_size))
        camera = get_3dpw_camera(seq_filepath) if FLAGS.real_intrinsics else None
        cameras = itertools.repeat(camera, times=n_frames) if camera is not None else None
        if not FLAGS.gtassoc:
            masks = spu.load_pickle(f'{DATA_ROOT}/3dpw-more/stcn-pred/{seq_name}.pkl')
        else:
            masks = None

        tracks = predict_sequence(
            predict_fn,
            frames_cpu_gpu,
            n_frames,
            poses2d_true,
            masks,
            ji2d,
            ji3d,
            viz,
            cameras,
        )
        save_result_file(seq_name, FLAGS.output_path, tracks)

    if viz is not None:
        viz.close()


def predict_sequence(
    predict_fn,
    frame_batches_cpu_gpu,
    n_frames,
    poses2d_true,
    masks,
    joint_info2d,
    joint_info3d,
    viz,
    cameras=None,
):
    n_tracks = poses2d_true.shape[1]
    prev_poses2d_pred_ordered = np.zeros((n_tracks, joint_info3d.n_joints, 2))
    tracks = [[] for _ in range(n_tracks)]
    if cameras is not None:
        camera_batches = more_itertools.chunked(cameras, FLAGS.batch_size)
    else:
        camera_batches = itertools.repeat(None)
    progbar = spu.progressbar(total=n_frames)
    i_frame = 0
    for (frames_cpu, frames_gpu), camera_batch in zip(frame_batches_cpu_gpu, camera_batches):
        if camera_batch is not None:
            extr = torch.as_tensor(
                np.stack([c.intrinsic_matrix for c in camera_batch], dtype=np.float32)
            )
            intr = torch.as_tensor(
                np.stack([c.get_extrinsic_matrix() for c in camera_batch], dtype=np.float32)
            )

            pred = predict_fn(frames_gpu, extrinsic_matrix=extr, intrinsic_matrix=intr)
        else:
            camera_batch = itertools.repeat(None)
            pred = predict_fn(frames_gpu)

        pred['poses3d'] = ragged_concat(
            pred['poses3d'], [x[..., torch.newaxis] for x in pred['uncertainties']], dim=-1
        )
        pred = to_np(pred)

        for frame, boxes, poses3d, poses2d, camera in zip(
            frames_cpu, pred['boxes'], pred['poses3d'], pred['poses2d'], camera_batch
        ):
            if FLAGS.gtassoc:
                poses3d_ordered, prev_poses2d_pred_ordered = associate_predictions(
                    poses3d,
                    poses2d[:, -24:],
                    poses2d_true[i_frame],
                    prev_poses2d_pred_ordered,
                    joint_info3d,
                    joint_info2d,
                )
            else:
                poses3d_ordered = associate_predictions_to_masks_mesh(
                    poses3d, frame.shape[:2], masks[i_frame], camera
                )

            for pose, track in zip(poses3d_ordered, tracks):
                if not np.any(np.isnan(pose)):
                    track.append((i_frame, pose))

            poses3d = np.array([t[-1][1] for t in tracks if t])
            if viz is not None:
                if camera is None:
                    camera = cameralib.Camera.from_fov(FLAGS.default_fov, frame.shape)

                # draw the frame number onto the frame image
                frame = cv2.putText(
                    frame,
                    f'{i_frame}',
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                viz.update(
                    frame,
                    boxes[:, :4],
                    poses3d[:, -24:, :3],
                    camera,
                    vertices=poses3d[:, :6890, :3],
                )
            progbar.update(1)
            i_frame += 1

    return tracks


def get_3dpw_camera(seq_filepath):
    intr = spu.load_pickle(seq_filepath)['cam_intrinsics']
    return cameralib.Camera(intrinsic_matrix=intr, world_up=[0, -1, 0])


def get_poses_3dpw(seq_name):
    seq_filepaths = glob.glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    filepath = next(p for p in seq_filepaths if osp.basename(p) == f'{seq_name}.pkl')
    seq = spu.load_pickle(filepath)
    return np.transpose(np.array(seq['poses2d']), [1, 0, 3, 2])  # [Frame, Track, Joint, Coord]


def pose2d_auc(pose2d_pred, pose2d_true, prev_pose2d_pred, joint_info3d, joint_info2d):
    pose2d_true = pose2d_true.copy()
    pose2d_true[pose2d_true[:, 2] < 0.2] = np.nan
    selected_joints = 'lsho,rsho,lelb,relb,lhip,rhip,lkne,rkne'.split(',')
    indices_true = [joint_info2d.ids[name] for name in selected_joints]
    indices_pred = [joint_info3d.ids[name] for name in selected_joints]
    size = np.linalg.norm(pose2d_pred[joint_info3d.ids.rsho] - pose2d_pred[joint_info3d.ids.lhip])
    dist = np.linalg.norm(pose2d_true[indices_true, :2] - pose2d_pred[indices_pred], axis=-1)
    if np.count_nonzero(~np.isnan(dist)) < 5:
        dist = np.linalg.norm(prev_pose2d_pred[indices_pred] - pose2d_pred[indices_pred], axis=-1)
    return np.nanmean(np.maximum(0, 1 - dist / size))


def associate_predictions(
    poses3d_pred, poses2d_pred, poses2d_true, prev_poses2d_pred_ordered, joint_info3d, joint_info2d
):
    auc_matrix = np.array(
        [
            [
                pose2d_auc(pose_pred, pose_true, prev_pose, joint_info3d, joint_info2d)
                for pose_pred in poses2d_pred
            ]
            for pose_true, prev_pose in zip(poses2d_true, prev_poses2d_pred_ordered)
        ]
    )

    true_indices, pred_indices = scipy.optimize.linear_sum_assignment(-auc_matrix)
    n_true_poses = len(poses2d_true)

    result = np.full((n_true_poses, 6890 + 24, 4), np.nan)

    poses2d_pred_ordered = np.array(prev_poses2d_pred_ordered).copy()
    for ti, pi in zip(true_indices, pred_indices):
        result[ti] = poses3d_pred[pi]
        poses2d_pred_ordered[ti] = poses2d_pred[pi]

    return result, poses2d_pred_ordered


def mask_iou(mask1, mask2):
    union = np.count_nonzero(np.logical_or(mask1, mask2))
    if union == 0:
        return 0
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    return intersection / union


@functools.lru_cache(1)
def get_renderer_cached():
    return Renderer()


def render(verts, faces, camera, seg, imshape):
    return get_renderer_cached().render(verts, camera, faces=faces, seg=seg, imshape=imshape)


@functools.lru_cache(1)
def get_render_fn_parallel(n_verts):
    vertex_subset = np.load(f'{DATA_ROOT}/body_models/smpl/vertex_subset_{n_verts}.npz')
    i_verts = vertex_subset['i_verts']
    faces = vertex_subset['faces']
    pool = multiprocessing.Pool(8)

    def _render(verts, camera, imshape):
        if len(verts) == 0:
            return np.zeros([0, *imshape[:2]], dtype=np.uint8)
        verts = verts[:, i_verts] / 1000
        results = pool.starmap(render, [(v, faces, camera, True, imshape) for v in verts])
        return np.stack(results)

    return _render


def associate_predictions_to_masks_mesh(
    poses3d_pred, frame_shape, masks, camera, n_points=6890 + 24, n_coords=4, iou_threshold=0
):
    masks = np.array([rlemasklib.decode(m) for m in masks])
    mask_shape = masks.shape[1:3]
    mask_size = np.array([mask_shape[1], mask_shape[0]], np.float32)
    frame_size = np.array([frame_shape[1], frame_shape[0]], np.float32)
    if camera is None:
        camera = cameralib.Camera.from_fov(55, frame_shape)
    scales = mask_size / frame_size
    camera_rescaled = camera.scale_output(scales, inplace=False)
    pose_masks = get_render_fn_parallel(n_verts=256)(
        poses3d_pred[:, :6890, :3], camera_rescaled, mask_shape
    )
    iou_matrix = np.array([[mask_iou(m1, m2) for m2 in pose_masks] for m1 in masks])
    true_indices, pred_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
    n_true_poses = len(masks)
    result = np.full((n_true_poses, n_points, n_coords), np.nan, dtype=np.float32)
    for ti, pi in zip(true_indices, pred_indices):
        if iou_matrix[ti, pi] >= iou_threshold:
            result[ti] = poses3d_pred[pi]
    return result


def complete_track(track, n_frames):
    track_dict = dict(track)
    results = []
    for i in range(n_frames):
        if i in track_dict:
            results.append(track_dict[i])
        elif results:
            # repeat last
            results.append(results[-1])
        else:
            # fill with nans
            results.append(np.full_like(track[0][1], fill_value=np.nan))
    return results


def save_result_file(seq_name, pred_dir, tracks):
    seq_filepaths = glob.glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    seq_path = next(p for p in seq_filepaths if osp.basename(p) == f'{seq_name}.pkl')
    rel_path = '/'.join(spu.split_path(seq_path)[-2:])
    out_path = f'{pred_dir}/{rel_path}'
    n_frames = len(glob.glob(f'{DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'))
    coords3d = np.array([complete_track(track, n_frames) for track in tracks]) / 1000
    coords3d = coords3d.astype(np.float32)
    verts, joints = np.split(coords3d, [6890], axis=-2)
    spu.dump_pickle(dict(jointPositions=joints, vertices=verts), out_path)


def get_joint_info(model, skeleton):
    joint_names = model.per_skeleton_joint_names[skeleton]
    edges = model.per_skeleton_joint_edges[skeleton]
    return JointInfo(joint_names, edges)


def iter_frame_batches(frame_paths, batch_size):
    for path_batch in more_itertools.chunked(frame_paths, batch_size):
        yield np.stack([cv2.imread(p)[..., ::-1] for p in path_batch])


def precuda(cpu_batches):
    for cur_cpu_batch in cpu_batches:
        cur_gpu_batch = torch.from_numpy(cur_cpu_batch).cuda(non_blocking=True).permute(0, 3, 1, 2)
        if cur_gpu_batch.dtype == torch.uint16:
            cur_gpu_batch = cur_gpu_batch.half().mul_(1.0 / 65536.0).nan_to_num_(posinf=1.0)
            cur_cpu_batch = cur_cpu_batch.view(torch.uint8)[..., 1::2]
            if FLAGS.viz:
                cur_cpu_batch = cur_cpu_batch.contiguous()
        elif cur_gpu_batch.dtype == torch.uint8:
            cur_gpu_batch = cur_gpu_batch.half().mul_(1.0 / 255.0)

        if FLAGS.clahe:
            import kornia
            yuv = kornia.color.rgb_to_yuv(cur_gpu_batch)
            y_clahe = kornia.enhance.equalization.equalize_clahe(
                yuv[:, 0], clip_limit=2.5, grid_size=(12, 12)
            )
            yuv[:, 0] = y_clahe
            cur_gpu_batch = kornia.color.yuv_to_rgb(yuv).clamp_(0, 1)

        yield cur_cpu_batch, cur_gpu_batch


def ragged_split(xs, sizes, dim):
    return zip(*[torch.split(x, sizes, dim=dim) for x in xs])


def ragged_concat(xs, ys, dim):
    return [torch.cat([x, y], dim=dim) for x, y in zip(xs, ys)]


def to_np(x):
    if isinstance(x, (list, tuple)):
        return [to_np(y) for y in x]

    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}

    return x.detach().cpu().numpy()


def nested_map(fn, xs):
    if isinstance(xs, (list, tuple)):
        return type(xs)(nested_map(fn, x) for x in xs)

    if isinstance(xs, dict):
        return {k: nested_map(fn, v) for k, v in xs.items()}

    return fn(xs)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
