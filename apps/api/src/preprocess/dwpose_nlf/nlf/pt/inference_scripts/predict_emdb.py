import vtk  # noqa: F401

"""separator"""
import argparse
import more_itertools
import collections
import functools
import os.path as osp

import boxlib
import cameralib
import numpy as np
import poseviz
import simplepyutils as spu
import torch
import torchvision  # noqa: F401
from simplepyutils import FLAGS, logger
from nlf.pt.inference_scripts.predict_tdpw import (
    get_joint_info,
    precuda,
    to_np,
    ragged_concat,
    ragged_split,
    iter_frame_batches,
)

from nlf.paths import DATA_ROOT, PROJDIR


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=0)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--clahe', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)


def main():
    initialize()
    model = torch.jit.load(FLAGS.model_path)

    ji3d = get_joint_info(model, 'smpl_24')
    faces = np.load(f'{PROJDIR}/smpl_faces.npy')

    cano_verts = np.load(f'{PROJDIR}/canonical_verts/smpl.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints/smpl.npy')
    cano_all = torch.cat([torch.as_tensor(cano_verts), torch.as_tensor(cano_joints)], dim=0).to(
        dtype=torch.float32, device='cuda'
    )

    predict_fn = functools.partial(
        model.estimate_poses_batched,
        default_fov_degrees=55,
        internal_batch_size=FLAGS.internal_batch_size,
        num_aug=FLAGS.num_aug,
        antialias_factor=2,
        weights=model.get_weights_for_canonical_points(cano_all),
    )

    detect_fn = functools.partial(
        model.detector, threshold=0.2, nms_iou_threshold=0.7, flip_aug=True
    )

    viz = (
        poseviz.PoseViz(
            ji3d.names,
            ji3d.stick_figure_edges,
            body_model_faces=faces,
            max_pixels_per_frame=1920 * 1440,
        )
        if FLAGS.viz
        else None
    )

    all_emdb_pkl_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/emdb/**/*_data.pkl')
    emdb1_sequence_roots = [
        osp.dirname(p) for p in all_emdb_pkl_paths if spu.load_pickle(p)['emdb1']
    ]

    results_all = collections.defaultdict(list)

    for seq_root in emdb1_sequence_roots:
        seq_name = osp.basename(seq_root)
        logger.info(f'Predicting {seq_name}...')
        subj = seq_root.split('/')[-2]
        seq_data = spu.load_pickle(f'{seq_root}/{subj}_{seq_name}_data.pkl')
        frame_paths = [
            f'{seq_root}/images/{i_frame:05d}.jpg' for i_frame in range(seq_data['n_frames'])
        ]
        bboxes = seq_data['bboxes']['bboxes']
        bboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], axis=1).astype(
            np.float32
        )

        frames_cpu_gpu = precuda(iter_frame_batches(frame_paths, FLAGS.batch_size))

        if FLAGS.viz:
            viz.reinit_camera_view()
            if FLAGS.out_video_dir:
                viz.new_sequence_output(f'{FLAGS.out_video_dir}/{seq_name}.mp4', fps=30)

        results_seq = predict_sequence(
            predict_fn, detect_fn, frames_cpu_gpu, bboxes, len(frame_paths), viz
        )
        results_all[f'{subj}_{seq_name}'] = results_seq

    spu.dump_pickle(results_all, FLAGS.output_path)

    if FLAGS.viz:
        viz.close()


def predict_sequence(predict_fn, detect_fn, frames_cpu_gpu, bboxes, n_frames, viz):
    result_batches = dict(vertices=[], joints=[])

    frames_cpu_gpu = more_itertools.peekable(frames_cpu_gpu)
    batch_cpu, batch_gpu = frames_cpu_gpu.peek()

    camera = cameralib.Camera.from_fov(55, imshape=batch_gpu.shape[2:4])

    box_batches = more_itertools.chunked(bboxes, FLAGS.batch_size)

    for box_b, (frames_b_cpu, frames_b) in zip(
        spu.progressbar(box_batches, total=n_frames, step=FLAGS.batch_size), frames_cpu_gpu
    ):
        boxes_det = detect_fn(frames_b)
        boxes_det = [x[..., :4].cpu().numpy() for x in boxes_det]
        boxes_selected = select_boxes(box_b, boxes_det)
        boxes_b = [torch.from_numpy(b[torch.newaxis]).cuda() for b in boxes_selected]
        pred = predict_fn(frames_b, boxes_b)

        pred['poses3d'] = ragged_concat(
            [x[0, ...] for x in pred['poses3d']],
            [x[0, :, torch.newaxis] for x in pred['uncertainties']],
            dim=-1,
        )
        vertices_b, joints_b = ragged_split(pred['poses3d'], [6890, 24], dim=-2)
        vertices_b = to_np(vertices_b)
        joints_b = to_np(joints_b)

        result_batches['vertices'].append(vertices_b)
        result_batches['joints'].append(joints_b)

        if FLAGS.viz:
            for frame, box, vertices, joints in zip(
                frames_b_cpu, boxes_selected, vertices_b, joints_b
            ):
                viz.update(
                    frame,
                    box[np.newaxis],
                    poses=joints[np.newaxis, :, :3],
                    vertices=vertices[np.newaxis, :, :3],
                    camera=camera,
                )

    return {k: np.concatenate(v, axis=0) for k, v in result_batches.items()}


def select_boxes(boxes_gt_batch, boxes_det_batch):
    result = []
    for box_gt, boxes_det in zip(boxes_gt_batch, boxes_det_batch):
        if boxes_det.shape[0] == 0:
            result.append(box_gt)
            continue

        ious = [boxlib.iou(box_gt, box_det) for box_det in boxes_det]
        max_iou = np.max(ious)
        if max_iou > 0.1:
            result.append(boxes_det[np.argmax(ious)])
        else:
            result.append(box_gt)
    return np.array(result)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
