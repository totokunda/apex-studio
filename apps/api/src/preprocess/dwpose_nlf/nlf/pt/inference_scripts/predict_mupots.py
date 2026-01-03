import vtk  # noqa: F401

"""separator"""
import argparse
import functools

import cameralib
import numpy as np
import poseviz
import simplepyutils as spu
import torch
import torchvision  # noqa: F401
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS
import nlf.matlabfile
from nlf.pt.inference_scripts.predict_tdpw import (
    get_joint_info,
    precuda,
    to_np,
    iter_frame_batches,
)


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=128)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--clahe', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)


def main():
    initialize()
    model = torch.jit.load(FLAGS.model_path)
    skeleton = 'mpi_inf_3dhp_17'
    ji3d = get_joint_info(model, skeleton)

    inds = model.per_skeleton_indices[skeleton]
    cano_points = model.crop_model.canonical_locs()[inds]

    predict_fn = functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        num_aug=FLAGS.num_aug, detector_threshold=0.2, detector_nms_iou_threshold=0.7,
        detector_flip_aug=True, antialias_factor=2, suppress_implausible_poses=False,
        weights=model.get_weights_for_canonical_points(cano_points))

    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, world_up=(0, -1, 0), downscale=4,
        queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    image_relpaths_all = []
    poses_all = []
    intrinsic_matrices = spu.load_json(f'{DATA_ROOT}/mupots/camera_intrinsics.json')

    for i_seq in range(1, 21):
        if FLAGS.viz:
            viz.reinit_camera_view()
            if FLAGS.out_video_dir:
                viz.new_sequence_output(f'{FLAGS.out_video_dir}/TS{i_seq}.mp4', fps=25)

        annotations = nlf.matlabfile.load(
            f'{DATA_ROOT}/mupots/TS{i_seq}/annot.mat')['annotations']
        camera = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrices[f'TS{i_seq}'], world_up=(0, -1, 0))
        frame_relpaths = [f'mupots/TS{i_seq}/img_{i:06d}.jpg' for i in range(annotations.shape[0])]
        frame_paths = [f'{DATA_ROOT}/{p}' for p in frame_relpaths]
        frames_cpu_gpu = precuda(iter_frame_batches(frame_paths, FLAGS.batch_size))
        poses_per_frames = predict_sequence(
            predict_fn, frames_cpu_gpu, len(frame_paths), camera, viz)

        for poses_of_frame, frame_relpath in zip(poses_per_frames, frame_relpaths):
            image_relpaths_all.extend([frame_relpath] * len(poses_of_frame))
            poses_all.extend(poses_of_frame)

    np.savez(
        FLAGS.output_path, image_path=np.stack(image_relpaths_all, axis=0),
        coords3d_pred_world=np.stack(poses_all, axis=0))

    if viz is not None:
        viz.close()


def predict_sequence(predict_fn, frames_cpu_gpu, n_frames, camera, viz):
    predict_fn = functools.partial(
        predict_fn, intrinsic_matrix=torch.from_numpy(camera.intrinsic_matrix[np.newaxis]).cuda(),
        distortion_coeffs=torch.from_numpy(camera.get_distortion_coeffs()[np.newaxis]).cuda())
    progbar = spu.progressbar(total=n_frames)
    poses_per_frame = []

    for frames_b_cpu, frames_b_gpu in frames_cpu_gpu:
        pred = predict_fn(frames_b_gpu)
        pred = to_np(pred)
        poses_per_frame.extend(pred['poses3d'])
        progbar.update(frames_b_gpu.shape[0])

        if viz is not None:
            for frame, boxes, poses3d in zip(
                    frames_b_cpu, pred['boxes'], pred['poses3d']):
                viz.update(frame, boxes, poses3d, camera)

    return poses_per_frame


if __name__ == '__main__':
    with torch.inference_mode():
        main()
