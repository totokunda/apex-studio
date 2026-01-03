import vtk  # noqa: F401

"""separator"""
import argparse
import functools

import h5py
import numpy as np
import posepile.ds.tdhp.main as tdhp_main
import poseviz
import simplepyutils as spu
import torch
import torchvision  # noqa: F401
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS
from nlf.pt.inference_scripts.predict_tdpw import (
    get_joint_info,
    precuda,
    to_np,
    ragged_concat,
    ragged_split,
    iter_frame_batches,
    nested_map
)

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
    skeleton = 'mpi_inf_3dhp_17'
    ji3d = get_joint_info(model, skeleton)

    inds = model.per_skeleton_indices[skeleton]
    cano_points = model.crop_model.canonical_locs()[inds]

    predict_fn = functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        num_aug=FLAGS.num_aug, detector_threshold=0, detector_flip_aug=True, antialias_factor=2,
        max_detections=1, suppress_implausible_poses=False,
        weights=model.get_weights_for_canonical_points(cano_points))

    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, world_up=(0, 1, 0), downscale=4,
        queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    image_relpaths_all = []
    coords_all = []
    cam1_4 = tdhp_main.get_test_camera_subj1_4()
    cam5_6 = tdhp_main.get_test_camera_subj5_6()

    for subj in range(1, 7):
        if FLAGS.viz:
            viz.reinit_camera_view()
            if FLAGS.out_video_dir:
                viz.new_sequence_output(f'{FLAGS.out_video_dir}/TS{subj}.mp4', fps=50)

        camera = cam1_4 if subj <= 4 else cam5_6
        with h5py.File(f'{DATA_ROOT}/3dhp/TS{subj}/annot_data.mat', 'r') as m:
            valid_frames = np.where(m['valid_frame'][:, 0])[0]

        frame_relpaths = [f'3dhp/TS{subj}/imageSequence/img_{i + 1:06d}.jpg' for i in valid_frames]
        frame_paths = [f'{DATA_ROOT}/{p}' for p in frame_relpaths]
        frames_cpu_gpu = precuda(iter_frame_batches(frame_paths, FLAGS.batch_size))

        coords3d_pred_world = predict_sequence(
            predict_fn, frames_cpu_gpu, len(frame_paths), camera, viz)
        image_relpaths_all.append(frame_relpaths)
        coords_all.append(coords3d_pred_world)

    np.savez(
        FLAGS.output_path, image_path=np.concatenate(image_relpaths_all, axis=0),
        coords3d_pred_world=np.concatenate(coords_all, axis=0))

    if FLAGS.viz:
        viz.close()


def predict_sequence(predict_fn, frames_cpu_gpu, n_frames, camera, viz):
    predict_fn = functools.partial(
        predict_fn,
        intrinsic_matrix=torch.from_numpy(camera.intrinsic_matrix[np.newaxis]).cuda(),
        distortion_coeffs=torch.from_numpy(camera.get_distortion_coeffs()[np.newaxis]).cuda(),
        extrinsic_matrix=torch.from_numpy(camera.get_extrinsic_matrix()[np.newaxis]).cuda(),
        world_up_vector=torch.from_numpy(camera.world_up).cuda())
    pose_batches = []

    for frames_b_cpu, frames_b in spu.progressbar(frames_cpu_gpu, total=n_frames, step=FLAGS.batch_size):
        pred = predict_fn(frames_b)
        pred = nested_map(lambda x: torch.squeeze(x, 0).cpu().numpy(), pred)
        pose_batches.append(pred['poses3d'])
        if FLAGS.viz:
            for frame, box, pose3d in zip(frames_b_cpu, pred['boxes'], pred['poses3d']):
                viz.update(frame, box[np.newaxis], pose3d[np.newaxis], camera)

    return np.concatenate(pose_batches, axis=0)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
