import vtk  # noqa: F401

"""separator"""
import more_itertools
import argparse
import functools
import itertools

import numpy as np
import posepile.ds.h36m.main as h36m_main
import poseviz
import simplepyutils as spu
import spacepy
import torch
import torchvision  # noqa: F401
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS, logger
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
    parser.add_argument('--num-joints', type=int, default=17)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--frame-step', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=0)
    parser.add_argument('--correct-S9', action=spu.argparse.BoolAction, default=False)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--clahe', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)


def main():
    initialize()
    model = torch.jit.load(FLAGS.model_path)
    assert FLAGS.num_joints in (17, 25)
    skeleton = f'h36m_{FLAGS.num_joints}'
    ji3d = get_joint_info(model, skeleton)

    inds = model.per_skeleton_indices[skeleton]
    cano_points = model.crop_model.canonical_locs()[inds]
    predict_fn = functools.partial(
        model.estimate_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        num_aug=FLAGS.num_aug, antialias_factor=2,
        weights=model.get_weights_for_canonical_points(cano_points))

    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, world_up=(0, 0, 1), ground_plane_height=0,
        queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    image_relpaths_all = []
    coords_all = []
    for i_subject in (9, 11):
        for activity_name, camera_id in itertools.product(
                h36m_main.get_activity_names(i_subject), range(4)):
            if FLAGS.viz:
                viz.reinit_camera_view()
                if FLAGS.out_video_dir:
                    viz.new_sequence_output(
                        f'{FLAGS.out_video_dir}/S{i_subject}/{activity_name}.{camera_id}.mp4',
                        fps=max(50 / FLAGS.frame_step, 2))

            logger.info(f'Predicting S{i_subject} {activity_name} {camera_id}...')
            video_relpath, frame_relpaths, bboxes, camera, world_coords_gt = get_sequence(
                i_subject, activity_name, camera_id)

            frame_paths = [f'{DATA_ROOT}/{p}' for p in frame_relpaths]

            frames_cpu_gpu = precuda(iter_frame_batches(frame_paths, FLAGS.batch_size))
            box_batches = more_itertools.chunked(bboxes, FLAGS.batch_size)
            # ds, frame_batches_cpu = tfinp.video_file(
            #    f'{DATA_ROOT}/{video_relpath}',
            #    extra_data=box_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz,
            #    video_slice=slice(0, None, FLAGS.frame_step))
            coords3d_pred_world = predict_sequence(predict_fn, frames_cpu_gpu, box_batches, camera, viz)
            image_relpaths_all.append(frame_relpaths)
            coords_all.append(coords3d_pred_world)

    np.savez(
        FLAGS.output_path, image_path=np.concatenate(image_relpaths_all, axis=0),
        coords3d_pred_world=np.concatenate(coords_all, axis=0))

    if FLAGS.viz:
        viz.close()


def predict_sequence(predict_fn, frames_cpu_gpu, box_batches, camera, viz):
    predict_fn = functools.partial(
        predict_fn, intrinsic_matrix=torch.from_numpy(camera.intrinsic_matrix[np.newaxis]).cuda(),
        extrinsic_matrix=torch.from_numpy(camera.get_extrinsic_matrix()[np.newaxis]).cuda(),
        distortion_coeffs=torch.from_numpy(camera.get_distortion_coeffs()[np.newaxis]).cuda(),
        world_up_vector=torch.from_numpy(camera.world_up).cuda())

    pose_batches = []

    for (frames_b_cpu, frames_b), box_b in zip(frames_cpu_gpu, box_batches):
        boxes_b = [torch.from_numpy(b[torch.newaxis]).cuda() for b in box_b]
        pred = predict_fn(frames_b, boxes_b)
        pred = nested_map(lambda x: torch.squeeze(x, 0).cpu().numpy(), pred)
        pose_batches.append(pred['poses3d'])

        if FLAGS.viz:
            for frame, box, pose3d in zip(frames_b_cpu, box_b, pred['poses3d']):
                viz.update(frame, box[np.newaxis], pose3d[np.newaxis], camera)

    return np.concatenate(pose_batches, axis=0)


# @spu.picklecache('h36m_sequence.pkl')
def get_sequence(i_subject, activity_name, i_camera):
    camera_name = ['54138969', '55011271', '58860488', '60457274'][i_camera]
    camera = h36m_main.get_cameras()[i_camera][i_subject - 1]
    bbox_path = f'{DATA_ROOT}/h36m/S{i_subject}/BBoxes/{activity_name}.{camera_name}.npy'
    bboxes_all = np.load(bbox_path)
    n_total_frames = bboxes_all.shape[0]
    bboxes = bboxes_all[::FLAGS.frame_step]
    image_relfolder = f'h36m/S{i_subject}/Images/{activity_name}.{camera_name}'
    video_relpath = f'h36m/S{i_subject}/Videos/{activity_name}.{camera_name}.mp4'
    image_relpaths = [f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                      for i_frame in range(0, n_total_frames, FLAGS.frame_step)]

    coord_path = (f'{DATA_ROOT}/h36m/S{i_subject}/'
                  f'MyPoseFeatures/D3_Positions/{activity_name}.cdf')
    try:
        with spacepy.pycdf.CDF(coord_path) as cdf_file:
            coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
    except spacepy.pycdf.CDFError as e:
        raise ValueError(f"Error loading CDF file at {coord_path!r}") from e

    coords_raw = coords_raw_all[::FLAGS.frame_step]
    i_relevant_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    world_coords = coords_raw.reshape([coords_raw.shape[0], -1, 3])[:, i_relevant_joints]
    assert len(bboxes) == len(image_relpaths)  # == len(world_coords)

    if FLAGS.correct_S9:
        world_coords = h36m_main.correct_world_coords(world_coords, coord_path)
        bboxes = h36m_main.correct_boxes(bboxes, bbox_path, world_coords, camera)

    return video_relpath, image_relpaths, bboxes, camera, world_coords


if __name__ == '__main__':
    with torch.inference_mode():
        main()
