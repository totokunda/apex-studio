import vtk  # noqa: F401

"""separator"""
import argparse
import functools

import cameralib
import more_itertools
import numpy as np
import poseviz
import simplepyutils as spu
import torch
import torchvision  # noqa: F401

from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS, logger

from nlf.paths import PROJDIR
from nlf.pt.inference_scripts.predict_tdpw import (
    iter_frame_batches,
    nested_map,
    precuda,
    ragged_split,
)


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--default-fov', type=float, default=55)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=64)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--clahe', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)


def main():
    initialize()
    logger.info('Loading model...')
    model = torch.jit.load(FLAGS.model_path)
    logger.info('Model loaded.')

    body_model_name = 'smpl'
    cano_verts = np.load(f'{PROJDIR}/canonical_verts/{body_model_name}.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints/{body_model_name}.npy')
    vertex_subset = np.arange(6890)
    cano_all = torch.cat(
        [torch.as_tensor(cano_verts[vertex_subset]), torch.as_tensor(cano_joints)], dim=0
    ).to(dtype=torch.float32, device='cuda')

    K = np.array([[5000, 0.0, 512 / 2.0], [0.0, 5000, 512 / 2.0], [0.0, 0.0, 1.0]], np.float32)

    predict_fn = functools.partial(
        model.estimate_poses_batched,
        intrinsic_matrix=torch.from_numpy(K[np.newaxis]).cuda(),
        internal_batch_size=FLAGS.internal_batch_size,
        num_aug=FLAGS.num_aug,
        antialias_factor=FLAGS.antialias_factor,
        weights=model.get_weights_for_canonical_points(cano_all),
    )

    labels = np.load(f'{DATA_ROOT}/ssp_3d/labels.npz')
    centers = labels['bbox_centres']
    size = labels['bbox_whs'][:, np.newaxis]
    boxes = np.concatenate([centers - size / 2, size, size], axis=1).astype(np.float32)

    image_paths = [f'{DATA_ROOT}/ssp_3d/images/{n}' for n in labels['fnames']]
    extra_data = zip(
        more_itertools.chunked(image_paths, FLAGS.batch_size),
        more_itertools.chunked(boxes, FLAGS.batch_size),
    )

    # the letterboxing border should be white for this model
    frames_cpu_gpu = precuda(iter_frame_batches(image_paths, FLAGS.batch_size))

    result_verts_batches = []
    result_joints_batches = []
    result_vert_uncert_batches = []
    result_joint_uncert_batches = []

    camera = cameralib.Camera(intrinsic_matrix=K)
    faces = np.load(f'{PROJDIR}/smpl_faces.npy')
    viz = poseviz.PoseViz(body_model_faces=faces, paused=True) if FLAGS.viz else None


    for (frame_batch_cpu, frame_batch_gpu), (extra_batch, boxes_batch) in spu.progressbar(
        zip(frames_cpu_gpu, extra_data), total=len(image_paths), step=FLAGS.batch_size
    ):

        boxes_b = [torch.from_numpy(b[torch.newaxis]).cuda() for b in boxes_batch]
        pred = predict_fn(frame_batch_gpu, boxes_b)

        pred['vertices'], pred['joints'] = ragged_split(
            pred['poses3d'], [len(vertex_subset), len(cano_joints)], dim=-2
        )
        pred['vertex_uncertainties'], pred['joint_uncertainties'] = ragged_split(
            pred['uncertainties'], [len(vertex_subset), len(cano_joints)], dim=-1
        )

        pred = nested_map(lambda x: torch.squeeze(x, 0).cpu().numpy(), pred)
        result_verts_batches.append(pred['vertices'])
        result_joints_batches.append(pred['joints'])
        result_vert_uncert_batches.append(pred['vertex_uncertainties'])
        result_joint_uncert_batches.append(pred['joint_uncertainties'])
        if viz is not None:
            for frame_cpu, box, vertices in zip(frame_batch_cpu, boxes_batch, pred['vertices']):
                viz.update(
                    frame=frame_cpu,
                    boxes=box[np.newaxis],
                    vertices=vertices[np.newaxis],
                    # vertices_alt=fit_vertices[np.newaxis],
                    camera=camera,
                )
    if viz is not None:
        viz.close()

    np.savez(
        FLAGS.output_path,
        vertices=np.concatenate(result_verts_batches, axis=0) / 1000,
        joints=np.concatenate(result_joints_batches, axis=0) / 1000,
        vertex_uncertainties=np.concatenate(result_vert_uncert_batches, axis=0),
        joint_uncertainties=np.concatenate(result_joint_uncert_batches, axis=0),
    )


if __name__ == '__main__':
    with torch.inference_mode():
        main()
