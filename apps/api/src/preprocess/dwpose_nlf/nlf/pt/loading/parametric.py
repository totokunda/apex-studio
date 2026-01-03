import functools

import boxlib
import cameralib
import cv2
import numpy as np
import scipy.sparse as sps
from simplepyutils import FLAGS

import nlf.common.augmentation.appearance as appearance_aug
import nlf.mmapped_smpl
from nlf.common import improc, util
from nlf.paths import PROJDIR
from nlf.pt.loading.common import (
    MemMappedCSR,
    augment_background,
    load_array,
    look_at_box,
    recolor_border,
    make_marker_plus,
)
from nlf.pt.util import TRAIN


def load_parametric(ex, n_points_surface, n_points_internal, learning_phase, rng):
    ex = ex.load()

    # For these datasets we only use the skeleton, not the shape because surface is not accurate.
    is_shape_valid = not ex.image_path.startswith(('genebody', 'egobody', 'humman'))
    is_shape_perfect = ex.image_path.startswith(('agora', 'dfaust', 'bedlam', 'spec', 'surreal'))

    # Get the random number generators for the different augmentations to make it reproducibile
    point_sampling_rng = util.new_rng(rng)
    appearance_rng = util.new_rng(rng)
    background_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)

    if 'kid_factor' not in ex.parameters:
        ex.parameters['kid_factor'] = np.array(0, np.float32)

    if 'scale' not in ex.parameters:
        ex.parameters['scale'] = np.array(1, np.float32)

    (
        canonical_joints,
        canonical_points_intern_all,
        canonical_points_surf_all,
        face_probs,
        interp_weights_intern_all,
        faces_all,
    ) = load_arrays(ex.parameters)
    if learning_phase == TRAIN:
        face_probs = face_probs / np.sum(face_probs)

    n_joints = canonical_joints.shape[0]

    n_sampled_surf = n_points_surface - (n_joints - n_joints // 2)
    n_sampled_internal = n_points_internal - n_joints // 2
    if 'all_surf' in FLAGS.custom:
        n_sampled_surf += n_sampled_internal
        n_sampled_internal = 0

    canonical_points, interp_weights = get_points(
        canonical_points_surf_all,
        canonical_points_intern_all,
        canonical_joints,
        interp_weights_intern_all,
        n_sampled_surf,
        n_sampled_internal,
        faces_all,
        face_probs,
        point_sampling_rng,
    )

    imshape = [FLAGS.proc_side, FLAGS.proc_side]

    orig_cam = ex.camera
    if 'surreal' in ex.image_path.lower():
        orig_cam.intrinsic_matrix = np.array(
            [[600, 0, 159.5], [0, 600, 119.5], [0, 0, 1]], dtype=np.float32
        )

    is_single_person_example = ex.image_path.startswith(
        (
            'rich',
            'behave',
            'surreal',
            'moyo',
            'arctic',
            'intercap',
            'genebody',
            'humman',
            'synbody_humannerf',
            'thuman2',
            'zjumocap',
            'dfaust_render',
        )
    )

    bbox = ex.bbox
    if (
        learning_phase == TRAIN
        and partial_visi_rng.random() < FLAGS.partial_visibility_prob
        and is_single_person_example
    ):
        bbox = boxlib.random_partial_subbox(boxlib.expand_to_square(bbox), partial_visi_rng)

    center_point = boxlib.center(bbox)
    if (learning_phase == TRAIN and FLAGS.geom_aug) or (
        learning_phase != TRAIN and FLAGS.test_aug and FLAGS.geom_aug
    ):
        center_point += (
                util.random_uniform_disc(geom_rng) * FLAGS.shift_aug / 100 * np.max(bbox[2:])
        )
    bbox = boxlib.box_around(center_point, bbox[2:])
    cam = look_at_box(orig_cam, bbox, imshape)

    if FLAGS.geom_aug and (learning_phase == TRAIN or FLAGS.test_aug):
        s1 = FLAGS.scale_aug_down / 100
        s2 = FLAGS.scale_aug_up / 100
        zoom = geom_rng.uniform(1 - s1, 1 + s2)
        cam.zoom(zoom)
        r = (
            np.pi
            if FLAGS.full_rot_aug_prob and geom_rng.random() < FLAGS.full_rot_aug_prob
            else np.deg2rad(FLAGS.rot_aug)
        )
        cam.rotate(roll=geom_rng.uniform(-r, r))

    im = ex.get_image()
    border_value = (FLAGS.border_value, FLAGS.border_value, FLAGS.border_value)
    if FLAGS.border_value != 0:
        im = recolor_border(im, border_value=border_value)

    interp_str = (
        FLAGS.image_interpolation_train
        if learning_phase == TRAIN
        else FLAGS.image_interpolation_test
    )
    antialias = FLAGS.antialias_train if learning_phase == TRAIN else FLAGS.antialias_test
    interp = getattr(cv2, 'INTER_' + interp_str.upper())
    im = cameralib.reproject_image(
        im,
        orig_cam,
        cam,
        imshape,
        antialias_factor=antialias,
        interp=interp,
        border_value=FLAGS.border_value,
    )

    # Background augmentation
    if hasattr(ex, 'mask') and ex.mask is not None:
        im = augment_background(
            ex, im, orig_cam, cam, imshape, learning_phase, antialias, interp, background_rng
        )

    # Occlusion and color augmentation
    im = appearance_aug.augment_appearance(
        im, learning_phase, FLAGS.occlude_aug_prob, border_value, appearance_rng
    )
    im = improc.normalize01(im)

    bm = nlf.mmapped_smpl.get_cached_body_model(ex.parameters['type'], ex.parameters['gender'])
    pose_cam, trans_cam = bm.rototranslate(
        cam.R,
        cam.t / 1000 / ex.parameters['scale'],
        ex.parameters['pose'],
        ex.parameters['shape'],
        ex.parameters['trans'],
        ex.parameters['kid_factor'],
        post_translate=False,
    )

    def pad(x, n):
        return np.concatenate([x, np.zeros([np.maximum(0, n - x.shape[0])])], axis=0)

    # Fix RICH params, where the expression coeffs have been mixed into the shape coeffs
    shape_betas = (
        ex.parameters['shape'][:10] if ex.image_path.startswith('rich') else ex.parameters['shape']
    )

    n_betas = 128
    shape_betas = shape_betas[:n_betas]
    shape_betas = pad(shape_betas, n_betas)
    pose_cam = pad(pose_cam, 55 * 3)
    kid_factor = np.float32(ex.parameters['kid_factor'])
    scale = np.float32(ex.parameters['scale'])
    point_validity_mask = np.ones([FLAGS.num_points], bool)
    if not is_shape_valid:
        point_validity_mask[:-n_joints] = False

    # if not is_shape_perfect:
    #     start = FLAGS.proc_side // 2 -7
    #     end = start + 14
    #     im[start:end, start:end, :] = make_marker_plus(14).astype(np.float32) / 255.0

    return dict(
        param=dict(
            image_path=ex.image_path,
            image=np.float32(im).transpose(2, 0, 1),
            intrinsics=np.float32(cam.intrinsic_matrix),
            extrinsics=np.float32(cam.get_extrinsic_matrix()),
            canonical_points=canonical_points.astype(np.float32),
            body_model=f'{ex.parameters["type"]}_{ex.parameters["gender"][0]}',
            pose=np.float32(pose_cam),
            shape=np.float32(shape_betas),
            trans=np.float32(trans_cam),
            kid_factor=np.float32(kid_factor),
            scale=np.float32(scale),
            is_shape_valid=is_shape_valid,
            point_validity=point_validity_mask,
            interp_weights=interp_weights,
            root_index=np.int32(len(canonical_points) - n_joints),
        )
    )


def globally_rotate_inplace(pose_rotvec, rot_change):
    current_rotmat = cv2.Rodrigues(pose_rotvec[:3])[0]
    new_rotmat = rot_change @ current_rotmat
    pose_rotvec[:3] = cv2.Rodrigues(new_rotmat)[0][:, 0]


def get_points(
    canonical_points_surf_all,
    canonical_points_intern_all,
    canonical_joints,
    interp_weights_intern_all,
    n_points_surface,
    n_points_internal,
    faces_all,
    face_probs,
    rng,
):
    # Internal point sampling
    start_index = rng.integers(0, len(canonical_points_intern_all) - n_points_internal)
    canonical_points_intern = canonical_points_intern_all[
        start_index : start_index + n_points_internal
    ]
    interp_weights_intern = interp_weights_intern_all[
        start_index : start_index + n_points_internal
    ]

    # Surface point sampling
    faces = rng.choice(faces_all, n_points_surface, p=face_probs, replace=True)
    barycentric_coords, canonical_points_surf = sample_points_on_faces(
        canonical_points_surf_all, faces, n_points_surface, rng
    )
    # create sparse matrix for the barycentric weights on the mesh, shape [n_points_out, n_verts]
    interp_weights_surf = sps.csr_matrix(
        (
            barycentric_coords.reshape(-1),
            (np.repeat(np.arange(n_points_surface), 3), np.reshape(faces, -1)),
        ),
        shape=[n_points_surface, interp_weights_intern.shape[1]],
    )

    canonical_points = np.concatenate(
        [canonical_points_surf, canonical_points_intern, canonical_joints], axis=0
    )

    interp_weights = sps.vstack(
        [
            interp_weights_surf,
            interp_weights_intern,
            get_bottom_rows(interp_weights_intern.shape[1], canonical_joints.shape[0]),
        ]
    )

    return np.float32(canonical_points), interp_weights


def sample_points_on_faces(canonical_points_surf_all, faces, n_points_surface, rng):
    ab = rng.uniform(0, 1, size=(n_points_surface, 2)).astype(np.float32)
    sqrt_a = np.sqrt(ab[:, :1])
    b = ab[:, 1:]
    barycentric_coords = np.concatenate([(1 - sqrt_a), sqrt_a * (1 - b), sqrt_a * b], axis=1)
    canonical_points_surf = np.einsum(
        'fpc,fp->fc', canonical_points_surf_all[faces], barycentric_coords
    )
    return barycentric_coords, canonical_points_surf


def load_arrays(parameters):
    canonical_points_intern_all = load_array(f'{PROJDIR}/canonical_internal_points_smpl.npy')

    if parameters['type'] in ['smpl', 'smplh']:
        faces_all = load_array(f'{PROJDIR}/smpl_faces.npy')
        face_probs = load_array(f'{PROJDIR}/smpl_face_probs.npy')
        canonical_points_surf_all = load_array(f'{PROJDIR}/canonical_vertices_smpl.npy')

        if parameters['type'] == 'smpl':
            interp_weights_intern_all = MemMappedCSR(f'{PROJDIR}/internal_regressor_m.csr')
            canonical_joints = load_array(f'{PROJDIR}/canonical_joints_m.npy')
        else:
            canonical_joints = load_array(f'{PROJDIR}/canonical_joints_smplh_f.npy')
            # Add zeros for weighting of the extra joints for SMPL-H, as the regressor
            # was made by natinterp for SMPL
            interp_weights_intern_all = MemMappedCSR(f'{PROJDIR}/internal_regressor_m.csr')
            interp_weights_intern_all.shape = (
                interp_weights_intern_all.shape[0],
                interp_weights_intern_all.shape[1] - 24 + 52,
            )
    elif parameters['type'].startswith('smplx'):
        faces_all = load_array(f'{PROJDIR}/smplx_faces.npy')
        canonical_points_surf_all = load_array(f'{PROJDIR}/canonical_vertices_smplx.npy')
        interp_weights_intern_all = MemMappedCSR(f'{PROJDIR}/internal_regressor_smplx_n.csr')
        canonical_joints = load_array(f'{PROJDIR}/canonical_joints_smplx_n.npy')
        face_probs = load_array(f'{PROJDIR}/smplx_face_probs_new.npy')
    else:
        raise ValueError(f'Unknown type {parameters["type"]}')

    return (
        canonical_joints,
        canonical_points_intern_all,
        canonical_points_surf_all,
        face_probs,
        interp_weights_intern_all,
        faces_all,
    )


@functools.lru_cache()
def get_bottom_rows(width, n_joints=55):
    zero_bottom = sps.csr_matrix((n_joints, width - n_joints), dtype=np.float32)
    return sps.hstack([zero_bottom, sps.eye(n_joints, dtype=np.float32)]).tocsr()


def random_canonical_points(n_points_surface, n_points_internal, rng):
    (
        canonical_joints,
        canonical_points_intern_all,
        canonical_points_surf_all,
        face_probs,
        interp_weights_intern_all,
        faces_all,
    ) = load_arrays(dict(gender='neutral', type='smplx'))
    return get_points(
        canonical_points_surf_all,
        canonical_points_intern_all,
        np.zeros([0, 3], np.float32),
        interp_weights_intern_all,
        n_points_surface,
        n_points_internal,
        faces_all,
        face_probs,
        rng,
    )
