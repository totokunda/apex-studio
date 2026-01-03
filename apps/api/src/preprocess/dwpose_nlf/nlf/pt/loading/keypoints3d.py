import re

import boxlib
import cameralib
import cv2
import numpy as np
from simplepyutils import FLAGS

import nlf.common.augmentation.appearance as appearance_aug
from nlf.common import improc, util
from nlf.common.util import TRAIN
from nlf.pt.loading.common import (
    augment_background,
    look_at_box,
    recolor_border,
    make_marker,
)
from nlf.pt.loading.parametric import random_canonical_points


def load_kp(ex, joint_info, learning_phase, rng):
    ex = ex.load()

    # Get the random number generators for the different augmentations to make it reproducibile
    appearance_rng = util.new_rng(rng)
    background_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    point_sampler_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)

    imshape = [FLAGS.proc_side, FLAGS.proc_side]

    world_coords = ex.world_coords

    if 'sailvos' in ex.image_path.lower():
        # This is needed in order not to lose precision in later operations.
        # Background: In the Sailvos dataset (GTA V), some world coordinates
        # are crazy large (several kilometers, i.e. millions of millimeters, which becomes
        # hard to process with the limited simultaneous dynamic range of float32).
        # They are stored in float64 but the processing is done in float32 here.
        world_coords -= ex.camera.t
        world_coords = world_coords.astype(np.float32)
        ex.camera.t[:] = 0
        ex.camera.t = ex.camera.t.astype(np.float32)

    orig_cam = ex.camera
    bbox = ex.bbox
    single_person_ds_names = [
        'h36m_',
        '3doh_down',
        'aist_',
        'aspset_',
        'gpa_',
        '3dpeople',
        'bml_movi',
        'mads_down',
        'bmhad_down',
        '3dhp_full_down',
        'totalcapture',
        'ikea_down',
        'human4d',
        'fit3d_',
        'freeman_',
        'dna_rendering',
    ]
    is_single_person_example = any(
        ex.image_path.startswith(name) for name in single_person_ds_names
    )
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

    if FLAGS.hflip_aug and learning_phase == TRAIN and geom_rng.random() < 0.5:
        cam.horizontal_flip()
        # Must reorder the joints due to left and right flip
        world_coords = world_coords[joint_info.mirror_mapping]

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

    # Color adjustment
    if re.match('.*mupots/TS[1-5]/.+', ex.image_path):
        im = improc.adjust_gamma(im, 0.67, inplace=True)
    elif '3dhp' in ex.image_path and re.match('.+/(TS[1-4])/', ex.image_path):
        im = improc.adjust_gamma(im, 0.67, inplace=True)
        im = improc.white_balance(im, 110, 145)
    elif 'panoptic' in ex.image_path.lower():
        im = improc.white_balance(im, 120, 138)

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

    camcoords = cam.world_to_camera(world_coords)

    joint_validity_mask = ~np.any(np.isnan(camcoords), axis=-1)
    i_valid_coords = np.argwhere(joint_validity_mask).squeeze(1)
    n_labeled_points = len(i_valid_coords)
    n_unused_point_slots = FLAGS.num_points - n_labeled_points

    root_index = next(
        (i for i, j in enumerate(i_valid_coords) if joint_info.names[j].startswith('pelv')), -1
    )
    camcoords = camcoords[i_valid_coords]
    canonical_points, interp_weights = random_canonical_points(
        n_points_surface=n_unused_point_slots // 2,
        n_points_internal=n_unused_point_slots - n_unused_point_slots // 2,
        rng=point_sampler_rng,
    )

    if 'mark_also3d' in FLAGS.custom:
        start = FLAGS.proc_side // 2 -7
        end = start + 14
        im[start:end, start:end, :] = (make_marker(14).astype(np.float32) / 255.0)[..., ::-1]

    return dict(
        kp3d=dict(
            image_path=ex.image_path,
            image=np.float32(im).transpose(2, 0, 1),
            extrinsics=np.float32(cam.get_extrinsic_matrix()),
            intrinsics=np.float32(cam.intrinsic_matrix),
            root_index=np.int32(root_index),
            _ragged_coords3d_true=np.float32(camcoords) / 1000,
            _ragged_point_ids=np.int32(i_valid_coords),
            _ragged_canonical_points=canonical_points,
        )
    )
