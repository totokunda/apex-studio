import boxlib
import cameralib
import cv2
import numpy as np
from simplepyutils import FLAGS

import nlf.common.augmentation.appearance as appearance_aug
from nlf.common import improc, util
from nlf.common.util import TRAIN
from nlf.paths import PROJDIR
from nlf.pt.loading.common import load_array, recolor_border, make_marker
from nlf.pt.loading.parametric import random_canonical_points


def load_dense(ex, joint_info, full_joint_info, learning_phase, rng):
    ex = ex.load()

    # Get the random number generators for the different augmentations to make it reproducibile
    appearance_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)
    background_rng = util.new_rng(rng)
    point_sampler_rng = util.new_rng(rng)

    # Load the image
    im_from_file = ex.get_image()

    border_value = (FLAGS.border_value, FLAGS.border_value, FLAGS.border_value)
    if FLAGS.border_value != 0:
        im_from_file = recolor_border(im_from_file, border_value=border_value)

    # Determine bounding box
    bbox = ex.bbox
    # if learning_phase == TRAIN and partial_visi_rng.random() < FLAGS.partial_visibility_prob:
    #    bbox = boxlib.random_partial_subbox(boxlib.expand_to_square(bbox), partial_visi_rng)

    crop_side = np.max(bbox[2:])
    center_point = boxlib.center(bbox)

    if FLAGS.geom_aug:
        center_point += util.random_uniform_disc(geom_rng) * FLAGS.shift_aug / 100 * crop_side

    has_3d_camera = hasattr(ex, 'camera') and ex.camera is not None
    orig_cam = ex.camera if has_3d_camera else cameralib.Camera.from_fov(8, im_from_file.shape)
    cam = orig_cam.copy()

    if has_3d_camera:
        if bbox[2] < bbox[3]:
            # Tall box: take midpoints of top and bottom sides
            delta_y = np.array([0, bbox[3] / 2])
            sidepoints = center_point + np.stack([-delta_y, delta_y])
        else:
            # Wide box: take midpoints of left and right sides
            delta_x = np.array([bbox[2] / 2, 0])
            sidepoints = center_point + np.stack([-delta_x, delta_x])

        cam.turn_towards(target_image_point=center_point)
        cam.undistort()
        cam.square_pixels()
        cam_sidepoints = cameralib.reproject_image_points(sidepoints, ex.camera, cam)
        crop_side = np.linalg.norm(cam_sidepoints[0] - cam_sidepoints[1])

    cam.zoom(FLAGS.proc_side / crop_side)

    if FLAGS.geom_aug:
        s1 = FLAGS.scale_aug_down / 100
        s2 = FLAGS.scale_aug_up / 100
        cam.zoom(geom_rng.uniform(1 - s1, 1 + s2))
        r = (
            np.pi
            if FLAGS.full_rot_aug_prob and geom_rng.random() < FLAGS.full_rot_aug_prob
            else np.deg2rad(FLAGS.rot_aug)
        )
        cam.rotate(roll=geom_rng.uniform(-r, r))

    dense_coords, faces, barycoords = ex.densepose
    dense_coords = np.array(dense_coords, np.float32)
    direc = f'{PROJDIR}/'
    canonical_points_surf_all = load_array(f'{direc}/canonical_vertices_smpl.npy')
    canonical_points = np.einsum('fpc,fp->fc', canonical_points_surf_all[faces], barycoords)

    if FLAGS.hflip_aug and learning_phase == TRAIN and geom_rng.random() < 0.5:
        cam.horizontal_flip()
        # Must reorder the joints due to left and right flip
        imcoords = ex.coords[joint_info.mirror_mapping]
        # Also flip the canonical points
        canonical_points[:, 0] *= -1
    else:
        imcoords = ex.coords

    if has_3d_camera:
        cam.center_principal_point((FLAGS.proc_side, FLAGS.proc_side))
    else:
        new_center_point = cameralib.reproject_image_points(center_point, orig_cam, cam)
        cam.shift_to_center(new_center_point, (FLAGS.proc_side, FLAGS.proc_side))

    # TODO check that invalid may be also signified by zero as coords
    is_annotation_invalid = np.nan_to_num(dense_coords[:, 1]) > im_from_file.shape[0] * 0.95
    dense_coords[is_annotation_invalid] = np.nan
    dense_coords = cameralib.reproject_image_points(dense_coords, orig_cam, cam)

    is_annotation_invalid = np.nan_to_num(imcoords[:, 1]) > im_from_file.shape[0] * 0.95
    imcoords[is_annotation_invalid] = np.nan
    imcoords = cameralib.reproject_image_points(imcoords, orig_cam, cam)

    interp_str = (
        FLAGS.image_interpolation_train
        if learning_phase == TRAIN
        else FLAGS.image_interpolation_test
    )
    antialias = FLAGS.antialias_train if learning_phase == TRAIN else FLAGS.antialias_test
    interp = getattr(cv2, 'INTER_' + interp_str.upper())

    im = cameralib.reproject_image(
        im_from_file,
        orig_cam,
        cam,
        (FLAGS.proc_side, FLAGS.proc_side),
        antialias_factor=antialias,
        interp=interp,
        border_value=FLAGS.border_value,
    )

    # Occlusion and color augmentation
    im = appearance_aug.augment_appearance(
        im, learning_phase, FLAGS.occlude_aug_prob, border_value, appearance_rng
    )
    im = improc.normalize01(im)

    dense_coords = np.nan_to_num(dense_coords)

    joint_validity_mask = ~np.any(np.isnan(imcoords), axis=-1)
    i_valid_coords = np.argwhere(joint_validity_mask).squeeze(1)
    imcoords = np.nan_to_num(imcoords)
    imcoords = imcoords[i_valid_coords]
    indices = [full_joint_info.ids[joint_info.names[i]] for i in i_valid_coords]

    true_points_both = np.concatenate([imcoords, dense_coords], axis=0)

    unused_point_slots = FLAGS.num_points - len(true_points_both)
    canonical_points_random, interp_weights_random = random_canonical_points(
        n_points_surface=unused_point_slots // 2,
        n_points_internal=unused_point_slots - unused_point_slots // 2,
        rng=point_sampler_rng,
    )

    cano_points_all = np.concatenate([canonical_points, canonical_points_random], axis=0)

    start = FLAGS.proc_side // 2 - 7
    end = start + 14
    im[start:end, start:end, :] = make_marker(14).astype(np.float32) / 255.0

    return dict(
        dense=dict(
            image=np.float32(im).transpose(2, 0, 1),
            intrinsics=np.float32(cam.intrinsic_matrix),
            _ragged_point_ids=np.int32(indices),  # kp
            _ragged_canonical_points=np.float32(cano_points_all),  # dense + random
            _ragged_coords2d_true=np.float32(true_points_both),  # kp + dense
        )
    )
