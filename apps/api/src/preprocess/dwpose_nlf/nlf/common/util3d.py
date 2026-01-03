import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from nlf.common import procrustes
from simplepyutils import logger


def rigid_align(coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
                reflection_align=False):
    """Returns the predicted coordinates after rigid alignment to the ground truth."""

    if joint_validity_mask is None:
        joint_validity_mask = np.ones_like(coords_pred[..., 0], dtype=bool)

    valid_coords_pred = coords_pred[joint_validity_mask]
    valid_coords_true = coords_true[joint_validity_mask]
    try:
        d, Z, tform = procrustes.procrustes(
            valid_coords_true, valid_coords_pred, scaling=scale_align,
            reflection='best' if reflection_align else False)
    except np.linalg.LinAlgError:
        logger.error('Cannot do Procrustes alignment, returning original prediction.')
        return coords_pred

    T = tform['rotation']
    b = tform['scale']
    c = tform['translation']
    return b * coords_pred @ T + c


def rigid_align_many(
        coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
        reflection_align=False):
    if joint_validity_mask is None:
        joint_validity_mask = np.ones_like(coords_pred[..., 0], dtype=bool)

    return np.stack([
        rigid_align(p, t, joint_validity_mask=jv, scale_align=scale_align,
                    reflection_align=reflection_align)
        for p, t, jv in zip(coords_pred, coords_true, joint_validity_mask)])



def scale_align(poses):
    mean_scale = np.sqrt(np.mean(np.square(poses), axis=(-3, -2, -1), keepdims=True))
    scales = np.sqrt(np.mean(np.square(poses), axis=(-2, -1), keepdims=True))
    return poses / scales * mean_scale


def relu(x):
    return np.maximum(0, x)


def auc(x, t1, t2):
    return relu(np.float32(1) - relu(x - t1) / (t2 - t1))


def are_joints_valid(coords):
    return np.logical_not(np.any(np.isnan(coords), axis=-1))


def unit_vector(vectors, axis=-1):
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / norm


def get_joint2bone_mat(joint_info):
    n_bones = len(joint_info.stick_figure_edges)
    joints2bones = np.zeros([n_bones, joint_info.n_joints], np.float32)
    for i_bone, (i_joint1, i_joint2) in enumerate(joint_info.stick_figure_edges):
        joints2bones[i_bone, i_joint1] = 1
        joints2bones[i_bone, i_joint2] = -1
    return joints2bones
