from typing import List, Optional

import numpy as np
import torch

from pt import ptu


def reconstruct_absolute(
    coords2d: torch.Tensor,
    coords3d_rel: torch.Tensor,
    intrinsics: torch.Tensor,
    proc_side: int,
    stride: int,
    centered_stride: bool,
    mix_3d_inside_fov: Optional[float] = None,
    weak_perspective: bool = None,
    point_validity_mask: Optional[torch.Tensor] = None,
    border_factor1: float = 0.75,
    border_factor2: Optional[float] = None,
    mix_based_on_3d: bool = True,
):
    inv_intrinsics = torch.linalg.inv(intrinsics.to(coords2d.dtype))
    coords2d_normalized = (to_homogeneous(coords2d) @ inv_intrinsics.transpose(1, 2))[
        ..., :2
    ]

    if border_factor2 is None:
        border_factor2 = border_factor1

    is_predicted_to_be_in_fov1 = is_within_fov(
        coords2d, proc_side, stride, centered_stride, border_factor1
    )

    if point_validity_mask is not None:
        is_predicted_to_be_in_fov1 = torch.logical_and(
            is_predicted_to_be_in_fov1, point_validity_mask
        )

    if weak_perspective:
        ref = reconstruct_ref_weakpersp(
            coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov1
        )
    else:
        ref = reconstruct_ref_fullpersp(
            coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov1
        )
    # coords_abs_3d_based = coords3d_rel + tf.expand_dims(ref, 1)
    coords_abs_3d_based = coords3d_rel + ref.unsqueeze(1)
    reference_depth = ref[:, 2]
    relative_depths = coords3d_rel[..., 2]
    coords_abs_2d_based = back_project(
        coords2d_normalized, relative_depths, reference_depth
    )

    if mix_3d_inside_fov is not None:
        coords_abs_2d_based = (
            mix_3d_inside_fov * coords_abs_3d_based
            + (1 - mix_3d_inside_fov) * coords_abs_2d_based
        )

    if mix_based_on_3d:
        coords_proj_for_mix_decision = project_pose(coords_abs_3d_based, intrinsics)
    else:
        coords_proj_for_mix_decision = project_pose(coords_abs_2d_based, intrinsics)

    is_predicted_to_be_in_fov2 = torch.logical_and(
        is_within_fov(
            coords_proj_for_mix_decision,
            proc_side,
            stride,
            centered_stride,
            border_factor2,
        ),
        coords_abs_3d_based[..., 2] > 0.001,
    )

    return torch.where(
        is_predicted_to_be_in_fov2[..., np.newaxis],
        coords_abs_2d_based,
        coords_abs_3d_based,
    )


def reconstruct_ref_weakpersp(normalized_2d, coords3d_rel, validity_mask):
    mean3d, stdev3d = ptu.mean_stdev_masked(
        coords3d_rel[..., :2], validity_mask, items_dim=1, dimensions_dim=2
    )

    mean2d, stdev2d = ptu.mean_stdev_masked(
        normalized_2d[..., :2], validity_mask, items_dim=1, dimensions_dim=2
    )

    stdev2d = torch.maximum(stdev2d, torch.tensor(1e-5))
    stdev3d = torch.maximum(stdev3d, torch.tensor(1e-5))

    old_mean = ptu.reduce_mean_masked(
        coords3d_rel, validity_mask, dim=[1], keepdim=True
    )
    new_mean_z = torch.nan_to_num(stdev3d / stdev2d)
    new_mean = to_homogeneous(mean2d) * new_mean_z
    return torch.squeeze(new_mean - old_mean, 1)


def to_homogeneous(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def reconstruct_ref_fullpersp(normalized_2d, coords3d_rel, validity_mask):
    """Reconstructs the reference point location.

    Args:
      normalized_2d: normalized image coordinates of the joints
         (without intrinsics applied), shape [batch_size, n_points, 2]
      coords3d_rel: 3D camera coordinate offsets relative to the unknown reference
         point which we want to reconstruct, shape [batch_size, n_points, 3]
      validity_mask: boolean mask of shape [batch_size, n_points] containing True
         where the point is reliable and should be used in the reconstruction

    Returns:
      The 3D reference point in camera coordinates, shape [batch_size, 3]
    """

    n_batch = normalized_2d.shape[0]
    n_points = normalized_2d.shape[1]

    eyes2 = (
        torch.eye(2, device=normalized_2d.device, dtype=normalized_2d.dtype)
        .unsqueeze(0)
        .repeat(n_batch, n_points, 1)
    )
    scale2d, reshaped2d = rms_normalize_and_reshape(
        normalized_2d, validity_mask, n_points
    )
    A = torch.cat([eyes2, -reshaped2d], dim=2)

    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj, b = rms_normalize_and_reshape(
        rel_backproj, validity_mask, n_points
    )

    weights = validity_mask.to(normalized_2d.dtype) + 1e-8
    weights = torch.repeat_interleave(weights, 2, 1)
    ref = lstsq_cholesky(A, b, weights, l2_regularizer=1e-4)

    ref = torch.cat(
        [ref[:, :2] * scale_rel_backproj, ref[:, 2:] * (scale_rel_backproj / scale2d)],
        dim=1,
    )
    return torch.squeeze(ref, dim=-1)


def rms_normalize_and_reshape(
    x: torch.Tensor, validity_mask: torch.Tensor, n_points: int
):
    scale = torch.sqrt(
        ptu.reduce_mean_masked(torch.square(x), validity_mask, dim=[1, 2], keepdim=True)
        + 1e-10
    )
    normalized = x / scale
    reshaped = normalized.reshape(-1, n_points * 2, 1)
    return scale, reshaped


def lstsq_cholesky(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    weights: torch.Tensor,
    l2_regularizer: Optional[float] = None,
) -> torch.Tensor:
    weighted_matrix = weights.unsqueeze(-1) * matrix
    regularized_gramian = weighted_matrix.mT @ matrix
    if l2_regularizer is not None:
        regularized_gramian.diagonal(dim1=-2, dim2=-1).add_(l2_regularizer)

    ATb = weighted_matrix.mT @ rhs
    chol, _ = torch.linalg.cholesky_ex(regularized_gramian)
    return torch.cholesky_solve(ATb, chol)


def back_project(camcoords2d, delta_z, z_offset):
    return to_homogeneous(camcoords2d) * torch.unsqueeze(
        delta_z + torch.unsqueeze(z_offset, -1), -1
    )


def is_within_fov(
    imcoords: torch.Tensor,
    proc_side: int,
    stride: int,
    centered_stride: bool,
    border_factor: float = 0.75,
):
    offset = -stride / 2.0 if not centered_stride else 0.0

    lower = stride * border_factor + offset
    upper = proc_side - stride * border_factor + offset
    proj_in_fov = torch.all(
        torch.logical_and(imcoords >= lower, imcoords <= upper), dim=-1
    )
    return proj_in_fov


def project_pose(coords3d, intrinsic_matrix):
    projected = coords3d / torch.maximum(
        torch.tensor(0.1, device=coords3d.device), coords3d[..., 2:]
    )
    return torch.einsum("bnk,bjk->bnj", projected, intrinsic_matrix[..., :2, :])


def lookat_matrix(forward_vector, up_vector):
    # Z will point forwards, towards the box center
    new_z = forward_vector / torch.linalg.norm(forward_vector, dim=-1, keepdim=True)
    # Get the X (right direction) as the cross of forward and up.
    new_x = torch.linalg.cross(new_z, up_vector)
    # Get alternative X by rotating the new Z around the old Y by 90 degrees
    # in case lookdir happens to align with the up vector and the above cross product is zero.
    new_x_alt = torch.stack(
        [new_z[:, 2], torch.zeros_like(new_z[:, 2]), -new_z[:, 0]], dim=1
    )
    new_x = torch.where(
        torch.linalg.norm(new_x, dim=-1, keepdim=True) == 0, new_x_alt, new_x
    )
    new_x = new_x / torch.linalg.norm(new_x, dim=-1, keepdim=True)
    # Complete the right-handed coordinate system to get Y
    new_y = torch.linalg.cross(new_z, new_x)
    # Stack the axis vectors to get the rotation matrix
    return torch.stack([new_x, new_y, new_z], dim=1)


def project(points):
    return points[..., :2] / points[..., 2:3]


def intrinsic_matrix_from_field_of_view(
    fov_degrees: float, imshape: List[int], device: Optional[torch.device] = None
):
    imshape = torch.tensor(imshape, dtype=torch.float32, device=device)
    fov_radians = fov_degrees * torch.tensor(
        torch.pi / 180, dtype=torch.float32, device=device
    )
    larger_side = torch.max(imshape)
    focal_length = larger_side / (torch.tan(fov_radians / 2) * 2)
    _0 = torch.tensor(0, dtype=torch.float32, device=device)
    _1 = torch.tensor(1, dtype=torch.float32, device=device)

    # print(torch.stack([focal_length, _0, imshape[1] / 2], dim=-1))
    return (
        torch.stack(
            [
                focal_length,
                _0,
                (imshape[1] - 1) / 2,
                _0,
                focal_length,
                (imshape[0] - 1) / 2,
                _0,
                _0,
                _1,
            ],
            dim=-1,
        )
        .unflatten(-1, (3, 3))
        .unsqueeze(0)
    )


def rotation_mat(angle: torch.Tensor, rot_axis: str):
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    _0 = torch.zeros_like(angle)
    _1 = torch.ones_like(angle)

    if rot_axis == "x":
        elems = [_1, _0, _0, _0, cos, sin, _0, -sin, cos]
    elif rot_axis == "y":
        elems = [cos, _0, -sin, _0, _1, _0, sin, _0, cos]
    else:
        elems = [cos, -sin, _0, sin, cos, _0, _0, _0, _1]

    return torch.stack(elems, dim=-1).unflatten(-1, (3, 3))


def center_relative_pose(
    coords3d, joint_validity_mask=None, center_is_mean=False, center_joints=None
):
    if center_is_mean:
        if joint_validity_mask is None:
            center = torch.mean(coords3d, dim=1, keepdim=True)
        else:
            if center_joints is not None:
                center = ptu.reduce_mean_masked(
                    coords3d[:, center_joints],
                    joint_validity_mask[:, center_joints],
                    dim=[1],
                    keepdim=True,
                )
            else:
                center = ptu.reduce_mean_masked(
                    coords3d, joint_validity_mask, dim=[1], keepdim=True
                )
    else:
        center = coords3d[:, -1:]
    return coords3d - center


def rigid_align(
    coords_pred,
    coords_true,
    *,
    joint_validity_mask=None,
    scale_align=False,
    reflection_align=False
):
    """Returns the predicted coordinates after rigid alignment to the ground truth."""
    if joint_validity_mask is None:
        joint_validity_mask = torch.ones_like(coords_pred[..., 0], dtype=torch.bool)

    return procrustes(
        coords_true,
        coords_pred,
        joint_validity_mask,
        allow_scaling=scale_align,
        allow_reflection=reflection_align,
    )


def procrustes(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    """Register the points in Y by rotation, translation, uniform scaling (optional) and
    reflection (optional) to be closest to the corresponding points in X, in a least-squares sense.

    This function operates on batches. For each item in the batch a separate
    transform is computed independently of the others.

    Arguments:
       X: Tensor with shape [batch_size, n_points, point_dimensionality]
       Y: Tensor with shape [batch_size, n_points, point_dimensionality]
       validity_mask: Boolean Tensor with shape [batch_size, n_points] indicating
         whether a point is valid in Y
       allow_scaling: boolean, specifying whether uniform scaling is allowed
       allow_reflection: boolean, specifying whether reflections are allowed

    Returns the transformed version of Y.
    """
    meanY, T, output_scale, meanX = procrustes_transf(
        X, Y, validity_mask, allow_scaling, allow_reflection
    )

    return ((Y - meanY) @ T) * output_scale + meanX


def procrustes_transf(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    validity_mask = validity_mask.unsqueeze(-1)
    _0 = torch.tensor(0, dtype=X.dtype, device=X.device)

    n_points_per_example = (
        validity_mask.count_nonzero(dim=1).to(dtype=X.dtype).unsqueeze(1)
    )
    denominator_correction_factor = validity_mask.shape[1] / n_points_per_example

    def normalize(Z):
        Z = torch.where(validity_mask, Z, _0)
        mean = Z.mean(dim=1, keepdim=True) * denominator_correction_factor
        centered = torch.where(validity_mask, Z - mean, _0)
        norm = torch.norm(centered, p="fro", dim=(1, 2), keepdim=True)
        normalized = centered / norm
        return mean, norm, normalized

    meanX, normX, normalizedX = normalize(X)
    meanY, normY, normalizedY = normalize(Y)

    A = torch.matmul(normalizedY.transpose(1, 2), normalizedX)
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)

    T = U @ Vh
    s = s.unsqueeze(-1)

    if allow_scaling:
        relative_scale = normX / normY
        output_scale = relative_scale * s.sum(dim=1, keepdim=True)
    else:
        relative_scale = None
        output_scale = torch.tensor(1.0, dtype=X.dtype, device=X.device)

    if not allow_reflection:
        # Check if T has a reflection component. If so, remove it by flipping
        has_reflection = (torch.det(T) < 0)[..., torch.newaxis, torch.newaxis]
        T_mirror = T - 2 * U[..., -1:] @ Vh[..., -1:, :]
        T = torch.where(has_reflection, T_mirror, T)

        if allow_scaling:
            output_scale_mirror = output_scale - 2 * relative_scale * s[:, -1:]
            output_scale = torch.where(
                has_reflection, output_scale_mirror, output_scale
            )

    return meanY, T, output_scale, meanX
