from typing import Optional, Sequence, TYPE_CHECKING

import florch
import numpy as np
import smplfitter.pt
import torch
from florch import EasyDict
from simplepyutils import FLAGS
import torch.nn as nn
from nlf.pt import ptu, ptu3d

if TYPE_CHECKING:
    from nlf.pt.models.nlf_model import NLFModel


class NLFTrainer(florch.ModelTrainer):
    def __init__(self, model: "NLFModel", **kwargs):
        super().__init__(model, **kwargs)
        self.model: NLFModel
        self.body_models = nn.ModuleDict(
            {
                f"{n}_{g[0]}": smplfitter.pt.BodyModel(
                    model_name=n, gender=g, num_betas=128
                )
                for n, g in [
                    ("smpl", "female"),
                    ("smpl", "male"),
                    ("smpl", "neutral"),
                    ("smplh", "female"),
                    ("smplh", "male"),
                    ("smplx", "female"),
                    ("smplx", "male"),
                    ("smplx", "neutral"),
                    ("smplxmoyo", "female"),
                ]
            }
        )

    @torch.jit.export
    def prepare_inputs(self, inps):
        # FORWARD BODY MODELS ON GPU
        # First we need to group the different body model types (smpl, smplh, smplx) and
        # genders together, so we can forward them in a batched way.
        bm_kinds = list(self.body_models.keys())

        # These selectors contain the indices where a certain body model appears in the batch
        # e.g. body_model_selectors['smpl', 'neutral'] == [0, 2, 5, 7, ...]
        body_model_selectors = {
            k: [i for i, t in enumerate(inps.param.body_model) if t == k]
            for k in bm_kinds
        }
        # The permutation is the one needed to sort the batch so that all body models of the same
        # kind are grouped together.
        # permutation = torch.cat([body_model_selectors[k] for k in bm_kinds], dim=0)
        # The inverse permutation is the one needed to undo the permutation, we will use it later
        # invperm = torch.argsort(permutation)
        # The sizes tensor contains how many of each body model type we have in the batch
        # sizes = [len(body_model_selectors[k]) for k in bm_kinds]

        invperm = list(
            np.argsort(np.concatenate([body_model_selectors[k] for k in bm_kinds]))
        )

        # This function is similar to tf.dynamic_partition
        def permute_and_split(x):
            return {k: x[body_model_selectors[k]] for k in bm_kinds}

        # Each of these become dictionaries that map to e.g. the pose of each body model type
        # Since different body models have different number of joints, the tensor sizes
        # may be different in each value of the dictionary.
        pose = permute_and_split(inps.param.pose)
        shape = permute_and_split(inps.param.shape)
        trans = permute_and_split(inps.param.trans)
        kid_factor = permute_and_split(inps.param.kid_factor)
        scale = permute_and_split(inps.param.scale)
        interp_weights = {
            k: [inps.param.interp_weights[i] for i in body_model_selectors[k]]
            for k in bm_kinds
        }

        # Now we determine the GT points for each body model type
        def decode_points_for_body_model(k):
            bm = self.body_models[k]
            # We first forward the corresponding body model to get the vertices and joints
            result = bm(
                pose[k][:, : bm.num_joints * 3], shape[k], trans[k], kid_factor[k]
            )
            # We concatenate the vertices and joints
            # (and scale them, which is not part of the body model definition, but some datasets
            # specify a scale factor for their fits, so we have to use it. AGORA's kid factor
            # is probably a better way.)
            verts_and_joints = (
                torch.cat([result["vertices"], result["joints"]], dim=1)
                * scale[k][:, torch.newaxis, torch.newaxis]
            )

            # The random internal and surface points of the current batch are specified by
            # nlf.pt.loading.parametric as vertex and joint weightings. The weights are
            # the Sibson coordinates of the points in canonical space w.r.t. the mesh vertices
            # and joints. About a million points were precomputed and the data loader
            # samples from that. Anyways, we now apply the same weights (Sibson coordinates)
            # to the posed and shaped vertices and joints to get the GT points that NLF
            # should learn to predict based on the image and the canonical points.
            # This function does the interpolation using CSR sparse representation.
            return interpolate_sparse(
                verts_and_joints,
                interp_weights[k],
            )

        # We now put the GT points of the different body model kinds back together
        # in the original order before we sorted them according to body model type.
        # For this we use the inverse permutation.
        gt_points = torch.cat(
            [decode_points_for_body_model(k) for k in bm_kinds], dim=0
        )
        inps.param.coords3d_true = gt_points[invperm]

        # LOOK UP CANONICAL POINTS
        # This gets the canonical points of the joints of all the different skeleton definitions.
        # We enforce symmetry on these (i.e., only the right side is simply the mirror of the left,
        # there are no variables for the right side; and the x of the middle joints like the spine
        # are forced to be zero, there is no variable for that).
        # model.canonical_locs() assembles the canonical points from the underlying trainable
        # variables and constructs this symmetric version.
        canonical_locs = self.model.canonical_locs()

        # Since each example can define a custom set of points (from the global pool of possible
        # skeleton points for which we have trainable canonicals), we now need to pick out the ones
        # for which each example provides GT annotation. The idea is that each skeleton-based
        # example specifies *indices* into the global set of canonical points.
        inps.kp3d.canonical_points = nested_cat(
            index_select_with_nested_indices(canonical_locs, inps.kp3d.point_ids),
            inps.kp3d.canonical_points,
        )
        inps.dense.canonical_points = nested_cat(
            index_select_with_nested_indices(canonical_locs, inps.dense.point_ids),
            inps.dense.canonical_points,
        )
        inps.kp2d.canonical_points = nested_cat(
            index_select_with_nested_indices(canonical_locs, inps.kp2d.point_ids),
            inps.kp2d.canonical_points,
        )

        # STACKING AND SPLITTING
        # We have four different sub-batches corresponding to the different annotation categories.
        # For efficient handling, e.g. passing though the backbone network, we have to concat
        # them all.
        inps.image = torch.cat(
            [inps.param.image, inps.kp3d.image, inps.dense.image, inps.kp2d.image],
            dim=0,
        )
        inps.intrinsics = torch.cat(
            [
                inps.param.intrinsics,
                inps.kp3d.intrinsics,
                inps.dense.intrinsics,
                inps.kp2d.intrinsics,
            ],
            dim=0,
        )
        inps.canonical_points = torch.cat(
            [
                torch.nested.as_nested_tensor(inps.param.canonical_points),
                inps.kp3d.canonical_points,
                inps.dense.canonical_points,
                inps.kp2d.canonical_points,
            ],
            dim=0,
        )

        # Some of the tensors are actually ragged tensors, which need to be converted to
        # dense tensors for efficient processing.
        # Besides the dense (padded) version, also produce dense (non-ragged) boolean mask tensors
        # that specify which elements were part of the ragged tensor and which are just paddings
        # This will be important to know when averaging in the loss computation.

        inps.kp3d.coords3d_true, inps.kp3d.point_validity = nested_to_tensor_and_mask(
            inps.kp3d.coords3d_true
        )
        inps.dense.coords2d_true, inps.dense.point_validity = nested_to_tensor_and_mask(
            inps.dense.coords2d_true
        )
        inps.kp2d.coords2d_true, inps.kp2d.point_validity = nested_to_tensor_and_mask(
            inps.kp2d.coords2d_true
        )

        inps.point_validity_mask = torch.cat(
            [
                inps.param.point_validity,
                inps.kp3d.point_validity,
                inps.dense.point_validity,
                inps.kp2d.point_validity,
            ],
            dim=0,
        )
        inps.canonical_points_tensor = nested_to_tensor_and_mask(inps.canonical_points)[
            0
        ]

        inps.coords2d_true = torch.cat(
            [
                ptu3d.project_pose(inps.param.coords3d_true, inps.param.intrinsics),
                ptu3d.project_pose(inps.kp3d.coords3d_true, inps.kp3d.intrinsics),
                inps.dense.coords2d_true,
                inps.kp2d.coords2d_true,
            ],
            dim=0,
        )

        # Check if Z coord is larger than 1 mm. this will be used to filter out points behind the
        # the camera, which are not visible in the image, and should not be part of the loss.
        is_in_front_true = torch.cat(
            [
                inps.param.coords3d_true[..., 2] > 0.001,
                inps.kp3d.coords3d_true[..., 2] > 0.001,
                torch.ones_like(inps.dense.coords2d_true[..., 0], dtype=torch.bool),
                torch.ones_like(inps.kp2d.coords2d_true[..., 0], dtype=torch.bool),
            ],
            dim=0,
        )

        # Check if the 2D GT points are within the field of view of the camera.
        # The border factor parameter is measured in terms of stride units of the backbone.
        # E.g., 0.5 means that we only consider points "within the fov" if it's at least half
        # a stride inside the image border. This is because the network is not able to output
        # 2D coordinates at the border of the image.
        inps.is_within_fov_true = torch.logical_and(
            torch.logical_and(
                ptu3d.is_within_fov(
                    inps.coords2d_true,
                    border_factor=0.5,
                    proc_side=FLAGS.proc_side,
                    stride=FLAGS.stride_train,
                    centered_stride=FLAGS.centered_stride,
                ),
                inps.point_validity_mask,
            ),
            is_in_front_true,
        )
        return inps

    @torch.jit.export
    def forward_train(self, inps):
        preds = EasyDict(
            param=EasyDict(), kp3d=EasyDict(), dense=EasyDict(), kp2d=EasyDict()
        )

        x_mirror_tensor = torch.tensor(
            [-1, 1, 1], dtype=torch.float32, device=inps.image.device
        )

        def backbone_and_head(image, canonical_points):
            # We perform horizontal flipping augmentation here.
            # We randomly decide to flip each example. This means that
            # 1) the image is flipped horizontally (tf.image.flip_left_right) and
            # 2) the canonical points' x coord is flipped (canonical_points * [-1, 1, 1]).
            # At the end the predictions will also be flipped back accordingly.
            flip_mask = torch.rand([image.shape[0], 1, 1], device=image.device) > 0.5
            image = torch.where(
                flip_mask[..., torch.newaxis], torch.flip(image, dims=[3]), image
            )

            canonical_points = torch.where(
                flip_mask,
                canonical_points * x_mirror_tensor,
                canonical_points,
            )

            dtype = dict(
                float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16
            )[FLAGS.dtype]

            # Flipped or not, the image is passed through the backbone network.
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                features = self.model.backbone(image.to(dtype=dtype))
                assert torch.isfinite(features).all(), "Nonfinite features!"
                assert (
                    features.dtype == dtype
                ), f"Features not {dtype} but {features.dtype}"
                assert canonical_points.dtype == torch.float32

                # The heatmap head (which contains the localizer field that dynamically constructs
                # the weights) now outputs the 2D and 3D predictions, as well as the uncertainties.
                head2d, head3d, uncertainties = self.model.heatmap_head(
                    features, canonical_points
                )

            # The results need to be flipped back if they were flipped for the input at the start
            # of this function.
            head3d = torch.where(flip_mask, head3d * x_mirror_tensor, head3d)
            head2d = torch.where(
                flip_mask,
                torch.cat(
                    [FLAGS.proc_side - 1 - head2d[..., :1], head2d[..., 1:]], dim=-1
                ),
                head2d,
            )
            return head2d, head3d, uncertainties

        head2d, head3d, uncertainties = backbone_and_head(
            inps.image, inps.canonical_points_tensor
        )

        # Now we perform the reconstruction of the absolute 3D coordinates.
        # We have 2D coords in pixel space and 3D coords at metric scale but up to unknown
        # translation. These are redundant to some extent and we have two choices to create the
        # output for the points inside the field of view: either back-projectingt the 2D points
        # or translating the 3D points. The mix factor here is controlling the weighting of
        # these two options.

        mix_3d_inside_fov = (
            torch.rand([head3d.shape[0], 1, 1], device=head3d.device)
            if self.training
            else FLAGS.mix_3d_inside_fov
        )

        # The reconstruction is only performed using those points that are within the field of
        # view according to the GT.
        validity = inps.is_within_fov_true
        if FLAGS.nll_loss:
            # Furthermore, if we have reliable uncertainties (not too early in training)
            # then we also pick only those points where the uncertainty is below a certain
            # threshold.
            is_early = torch.tensor(
                self.adjusted_train_counter < 0.1 * FLAGS.training_steps,
                device=validity.device,
            )
            validity = torch.logical_and(
                validity, torch.logical_or(is_early, uncertainties < 0.3)
            )

        if "validrecons" in FLAGS.custom:
            validity = torch.logical_and(validity, inps.point_validity_mask)

        preds.coords3d_abs = self.reconstruct_absolute(
            head2d, head3d, inps.intrinsics, mix_3d_inside_fov, validity
        )

        # SPLITTING
        # The prediction was more efficient to do in one big batch, but now
        # for loss and metric computation it's better to split the different annotation
        # categories (i.e., parametric, 3D skeleton, densepose and 2D skeleton) into separate
        # tensors again.
        batch_parts = [inps.param, inps.kp3d, inps.dense, inps.kp2d]
        batch_sizes = [p.image.shape[0] for p in batch_parts]
        (
            preds.param.coords3d_abs,
            preds.kp3d.coords3d_abs,
            preds.dense.coords3d_abs,
            preds.kp2d.coords3d_abs,
        ) = torch.split(preds.coords3d_abs, batch_sizes, dim=0)

        preds.param.uncert, preds.kp3d.uncert, preds.dense.uncert, preds.kp2d.uncert = (
            torch.split(uncertainties, batch_sizes, dim=0)
        )

        return preds

    @torch.jit.export
    def compute_losses(self, inps, preds):
        losses = EasyDict()

        # Parametric
        losses_parambatch, losses.loss_parambatch = self.compute_loss_with_3d_gt(
            inps.param, preds.param
        )
        # We also store the individual component losses for each category to track them in
        # Weights and Biases (wandb).
        losses.loss_abs_param = losses_parambatch.loss3d_abs
        losses.loss_rel_param = losses_parambatch.loss3d
        losses.loss_px_param = losses_parambatch.loss2d

        # 3D keypoints
        losses_kpbatch, losses.loss_kpbatch = self.compute_loss_with_3d_gt(
            inps.kp3d, preds.kp3d
        )
        losses.loss_abs_kp = losses_kpbatch.loss3d_abs
        losses.loss_rel_kp = losses_kpbatch.loss3d
        losses.loss_px_kp = losses_kpbatch.loss2d

        # Densepose
        losses.loss_densebatch = self.compute_loss_with_2d_gt(inps.dense, preds.dense)

        # 2D keypoints
        losses.loss_2dbatch = self.compute_loss_with_2d_gt(inps.kp2d, preds.kp2d)

        # REGULARIZATION
        losses.eigval_regul = (
            self.model.heatmap_head.weight_field.first_layer_regularization()
        )

        # AGGREGATE
        # The final loss is a weighted sum of the losses computed on the four different
        # training example categories.
        losses.loss = (
            FLAGS.loss_factor_param * torch.nan_to_num(losses.loss_parambatch)
            + FLAGS.loss_factor_kp * torch.nan_to_num(losses.loss_kpbatch)
            + FLAGS.loss_factor_dense * torch.nan_to_num(losses.loss_densebatch)
            + FLAGS.loss_factor_2d * torch.nan_to_num(losses.loss_2dbatch)
            + FLAGS.lbo_eigval_regul * losses.eigval_regul
        )

        for name, value in losses.items():
            if not torch.isfinite(value).all():
                print(f"Nonfinite {name}!, {value}")

        return losses

    @torch.jit.export
    def compute_loss_with_3d_gt(self, inps, preds):
        losses = EasyDict()

        if inps.point_validity is None:
            inps.point_validity = torch.ones_like(
                preds.coords3d_abs[..., 0], dtype=torch.bool
            )

        diff = inps.coords3d_true - preds.coords3d_abs

        # CENTER-RELATIVE 3D LOSS
        # We now compute a "center-relative" error, which is either root-relative
        # (if there is a root joint present), or mean-relative (i.e. the mean is subtracted).
        meanrel_diff = ptu3d.center_relative_pose(
            diff, joint_validity_mask=inps.point_validity, center_is_mean=True
        )
        # root_index is a [batch_size] int tensor that holds which one is the root
        # we now need to select the root joint from each batch element
        # diff has shape N,P,3 for batch, point, coord
        sanitized_root_index = torch.where(
            inps.root_index == -1, torch.zeros_like(inps.root_index), inps.root_index
        )
        root_diff = diff[
            torch.arange(diff.shape[0], device=diff.device), sanitized_root_index
        ].unsqueeze(1)

        rootrel_diff = diff - root_diff
        # Some elements of the batch do not have a root joint, which is marked as -1 as root_index.
        # For these elements we use the mean-relative error.
        center_relative_diff = torch.where(
            inps.root_index[:, torch.newaxis, torch.newaxis] == -1,
            meanrel_diff,
            rootrel_diff,
        )

        losses.loss3d = ptu.reduce_mean_masked(
            custom_norm(center_relative_diff, preds.uncert), inps.point_validity
        )

        # ABSOLUTE 3D LOSS (camera-space)
        absdiff = torch.abs(diff)

        # Since the depth error will naturally scale linearly with distance, we scale the z-error
        # down to the level that we would get if the person was 5 m away.
        scale_factor_for_far = torch.clamp_max(
            5.0 / torch.abs(inps.coords3d_true[..., 2:]), 1.0
        )
        absdiff_scaled = torch.cat(
            [absdiff[..., :2], absdiff[..., 2:] * scale_factor_for_far], dim=-1
        )

        # There are numerical difficulties for points too close to the camera, so we only
        # apply the absolute loss for points at least 30 cm away from the camera.
        is_far_enough = inps.coords3d_true[..., 2] > 0.3
        is_valid_and_far_enough = torch.logical_and(inps.point_validity, is_far_enough)

        # To make things simpler, we estimate one uncertainty and automatically
        # apply a factor of 4 to get the uncertainty for the absolute prediction
        # this is just an approximation, but it works well enough.
        # The uncertainty does not need to be perfect, it merely serves as a
        # self-gating mechanism, and the actual value of it is less important
        # compared to the relative values between different points.
        losses.loss3d_abs = ptu.reduce_mean_masked(
            custom_norm(absdiff_scaled, preds.uncert * 4.0), is_valid_and_far_enough
        )

        # 2D PROJECTION LOSS (pixel-space)
        # We also compute a loss in pixel space to encourage good image-alignment in the model.
        coords2d_pred = ptu3d.project_pose(preds.coords3d_abs, inps.intrinsics)
        coords2d_true = ptu3d.project_pose(inps.coords3d_true, inps.intrinsics)

        # Balance factor which considers the 2D image size equivalent to the 3D box size of the
        # volumetric heatmap. This is just a factor to get a rough ballpark.
        # It could be tuned further.
        scale_2d = 1.0 / FLAGS.proc_side * FLAGS.box_size_m

        # We only use the 2D loss for points that are in front of the camera and aren't
        # very far out of the field of view. It's not a problem that the point is outside
        # to a certain extent, because this will provide training signal to move points which
        # are outside the image, toward the image border. Therefore those point predictions
        # will gather up near the border and we can mask them out when doing the absolute
        # reconstruction.
        is_in_fov_pred = torch.logical_and(
            ptu3d.is_within_fov(
                coords2d_pred,
                border_factor=-20 * (FLAGS.proc_side / 256),
                centered_stride=FLAGS.centered_stride,
                stride=FLAGS.stride_train,
                proc_side=FLAGS.proc_side,
            ),
            preds.coords3d_abs[..., 2] > 0.001,
        )
        is_near_fov_true = torch.logical_and(
            ptu3d.is_within_fov(
                coords2d_true,
                border_factor=-20 * (FLAGS.proc_side / 256),
                centered_stride=FLAGS.centered_stride,
                stride=FLAGS.stride_train,
                proc_side=FLAGS.proc_side,
            ),
            inps.coords3d_true[..., 2] > 0.001,
        )

        if "fovtrueonly" in FLAGS.custom:
            is_near_fov = is_near_fov_true
        else:
            is_near_fov = torch.logical_and(is_near_fov_true, is_in_fov_pred)

        losses.loss2d = ptu.reduce_mean_masked(
            custom_norm((coords2d_true - coords2d_pred) * scale_2d, preds.uncert),
            torch.logical_and(is_valid_and_far_enough, is_near_fov),
        )

        return losses, (
            losses.loss3d
            + losses.loss2d
            + FLAGS.absloss_factor
            * self.stop_grad_before_step(losses.loss3d_abs, FLAGS.absloss_start_step)
        )

    @torch.jit.export
    def compute_loss_with_2d_gt(self, inps, preds):
        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_m
        coords2d_pred = ptu3d.project_pose(preds.coords3d_abs, inps.intrinsics)

        is_in_fov_pred2d = ptu3d.is_within_fov(
            coords2d_pred,
            border_factor=-20 * (FLAGS.proc_side / 256),
            centered_stride=FLAGS.centered_stride,
            stride=FLAGS.stride_train,
            proc_side=FLAGS.proc_side,
        )
        is_near_fov_true2d = ptu3d.is_within_fov(
            inps.coords2d_true,
            border_factor=-20 * (FLAGS.proc_side / 256),
            centered_stride=FLAGS.centered_stride,
            stride=FLAGS.stride_train,
            proc_side=FLAGS.proc_side,
        )

        if "fovtrueonly" in FLAGS.custom:
            is_near_fov = is_near_fov_true2d
        else:
            is_near_fov = torch.logical_and(is_near_fov_true2d, is_in_fov_pred2d)

        return ptu.reduce_mean_masked(
            custom_norm((inps.coords2d_true - coords2d_pred) * scale_2d, preds.uncert),
            torch.logical_and(is_near_fov, inps.point_validity),
        )

    @torch.jit.export
    def compute_metrics(self, inps, preds):
        metrics = EasyDict()
        metrics_parambatch = self.compute_metrics_with_3d_gt(inps.param, preds.param)
        if not self.training:
            return metrics_parambatch

        metrics_kp = self.compute_metrics_with_3d_gt(inps.kp3d, preds.kp3d, "_kp")
        metrics_dense = self.compute_metrics_with_2d_gt(
            inps.dense, preds.dense, "_dense"
        )
        metrics_2d = self.compute_metrics_with_2d_gt(inps.kp2d, preds.kp2d, "_2d")

        metrics.update(
            **metrics_parambatch, **metrics_kp, **metrics_dense, **metrics_2d
        )
        return metrics

    def compute_metrics_with_3d_gt(self, inps, preds, suffix: str = ""):
        metrics = EasyDict()
        diff = inps.coords3d_true - preds.coords3d_abs

        # ABSOLUTE
        metrics["mean_error_abs" + suffix] = (
            ptu.reduce_mean_masked(torch.norm(diff, dim=-1), inps.point_validity) * 1000
        )

        # RELATIVE
        meanrel_absdiff = torch.abs(
            ptu3d.center_relative_pose(
                diff, joint_validity_mask=inps.point_validity, center_is_mean=True
            )
        )
        dist = torch.norm(meanrel_absdiff, dim=-1)
        metrics["mean_error" + suffix] = (
            ptu.reduce_mean_masked(dist, inps.point_validity) * 1000
        )

        # PCK/AUC
        auc_score = ptu.auc(dist, 0.0, 0.1)
        metrics["auc" + suffix] = (
            ptu.reduce_mean_masked(auc_score, inps.point_validity) * 100
        )
        is_correct = (dist <= 0.1).float()
        metrics["pck" + suffix] = (
            ptu.reduce_mean_masked(is_correct, inps.point_validity) * 100
        )

        # PROCRUSTES
        coords3d_pred_procrustes = ptu3d.rigid_align(
            preds.coords3d_abs,
            inps.coords3d_true,
            joint_validity_mask=inps.point_validity,
            scale_align=True,
        )
        dist_procrustes = torch.norm(
            coords3d_pred_procrustes - inps.coords3d_true, dim=-1
        )
        metrics["mean_error_procrustes" + suffix] = (
            ptu.reduce_mean_masked(dist_procrustes, inps.point_validity) * 1000
        )

        # PROJECTION
        coords2d_pred = ptu3d.project_pose(preds.coords3d_abs, inps.intrinsics)
        coords2d_true = ptu3d.project_pose(inps.coords3d_true, inps.intrinsics)
        scale = 256 / FLAGS.proc_side
        metrics["mean_error_px" + suffix] = ptu.reduce_mean_masked(
            torch.norm((coords2d_true - coords2d_pred) * scale, dim=-1),
            inps.point_validity,
        )

        return metrics

    def compute_metrics_with_2d_gt(self, inps, preds, suffix: str = ""):
        metrics = EasyDict()
        scale = 256 / FLAGS.proc_side
        coords2d_pred = ptu3d.project_pose(preds.coords3d_abs, inps.intrinsics)
        metrics["mean_error_px" + suffix] = ptu.reduce_mean_masked(
            torch.norm((inps.coords2d_true - coords2d_pred) * scale, dim=-1),
            inps.point_validity,
        )
        return metrics

    def reconstruct_absolute(
        self,
        head2d,
        head3d,
        intrinsics,
        mix_3d_inside_fov,
        point_validity_mask: Optional[torch.Tensor] = None,
    ):
        full_perspective_start_step = 500 if FLAGS.dual_finetune_lr else 5000
        if (
            FLAGS.weak_perspective
            or self.adjusted_train_counter < full_perspective_start_step
        ):
            return ptu3d.reconstruct_absolute(
                head2d,
                head3d,
                intrinsics,
                mix_3d_inside_fov=mix_3d_inside_fov,
                weak_perspective=True,
                point_validity_mask=point_validity_mask,
                border_factor1=1,
                border_factor2=0.55,
                mix_based_on_3d=False,
                proc_side=FLAGS.proc_side,
                stride=FLAGS.stride_train,
                centered_stride=FLAGS.centered_stride,
            )
        else:
            return ptu3d.reconstruct_absolute(
                head2d,
                head3d,
                intrinsics,
                mix_3d_inside_fov=mix_3d_inside_fov,
                weak_perspective=False,
                point_validity_mask=point_validity_mask,
                border_factor1=1,
                border_factor2=0.55,
                mix_based_on_3d=False,
                proc_side=FLAGS.proc_side,
                stride=FLAGS.stride_train,
                centered_stride=FLAGS.centered_stride,
            )

    def stop_grad_before_step(self, x, step: int):
        if self.adjusted_train_counter >= step:
            return x

        if torch.all(torch.isfinite(x)):
            return x.detach()
        else:
            return torch.zeros_like(x)


def scaled_euclidean_norm(x: torch.Tensor):
    dim = torch.tensor(x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.linalg.norm(x, dim=-1, keepdim=True) * torch.rsqrt(dim)


def custom_norm(x: torch.Tensor, uncert: Optional[torch.Tensor] = None):
    if uncert is None:
        return ptu.charbonnier(x, epsilon=2e-2, dim=-1)

    if FLAGS.nll_loss:
        dim = torch.tensor(x.shape[-1], dtype=x.dtype, device=x.device)
        beta_comp_factor = uncert.detach() ** FLAGS.beta_nll if FLAGS.beta_nll else 1.0

        factor = torch.rsqrt(dim) if FLAGS.fix_uncert_factor else torch.sqrt(dim)
        return (
            ptu.charbonnier(
                x / torch.unsqueeze(uncert, -1), epsilon=FLAGS.charb_eps, dim=-1
            )
            + factor * torch.log(uncert)
        ) * beta_comp_factor
    else:
        return ptu.charbonnier(x, epsilon=FLAGS.charb_eps, dim=-1)


def interpolate_sparse(
    source_points: torch.Tensor, sparse_mats: Sequence[torch.Tensor]
):
    if source_points.shape[0] == 0:
        return torch.zeros((0, FLAGS.num_points, 3), device=source_points.device)
    return (block_diag_csr(sparse_mats) @ source_points.reshape(-1, 3)).reshape(
        source_points.shape[0], -1, 3
    )


def block_diag_csr(csr_mats: Sequence[torch.Tensor]):
    """Returns a block diagonal CSR tensor built from the given CSR tensors as the blocks."""
    device = csr_mats[0].device
    crow_dtype = csr_mats[0].crow_indices().dtype
    col_dtype = csr_mats[0].col_indices().dtype

    row_offsets = torch.tensor(
        [0] + [csr.col_indices().shape[0] for csr in csr_mats],
        dtype=crow_dtype,
        device=device,
    ).cumsum(0, dtype=crow_dtype)
    col_offsets = torch.tensor(
        [0] + [csr.shape[1] for csr in csr_mats], dtype=col_dtype, device=device
    ).cumsum(0, dtype=col_dtype)
    return torch.sparse_csr_tensor(
        torch.cat(
            [
                csr.crow_indices()[:-1] + offset
                for csr, offset in zip(csr_mats, row_offsets)
            ]
            + [row_offsets[-1:]]
        ),
        torch.cat(
            [csr.col_indices() + offset for csr, offset in zip(csr_mats, col_offsets)]
        ),
        torch.cat([csr.values() for csr in csr_mats]),
        (sum(c.shape[0] for c in csr_mats), sum(c.shape[1] for c in csr_mats)),
    )


def nested_cat(a: torch.Tensor, b: torch.Tensor):
    return torch.nested.nested_tensor(
        [torch.cat(xy, dim=0) for xy in zip(a.unbind(0), b.unbind(0))]
    )


def index_select_with_nested_indices(tensor: torch.Tensor, index: torch.Tensor):
    return torch.nested.nested_tensor(
        [torch.index_select(tensor, 0, i) for i in index.unbind(0)]
    )


def nested_to_tensor_and_mask(x: torch.Tensor):
    padded = x.to_padded_tensor(
        float("nan"), output_size=(x.size(0), FLAGS.num_points, x.size(2))
    )
    mask = torch.isfinite(padded).all(dim=-1)
    return padded.nan_to_num(0.0), mask
