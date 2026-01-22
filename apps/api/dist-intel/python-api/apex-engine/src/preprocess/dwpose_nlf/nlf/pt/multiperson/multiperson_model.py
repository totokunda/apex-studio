import math
from typing import Dict, List, Optional
import smplfitter.pt
import torch
import torchvision.transforms.v2.functional
from pt import ptu, ptu3d
from pt.multiperson import plausibility_check as plausib, warping
from pt.models.bodymodel import BodyModel

# Dummy value which will mean that the intrinsic_matrix are unknown
UNKNOWN_INTRINSIC_MATRIX = ((-1, -1, -1), (-1, -1, -1), (-1, -1, -1))
DEFAULT_EXTRINSIC_MATRIX = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
DEFAULT_DISTORTION = (0, 0, 0, 0, 0)
DEFAULT_WORLD_UP = (0, -1, 0)


class MultipersonNLF(torch.nn.Module):
    def __init__(
        self,
        crop_model,
        detector,
        pad_white_pixels=True,
        smpl_config=None,
        smplx_config=None,
        fitter_smpl_config=None,
        fitter_smplx_config=None,
        num_betas=10,
    ):
        super().__init__()

        self.crop_model = crop_model
        self.detector = detector
        device = next(self.crop_model.parameters()).device

        self.body_models = torch.nn.ModuleDict(
            dict(
                smpl=BodyModel(
                    model_name="smpl", num_betas=num_betas, model_config=smpl_config
                ),
                smplx=BodyModel(
                    model_name="smplx", num_betas=num_betas, model_config=smplx_config
                ),
            )
        )

        body_models_partial = {
            k: BodyModel(
                model_name=k,
                num_betas=num_betas,
                vertex_subset_size=1024,
                model_config=fitter_smpl_config if k == "smpl" else fitter_smplx_config,
            )
            for k in self.body_models
        }

        self.cano_all = {}
        for k in self.body_models:
            buffer = torch.zeros(
                self.body_models[k].model_config["cano_all"],
                dtype=torch.float32,
                device=device,
            )
            self.register_buffer(f"cano_all_{k}", buffer, persistent=True)
            self.cano_all[k] = buffer

        nothing = torch.zeros([], device=device, dtype=torch.float32)

        # This is not initialized here to save disk space. It will be computed on first use.
        # They will be the weights read out from the localizer field for self.cano_all
        self.weights = {
            k: dict(
                w_tensor=nothing,
                b_tensor=nothing,
                w_tensor_flipped=nothing,
                b_tensor_flipped=nothing,
            )
            for k in self.body_models
        }

        self.fitters = torch.nn.ModuleDict(
            dict(
                smpl=smplfitter.pt.BodyFitter(body_models_partial["smpl"]),
                smplx=smplfitter.pt.BodyFitter(body_models_partial["smplx"]),
            )
        )

        self.pad_white_pixels = pad_white_pixels

    def _apply(self, fn):
        # PyTorch moves registered buffers/params by replacing the tensor objects.
        # Any extra Python references (like entries inside dicts) will still point to the
        # old tensors unless we refresh them after the move.
        ret = super()._apply(fn)
        if hasattr(self, "cano_all") and hasattr(self, "body_models"):
            for k in self.body_models:
                name = f"cano_all_{k}"
                if hasattr(self, name):
                    self.cano_all[k] = getattr(self, name)
        return ret

    @torch.jit.export
    def detect_parametric_batched(
        self,
        images: torch.Tensor,
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        detector_threshold: float = 0.3,
        detector_nms_iou_threshold: float = 0.7,
        max_detections: int = 150,
        detector_flip_aug: bool = False,
        detector_both_flip_aug: bool = False,
        suppress_implausible_poses: bool = True,
        beta_regularizer: float = 10.0,
        beta_regularizer2: float = 0.0,
        model_name: str = "smpl",
        extra_boxes: Optional[List[torch.Tensor]] = None,
    ):
        images = im_to_linear(images)

        boxes = self.detector(
            images=images,
            threshold=detector_threshold,
            nms_iou_threshold=detector_nms_iou_threshold,
            max_detections=max_detections,
            extrinsic_matrix=extrinsic_matrix,
            world_up_vector=world_up_vector,
            flip_aug=detector_flip_aug,
            bothflip_aug=detector_both_flip_aug,
            extra_boxes=extra_boxes,
        )
        return self._estimate_parametric_batched(
            images,
            boxes,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses,
            beta_regularizer,
            beta_regularizer2,
            model_name,
        )

    @torch.jit.export
    def estimate_parametric_batched(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        beta_regularizer: float = 10.0,
        beta_regularizer2: float = 0.0,
        model_name: str = "smpl",
    ):
        return self._estimate_parametric_batched(
            images,
            boxes,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=False,
            beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2,
            model_name=model_name,
        )

    detect_smpl_batched = detect_parametric_batched
    estimate_smpl_batched = estimate_parametric_batched

    def _estimate_parametric_batched(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        suppress_implausible_poses: bool = True,
        beta_regularizer: float = 10.0,
        beta_regularizer2: float = 0.0,
        model_name: str = "smpl",
    ):
        if model_name not in self.body_models:
            raise ValueError(
                f"Unknown model name {model_name}, use one of {self.body_models.keys()}"
            )

        if self.weights[model_name]["w_tensor"].ndim == 0:
            self.weights[model_name] = self.get_weights_for_canonical_points(
                self.cano_all[model_name]
            )

        result = self._estimate_poses_batched(
            images,
            boxes,
            self.weights[model_name],
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=suppress_implausible_poses,
        )
        boxes = result["boxes"]
        n_pose_per_image_list = [len(b) for b in boxes]
        if sum(n_pose_per_image_list) == 0:
            return self._predict_empty_parametric(images, model_name)

        poses3d_flat = torch.cat(result["poses3d"], dim=0)
        mean_poses = torch.mean(poses3d_flat, dim=-2, keepdim=True)
        poses3d_flat = poses3d_flat - mean_poses
        poses2d_flat = torch.cat(result["poses2d"], dim=0)
        uncertainties_flat = torch.cat(result["uncertainties"], dim=0)

        fitter = self.fitters["smpl"] if model_name == "smpl" else self.fitters["smplx"]
        vertices_flat, joints_flat = torch.split(
            poses3d_flat,
            [fitter.body_model.num_vertices, fitter.body_model.num_joints],
            dim=-2,
        )
        vertex_uncertainties_flat, joint_uncertainties_flat = torch.split(
            uncertainties_flat,
            [fitter.body_model.num_vertices, fitter.body_model.num_joints],
            dim=-1,
        )

        vertex_weights = vertex_uncertainties_flat**-1.5
        vertex_weights = vertex_weights / torch.mean(
            vertex_weights, dim=-1, keepdim=True
        )
        joint_weights = joint_uncertainties_flat**-1.5
        joint_weights = joint_weights / torch.mean(joint_weights, dim=-1, keepdim=True)

        fit = fitter.fit(
            vertices_flat / 1000,
            joints_flat / 1000,
            vertex_weights=vertex_weights,
            joint_weights=joint_weights,
            num_iter=3,
            beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2,
            final_adjust_rots=True,
            requested_keys=["pose_rotvecs", "shape_betas", "trans"],
        )

        result["pose"] = torch.split(fit["pose_rotvecs"], n_pose_per_image_list)
        result["betas"] = torch.split(fit["shape_betas"], n_pose_per_image_list)
        result["trans"] = torch.split(
            fit["trans"] + mean_poses.squeeze(-2) / 1000, n_pose_per_image_list
        )

        body_model: smplfitter.pt.BodyModel = (
            self.body_models["smpl"]
            if model_name == "smpl"
            else self.body_models["smplx"]
        )
        fit_res = body_model.forward(
            fit["pose_rotvecs"],
            fit["shape_betas"],
            fit["trans"] + mean_poses.squeeze(-2) / 1000,
        )
        fit_vertices_flat = fit_res["vertices"] * 1000
        fit_joints_flat = fit_res["joints"] * 1000
        result["vertices3d"] = torch.split(fit_vertices_flat, n_pose_per_image_list)
        result["joints3d"] = torch.split(fit_joints_flat, n_pose_per_image_list)

        result["vertices2d"] = project_ragged(
            images,
            [x for x in result["vertices3d"]],
            extrinsic_matrix,
            intrinsic_matrix,
            distortion_coeffs,
            default_fov_degrees,
        )
        result["joints2d"] = project_ragged(
            images,
            [x for x in result["joints3d"]],
            extrinsic_matrix,
            intrinsic_matrix,
            distortion_coeffs,
            default_fov_degrees,
        )

        result["vertices3d_nonparam"] = torch.split(
            vertices_flat + mean_poses, n_pose_per_image_list
        )
        result["joints3d_nonparam"] = torch.split(
            joints_flat + mean_poses, n_pose_per_image_list
        )
        vertices2d, joints2d = torch.split(
            poses2d_flat,
            [fitter.body_model.num_vertices, fitter.body_model.num_joints],
            dim=-2,
        )

        result["vertices2d_nonparam"] = torch.split(vertices2d, n_pose_per_image_list)
        result["joints2d_nonparam"] = torch.split(joints2d, n_pose_per_image_list)

        result["vertex_uncertainties"] = torch.split(
            vertex_uncertainties_flat * 1000, n_pose_per_image_list
        )
        result["joint_uncertainties"] = torch.split(
            joint_uncertainties_flat * 1000, n_pose_per_image_list
        )

        del result["poses3d"]
        del result["poses2d"]
        del result["uncertainties"]
        return result

    @torch.jit.export
    def detect_poses_batched(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        detector_threshold: float = 0.3,
        detector_nms_iou_threshold: float = 0.7,
        max_detections: int = 150,
        detector_flip_aug: bool = False,
        detector_both_flip_aug: bool = False,
        suppress_implausible_poses: bool = True,
        extra_boxes: Optional[List[torch.Tensor]] = None,
    ):

        images = im_to_linear(images)

        boxes = self.detector(
            images=images,
            threshold=detector_threshold,
            nms_iou_threshold=detector_nms_iou_threshold,
            max_detections=max_detections,
            extrinsic_matrix=extrinsic_matrix,
            world_up_vector=world_up_vector,
            flip_aug=detector_flip_aug,
            bothflip_aug=detector_both_flip_aug,
            extra_boxes=extra_boxes,
        )

        return self._estimate_poses_batched(
            images,
            boxes,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses,
        )

    @torch.jit.export
    def estimate_poses_batched(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 5,
        rot_aug_max_degrees: float = 25.0,
    ):
        boxes = [torch.cat([b, torch.ones_like(b[..., :1])], dim=-1) for b in boxes]
        pred = self._estimate_poses_batched(
            images,
            boxes,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            suppress_implausible_poses=False,
        )
        del pred["boxes"]
        return pred

    def _estimate_poses_batched(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor],
        distortion_coeffs: Optional[torch.Tensor],
        extrinsic_matrix: Optional[torch.Tensor],
        world_up_vector: Optional[torch.Tensor],
        default_fov_degrees: float,
        internal_batch_size: int,
        antialias_factor: int,
        num_aug: int,
        rot_aug_max_degrees: float,
        suppress_implausible_poses: bool,
    ):
        if sum(len(b) for b in boxes) == 0:
            return self._predict_empty(images, weights)

        images = im_to_linear(images)

        n_images = len(images)
        device = images.device
        # If one intrinsic matrix is given, repeat it for all images

        if intrinsic_matrix is None:
            intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
                default_fov_degrees, images.shape[2:4], device=device
            )

        if len(intrinsic_matrix) == 1:
            # If intrinsic_matrix is not given, fill it in based on field of view
            intrinsic_matrix = torch.repeat_interleave(
                intrinsic_matrix, n_images, dim=0
            )

        if distortion_coeffs is None:
            distortion_coeffs = torch.zeros((n_images, 5), device=device)
        # If one distortion coeff/extrinsic matrix is given, repeat it for all images
        if len(distortion_coeffs) == 1:
            distortion_coeffs = torch.repeat_interleave(
                distortion_coeffs, n_images, dim=0
            )

        if extrinsic_matrix is None:
            extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
        if len(extrinsic_matrix) == 1:
            extrinsic_matrix = torch.repeat_interleave(
                extrinsic_matrix, n_images, dim=0
            )

        # Now repeat these camera params for each box
        n_box_per_image_list = [len(b) for b in boxes]
        n_box_per_image = torch.tensor([len(b) for b in boxes], device=device)

        intrinsic_matrix = torch.repeat_interleave(
            intrinsic_matrix, n_box_per_image, dim=0
        )
        distortion_coeffs = torch.repeat_interleave(
            distortion_coeffs, n_box_per_image, dim=0
        )

        # Up-vector in camera-space
        if world_up_vector is None:
            world_up_vector = torch.tensor(
                [0, -1, 0], device=device, dtype=torch.float32
            )

        camspace_up = torch.einsum(
            "c,bCc->bC", world_up_vector, extrinsic_matrix[..., :3, :3]
        )
        camspace_up = torch.repeat_interleave(camspace_up, n_box_per_image, dim=0)

        # Set up the test-time augmentation parameters
        aug_gammas = ptu.linspace(0.6, 1.0, num_aug, dtype=torch.float32, device=device)

        aug_angle_range = rot_aug_max_degrees * (torch.pi / 180.0)
        aug_angles = ptu.linspace(
            -aug_angle_range,
            aug_angle_range,
            num_aug,
            dtype=torch.float32,
            device=device,
        )

        if num_aug == 1:
            aug_scales = torch.tensor([1.0], device=device, dtype=torch.float32)
        else:
            aug_scales = torch.cat(
                [
                    ptu.linspace(
                        0.8,
                        1.0,
                        num_aug // 2,
                        endpoint=False,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.linspace(
                        1.0,
                        1.1,
                        num_aug - num_aug // 2,
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=0,
            )
        aug_should_flip = (
            torch.arange(0, num_aug, device=device) - num_aug // 2
        ) % 2 != 0
        aug_flipmat = torch.tensor(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device
        )
        aug_maybe_flipmat = torch.where(
            aug_should_flip[:, torch.newaxis, torch.newaxis],
            aug_flipmat,
            torch.eye(3, device=device),
        )
        aug_rotmat = ptu3d.rotation_mat(-aug_angles, rot_axis="z")
        aug_rotflipmat = aug_maybe_flipmat @ aug_rotmat

        # crops_flat, poses3d_flat = self._predict_in_batches(
        poses3d_flat, uncert_flat = self._predict_in_batches(
            images,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            camspace_up,
            boxes,
            internal_batch_size,
            aug_should_flip,
            aug_rotflipmat,
            aug_gammas,
            aug_scales,
            antialias_factor,
        )
        poses3d_flat = plausib.scale_align(poses3d_flat)
        mean = torch.mean(poses3d_flat, dim=(-3, -2), keepdim=True)
        poses3d_flat_submean = (poses3d_flat - mean).float()
        poses3d_flat_submean, final_weights = weighted_geometric_median(
            poses3d_flat_submean, uncert_flat**-1.5, dim=-3, n_iter=10, eps=50.0
        )
        # MPS doesn't support double precision, so use float32 for MPS devices
        is_mps = poses3d_flat.device.type == "mps"
        if is_mps:
            poses3d_flat = poses3d_flat_submean.float() + mean.squeeze(1).float()
        else:
            poses3d_flat = poses3d_flat_submean.double() + mean.squeeze(1).double()

        uncert_flat = weighted_mean(uncert_flat, final_weights, dim=-2)

        # Project the 3D poses to get the 2D poses
        poses2d_flat_normalized = ptu3d.to_homogeneous(
            warping.distort_points(
                ptu3d.project(poses3d_flat.float()), distortion_coeffs
            )
        )
        poses2d_flat = torch.einsum(
            "bnk,bjk->bnj", poses2d_flat_normalized, intrinsic_matrix[:, :2, :]
        )

        # Arrange the results back into ragged tensors
        poses3d = torch.split(poses3d_flat, n_box_per_image_list)
        poses2d = torch.split(poses2d_flat, n_box_per_image_list)
        uncert = torch.split(uncert_flat, n_box_per_image_list)

        if suppress_implausible_poses:
            # Filter the resulting poses for individual plausibility to reduce false positives
            boxes, poses3d, poses2d, uncert = self._filter_poses(
                boxes, poses3d, poses2d, uncert
            )

        n_box_per_image_list = [len(b) for b in boxes]

        if sum(n_box_per_image_list) == 0:
            return self._predict_empty(images, weights)

        n_box_per_image = torch.tensor(n_box_per_image_list, device=device)
        # Convert to world coordinates
        # MPS doesn't support double precision, so use float32 for MPS devices
        is_mps = device.type == "mps"
        if is_mps:
            extrinsic_matrix_dtype = extrinsic_matrix.float()
        else:
            extrinsic_matrix_dtype = extrinsic_matrix.double()
        inv_extrinsic_matrix = torch.repeat_interleave(
            torch.linalg.inv(extrinsic_matrix_dtype), n_box_per_image, dim=0
        )
        poses3d_flat = torch.einsum(
            "bnk,bjk->bnj",
            ptu3d.to_homogeneous(torch.cat(poses3d)),
            inv_extrinsic_matrix[:, :3, :],
        )
        poses3d = torch.split(poses3d_flat.float(), n_box_per_image_list)

        result = dict(
            boxes=boxes, poses3d=poses3d, poses2d=poses2d, uncertainties=uncert
        )
        return result

    def _filter_poses(
        self,
        boxes: List[torch.Tensor],
        poses3d: List[torch.Tensor],
        poses2d: List[torch.Tensor],
        uncert: List[torch.Tensor],
    ):
        boxes_out = []
        poses3d_out = []
        poses2d_out = []
        uncert_out = []
        for boxes_, poses3d_, poses2d_, uncert_ in zip(boxes, poses3d, poses2d, uncert):
            is_uncert_low = plausib.is_uncertainty_low(uncert_)
            is_pose_consi = plausib.is_pose_consistent_with_box(poses2d_, boxes_)
            plausible_mask = torch.logical_and(is_uncert_low, is_pose_consi)
            nms_indices = plausib.pose_non_max_suppression(
                poses3d_, boxes_[..., 4] / torch.mean(uncert_, dim=-1), plausible_mask
            )
            boxes_out.append(boxes_[nms_indices])
            poses3d_out.append(poses3d_[nms_indices])
            poses2d_out.append(poses2d_[nms_indices])
            uncert_out.append(uncert_[nms_indices])
        return boxes_out, poses3d_out, poses2d_out, uncert_out

    def _predict_in_batches(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: List[torch.Tensor],
        internal_batch_size: int,
        aug_should_flip: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_gammas: torch.Tensor,
        aug_scales: torch.Tensor,
        antialias_factor: int,
    ):
        num_aug = len(aug_gammas)
        boxes_per_batch = internal_batch_size // num_aug
        boxes_flat = torch.cat(boxes, dim=0)
        image_id_per_box = torch.repeat_interleave(
            torch.arange(len(boxes)), torch.tensor([len(b) for b in boxes])
        )

        if boxes_per_batch == 0:
            # Run all as a single batch
            return self._predict_single_batch(
                images,
                weights,
                intrinsic_matrix,
                distortion_coeffs,
                camspace_up,
                boxes_flat,
                image_id_per_box,
                aug_rotflipmat,
                aug_should_flip,
                aug_scales,
                aug_gammas,
                antialias_factor,
            )
        else:
            # Chunk the image crops into batches and predict them one by one
            n_total_boxes = len(boxes_flat)
            n_batches = int(math.ceil(n_total_boxes / boxes_per_batch))
            poses3d_batches = []
            uncert_batches = []
            # CROP
            # crop_batches = []
            for i in range(n_batches):
                batch_slice = slice(i * boxes_per_batch, (i + 1) * boxes_per_batch)
                # CROP
                poses3d, uncert = self._predict_single_batch(
                    images,
                    weights,
                    intrinsic_matrix[batch_slice],
                    distortion_coeffs[batch_slice],
                    camspace_up[batch_slice],
                    boxes_flat[batch_slice],
                    image_id_per_box[batch_slice],
                    aug_rotflipmat,
                    aug_should_flip,
                    aug_scales,
                    aug_gammas,
                    antialias_factor,
                )
                poses3d_batches.append(poses3d)
                uncert_batches.append(uncert)
            return torch.cat(poses3d_batches, dim=0), torch.cat(uncert_batches, dim=0)

    def _predict_single_batch(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: torch.Tensor,
        image_ids: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_should_flip: torch.Tensor,
        aug_scales: torch.Tensor,
        aug_gammas: torch.Tensor,
        antialias_factor: int,
    ):
        # Get crops and info about the transformation used to create them
        # Each has shape [num_aug, n_boxes, ...]
        crops, new_intrinsic_matrix, R = self._get_crops(
            images,
            intrinsic_matrix,
            distortion_coeffs,
            camspace_up,
            boxes,
            image_ids,
            aug_rotflipmat,
            aug_scales,
            aug_gammas,
            antialias_factor,
        )

        # Flatten each and predict the pose with the crop model
        new_intrinsic_matrix_flat = torch.reshape(new_intrinsic_matrix, (-1, 3, 3))
        res = self.crop_model.input_resolution
        crops_flat = torch.reshape(crops, (-1, 3, res, res))

        n_cases = crops.shape[1]
        aug_should_flip_flat = torch.repeat_interleave(aug_should_flip, n_cases, dim=0)

        poses_flat, uncert_flat = self.crop_model.predict_multi_same_weights(
            crops_flat, new_intrinsic_matrix_flat, weights, aug_should_flip_flat
        )
        # MPS doesn't support double precision, so use float32 for MPS devices
        is_mps = poses_flat.device.type == "mps"
        if is_mps:
            poses_flat = poses_flat.float()
            R_dtype = R.float()
        else:
            poses_flat = poses_flat.double()
            R_dtype = R.double()
        n_joints = poses_flat.shape[-2]

        poses = torch.reshape(poses_flat, [-1, n_cases, n_joints, 3])
        uncert = torch.reshape(uncert_flat, [-1, n_cases, n_joints])
        poses_orig_camspace = poses @ R_dtype

        # Transpose to [n_boxes, num_aug, ...]
        return poses_orig_camspace.transpose(0, 1), uncert.transpose(0, 1)
        # CROP
        # crops = torch.reshape(crops_flat, [num_aug, -1, 3, res, res])
        # return crops.transpose(0, 1), poses_orig_camspace.transpose(0, 1)

    def _get_crops(
        self,
        images: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: torch.Tensor,
        image_ids: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_scales: torch.Tensor,
        aug_gammas: torch.Tensor,
        antialias_factor: int,
    ):
        R_noaug, box_scales = self._get_new_rotation_and_scale(
            intrinsic_matrix, distortion_coeffs, camspace_up, boxes
        )

        device = images.device
        # How much we need to scale overall, taking scale augmentation into account
        # From here on, we introduce the dimension of augmentations
        crop_scales = aug_scales[:, torch.newaxis] * box_scales[torch.newaxis, :]
        # Build the new intrinsic matrix
        num_box = boxes.shape[0]
        num_aug = aug_gammas.shape[0]
        res = self.crop_model.input_resolution
        new_intrinsic_matrix = torch.cat(
            [
                torch.cat(
                    [
                        # Top-left of original intrinsic matrix gets scaled
                        intrinsic_matrix[torch.newaxis, :, :2, :2]
                        * crop_scales[:, :, torch.newaxis, torch.newaxis],
                        # Principal point is the middle of the new image size
                        torch.full(
                            (num_aug, num_box, 2, 1),
                            res / 2,
                            dtype=torch.float32,
                            device=device,
                        ),
                    ],
                    dim=3,
                ),
                torch.cat(
                    [
                        # [0, 0, 1] as the last row of the intrinsic matrix:
                        torch.zeros(
                            (num_aug, num_box, 1, 2), dtype=torch.float32, device=device
                        ),
                        torch.ones(
                            (num_aug, num_box, 1, 1), dtype=torch.float32, device=device
                        ),
                    ],
                    dim=3,
                ),
            ],
            dim=2,
        )
        R = aug_rotflipmat[:, torch.newaxis] @ R_noaug
        new_invprojmat = torch.linalg.inv(new_intrinsic_matrix @ R)

        # If we perform antialiasing through output scaling, we render a larger image first and then
        # shrink it. So we scale the homography first.
        if antialias_factor > 1:
            scaling_mat = warping.corner_aligned_scale_mat(1 / antialias_factor)
            new_invprojmat = new_invprojmat @ scaling_mat.to(new_invprojmat.device)

        # Note that x.neg_().add_(1) is equivalent to 1 - x, but it's done in-place
        # The 1-x is done because torch's warping pads with 0, but we want to pad with 1
        # because empirically it performs better on the trained model
        # (dependent on training config). Not needed for latest recipe.

        if self.pad_white_pixels:
            images.neg_().add_(1)

        crops = warping.warp_images_with_pyramid(
            images,
            intrinsic_matrix=torch.tile(intrinsic_matrix, [num_aug, 1, 1]),
            new_invprojmats=torch.reshape(new_invprojmat, [-1, 3, 3]),
            distortion_coeffs=torch.tile(distortion_coeffs, [num_aug, 1]),
            crop_scales=torch.reshape(crop_scales, [-1]) * antialias_factor,
            output_shape=(res * antialias_factor, res * antialias_factor),
            image_ids=torch.tile(image_ids, [num_aug]),
        )
        if self.pad_white_pixels:
            crops.neg_().add_(1).clamp_(0, 1)

        # Downscale the result if we do antialiasing through output scaling
        if antialias_factor == 2:
            crops = torch.nn.functional.avg_pool2d(crops, 2, 2)
        elif antialias_factor == 4:
            crops = torch.nn.functional.avg_pool2d(crops, 4, 4)
        elif antialias_factor > 4:
            crops = torchvision.transforms.v2.functional.resize(
                crops,
                (res, res),
                torchvision.transforms.v2.functional.InterpolationMode.BILINEAR,
                antialias=True,
            )
        crops = torch.reshape(crops, [num_aug, num_box, 3, res, res])
        # The division by 2.2 cancels the original gamma decoding from earlier
        crops **= torch.reshape(aug_gammas.to(crops.dtype) / 2.2, [-1, 1, 1, 1, 1])
        return crops, new_intrinsic_matrix, R

    def _get_new_rotation_and_scale(
        self, intrinsic_matrix, distortion_coeffs, camspace_up, boxes
    ):
        # Transform five points on each box: the center and the midpoints of the four sides
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxpoints_homog = ptu3d.to_homogeneous(
            torch.stack(
                [
                    torch.stack([x + w / 2, y + h / 2], dim=1),  # center
                    torch.stack([x + w / 2, y], dim=1),
                    torch.stack([x + w, y + h / 2], dim=1),
                    torch.stack([x + w / 2, y + h], dim=1),
                    torch.stack([x, y + h / 2], dim=1),
                ],
                dim=1,
            )
        )

        boxpoints_camspace = torch.einsum(
            "bpc,bCc->bpC", boxpoints_homog, torch.linalg.inv(intrinsic_matrix)
        )
        boxpoints_camspace = ptu3d.to_homogeneous(
            warping.undistort_points(boxpoints_camspace[:, :, :2], distortion_coeffs)
        )
        # Create a rotation matrix that will put the box center to the principal point
        # and apply the augmentation rotation and flip, to get the new coordinate frame
        box_center_camspace = boxpoints_camspace[:, 0]
        R_noaug = ptu3d.lookat_matrix(
            forward_vector=box_center_camspace, up_vector=camspace_up
        )

        # Transform the side midpoints of the box to the new coordinate frame
        sidepoints_camspace = boxpoints_camspace[:, 1:5]
        sidepoints_new = ptu3d.project(
            torch.einsum(
                "bpc,bCc->bpC", sidepoints_camspace, intrinsic_matrix @ R_noaug
            )
        )

        # Measure the size of the reprojected boxes
        vertical_size = torch.linalg.norm(
            sidepoints_new[:, 0] - sidepoints_new[:, 2], dim=-1
        )
        horiz_size = torch.linalg.norm(
            sidepoints_new[:, 1] - sidepoints_new[:, 3], dim=-1
        )
        box_size_new = torch.maximum(vertical_size, horiz_size)

        # How much we need to scale (zoom) to have the boxes fill out the final crop
        box_scales = (
            torch.tensor(self.crop_model.input_resolution, dtype=box_size_new.dtype)
            / box_size_new
        )
        return R_noaug, box_scales

    def _predict_empty(self, image: torch.Tensor, weights: Dict[str, torch.Tensor]):
        device = image.device
        n_joints = weights["w_tensor"].shape[0]
        poses3d = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        poses2d = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        uncert = torch.zeros((0, n_joints), dtype=torch.float32, device=device)
        boxes = torch.zeros((0, 5), dtype=torch.float32, device=device)
        n_images = image.shape[0]

        result = dict(
            boxes=[boxes] * n_images,
            poses3d=[poses3d] * n_images,
            poses2d=[poses2d] * n_images,
            uncertainties=[uncert] * n_images,
        )
        return result

    def _predict_empty_parametric(self, image: torch.Tensor, model_name: str):
        device = image.device
        fitter = self.fitters["smpl"] if model_name == "smpl" else self.fitters["smplx"]
        n_joints = fitter.body_model.num_joints
        n_verts = fitter.body_model.num_vertices
        pose = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        betas = torch.zeros(
            (0, fitter.body_model.num_betas), dtype=torch.float32, device=device
        )
        trans = torch.zeros((0, 3), dtype=torch.float32, device=device)
        vertices3d = torch.zeros((0, n_verts, 3), dtype=torch.float32, device=device)
        joints3d = torch.zeros((0, n_joints, 3), dtype=torch.float32, device=device)
        vertices2d = torch.zeros((0, n_verts, 2), dtype=torch.float32, device=device)
        joints2d = torch.zeros((0, n_joints, 2), dtype=torch.float32, device=device)
        vertices3d_nonparam = torch.zeros(
            (0, n_verts, 3), dtype=torch.float32, device=device
        )
        joints3d_nonparam = torch.zeros(
            (0, n_joints, 3), dtype=torch.float32, device=device
        )
        vertices2d_nonparam = torch.zeros(
            (0, n_verts, 2), dtype=torch.float32, device=device
        )
        joints2d_nonparam = torch.zeros(
            (0, n_joints, 2), dtype=torch.float32, device=device
        )
        vertex_uncertainties = torch.zeros(
            (0, n_verts), dtype=torch.float32, device=device
        )
        joint_uncertainties = torch.zeros(
            (0, n_joints), dtype=torch.float32, device=device
        )
        n_images = image.shape[0]
        result = dict(
            pose=[pose] * n_images,
            betas=[betas] * n_images,
            trans=[trans] * n_images,
            vertices3d=[vertices3d] * n_images,
            joints3d=[joints3d] * n_images,
            vertices2d=[vertices2d] * n_images,
            joints2d=[joints2d] * n_images,
            vertices3d_nonparam=[vertices3d_nonparam] * n_images,
            joints3d_nonparam=[joints3d_nonparam] * n_images,
            vertices2d_nonparam=[vertices2d_nonparam] * n_images,
            joints2d_nonparam=[joints2d_nonparam] * n_images,
            vertex_uncertainties=[vertex_uncertainties] * n_images,
            joint_uncertainties=[joint_uncertainties] * n_images,
        )
        return result

    @torch.jit.export
    def detect_poses(
        self,
        image: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 5,
        rot_aug_max_degrees: float = 25.0,
        detector_threshold: float = 0.3,
        detector_nms_iou_threshold: float = 0.7,
        max_detections: int = 150,
        detector_flip_aug: bool = False,
        detector_both_flip_aug: bool = False,
        suppress_implausible_poses: bool = True,
        extra_boxes: Optional[torch.Tensor] = None,
    ):

        images = image[torch.newaxis]
        intrinsic_matrix = (
            intrinsic_matrix[torch.newaxis] if intrinsic_matrix is not None else None
        )
        distortion_coeffs = (
            distortion_coeffs[torch.newaxis] if distortion_coeffs is not None else None
        )
        extrinsic_matrix = (
            extrinsic_matrix[torch.newaxis] if extrinsic_matrix is not None else None
        )
        extra_boxes = [extra_boxes] if extra_boxes is not None else None

        result = self.detect_poses_batched(
            images,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
            detector_threshold,
            detector_nms_iou_threshold,
            max_detections,
            detector_flip_aug,
            detector_both_flip_aug,
            suppress_implausible_poses,
            extra_boxes,
        )
        return {k: v[0] for k, v in result.items()}

    @torch.jit.export
    def estimate_poses(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: Optional[torch.Tensor] = None,
        distortion_coeffs: Optional[torch.Tensor] = None,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
    ):

        images = image[torch.newaxis]
        boxes = [boxes]
        intrinsic_matrix = (
            intrinsic_matrix[torch.newaxis] if intrinsic_matrix is not None else None
        )
        distortion_coeffs = (
            distortion_coeffs[torch.newaxis] if distortion_coeffs is not None else None
        )
        extrinsic_matrix = (
            extrinsic_matrix[torch.newaxis] if extrinsic_matrix is not None else None
        )

        result = self.estimate_poses_batched(
            images,
            boxes,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            extrinsic_matrix,
            world_up_vector,
            default_fov_degrees,
            internal_batch_size,
            antialias_factor,
            num_aug,
            rot_aug_max_degrees,
        )
        return {k: v[0] for k, v in result.items()}

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        return self.crop_model.get_weights_for_canonical_points(canonical_points)


def im_to_linear(im: torch.Tensor):
    if im.dtype == torch.uint8:
        return im.to(dtype=torch.float16).mul_(1.0 / 255.0).pow_(2.2)
    elif im.dtype == torch.uint16:
        return (
            im.to(dtype=torch.float16)
            .mul_(1.0 / 65504.0)
            .nan_to_num_(posinf=1.0)
            .pow_(2.2)
        )
        # return im.to(dtype=torch.float16).nan_to_num_(posinf=65504.0).div_(65504.0).pow_(2.2)
    elif im.dtype == torch.float16:
        return im**2.2
    else:
        return im.to(dtype=torch.float16).pow_(2.2)


def project_ragged(
    images: torch.Tensor,
    poses3d: List[torch.Tensor],
    extrinsic_matrix: Optional[torch.Tensor],
    intrinsic_matrix: Optional[torch.Tensor],
    distortion_coeffs: Optional[torch.Tensor],
    default_fov_degrees: float = 55.0,
):
    device = images.device
    n_box_per_image_list = [len(b) for b in poses3d]
    n_box_per_image = torch.tensor([len(b) for b in poses3d], device=device)

    n_images = images.shape[0]

    if intrinsic_matrix is None:
        intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
            default_fov_degrees, images.shape[2:4], device=device
        )

    # If one intrinsic matrix is given, repeat it for all images
    if intrinsic_matrix.shape[0] == 1:
        intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_images, dim=0)

    if distortion_coeffs is None:
        distortion_coeffs = torch.zeros((n_images, 5), device=device)

    # If one distortion coeff/extrinsic matrix is given, repeat it for all images
    if distortion_coeffs.shape[0] == 1:
        distortion_coeffs = torch.repeat_interleave(distortion_coeffs, n_images, dim=0)

    if extrinsic_matrix is None:
        extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
    if extrinsic_matrix.shape[0] == 1:
        extrinsic_matrix = torch.repeat_interleave(extrinsic_matrix, n_images, dim=0)

    intrinsic_matrix_rep = torch.repeat_interleave(
        intrinsic_matrix, n_box_per_image, dim=0
    )
    distortion_coeffs_rep = torch.repeat_interleave(
        distortion_coeffs, n_box_per_image, dim=0
    )
    extrinsic_matrix_rep = torch.repeat_interleave(
        extrinsic_matrix, n_box_per_image, dim=0
    )

    poses3d_flat = torch.cat(poses3d, dim=0)
    poses3d_flat = torch.einsum(
        "bnk,bjk->bnj",
        ptu3d.to_homogeneous(poses3d_flat),
        extrinsic_matrix_rep[:, :3, :],
    )

    poses2d_flat_normalized = ptu3d.to_homogeneous(
        warping.distort_points(ptu3d.project(poses3d_flat), distortion_coeffs_rep)
    )
    poses2d_flat = torch.einsum(
        "bnk,bjk->bnj", poses2d_flat_normalized, intrinsic_matrix_rep[:, :2, :]
    )
    poses2d = torch.split(poses2d_flat, n_box_per_image_list)
    return poses2d


def weighted_geometric_median(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    n_iter: int = 10,
    dim: int = -2,
    eps: float = 1e-1,
    keepdim: bool = False,
):
    if dim < 0:
        dim = len(x.shape) + dim

    if w is None:
        w = torch.ones_like(x[..., :1])
    else:
        w = w.unsqueeze(-1)

    new_weights = w
    y = weighted_mean(x, new_weights, dim=dim, keepdim=True)
    for _ in range(n_iter):
        dist = torch.norm(x - y, dim=-1, keepdim=True)
        new_weights = w / (dist + eps)
        y = weighted_mean(x, new_weights, dim=dim, keepdim=True)

    if not keepdim:
        y = y.squeeze(dim)

    return y, new_weights.squeeze(-1)


def weighted_mean(
    x: torch.Tensor, w: torch.Tensor, dim: int = -2, keepdim: bool = False
):
    return (x * w).sum(dim=dim, keepdim=keepdim) / w.sum(dim=dim, keepdim=keepdim)
