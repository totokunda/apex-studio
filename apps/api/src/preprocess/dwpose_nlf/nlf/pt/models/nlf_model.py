from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pt import ptu, ptu3d
from pt.models import util as model_util

class NLFModel(nn.Module):
    def __init__(self, config, backbone, weight_field, normalizer, backbone_channels=1280, permutation=867, i_left_joints=360, i_center_joints=147):
        super().__init__()
        self.backbone = backbone
        self.heatmap_head = LocalizerHead(config, weight_field, normalizer, in_channels=backbone_channels)
        self.input_resolution = config["proc_side"]
        
        # These buffers are part of the trained model and are expected to be present in checkpoints.
        # Keep them persistent so they round-trip through state_dict (no missing/unexpected keys).
        self.inv_permutation = nn.Buffer(torch.ones(permutation), persistent=True)
        
        self.canonical_lefts = nn.Parameter(
            torch.zeros(i_left_joints, 3, dtype=torch.float32), requires_grad=False)
        
        self.canonical_centers = nn.Parameter(
            torch.zeros(i_center_joints, 2, dtype=torch.float32), requires_grad=False)

        self.canonical_locs_init = nn.Buffer(
            torch.ones(permutation, 3, dtype=torch.float32),
            persistent=True,
        )
        self.canonical_delta_mask = nn.Buffer(
            torch.ones(permutation, dtype=torch.float32),
            persistent=True,
        )

    @torch.jit.export
    def canonical_locs(self):
        canonical_rights = torch.cat(
            [-self.canonical_lefts[:, :1], self.canonical_lefts[:, 1:]], dim=1
        )
        canonical_centers = torch.cat(
            [torch.zeros_like(self.canonical_centers[:, :1]), self.canonical_centers], dim=1
        )
        permuted = torch.cat([self.canonical_lefts, canonical_rights, canonical_centers], dim=0)
        return (
            permuted.index_select(0, self.inv_permutation)
            * self.canonical_delta_mask[:, torch.newaxis]
            + self.canonical_locs_init
        )

    @torch.jit.export
    def predict_multi_same_canonicals(
        self, image: torch.Tensor, intrinsic_matrix: torch.Tensor, canonical_points: torch.Tensor
    ):  # , flip_canonicals_per_image=()):
        features = self.backbone(image)
        coords2d, coords3d, uncertainties = self.heatmap_head.predict_same_canonicals(
            features, canonical_points
        )
        with torch.amp.autocast('cuda', enabled=False):
            return self.heatmap_head.reconstruct_absolute(
                coords2d.float(), coords3d.float(), uncertainties.float(), intrinsic_matrix.float()
            )

    @torch.jit.export
    def get_features(self, image: torch.Tensor):
        f = self.backbone(image)
        return self.heatmap_head.layer(f)

    @torch.jit.export
    def predict_multi_same_weights(
        self,
        image: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        flip_canonicals_per_image: torch.Tensor,
    ):
        features_processed = self.get_features(image)
        coords2d, coords3d, uncertainties = self.heatmap_head.decode_features_multi_same_weights(
            features_processed, weights, flip_canonicals_per_image
        )

        with torch.amp.autocast('cuda', enabled=False):
            return self.heatmap_head.reconstruct_absolute(
                coords2d.float(), coords3d.float(), uncertainties.float(), intrinsic_matrix.float()
            )

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        return self.heatmap_head.get_weights_for_canonical_points(canonical_points)


class LocalizerHead(nn.Module):
    def __init__(self, config, weight_field, normalizer, in_channels=1280):
        super().__init__()
        self.uncert_bias = config["uncert_bias"]
        self.uncert_bias2 = config["uncert_bias2"]
        self.depth = config["depth"]
        self.weight_field = weight_field
        self.stride_test = config["stride_test"]
        self.centered_stride = config["centered_stride"]
        self.box_size_m = config["box_size_m"]
        self.proc_side = config["proc_side"]
        self.backbone_link_dim = config["backbone_link_dim"]
        self.fix_uncert_factor = config["fix_uncert_factor"]
        self.mix_3d_inside_fov = config["mix_3d_inside_fov"]
        self.weak_perspective = config["weak_perspective"]
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, self.backbone_link_dim, kernel_size=1, bias=False),
            normalizer(self.backbone_link_dim),
            nn.SiLU(),
        )

    @torch.jit.export
    def forward(self, features: torch.Tensor, canonical_positions: Optional[torch.Tensor] = None):
        assert canonical_positions is not None
        weights = self.weight_field(canonical_positions)  # NP[C(c+1)]
        return self.call_with_weights(features, weights)

    @torch.jit.export
    def call_with_weights(self, features, weights):
        features_processed = self.layer(features)  # NHWc
        coords2d, coords3d_rel_pred, uncertainties = self.apply_weights3d(
            features_processed, weights, n_out_channels=2 + self.depth
        )
        coords2d_pred = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride
        )
        coords3d_rel_pred = model_util.heatmap_to_metric(
            coords3d_rel_pred,
            self.proc_side,
            self.stride_test,
            self.centered_stride,
            self.box_size_m,
        )
        return coords2d_pred, coords3d_rel_pred, uncertainties

    @torch.jit.export
    def predict_same_canonicals(self, features: torch.Tensor, canonical_positions: torch.Tensor):
        weights = self.weight_field(canonical_positions)  # NP[C(c+1)]
        features_processed = self.layer(features)  # NcHW

        coords2d, coords3d_rel_pred, uncertainties = self.apply_weights3d_same_canonicals(
            features_processed, weights
        )
        coords2d_pred = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride
        )
        coords3d_rel_pred = model_util.heatmap_to_metric(
            coords3d_rel_pred,
            self.proc_side,
            self.stride_test,
            self.centered_stride,
            self.box_size_m,
        )
        return coords2d_pred, coords3d_rel_pred, uncertainties

    @torch.jit.export
    def apply_weights3d(self, features: torch.Tensor, weights: torch.Tensor, n_out_channels: int):
        # features: NcHW 128,1280,8,8
        # weights:  NP[(c+1)C] 128,768,10*1281
        weights = weights.to(features.dtype)
        weights_resh = torch.unflatten(
            weights, -1, (features.shape[1] + 1, n_out_channels)
        )  # NPC(c+1)
        w_tensor = weights_resh[..., :-1, :]  # NPCc
        b_tensor = weights_resh[..., -1, :]  # NPC
        # TODO: check if a different einsum order would be faster
        logits = (
            torch.einsum('nchw,npcC->npChw', features, w_tensor)
            + b_tensor[:, :, :, torch.newaxis, torch.newaxis]
        ).float()
        # This is an alternative to the einsum above (perhaps faster, perhaps not; because
        # the addition is done in the same call)
        # features_ = features.flatten(2, 3)
        # w_tensor_ = w_tensor.mT.flatten(1, 2)
        # b_tensor_ = b_tensor.flatten(1, 2).unsqueeze(-1)
        # logits = (
        #     torch.baddbmm(b_tensor_, w_tensor_, features_)
        #     .unflatten(1, (-1, n_out_channels))
        #     .unflatten(-1, (features.shape[2], features.shape[3]))
        # )

        uncertainty_map = logits[:, :, 0]
        coords_metric_xy = ptu.soft_argmax(logits[:, :, 1], dim=[3, 2])
        heatmap25d = ptu.softmax(logits[:, :, 2:], dim=[4, 3, 2])
        heatmap2d = torch.sum(heatmap25d, dim=2)

        # earlier slower version: torch.sum(uncertainty_map * heatmap2d.detach(), dim=[3, 2])
        uncertainties = torch.einsum('nphw,nphw->np', uncertainty_map, heatmap2d.detach())
        uncertainties = F.softplus(uncertainties + self.uncert_bias) + self.uncert_bias2
        coords25d = ptu.decode_heatmap(heatmap25d, dim=[4, 3, 2])
        coords2d = coords25d[..., :2]
        coords3d = torch.cat([coords_metric_xy, coords25d[..., 2:]], dim=-1)
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def transpose_weights(self, weights: torch.Tensor, n_in_channels: int):
        n_out_channels = 2 + self.depth
        weights_resh = torch.unflatten(weights, -1, (n_in_channels + 1, n_out_channels))  # P(c+1)C
        w_tensor = weights_resh[..., :-1, :]  # PcC
        b_tensor = weights_resh[..., -1, :]  # PC
        # old: w_tensor = w_tensor.permute(1, 0, 2)  # PcC-> cPC
        w_tensor = w_tensor.permute(0, 2, 1)  # PcC-> PCc
        return w_tensor.contiguous(), b_tensor.contiguous()

    @torch.jit.export
    def apply_weights3d_same_canonicals(self, features: torch.Tensor, weights: torch.Tensor):
        # features: NcHW 128,1280,8,8
        # weights:  P[(c+1)C] 768,1281*10
        w_tensor, b_tensor = self.transpose_weights(weights.to(features.dtype), features.shape[1])
        return self.apply_weights3d_same_canonicals_impl(features, w_tensor, b_tensor)

    @torch.jit.export
    def apply_weights3d_same_canonicals_impl(
        self, features: torch.Tensor, w_tensor: torch.Tensor, b_tensor: torch.Tensor
    ):
        # features: nchw 128,1280,8,8
        # w_tensor: cpC 1280,768,10
        n_out_channels = 2 + self.depth
        # old:
        # w_tensor = torch.flatten(w_tensor, start_dim=-2, end_dim=-1).mT.unsqueeze(-1).unsqueeze(-1)
        # w_tensor = w_tensor.contiguous()  # (pC)c11

        # new:
        w_tensor = torch.flatten(w_tensor, start_dim=0, end_dim=1).unsqueeze(-1).unsqueeze(-1)
        b_tensor = b_tensor.reshape(-1)
        #

        logits = F.conv2d(features, w_tensor, bias=b_tensor).float()
        logits = torch.unflatten(logits, 1, (-1, n_out_channels))  # npChw
        uncertainty_map = logits[:, :, 0]

        coords_metric_xy = ptu.soft_argmax(logits[:, :, 1], dim=[3, 2])
        heatmap25d = ptu.softmax(logits[:, :, 2:], dim=[4, 3, 2])
        heatmap2d = torch.sum(heatmap25d, dim=2)

        # old slower: uncertainties = torch.sum(uncertainty_map * heatmap2d.detach(), dim=[3, 2])
        uncertainties = torch.einsum('nphw,nphw->np', uncertainty_map, heatmap2d.detach())
        uncertainties = F.softplus(uncertainties + self.uncert_bias) + self.uncert_bias2
        coords25d = ptu.decode_heatmap(heatmap25d, dim=[4, 3, 2])
        coords2d = coords25d[..., :2]
        coords3d = torch.cat([coords_metric_xy, coords25d[..., 2:]], dim=-1)
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def get_weights_for_canonical_points(self, canonical_points: torch.Tensor):
        weights = self.weight_field(canonical_points)
        w_tensor, b_tensor = self.transpose_weights(weights.half(), self.backbone_link_dim)
        weights_fl = self.weight_field(
            canonical_points
            * torch.tensor([-1, 1, 1], dtype=torch.float32, device=canonical_points.device)
        )
        w_tensor_fl, b_tensor_fl = self.transpose_weights(weights_fl.half(), self.backbone_link_dim)
        return dict(
            w_tensor=w_tensor,
            b_tensor=b_tensor,
            w_tensor_flipped=w_tensor_fl,
            b_tensor_flipped=b_tensor_fl,
        )

    @torch.jit.export
    def decode_features_multi_same_weights(
        self,
        features: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        flip_canonicals_per_image: torch.Tensor,
    ):
        features_processed = features
        flip_canonicals_per_image_ind = flip_canonicals_per_image.to(torch.int32)

        nfl_features_processed, fl_features_processed = ptu.dynamic_partition(
            features_processed, flip_canonicals_per_image_ind, 2
        )
        partitioned_indices = ptu.dynamic_partition(
            torch.arange(features_processed.shape[0], device=flip_canonicals_per_image_ind.device),
            flip_canonicals_per_image_ind,
            2,
        )
        nfl_coords2d, nfl_coords3d, nfl_uncertainties = self.apply_weights3d_same_canonicals_impl(
            nfl_features_processed, weights['w_tensor'], weights['b_tensor']
        )
        fl_coords2d, fl_coords3d, fl_uncertainties = self.apply_weights3d_same_canonicals_impl(
            fl_features_processed, weights['w_tensor_flipped'], weights['b_tensor_flipped']
        )
        coords2d = ptu.dynamic_stitch(partitioned_indices, [nfl_coords2d, fl_coords2d])
        coords3d = ptu.dynamic_stitch(partitioned_indices, [nfl_coords3d, fl_coords3d])
        uncertainties = ptu.dynamic_stitch(
            partitioned_indices, [nfl_uncertainties, fl_uncertainties]
        )

        coords2d = model_util.heatmap_to_image(
            coords2d, self.proc_side, self.stride_test, self.centered_stride
        )
        coords3d = model_util.heatmap_to_metric(
            coords3d,
            self.proc_side,
            self.stride_test,
            self.centered_stride,
            self.box_size_m,
        )
        return coords2d, coords3d, uncertainties

    @torch.jit.export
    def reconstruct_absolute(
        self,
        coords2d: torch.Tensor,
        coords3d: torch.Tensor,
        uncertainties: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
    ):
        coords3d_abs = (
            ptu3d.reconstruct_absolute(
                coords2d,
                coords3d,
                intrinsic_matrix,
                proc_side=self.proc_side,
                stride=self.stride_test,
                centered_stride=self.centered_stride,
                weak_perspective=self.weak_perspective,
                mix_3d_inside_fov=0.5,
                point_validity_mask=uncertainties < 0.3,
                border_factor1=1.0,
                border_factor2=0.6,
                mix_based_on_3d=True,
            )
            * 1000
        )
        factor = 1 if self.fix_uncert_factor else 3
        return coords3d_abs, uncertainties * factor


def is_hand_joint(name):
    n = name.partition('_')[0]
    if any(x in n for x in ['thumb', 'index', 'middle', 'ring', 'pinky']):
        return True

    return n.startswith(('lhan', 'rhan')) and len(n) > 4
