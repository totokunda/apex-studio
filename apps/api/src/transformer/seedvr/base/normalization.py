# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Callable, Optional
from diffusers.models.normalization import RMSNorm
from torch import nn

# (dim: int, eps: float, elementwise_affine: bool)
norm_layer_type = Callable[[int, float, bool], nn.Module]


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()

        if norm_type == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "rms":
            return RMSNorm(
                dim=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "fusedln":
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        if norm_type == "fusedrms":
            from apex.normalization import FusedRMSNorm

            return FusedRMSNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer
