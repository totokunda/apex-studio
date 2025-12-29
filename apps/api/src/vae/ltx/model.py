# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
import math
import inspect

import json
import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Tuple, Union, List
from pathlib import Path

import torch
import numpy as np
from einops import rearrange
from torch import nn
from diffusers.utils import logging
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from safetensors import safe_open
from src.transformer.ltx.base.model import LTXVideoAttention as Attention


PER_CHANNEL_STATISTICS_PREFIX = "per_channel_statistics."
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def make_hashable_key(dict_key):
    def convert_value(value):
        if isinstance(value, list):
            return tuple(value)
        elif isinstance(value, dict):
            return tuple(sorted((k, convert_value(v)) for k, v in value.items()))
        else:
            return value

    return tuple(sorted((k, convert_value(v)) for k, v in dict_key.items()))


DIFFUSERS_VAE_CONFIG = {
    "_class_name": "AutoencoderKLLTXVideo",
    "_diffusers_version": "0.32.0.dev0",
    "block_out_channels": [128, 256, 512, 512],
    "decoder_causal": False,
    "encoder_causal": True,
    "in_channels": 3,
    "latent_channels": 128,
    "layers_per_block": [4, 3, 3, 3, 4],
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-06,
    "scaling_factor": 1.0,
    "spatio_temporal_scaling": [True, True, True, False],
}

VAE_KEYS_RENAME_DICT = {
    "decoder.up_blocks.3.conv_in": "decoder.up_blocks.7",
    "decoder.up_blocks.3.upsamplers.0": "decoder.up_blocks.8",
    "decoder.up_blocks.3": "decoder.up_blocks.9",
    "decoder.up_blocks.2.upsamplers.0": "decoder.up_blocks.5",
    "decoder.up_blocks.2.conv_in": "decoder.up_blocks.4",
    "decoder.up_blocks.2": "decoder.up_blocks.6",
    "decoder.up_blocks.1.upsamplers.0": "decoder.up_blocks.2",
    "decoder.up_blocks.1": "decoder.up_blocks.3",
    "decoder.up_blocks.0": "decoder.up_blocks.1",
    "decoder.mid_block": "decoder.up_blocks.0",
    "encoder.down_blocks.3": "encoder.down_blocks.8",
    "encoder.down_blocks.2.downsamplers.0": "encoder.down_blocks.7",
    "encoder.down_blocks.2": "encoder.down_blocks.6",
    "encoder.down_blocks.1.downsamplers.0": "encoder.down_blocks.4",
    "encoder.down_blocks.1.conv_out": "encoder.down_blocks.5",
    "encoder.down_blocks.1": "encoder.down_blocks.3",
    "encoder.down_blocks.0.conv_out": "encoder.down_blocks.2",
    "encoder.down_blocks.0.downsamplers.0": "encoder.down_blocks.1",
    "encoder.down_blocks.0": "encoder.down_blocks.0",
    "encoder.mid_block": "encoder.down_blocks.9",
    "conv_shortcut.conv": "conv_shortcut",
    "resnets": "res_blocks",
    "norm3": "norm3.norm",
    "latents_mean": "per_channel_statistics.mean-of-means",
    "latents_std": "per_channel_statistics.std-of-means",
}

OURS_VAE_CONFIG = {
    "_class_name": "CausalVideoAutoencoder",
    "dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "blocks": [
        ["res_x", 4],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x", 3],
        ["res_x", 4],
    ],
    "scaling_factor": 1.0,
    "norm_layer": "pixel_norm",
    "patch_size": 4,
    "latent_log_var": "uniform",
    "use_quant_conv": False,
    "causal_decoder": False,
}

diffusers_and_ours_config_mapping = {
    make_hashable_key(DIFFUSERS_VAE_CONFIG): OURS_VAE_CONFIG,
}


def make_conv_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    causal=False,
    spatial_padding_mode="zeros",
    temporal_padding_mode="zeros",
):
    if not (spatial_padding_mode == temporal_padding_mode or causal):
        raise NotImplementedError("spatial and temporal padding modes must be equal")
    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias=True,
):
    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
    elif dims == 3 or dims == (2, 1):
        return torch.nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def patchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)


class PixelShuffleND(nn.Module):
    def __init__(self, dims, upscale_factors=(2, 2, 2)):
        super().__init__()
        assert dims in [1, 2, 3], "dims must be 1, 2, or 3"
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x):
        if self.dims == 3:
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        elif self.dims == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        elif self.dims == 1:
            return rearrange(
                x,
                "b (c p1) f h w -> b c (f p1) h w",
                p1=self.upscale_factors[0],
            )


class DualConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(DualConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        # Ensure kernel_size, stride, padding, and dilation are tuples of length 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError(
                "kernel_size must be greater than 1. Use make_linear_nd instead."
            )
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        # Set parameters for convolutions
        self.groups = groups
        self.bias = bias

        # Define the size of the channels after the first convolution
        intermediate_channels = (
            out_channels if in_channels < out_channels else in_channels
        )

        # Define parameters for the first convolution
        self.weight1 = nn.Parameter(
            torch.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Define parameters for the second convolution
        self.weight2 = nn.Parameter(
            torch.Tensor(
                out_channels, intermediate_channels // groups, kernel_size[0], 1, 1
            )
        )
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / math.sqrt(fan_in1)
            nn.init.uniform_(self.bias1, -bound1, bound1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / math.sqrt(fan_in2)
            nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x, use_conv3d=False, skip_time_conv=False):
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x, skip_time_conv):
        # First convolution
        x = F.conv3d(
            x,
            self.weight1,
            self.bias1,
            self.stride1,
            self.padding1,
            self.dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        if skip_time_conv:
            return x

        # Second convolution
        x = F.conv3d(
            x,
            self.weight2,
            self.bias2,
            self.stride2,
            self.padding2,
            self.dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )

        return x

    def forward_with_2d(self, x, skip_time_conv):
        b, c, d, h, w = x.shape

        # First 2D convolution
        x = rearrange(x, "b c d h w -> (b d) c h w")
        # Squeeze the depth dimension out of weight1 since it's 1
        weight1 = self.weight1.squeeze(2)
        # Select stride, padding, and dilation for the 2D convolution
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(
            x,
            weight1,
            self.bias1,
            stride1,
            padding1,
            dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        _, _, h, w = x.shape

        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x

        # Second convolution which is essentially treated as a 1D convolution across the 'd' dimension
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)

        # Reshape weight2 to match the expected dimensions for conv1d
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        # Use only the relevant dimension for stride, padding, and dilation for the 1D convolution
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]
        x = F.conv1d(
            x,
            weight2,
            self.bias2,
            stride2,
            padding2,
            dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)

        return x

    @property
    def weight(self):
        return self.weight2


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode,
            groups=groups,
        )

    def forward(self, x, causal: bool = True):
        if causal:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x), dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            )
            last_frame_pad = x[:, :, -1:, :, :].repeat(
                (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)
        x = self.conv(x)
        return x

    @property
    def weight(self):
        return self.conv.weight


class AutoencoderKLWrapper(ModelMixin, ConfigMixin):
    """Variational Autoencoder (VAE) model with KL loss.

    VAE from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma and Max Welling.
    This model is a wrapper around an encoder and a decoder, and it adds a KL loss term to the reconstruction loss.

    Args:
        encoder (`nn.Module`):
            Encoder module.
        decoder (`nn.Module`):
            Decoder module.
        latent_channels (`int`, *optional*, defaults to 4):
            Number of latent channels.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_channels: int = 4,
        spatial_compression_ratio: int = 32,
        temporal_compression_ratio: int = 8,
        dims: int = 2,
        sample_size=512,
        use_quant_conv: bool = True,
        normalize_latent_channels: bool = False,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = encoder
        self.use_quant_conv = use_quant_conv
        self.normalize_latent_channels = normalize_latent_channels
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # pass init params to Decoder
        quant_dims = 2 if dims == 2 else 3
        self.decoder = decoder
        if use_quant_conv:
            self.quant_conv = make_conv_nd(
                quant_dims, 2 * latent_channels, 2 * latent_channels, 1
            )
            self.post_quant_conv = make_conv_nd(
                quant_dims, latent_channels, latent_channels, 1
            )
        else:
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()

        if normalize_latent_channels:
            if dims == 2:
                self.latent_norm_out = nn.BatchNorm2d(latent_channels, affine=False)
            else:
                self.latent_norm_out = nn.BatchNorm3d(latent_channels, affine=False)
        else:
            self.latent_norm_out = nn.Identity()
        self.use_z_tiling = False
        self.use_hw_tiling = False
        self.dims = dims
        self.z_sample_size = 1

        self.decoder_params = inspect.signature(self.decoder.forward).parameters

        # only relevant if vae tiling is enabled
        self.set_tiling_params(sample_size=sample_size, overlap_factor=0.25)

    def set_tiling_params(self, sample_size: int = 512, overlap_factor: float = 0.25):
        self.tile_sample_min_size = sample_size
        num_blocks = len(self.encoder.down_blocks)
        self.tile_latent_min_size = int(sample_size / (2 ** (num_blocks - 1)))
        self.tile_overlap_factor = overlap_factor

    def enable_z_tiling(self, z_sample_size: int = 8):
        r"""
        Enable tiling during VAE decoding.

        When this option is enabled, the VAE will split the input tensor in tiles to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_z_tiling = z_sample_size > 1
        self.z_sample_size = z_sample_size
        assert (
            z_sample_size % 8 == 0 or z_sample_size == 1
        ), f"z_sample_size must be a multiple of 8 or 1. Got {z_sample_size}."

    def disable_z_tiling(self):
        r"""
        Disable tiling during VAE decoding. If `use_tiling` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_z_tiling = False

    def enable_hw_tiling(self):
        r"""
        Enable tiling during VAE decoding along the height and width dimension.
        """
        self.use_hw_tiling = True

    def disable_hw_tiling(self):
        r"""
        Disable tiling during VAE decoding along the height and width dimension.
        """
        self.use_hw_tiling = False

    def _hw_tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        return moments

    def blend_z(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for z in range(blend_extent):
            b[:, :, z, :, :] = a[:, :, -blend_extent + z, :, :] * (
                1 - z / blend_extent
            ) + b[:, :, z, :, :] * (z / blend_extent)
        return b

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def _hw_tiled_decode(self, z: torch.FloatTensor, target_shape):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        tile_target_shape = (
            *target_shape[:3],
            self.tile_sample_min_size,
            self.tile_sample_min_size,
        )
        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, target_shape=tile_target_shape)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def encode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_z_tiling and z.shape[2] > self.z_sample_size > 1:
            num_splits = z.shape[2] // self.z_sample_size
            sizes = [self.z_sample_size] * num_splits
            sizes = (
                sizes + [z.shape[2] - sum(sizes)]
                if z.shape[2] - sum(sizes) > 0
                else sizes
            )
            tiles = z.split(sizes, dim=2)
            moments_tiles = [
                (
                    self._hw_tiled_encode(z_tile, return_dict)
                    if self.use_hw_tiling
                    else self._encode(z_tile)
                )
                for z_tile in tiles
            ]
            moments = torch.cat(moments_tiles, dim=2)

        else:
            moments = (
                self._hw_tiled_encode(z, return_dict)
                if self.use_hw_tiling
                else self._encode(z)
            )

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _normalize_latent_channels(self, z: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(self.latent_norm_out, nn.BatchNorm3d):
            _, c, _, _, _ = z.shape
            z = torch.cat(
                [
                    self.latent_norm_out(z[:, : c // 2, :, :, :]),
                    z[:, c // 2 :, :, :, :],
                ],
                dim=1,
            )
        elif isinstance(self.latent_norm_out, nn.BatchNorm2d):
            raise NotImplementedError("BatchNorm2d not supported")
        return z

    def _unnormalize_latent_channels(self, z: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(self.latent_norm_out, nn.BatchNorm3d):
            running_mean = self.latent_norm_out.running_mean.view(1, -1, 1, 1, 1)
            running_var = self.latent_norm_out.running_var.view(1, -1, 1, 1, 1)
            eps = self.latent_norm_out.eps

            z = z * torch.sqrt(running_var + eps) + running_mean
        elif isinstance(self.latent_norm_out, nn.BatchNorm3d):
            raise NotImplementedError("BatchNorm2d not supported")
        return z

    def _encode(self, x: torch.FloatTensor) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        moments = self._normalize_latent_channels(moments)
        return moments

    def _decode(
        self,
        z: torch.FloatTensor,
        target_shape=None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self._unnormalize_latent_channels(z)
        z = self.post_quant_conv(z)

        if "timestep" in self.decoder_params:
            dec = self.decoder(z, target_shape=target_shape, timestep=timestep)
        else:
            dec = self.decoder(z, target_shape=target_shape)
        return dec

    def decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        target_shape=None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        if target_shape is None:
            *_, fl, hl, wl = z.shape
            target_shape = (
                1,
                3,
                fl * self.spatial_compression_ratio,
                hl * self.spatial_compression_ratio,
                wl * self.spatial_compression_ratio,
            )
        assert target_shape is not None, "target_shape must be provided for decoding"
        if self.use_z_tiling and z.shape[2] > self.z_sample_size > 1:
            reduction_factor = int(
                self.encoder.patch_size_t
                * 2
                ** (
                    len(self.encoder.down_blocks)
                    - 1
                    - math.sqrt(self.encoder.patch_size)
                )
            )
            split_size = self.z_sample_size // reduction_factor
            num_splits = z.shape[2] // split_size

            # copy target shape, and divide frame dimension (=2) by the context size
            target_shape_split = list(target_shape)
            target_shape_split[2] = target_shape[2] // num_splits

            decoded_tiles = [
                (
                    self._hw_tiled_decode(z_tile, target_shape_split)
                    if self.use_hw_tiling
                    else self._decode(z_tile, target_shape=target_shape_split)
                )
                for z_tile in torch.tensor_split(z, num_splits, dim=2)
            ]
            decoded = torch.cat(decoded_tiles, dim=2)
        else:
            decoded = (
                self._hw_tiled_decode(z, target_shape)
                if self.use_hw_tiling
                else self._decode(z, target_shape=target_shape, timestep=timestep)
            )

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                Generator used to sample from the posterior.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, target_shape=sample.shape).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


class AutoencoderKLLTXVideo(AutoencoderKLWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs,
    ):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        if (
            pretrained_model_name_or_path.is_dir()
            and (pretrained_model_name_or_path / "autoencoder.pth").exists()
        ):
            config_local_path = pretrained_model_name_or_path / "config.json"
            config = cls.load_config(config_local_path, **kwargs)

            model_local_path = pretrained_model_name_or_path / "autoencoder.pth"
            state_dict = torch.load(model_local_path, map_location=torch.device("cpu"))

            statistics_local_path = (
                pretrained_model_name_or_path / "per_channel_statistics.json"
            )
            if statistics_local_path.exists():
                with open(statistics_local_path, "r") as file:
                    data = json.load(file)
                transposed_data = list(zip(*data["data"]))
                data_dict = {
                    col: torch.tensor(vals)
                    for col, vals in zip(data["columns"], transposed_data)
                }
                std_of_means = data_dict["std-of-means"]
                mean_of_means = data_dict.get(
                    "mean-of-means", torch.zeros_like(data_dict["std-of-means"])
                )
                state_dict[f"{PER_CHANNEL_STATISTICS_PREFIX}std-of-means"] = (
                    std_of_means
                )
                state_dict[f"{PER_CHANNEL_STATISTICS_PREFIX}mean-of-means"] = (
                    mean_of_means
                )

        elif pretrained_model_name_or_path.is_dir():
            config_path = pretrained_model_name_or_path / "vae" / "config.json"
            with open(config_path, "r") as f:
                config = make_hashable_key(json.load(f))

            assert config in diffusers_and_ours_config_mapping, (
                "Provided diffusers checkpoint config for VAE is not suppported. "
                "We only support diffusers configs found in Lightricks/LTX-Video."
            )

            config = diffusers_and_ours_config_mapping[config]

            state_dict_path = (
                pretrained_model_name_or_path
                / "vae"
                / "diffusion_pytorch_model.safetensors"
            )

            state_dict = {}
            with safe_open(state_dict_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            for key in list(state_dict.keys()):
                new_key = key
                for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
                    new_key = new_key.replace(replace_key, rename_key)

                state_dict[new_key] = state_dict.pop(key)

        elif pretrained_model_name_or_path.is_file() and str(
            pretrained_model_name_or_path
        ).endswith(".safetensors"):
            state_dict = {}
            with safe_open(
                pretrained_model_name_or_path, framework="pt", device="cpu"
            ) as f:
                metadata = f.metadata()
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            configs = json.loads(metadata["config"])

            config = configs["vae"]

        video_vae = cls.from_config(config)
        if "torch_dtype" in kwargs:
            video_vae.to(kwargs["torch_dtype"])
        video_vae.load_state_dict(state_dict)
        return video_vae

    @staticmethod
    def from_config(config):
        if isinstance(config["dims"], list):
            config["dims"] = tuple(config["dims"])

        assert config["dims"] in [2, 3, (2, 1)], "dims must be 2, 3 or (2, 1)"

        double_z = config.get("double_z", True)
        latent_log_var = config.get(
            "latent_log_var", "per_channel" if double_z else "none"
        )
        use_quant_conv = config.get("use_quant_conv", True)
        normalize_latent_channels = config.get("normalize_latent_channels", False)

        if use_quant_conv and latent_log_var in ["uniform", "constant"]:
            raise ValueError(
                f"latent_log_var={latent_log_var} requires use_quant_conv=False"
            )

        encoder = Encoder(
            dims=config["dims"],
            in_channels=config.get("in_channels", 3),
            out_channels=config["latent_channels"],
            blocks=config.get("encoder_blocks", config.get("blocks")),
            patch_size=config.get("patch_size", 1),
            latent_log_var=latent_log_var,
            norm_layer=config.get("norm_layer", "group_norm"),
            base_channels=config.get("encoder_base_channels", 128),
            spatial_padding_mode=config.get("spatial_padding_mode", "zeros"),
        )

        decoder = Decoder(
            dims=config["dims"],
            in_channels=config["latent_channels"],
            out_channels=config.get("out_channels", 3),
            blocks=config.get("decoder_blocks", config.get("blocks")),
            patch_size=config.get("patch_size", 1),
            norm_layer=config.get("norm_layer", "group_norm"),
            causal=config.get("causal_decoder", False),
            timestep_conditioning=config.get("timestep_conditioning", False),
            base_channels=config.get("decoder_base_channels", 128),
            spatial_padding_mode=config.get("spatial_padding_mode", "zeros"),
        )

        dims = config["dims"]
        return AutoencoderKLLTXVideo(
            encoder=encoder,
            decoder=decoder,
            latent_channels=config["latent_channels"],
            dims=dims,
            use_quant_conv=use_quant_conv,
            normalize_latent_channels=normalize_latent_channels,
        )

    @property
    def config(self):
        return SimpleNamespace(
            _class_name="AutoencoderKLLTXVideo",
            dims=self.dims,
            in_channels=self.encoder.conv_in.in_channels // self.encoder.patch_size**2,
            out_channels=self.decoder.conv_out.out_channels
            // self.decoder.patch_size**2,
            latent_channels=self.decoder.conv_in.in_channels,
            encoder_blocks=self.encoder.blocks_desc,
            decoder_blocks=self.decoder.blocks_desc,
            scaling_factor=1.0,
            spatial_compression_ratio=self.spatial_compression_ratio,
            temporal_compression_ratio=self.temporal_compression_ratio,
            norm_layer=self.encoder.norm_layer,
            patch_size=self.encoder.patch_size,
            latent_log_var=self.encoder.latent_log_var,
            use_quant_conv=self.use_quant_conv,
            causal_decoder=self.decoder.causal,
            timestep_conditioning=self.decoder.timestep_conditioning,
            normalize_latent_channels=self.normalize_latent_channels,
        )

    @property
    def is_video_supported(self):
        """
        Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
        """
        return self.dims != 2

    @property
    def spatial_downscale_factor(self):
        return (
            2
            ** len(
                [
                    block
                    for block in self.encoder.blocks_desc
                    if block[0]
                    in [
                        "compress_space",
                        "compress_all",
                        "compress_all_res",
                        "compress_space_res",
                    ]
                ]
            )
            * self.encoder.patch_size
        )

    @property
    def temporal_downscale_factor(self):
        return 2 ** len(
            [
                block
                for block in self.encoder.blocks_desc
                if block[0]
                in [
                    "compress_time",
                    "compress_all",
                    "compress_all_res",
                    "compress_time_res",
                ]
            ]
        )

    def to_json_string(self) -> str:
        import json

        return json.dumps(self.config.__dict__)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True
    ):
        if any([key.startswith("vae.") for key in state_dict.keys()]):
            state_dict = {
                key.replace("vae.", ""): value
                for key, value in state_dict.items()
                if key.startswith("vae.")
            }
        ckpt_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(PER_CHANNEL_STATISTICS_PREFIX)
        }

        model_keys = set(name for name, _ in self.named_modules())

        key_mapping = {
            ".resnets.": ".res_blocks.",
            "downsamplers.0": "downsample",
            "upsamplers.0": "upsample",
        }
        converted_state_dict = {}
        for key, value in ckpt_state_dict.items():
            for k, v in key_mapping.items():
                key = key.replace(k, v)

            key_prefix = ".".join(key.split(".")[:-1])
            if "norm" in key and key_prefix not in model_keys:
                logger.info(
                    f"Removing key {key} from state_dict as it is not present in the model"
                )
                continue

            converted_state_dict[key] = value

        super().load_state_dict(converted_state_dict, strict=strict, assign=assign)

        data_dict = {
            key.removeprefix(PER_CHANNEL_STATISTICS_PREFIX): value
            for key, value in state_dict.items()
            if key.startswith(PER_CHANNEL_STATISTICS_PREFIX)
        }
        if len(data_dict) > 0:
            self.register_buffer("std_of_means", data_dict["std-of-means"])
            self.register_buffer(
                "mean_of_means",
                data_dict.get(
                    "mean-of-means", torch.zeros_like(data_dict["std-of-means"])
                ),
            )

    def last_layer(self):
        if hasattr(self.decoder, "conv_out"):
            if isinstance(self.decoder.conv_out, nn.Sequential):
                last_layer = self.decoder.conv_out[-1]
            else:
                last_layer = self.decoder.conv_out
        else:
            last_layer = self.decoder.layers[-1]
        return last_layer

    def set_use_tpu_flash_attention(self):
        for block in self.decoder.up_blocks:
            if isinstance(block, UNetMidBlock3D) and block.attention_blocks:
                for attention_block in block.attention_blocks:
                    attention_block.set_use_tpu_flash_attention()

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = self.mean_of_means.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.std_of_means.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        scaling_factor = self.config.scaling_factor
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = self.mean_of_means.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.std_of_means.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        scaling_factor = self.config.scaling_factor
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def _temporal_tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (
                height > self.tile_sample_min_height
                or width > self.tile_sample_min_width
            ):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(
        self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )
        blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        )

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (
                tile.shape[-1] > tile_latent_min_width
                or tile.shape[-2] > tile_latent_min_height
            ):
                decoded = self.tiled_decode(tile, temb, return_dict=True).sample
            else:
                decoded = self.decoder(tile, temb)
            if i > 0:
                decoded = decoded[:, :, :-1, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                tile = tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                result_row.append(tile)
            else:
                result_row.append(
                    tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :]
                )

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        dims (`int` or `Tuple[int, int]`, *optional*, defaults to 3):
            The number of dimensions to use in convolutions.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        blocks (`List[Tuple[str, int]]`, *optional*, defaults to `[("res_x", 1)]`):
            The blocks to use. Each block is a tuple of the block name and the number of layers.
        base_channels (`int`, *optional*, defaults to 128):
            The number of output channels for the first convolutional layer.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var (`str`, *optional*, defaults to `per_channel`):
            The number of channels for the log variance. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]] = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        blocks: List[Tuple[str, int | dict]] = [("res_x", 1)],
        base_channels: int = 128,
        norm_num_groups: int = 32,
        patch_size: Union[int, Tuple[int]] = 1,
        norm_layer: str = "group_norm",  # group_norm, pixel_norm
        latent_log_var: str = "per_channel",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self.blocks_desc = blocks

        in_channels = in_channels * patch_size**2
        output_channel = base_channels

        self.conv_in = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in blocks:
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            if block_name == "res_x":
                block = UNetMidBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_eps=1e-6,
                    resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "res_x_y":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = ResnetBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    eps=1e-6,
                    groups=norm_num_groups,
                    norm_layer=norm_layer,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time":
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(2, 1, 1),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space":
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(1, 2, 2),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all":
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all_x_y":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = make_conv_nd(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=(2, 2, 2),
                    causal=True,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=(2, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=(1, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time_res":
                output_channel = block_params.get("multiplier", 2) * output_channel
                block = SpaceToDepthDownsample(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=(2, 1, 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"unknown block: {block_name}")

            self.down_blocks.append(block)

        # out
        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=output_channel, num_groups=norm_num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var == "uniform":
            conv_out_channels += 1
        elif latent_log_var == "constant":
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")
        self.conv_out = make_conv_nd(
            dims,
            output_channel,
            conv_out_channels,
            3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == "uniform":
            last_channel = sample[:, -1:, ...]
            num_dims = sample.dim()

            if num_dims == 4:
                # For shape (B, C, H, W)
                repeated_last_channel = last_channel.repeat(
                    1, sample.shape[1] - 2, 1, 1
                )
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            elif num_dims == 5:
                # For shape (B, C, F, H, W)
                repeated_last_channel = last_channel.repeat(
                    1, sample.shape[1] - 2, 1, 1, 1
                )
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            else:
                raise ValueError(f"Invalid input shape: {sample.shape}")
        elif self.latent_log_var == "constant":
            sample = sample[:, :-1, ...]
            approx_ln_0 = (
                -30
            )  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        dims (`int` or `Tuple[int, int]`, *optional*, defaults to 3):
            The number of dimensions to use in convolutions.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        blocks (`List[Tuple[str, int]]`, *optional*, defaults to `[("res_x", 1)]`):
            The blocks to use. Each block is a tuple of the block name and the number of layers.
        base_channels (`int`, *optional*, defaults to 128):
            The number of output channels for the first convolutional layer.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal (`bool`, *optional*, defaults to `True`):
            Whether to use causal convolutions or not.
    """

    def __init__(
        self,
        dims,
        in_channels: int = 3,
        out_channels: int = 3,
        blocks: List[Tuple[str, int | dict]] = [("res_x", 1)],
        base_channels: int = 128,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: int = 1,
        norm_layer: str = "group_norm",
        causal: bool = True,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.blocks_desc = blocks

        # Compute output channel to be product of all channel-multiplier blocks
        output_channel = base_channels
        for block_name, block_params in list(reversed(blocks)):
            block_params = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                output_channel = output_channel * block_params.get("multiplier", 2)
            if block_name.startswith("compress"):
                output_channel = output_channel * block_params.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims,
            in_channels,
            output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(blocks)):
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {"num_layers": block_params}

            if block_name == "res_x":
                block = UNetMidBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_eps=1e-6,
                    resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "attn_res_x":
                block = UNetMidBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    num_layers=block_params["num_layers"],
                    resnet_groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=timestep_conditioning,
                    attention_head_dim=block_params["attention_head_dim"],
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "res_x_y":
                output_channel = output_channel // block_params.get("multiplier", 2)
                block = ResnetBlock3D(
                    dims=dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    eps=1e-6,
                    groups=norm_num_groups,
                    norm_layer=norm_layer,
                    inject_noise=block_params.get("inject_noise", False),
                    timestep_conditioning=False,
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_time":
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(2, 1, 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_space":
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(1, 2, 2),
                    spatial_padding_mode=spatial_padding_mode,
                )
            elif block_name == "compress_all":
                output_channel = output_channel // block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    dims=dims,
                    in_channels=input_channel,
                    stride=(2, 2, 2),
                    residual=block_params.get("residual", False),
                    out_channels_reduction_factor=block_params.get("multiplier", 1),
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"unknown layer: {block_name}")

            self.up_blocks.append(block)

        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=output_channel, num_groups=norm_num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        elif norm_layer == "layer_norm":
            self.conv_norm_out = LayerNorm(output_channel, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims,
            output_channel,
            out_channels,
            3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(
                torch.tensor(1000.0, dtype=torch.float32)
            )
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                output_channel * 2, 0
            )
            self.last_scale_shift_table = nn.Parameter(
                torch.randn(2, output_channel) / output_channel**0.5
            )

    def forward(
        self,
        sample: torch.FloatTensor,
        target_shape,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        assert target_shape is not None, "target_shape must be provided"
        batch_size = sample.shape[0]

        sample = self.conv_in(sample, causal=self.causal)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        sample = sample.to(upscale_dtype)

        if self.timestep_conditioning:
            assert (
                timestep is not None
            ), "should pass timestep with timestep_conditioning=True"
            scaled_timestep = timestep * self.timestep_scale_multiplier

        for up_block in self.up_blocks:
            if self.timestep_conditioning and isinstance(up_block, UNetMidBlock3D):
                sample = checkpoint_fn(up_block)(
                    sample, causal=self.causal, timestep=scaled_timestep
                )
            else:
                sample = checkpoint_fn(up_block)(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=sample.shape[0],
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(
                batch_size, embedded_timestep.shape[-1], 1, 1, 1
            )
            ada_values = self.last_scale_shift_table[
                None, ..., None, None, None
            ] + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.

    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        inject_noise (`bool`, *optional*, defaults to `False`):
            Whether to inject noise into the hidden states.
        timestep_conditioning (`bool`, *optional*, defaults to `False`):
            Whether to condition the hidden states on the timestep.
        attention_head_dim (`int`, *optional*, defaults to -1):
            The dimension of the attention head. If -1, no attention is used.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: str = "group_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        attention_head_dim: int = -1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0
            )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_blocks = None

        if attention_head_dim > 0:
            if attention_head_dim > in_channels:
                raise ValueError(
                    "attention_head_dim must be less than or equal to in_channels"
                )

            self.attention_blocks = nn.ModuleList(
                [
                    Attention(
                        query_dim=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        bias=True,
                        out_bias=True,
                        qk_norm="rms_norm",
                        residual_connection=True,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        timestep_embed = None
        if self.timestep_conditioning:
            assert (
                timestep is not None
            ), "should pass timestep with timestep_conditioning=True"
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(
                batch_size, timestep_embed.shape[-1], 1, 1, 1
            )

        if self.attention_blocks:
            for resnet, attention in zip(self.res_blocks, self.attention_blocks):
                hidden_states = resnet(
                    hidden_states, causal=causal, timestep=timestep_embed
                )

                # Reshape the hidden states to be (batch_size, frames * height * width, channel)
                batch_size, channel, frames, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, frames * height * width
                ).transpose(1, 2)

                if attention.use_tpu_flash_attention:
                    # Pad the second dimension to be divisible by block_k_major (block in flash attention)
                    seq_len = hidden_states.shape[1]
                    block_k_major = 512
                    pad_len = (block_k_major - seq_len % block_k_major) % block_k_major
                    if pad_len > 0:
                        hidden_states = F.pad(
                            hidden_states, (0, 0, 0, pad_len), "constant", 0
                        )

                    # Create a mask with ones for the original sequence length and zeros for the padded indexes
                    mask = torch.ones(
                        (hidden_states.shape[0], seq_len),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    if pad_len > 0:
                        mask = F.pad(mask, (0, pad_len), "constant", 0)

                hidden_states = attention(
                    hidden_states,
                    attention_mask=(
                        None if not attention.use_tpu_flash_attention else mask
                    ),
                )

                if attention.use_tpu_flash_attention:
                    # Remove the padding
                    if pad_len > 0:
                        hidden_states = hidden_states[:, :-pad_len, :]

                # Reshape the hidden states back to (batch_size, channel, frames, height, width, channel)
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, frames, height, width
                )
        else:
            for resnet in self.res_blocks:
                hidden_states = resnet(
                    hidden_states, causal=causal, timestep=timestep_embed
                )

        return hidden_states


class SpaceToDepthDownsample(nn.Module):
    def __init__(self, dims, in_channels, out_channels, stride, spatial_padding_mode):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * np.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // np.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, x, causal: bool = True):
        if self.stride[0] == 2:
            x = torch.cat(
                [x[:, :, :1, :, :], x], dim=2
            )  # duplicate first frames for padding

        # skip connection
        x_in = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        # conv
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        x = x + x_in

        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        dims,
        in_channels,
        stride,
        residual=False,
        out_channels_reduction_factor=1,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = (
            np.prod(stride) * in_channels // out_channels_reduction_factor
        )
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.pixel_shuffle = PixelShuffleND(dims=dims, upscale_factors=stride)
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(self, x, causal: bool = True):
        if self.residual:
            # Reshape and duplicate the input to match the output shape
            x_in = self.pixel_shuffle(x)
            num_repeat = np.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]
        x = self.conv(x, causal=causal)
        x = self.pixel_shuffle(x)
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "group_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        if norm_layer == "group_norm":
            self.norm1 = nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        elif norm_layer == "pixel_norm":
            self.norm1 = PixelNorm()
        elif norm_layer == "layer_norm":
            self.norm1 = LayerNorm(in_channels, eps=eps, elementwise_affine=True)

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        if norm_layer == "group_norm":
            self.norm2 = nn.GroupNorm(
                num_groups=groups, num_channels=out_channels, eps=eps, affine=True
            )
        elif norm_layer == "pixel_norm":
            self.norm2 = PixelNorm()
        elif norm_layer == "layer_norm":
            self.norm2 = LayerNorm(out_channels, eps=eps, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        self.conv_shortcut = (
            make_linear_nd(
                dims=dims, in_channels=in_channels, out_channels=out_channels
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm3 = (
            LayerNorm(in_channels, eps=eps, elementwise_affine=True)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(
                torch.randn(4, in_channels) / in_channels**0.5
            )

    def _feed_spatial_noise(
        self, hidden_states: torch.FloatTensor, per_channel_scale: torch.FloatTensor
    ) -> torch.FloatTensor:
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # similar to the "explicit noise inputs" method in style-gan
        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise

        return hidden_states

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        hidden_states = self.norm1(hidden_states)
        if self.timestep_conditioning:
            assert (
                timestep is not None
            ), "should pass timestep with timestep_conditioning=True"
            ada_values = self.scale_shift_table[
                None, ..., None, None, None
            ] + timestep.reshape(
                batch_size,
                4,
                -1,
                timestep.shape[-3],
                timestep.shape[-2],
                timestep.shape[-1],
            )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)

            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states, self.per_channel_scale1
            )

        hidden_states = self.norm2(hidden_states)

        if self.timestep_conditioning:
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states, self.per_channel_scale2
            )

        input_tensor = self.norm3(input_tensor)

        batch_size = input_tensor.shape[0]

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor
