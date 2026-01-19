# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

import torch
from src.helpers.helpers import helpers as helper_registry
from typing import Optional
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from src.vae.ltx2.model import AutoencoderKLLTX2Video
from einops import rearrange
from typing import Tuple
import math
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
    def __init__(
        self, channels: int, mid_channels: Optional[int] = None, dims: int = 3
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        Conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.conv1 = Conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = Conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation(hidden_states + residual)
        return hidden_states


class PixelShuffleND(torch.nn.Module):
    def __init__(self, dims, upscale_factors=(2, 2, 2)):
        super().__init__()

        self.dims = dims
        self.upscale_factors = upscale_factors

        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3")

    def forward(self, x):
        if self.dims == 3:
            # spatiotemporal: b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:3]))
                .permute(0, 1, 5, 2, 6, 3, 7, 4)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        elif self.dims == 2:
            # spatial: b (c p1 p2) h w -> b c (h p1) (w p2)
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:2]))
                .permute(0, 1, 4, 2, 5, 3)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        elif self.dims == 1:
            # temporal: b (c p1) f h w -> b c (f p1) h w
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:1]))
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(2, 3)
            )


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    mapping = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
    if float(scale) not in mapping:
        raise ValueError(
            f"Unsupported scale {scale}. Choose from {list(mapping.keys())}"
        )
    return mapping[float(scale)]


class BlurDownsample(torch.nn.Module):
    """
    Anti-aliased spatial downsampling by integer stride using a fixed separable binomial kernel.
    Applies only on H,W. Works for dims=2 or dims=3 (per-frame).
    """

    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        assert dims in (2, 3)
        assert isinstance(stride, int)
        assert stride >= 1
        assert kernel_size >= 3
        assert kernel_size % 2 == 1
        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

        # 5x5 separable binomial kernel using binomial coefficients [1, 4, 6, 4, 1] from
        # the 4th row of Pascal's triangle. This kernel is used for anti-aliasing and
        # provides a smooth approximation of a Gaussian filter (often called a "binomial filter").
        # The 2D kernel is constructed as the outer product and normalized.
        k = torch.tensor([math.comb(kernel_size - 1, k) for k in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()  # shape (kernel_size, kernel_size)
        self.register_buffer(
            "kernel", k2d[None, None, :, :]
        )  # (1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        if self.dims == 2:
            return self._apply_2d(x)
        else:
            # dims == 3: apply per-frame on H,W
            b, _, f, _, _ = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self._apply_2d(x)
            h2, w2 = x.shape[-2:]
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)
            return x

    def _apply_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        c = x2d.shape[1]
        weight = self.kernel.expand(
            c, 1, self.kernel_size, self.kernel_size
        )  # depthwise
        x2d = F.conv2d(
            x2d,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=c,
        )
        return x2d


class SpatialRationalResampler(torch.nn.Module):
    """
    Fully-learned rational spatial scaling: up by 'num' via PixelShuffle, then anti-aliased
    downsample by 'den' using fixed blur + stride. Operates on H,W only.
    For dims==3, work per-frame for spatial scaling (temporal axis untouched).
    Args:
        mid_channels (`int`): Number of intermediate channels for the convolution layer
        scale (`float`): Spatial scaling factor. Supported values are:
            - 0.75: Downsample by 3/4 (reduce spatial size)
            - 1.5: Upsample by 3/2 (increase spatial size)
            - 2.0: Upsample by 2x (double spatial size)
            - 4.0: Upsample by 4x (quadruple spatial size)
            Any other value will raise a ValueError.
    """

    def __init__(self, mid_channels: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _rational_for_scale(self.scale)
        self.conv = torch.nn.Conv2d(
            mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1
        )
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x


@helper_registry("ltx2.upsampler")
class LTXLatentUpsamplerModel(ModelMixin, ConfigMixin):
    """
    Model to spatially upsample VAE latents.

    Args:
        in_channels (`int`, defaults to `128`):
            Number of channels in the input latent
        mid_channels (`int`, defaults to `512`):
            Number of channels in the middle layers
        num_blocks_per_stage (`int`, defaults to `4`):
            Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`, defaults to `3`):
            Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`, defaults to `True`):
            Whether to spatially upsample the latent
        temporal_upsample (`bool`, defaults to `False`):
            Whether to temporally upsample the latent
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        self.res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(
                    mid_channels, 8 * mid_channels, kernel_size=3, padding=1
                ),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = SpatialRationalResampler(
                    mid_channels=mid_channels, scale=self.spatial_scale
                )
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        mid_channels, 4 * mid_channels, kernel_size=3, padding=1
                    ),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(
                    mid_channels, 2 * mid_channels, kernel_size=3, padding=1
                ),
                PixelShuffleND(1),
            )
        else:
            raise ValueError(
                "Either spatial_upsample or temporal_upsample must be True"
            )

        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                # remove the first frame after upsampling.
                # This is done because the first frame encodes one pixel frame.
                x = x[:, :, 1:, :, :]
            elif isinstance(self.upsampler, SpatialRationalResampler):
                x = self.upsampler(x)
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


def upsample_video(
    video: torch.Tensor,
    video_vae: AutoencoderKLLTX2Video,
    upsampler: LTXLatentUpsamplerModel,
) -> torch.Tensor:
    video = video_vae.denormalize_latents(video)
    video = video.to(upsampler.dtype)
    video = upsampler(video)
    video = video.to(video_vae.dtype)
    video = video_vae.normalize_latents(video)
    return video
