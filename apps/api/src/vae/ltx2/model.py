from dataclasses import replace
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from src.vae.ltx2.normalization import PixelNorm
from src.transformer.ltx2.base.model import PixArtAlphaCombinedTimestepSizeEmbeddings
from src.vae.ltx2.convolution import make_conv_nd
from src.vae.ltx2.enums import LogVarianceType, NormLayerType, PaddingModeType
from src.vae.ltx2.ops import patchify, unpatchify
from src.vae.ltx2.resnet import ResnetBlock3D, UNetMidBlock3D
from src.vae.ltx2.sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from src.vae.ltx2.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    create_tiles,
)
from tqdm import tqdm

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from src.engine.ltx22.shared.types import SpatioTemporalScaleFactors, VideoLatentShape


def _ltx2_downsample_type_to_encoder_block_name(downsample_type: str) -> str:
    """
    Map diffusers-style LTX2 VAE `downsample_type` entries to this module's encoder block names.

    Observed values:
    - "spatial" -> downsample H,W
    - "temporal" -> downsample F
    - "spatiotemporal" -> downsample F,H,W
    """
    mapping = {
        "spatial": "compress_space_res",
        "temporal": "compress_time_res",
        "spatiotemporal": "compress_all_res",
    }
    if downsample_type not in mapping:
        raise ValueError(f"Unsupported downsample_type={downsample_type!r}. Expected one of {sorted(mapping)}.")
    return mapping[downsample_type]


def _ltx2_build_encoder_blocks(
    *,
    layers_per_block: List[int],
    downsample_type: List[str],
    spatio_temporal_scaling: Optional[List[bool]] = None,
    multiplier: int = 2,
) -> List[list[Any]]:
    """
    Convert diffusers-style encoder description into our `VideoEncoder.encoder_blocks` format.

    Pattern:
      res_x(layers_per_block[0]),
      downsample(downsample_type[0]),
      res_x(layers_per_block[1]),
      ...
      res_x(layers_per_block[-1])
    """
    if len(layers_per_block) != len(downsample_type) + 1:
        raise ValueError(
            "Expected len(layers_per_block) == len(downsample_type) + 1, "
            f"got {len(layers_per_block)} vs {len(downsample_type)}."
        )
    if spatio_temporal_scaling is not None and len(spatio_temporal_scaling) != len(downsample_type):
        raise ValueError(
            "If provided, expected len(spatio_temporal_scaling) == len(downsample_type), "
            f"got {len(spatio_temporal_scaling)} vs {len(downsample_type)}."
        )

    blocks: List[list[Any]] = []
    for i, num_layers in enumerate(layers_per_block):
        blocks.append(["res_x", {"num_layers": int(num_layers)}])
        if i >= len(downsample_type):
            continue

        should_scale = True if spatio_temporal_scaling is None else bool(spatio_temporal_scaling[i])
        if not should_scale:
            continue

        blocks.append(
            [
                _ltx2_downsample_type_to_encoder_block_name(downsample_type[i]),
                {"multiplier": int(multiplier)},
            ]
        )

    return blocks


def _ltx2_build_decoder_blocks(
    *,
    decoder_layers_per_block: List[int],
    decoder_inject_noise: List[bool],
    upsample_factor: List[int],
    upsample_residual: List[bool],
    decoder_spatio_temporal_scaling: Optional[List[bool]] = None,
) -> List[list[Any]]:
    """
    Convert diffusers-style decoder description into our `VideoDecoder.decoder_blocks` format.

    Pattern:
      res_x(...),
      compress_all(...),
      res_x(...),
      ...
    """
    if len(decoder_layers_per_block) != len(decoder_inject_noise):
        raise ValueError(
            "Expected len(decoder_layers_per_block) == len(decoder_inject_noise), "
            f"got {len(decoder_layers_per_block)} vs {len(decoder_inject_noise)}."
        )
    if len(upsample_factor) != len(upsample_residual):
        raise ValueError(
            "Expected len(upsample_factor) == len(upsample_residual), "
            f"got {len(upsample_factor)} vs {len(upsample_residual)}."
        )
    if len(upsample_factor) != len(decoder_layers_per_block) - 1:
        raise ValueError(
            "Expected len(upsample_factor) == len(decoder_layers_per_block) - 1, "
            f"got {len(upsample_factor)} vs {len(decoder_layers_per_block)}."
        )
    if decoder_spatio_temporal_scaling is not None and len(decoder_spatio_temporal_scaling) != len(upsample_factor):
        raise ValueError(
            "If provided, expected len(decoder_spatio_temporal_scaling) == len(upsample_factor), "
            f"got {len(decoder_spatio_temporal_scaling)} vs {len(upsample_factor)}."
        )

    blocks: List[list[Any]] = []
    for i, num_layers in enumerate(decoder_layers_per_block):
        blocks.append(
            [
                "res_x",
                {"num_layers": int(num_layers), "inject_noise": bool(decoder_inject_noise[i])},
            ]
        )
        if i >= len(upsample_factor):
            continue

        should_scale = True if decoder_spatio_temporal_scaling is None else bool(decoder_spatio_temporal_scaling[i])
        if not should_scale:
            continue

        blocks.append(
            [
                "compress_all",
                {
                    "residual": bool(upsample_residual[i]),
                    "multiplier": int(upsample_factor[i]),
                },
            ]
        )

    return blocks


def convert_diffusers_ltx2_vae_config_to_causal_video_autoencoder_config(diffusers_config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a diffusers-style LTX2 VAE config (AutoencoderKLLTX2Video) into a dict that looks like the
    CausalVideoAutoencoder config consumed by `VideoEncoderConfigurator`/`VideoDecoderConfigurator`.
    """
    in_channels = int(diffusers_config.get("in_channels", 3))
    out_channels = int(diffusers_config.get("out_channels", 3))
    latent_channels = int(diffusers_config.get("latent_channels", 128))
    patch_size = int(diffusers_config.get("patch_size", 4))

    encoder_blocks = _ltx2_build_encoder_blocks(
        layers_per_block=list(diffusers_config.get("layers_per_block", [])),
        downsample_type=list(diffusers_config.get("downsample_type", [])),
        spatio_temporal_scaling=diffusers_config.get("spatio_temporal_scaling"),
    )
    decoder_blocks = _ltx2_build_decoder_blocks(
        decoder_layers_per_block=list(diffusers_config.get("decoder_layers_per_block", [])),
        decoder_inject_noise=list(diffusers_config.get("decoder_inject_noise", [])),
        upsample_factor=list(diffusers_config.get("upsample_factor", [])),
        upsample_residual=list(diffusers_config.get("upsample_residual", [])),
        decoder_spatio_temporal_scaling=diffusers_config.get("decoder_spatio_temporal_scaling"),
    )

    block_out_channels = diffusers_config.get("block_out_channels") or []
    decoder_block_out_channels = diffusers_config.get("decoder_block_out_channels") or []
    encoder_base_channels = int(block_out_channels[0] // 2) if block_out_channels else int(latent_channels)
    decoder_base_channels = (
        int(decoder_block_out_channels[0] // 2) if decoder_block_out_channels else int(encoder_base_channels)
    )

    raw_scaling_factor = diffusers_config.get("scaling_factor", 1.0)
    try:
        scaling_factor_f = float(raw_scaling_factor)
        scaling_factor: float | int = int(scaling_factor_f) if scaling_factor_f.is_integer() else scaling_factor_f
    except (TypeError, ValueError):
        scaling_factor = raw_scaling_factor

    return {
        "_class_name": "CausalVideoAutoencoder",
        "dims": 3,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "latent_channels": latent_channels,
        "encoder_blocks": encoder_blocks,
        "decoder_blocks": decoder_blocks,
        "scaling_factor": scaling_factor,
        "norm_layer": "pixel_norm",
        "patch_size": patch_size,
        "latent_log_var": "uniform",
        "use_quant_conv": False,
        "causal_decoder": bool(diffusers_config.get("decoder_causal", False)),
        "timestep_conditioning": bool(diffusers_config.get("timestep_conditioning", False)),
        "normalize_latent_channels": False,
        "encoder_base_channels": encoder_base_channels,
        "decoder_base_channels": decoder_base_channels,
    }


def _as_padding_mode(mode: PaddingModeType | str) -> PaddingModeType:
    if isinstance(mode, PaddingModeType):
        return mode
    return PaddingModeType(mode)

def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Encoder. Encodes video frames into a latent representation.
    The encoder compresses the input video through a series of downsampling operations controlled by
    patch_size and encoder_blocks. The output is a normalized latent tensor with shape (B, 128, F', H', W').
    Compression Behavior:
        The total compression is determined by:
        1. Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        2. Sequential compression through encoder_blocks based on their stride patterns
        Compression blocks apply 2x compression in specified dimensions:
            - "compress_time" / "compress_time_res": temporal only
            - "compress_space" / "compress_space_res": spatial only (H and W)
            - "compress_all" / "compress_all_res": all dimensions (F, H, W)
            - "res_x" / "res_x_y": no compression
        Standard LTX Video configuration:
            - patch_size=4
            - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
            - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
            - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
            - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels. For RGB images, this is 3.
        out_channels: The number of output channels (latent channels). For latent channels, this is 128.
        encoder_blocks: The list of blocks to construct the encoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: The patch size for initial spatial compression. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var: The log variance mode. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int]] | List[Tuple[str, dict[str, Any]]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )

            self.down_blocks.append(block)

        # out
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""
        Encode video frames into normalized latent representation.
        Args:
            sample: Input video (B, C, F, H, W). F must be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )

        # Initial spatial compression: trade spatial resolution for channel depth
        # This reduces H,W by patch_size and increases channels, making convolutions more efficient
        # Example: (B, 3, F, 512, 512) -> (B, 48, F, 128, 128) with patch_size=4
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform Variance: model outputs N means and 1 shared log-variance channel.
            # We need to expand the single logvar to match the number of means channels
            # to create a format compatible with PER_CHANNEL (means + logvar, each with N channels).
            # Sample shape: (B, N+1, ...) where N = latent_channels (e.g., 128 means + 1 logvar = 129)
            # Target shape: (B, 2*N, ...) where first N are means, last N are logvar

            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2 channels "
                    f"(N means + 1 logvar), got {sample.shape[1]}"
                )

            # Extract means (first N channels) and logvar (last 1 channel)
            means = sample[:, :-1, ...]  # (B, N, ...)
            logvar = sample[:, -1:, ...]  # (B, 1, ...)

            # Repeat logvar N times to match means channels
            # Use expand/repeat pattern that works for both 4D and 5D tensors
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)  # (B, N, ...)

            # Concatenate to create (B, 2*N, ...) format: [means, repeated_logvar]
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.normalize_latents(means)


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


class VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Decoder. Decodes latent representation into video frames.
    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.
    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling
    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
        First frame removed after temporal upsampling in both modes. Output shape unchanged.
        Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512) for both modes.
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels (latent channels). Default is 128.
        out_channels: The number of output channels. For RGB images, this is 3.
        decoder_blocks: The list of blocks to construct the decoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: Final spatial expansion factor. For standard LTX Video, use 4 for 4x spatial expansion:
                    H -> Hx4, W -> Wx4. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal: Whether to use causal convolutions. For standard LTX Video, use False for symmetric padding.
                When True, uses causal padding (past/current frames only).
        timestep_conditioning: Whether to condition the decoder on timestep for denoising.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__()

        # Spatiotemporal downscaling between decoded video space and VAE latents.
        # According to the LTXV paper, the standard configuration downsamples
        # video inputs by a factor of 8 in the temporal dimension and 32 in
        # each spatial dimension (height and width). This parameter determines how
        # many video frames and pixels correspond to a single latent cell.
        self.video_downscale_factors = SpatioTemporalScaleFactors(
            time=8,
            width=32,
            height=32,
        )

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # Compute initial feature_channels by going through blocks in reverse
        # This determines the channel width at the start of the decoder
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Decode latent representation into video frames.
        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True). Uses default 0.05 if None.
            generator: Random generator for deterministic noise injection (if inject_noise=True in blocks).
        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512).
            Note: First frame is removed after temporal upsampling regardless of causal mode.
            When causal=False, allows future frame dependencies in convolutions but maintains same output shape.
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )

            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.denormalize_latents(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
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

        # Final spatial expansion: reverse the initial patchify from encoder
        # Moves pixels from channels back to spatial dimensions
        # Example: (B, 48, F, 128, 128) -> (B, 3, F, 512, 512) with patch_size=4
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample

    def _prepare_tiles(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> List[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION] * len(latent.shape)
        mappers = [DEFAULT_MAPPING_OPERATION] * len(latent.shape)
        if tiling_config is not None and tiling_config.spatial_config is not None:
            cfg = tiling_config.spatial_config
            long_side = max(latent.shape[3], latent.shape[4])

            def enable_on_axis(axis_idx: int, factor: int) -> None:
                size = cfg.tile_size_in_pixels // factor
                overlap = cfg.tile_overlap_in_pixels // factor
                axis_length = latent.shape[axis_idx]
                lower_threshold = max(2, overlap + 1)
                tile_size = max(lower_threshold, round(size * axis_length / long_side))
                splitters[axis_idx] = split_in_spatial(tile_size, overlap)
                mappers[axis_idx] = to_mapping_operation(map_spatial_slice, factor)

            enable_on_axis(3, self.video_downscale_factors.height)
            enable_on_axis(4, self.video_downscale_factors.width)

        if tiling_config is not None and tiling_config.temporal_config is not None:
            cfg = tiling_config.temporal_config
            tile_size = cfg.tile_size_in_frames // self.video_downscale_factors.time
            overlap = cfg.tile_overlap_in_frames // self.video_downscale_factors.time
            splitters[2] = split_in_temporal(tile_size, overlap)
            mappers[2] = to_mapping_operation(map_temporal_slice, self.video_downscale_factors.time)

        return create_tiles(latent.shape, splitters, mappers)

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode a latent tensor into video frames using tiled processing.
        Splits the latent tensor into tiles, decodes each tile individually,
        and yields video chunks as they become available.
        Args:
            latent: Input latent tensor (B, C, F', H', W').
            tiling_config: Tiling configuration for the latent tensor.
            timestep: Optional timestep for decoder conditioning.
            generator: Optional random generator for deterministic decoding.
        Yields:
            Video chunks (B, C, T, H, W) by temporal slices;
        """

        # Calculate full video shape from latent shape to get spatial dimensions
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
        tiles = self._prepare_tiles(latent, tiling_config)

        temporal_groups = self._group_tiles_by_temporal_slice(tiles)

        # State for temporal overlap handling
        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]

            # Calculate the shape of the temporal buffer for this group of tiles.
            # The temporal length depends on whether this is the first tile (starts at 0) or not.
            # - First tile: (frames - 1) * scale + 1
            # - Subsequent tiles: frames * scale
            # This logic is handled by TemporalAxisMapping and reflected in out_coords.
            temporal_tile_buffer_shape = full_video_shape._replace(
                frames=curr_temporal_slice.stop - curr_temporal_slice.start,
            )

            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=latent.device,
                dtype=latent.dtype,
            )

            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
            )

            # Blend with previous temporal chunk if it exists
            if previous_chunk is not None:
                # Check if current temporal slice overlaps with previous temporal slice
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)

                    # The overlap is already masked before it reaches this step. Each tile is accumulated into buffer
                    # with its trapezoidal mask, and curr_weights accumulates the same mask. In the overlap blend we add
                    # the masked values (buffer[...]) and the corresponding weights (curr_weights[...]) into the
                    # previous buffers, then later normalize by weights.
                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[:, :, slice(0, overlap_len), :, :]
                    previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[
                        :, :, slice(0, overlap_len), :, :
                    ]

                    buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]
                    curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[
                        :, :, temporal_overlap_slice, :, :
                    ]

                # Yield the non-overlapping part of the previous chunk
                previous_weights = previous_weights.clamp(min=1e-8)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                yield (previous_chunk / previous_weights)[:, :, :yield_len, :, :]

            # Update state for next iteration
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice

        # Yield any remaining chunk
        if previous_chunk is not None:
            previous_weights = previous_weights.clamp(min=1e-8)
            yield previous_chunk / previous_weights

    def _group_tiles_by_temporal_slice(self, tiles: List[Tile]) -> List[List[Tile]]:
        """Group tiles by their temporal output slice."""
        if not tiles:
            return []

        groups = []
        current_slice = tiles[0].out_coords[2]
        current_group = []

        for tile in tiles:
            tile_slice = tile.out_coords[2]
            if tile_slice == current_slice:
                current_group.append(tile)
            else:
                groups.append(current_group)
                current_slice = tile_slice
                current_group = [tile]

        # Add the final group
        if current_group:
            groups.append(current_group)

        return groups

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """
        Decode and accumulate all tiles of a temporal group into a local buffer.
        The buffer is local to the group and always starts at time 0; temporal coordinates
        are rebased by subtracting temporal_slice.start.
        """
        temporal_slice = group_tiles[0].out_coords[2]

        weights = torch.zeros_like(buffer)

        for tile in group_tiles:
            decoded_tile = self.forward(latent[tile.in_coords], timestep, generator)
            mask = tile.blend_mask.to(device=buffer.device, dtype=buffer.dtype)
            temporal_offset = tile.out_coords[2].start - temporal_slice.start
            # Use the tile's output coordinate length, not the decoded tile's length,
            # as the decoder may produce a different number of frames than expected
            expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
            decoded_temporal_len = decoded_tile.shape[2]

            # Ensure we don't exceed the buffer or decoded tile bounds
            actual_temporal_len = min(expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset)

            chunk_coords = (
                slice(None),  # batch
                slice(None),  # channels
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],  # height
                tile.out_coords[4],  # width
            )

            # Slice decoded_tile and mask to match the actual length we're writing
            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask

            buffer[chunk_coords] += decoded_slice * mask_slice
            weights[chunk_coords] += mask_slice

        return weights



class AutoencoderKLLTX2Video(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, 
                 block_out_channels: List[int] = [128, 256, 512, 512],
                 decoder_block_out_channels: List[int] = [128, 256, 512, 512],
                 decoder_causal: bool = False,
                 decoder_inject_noise: List[bool] = [False, False, False, False],  # noqa: B006
                 decoder_layers_per_block: List[int] = [2, 2, 2, 2],
                 decoder_spatial_padding_mode: PaddingModeType | str = PaddingModeType.REFLECT,
                 decoder_spatio_temporal_scaling: List[bool] = [True, True, True, False],
                 down_block_types: List[str] = ["DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"],
                 downsample_type: List[str] = ["spatial", "temporal", "spatial", "temporal"],
                 encoder_causal: bool = False, 
                 encoder_spatial_padding_mode: PaddingModeType | str = PaddingModeType.REFLECT,
                 in_channels: int = 3,
                 latent_channels: int = 128,
                 layers_per_block: List[int] = [2, 2, 2, 2],
                 out_channels: int = 3,
                 patch_size: int = 4,
                 patch_size_t: int = 1,
                 resnet_norm_eps: float = 1e-06,
                 scaling_factor: float = 1.0,
                 spatial_compression_ratio: int = 32,
                 spatio_temporal_scaling: List[bool] = [True, True, True, True],
                 temporal_compression_ratio: int = 8,
                 timestep_conditioning: bool = False,
                 upsample_factor: List[int] = [2, 2, 2],
                 upsample_residual: List[bool] = [True, True, True],
                 ):
    
        super().__init__()

        # Diffusers configs provide padding modes as strings; normalize to enums.
        encoder_spatial_padding_mode = _as_padding_mode(encoder_spatial_padding_mode)
        decoder_spatial_padding_mode = _as_padding_mode(decoder_spatial_padding_mode)

        # Back-compat: some configs may provide a single bool instead of a per-block list.
        if isinstance(decoder_inject_noise, bool):
            decoder_inject_noise = [decoder_inject_noise] * len(decoder_layers_per_block)
        
        converted_cfg = convert_diffusers_ltx2_vae_config_to_causal_video_autoencoder_config(
            {
                "block_out_channels": block_out_channels,
                "decoder_block_out_channels": decoder_block_out_channels,
                "decoder_causal": decoder_causal,
                "decoder_inject_noise": decoder_inject_noise,
                "decoder_layers_per_block": decoder_layers_per_block,
                "decoder_spatio_temporal_scaling": decoder_spatio_temporal_scaling,
                "downsample_type": downsample_type,
                "encoder_causal": encoder_causal,
                "in_channels": in_channels,
                "latent_channels": latent_channels,
                "layers_per_block": layers_per_block,
                "out_channels": out_channels,
                "patch_size": patch_size,
                "patch_size_t": patch_size_t,
                "resnet_norm_eps": resnet_norm_eps,
                "scaling_factor": scaling_factor,
                "spatio_temporal_scaling": spatio_temporal_scaling,
                "timestep_conditioning": timestep_conditioning,
                "upsample_factor": upsample_factor,
                "upsample_residual": upsample_residual,
            }
        )
        

        self.encoder = VideoEncoder(
            convolution_dimensions=converted_cfg["dims"],
            in_channels=converted_cfg["in_channels"],
            out_channels=converted_cfg["latent_channels"],
            encoder_blocks=converted_cfg["encoder_blocks"],
            patch_size=converted_cfg["patch_size"],
            norm_layer=NormLayerType(converted_cfg["norm_layer"]),
            latent_log_var=LogVarianceType(converted_cfg["latent_log_var"]),
            encoder_spatial_padding_mode=encoder_spatial_padding_mode,
        )
        
        self.decoder = VideoDecoder(
            convolution_dimensions=converted_cfg["dims"],
            in_channels=converted_cfg["latent_channels"],
            out_channels=converted_cfg["out_channels"],
            decoder_blocks=converted_cfg["decoder_blocks"],
            patch_size=converted_cfg["patch_size"],
            norm_layer=NormLayerType(converted_cfg["norm_layer"]),
            causal=converted_cfg["causal_decoder"],
            timestep_conditioning=converted_cfg["timestep_conditioning"],
            decoder_spatial_padding_mode=decoder_spatial_padding_mode,
        )

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)
        
        self.decoder.denormalize_latents = self.denormalize_latents
        self.decoder.normalize_latents = self.normalize_latents
        self.encoder.denormalize_latents = self.denormalize_latents
        self.encoder.normalize_latents = self.normalize_latents

        self.spatial_compression_ratio = (
            patch_size * 2 ** sum(spatio_temporal_scaling)
            if spatial_compression_ratio is None
            else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_size_t * 2 ** sum(spatio_temporal_scaling)
            if temporal_compression_ratio is None
            else temporal_compression_ratio
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def denormalize_latents(self, latents: torch.Tensor):
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_std / self.config.scaling_factor + latents_mean
        return latents

    def normalize_latents(self, latents: torch.Tensor):
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * self.config.scaling_factor / latents_std
        return latents

@torch.no_grad()
def decode_video(
    latent: torch.Tensor,
    video_decoder: VideoDecoder,
    tiling_config: TilingConfig | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """
    Decode a video latent tensor with the given decoder.
    Args:
        latent: Tensor [c, f, h, w]
        video_decoder: Decoder module.
        tiling_config: Optional tiling settings.
        generator: Optional random generator for deterministic decoding.
    Yields:
        Decoded chunk [f, h, w, c], uint8 in [0, 255].
    """
    
    video_decoder.eval()

    def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
        frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")
        return frames
    
    out_frames = []
    if tiling_config is not None:
        for frames in tqdm(video_decoder.tiled_decode(latent, tiling_config, generator=generator), desc="Decoding video"):
            out_frames.append(convert_to_uint8(frames))
    else:
        decoded_video = video_decoder(latent, generator=generator)
        out_frames.append(convert_to_uint8(decoded_video))

    return torch.cat(out_frames, dim=0)


def get_video_chunks_number(num_frames: int, tiling_config: TilingConfig | None = None) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.
    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.
    Returns:
        Number of video chunks.
    """
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    frame_stride = cfg.tile_size_in_frames - cfg.tile_overlap_in_frames
    return (num_frames - 1 + frame_stride - 1) // frame_stride


def split_in_spatial(size: int, overlap: int) -> SplitOperation:
    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def split_in_temporal(size: int, overlap: int) -> SplitOperation:
    non_causal_split = split_in_spatial(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = intervals.starts
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = intervals.left_ramps
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor]],
    scale: int,
) -> MappingOperation:
    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor | None] = []
        number_of_slices = len(intervals.starts)
        for i in range(number_of_slices):
            start = intervals.starts[i]
            end = intervals.ends[i]
            left_ramp = intervals.left_ramps[i]
            right_ramp = intervals.right_ramps[i]
            output_slice, mask_1d = map_func(start, end, left_ramp, right_ramp, scale)
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp = 1 + (left_ramp - 1) * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, True)


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp = left_ramp * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, False)
