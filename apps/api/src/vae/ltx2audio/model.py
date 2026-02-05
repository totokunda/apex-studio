from typing import Any, Set, Tuple, Optional

import torch
import torch.nn.functional as F

from src.engine.ltx22.shared.patchifiers import AudioPatchifier
from src.vae.ltx2audio.attention import AttentionType, make_attn
from src.vae.ltx2audio.causal_conv_2d import make_conv2d
from src.vae.ltx2audio.causality_axis import CausalityAxis
from src.vae.ltx2audio.downsample import build_downsampling_path
from src.vae.ltx2audio.ops import PerChannelStatistics
from src.vae.ltx2audio.resnet import ResnetBlock
from src.vae.ltx2audio.upsample import build_upsampling_path
from src.vae.ltx2audio.vocoder import Vocoder
from src.vae.ltx2audio.normalization import NormType, build_normalization_layer
from src.engine.ltx22.shared.types import AudioLatentShape
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import AutoencoderMixin

LATENT_DOWNSAMPLE_FACTOR = 4


def convert_diffusers_ltx2_audio_vae_config_to_preprocessing_model_config(diffusers_config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a diffusers-style `AutoencoderKLLTX2Audio` config into the nested config shape:

    - `preprocessing.audio`, `preprocessing.stft`, `preprocessing.mel`
    - `model.params.sampling_rate`, `model.params.embed_dim`, `model.params.ddconfig`

    This matches the schema used by LTX audio VAE checkpoints and `src/vae/ltx2audio/model_configurator.py`.
    """
    sample_rate = int(diffusers_config.get("sample_rate", 16000))
    mel_hop_length = int(diffusers_config.get("mel_hop_length", 160))
    mel_bins = int(diffusers_config.get("mel_bins", 64))
    is_causal = bool(diffusers_config.get("is_causal", True))

    latent_channels = int(diffusers_config.get("latent_channels", 8))

    attn_resolutions = diffusers_config.get("attn_resolutions")
    # Diffusers may store this as null; nested config uses [].
    attn_resolutions_list = [] if attn_resolutions is None else list(attn_resolutions)

    return {
        "preprocessing": {
            "audio": {
                "sampling_rate": sample_rate,
                "max_wav_value": 32768,
                "duration": 5.12,
                "stereo": True,
                "causal_padding": 3,
            },
            "stft": {
                "filter_length": 1024,
                "hop_length": mel_hop_length,
                "win_length": 1024,
                "causal": is_causal,
            },
            "mel": {
                "n_mel_channels": mel_bins,
                "mel_fmin": 0,
                "mel_fmax": sample_rate // 2,
            },
        },
        "model": {
            "params": {
                "sampling_rate": sample_rate,
                "embed_dim": latent_channels,
                "ddconfig": {
                    "double_z": bool(diffusers_config.get("double_z", True)),
                    "mel_bins": mel_bins,
                    "z_channels": latent_channels,
                    "resolution": int(diffusers_config.get("resolution", 256)),
                    "downsample_time": False,
                    "in_channels": int(diffusers_config.get("in_channels", 2)),
                    "out_ch": int(diffusers_config.get("output_channels", 2)),
                    "ch": int(diffusers_config.get("base_channels", 128)),
                    "ch_mult": list(diffusers_config.get("ch_mult", (1, 2, 4))),
                    "num_res_blocks": int(diffusers_config.get("num_res_blocks", 2)),
                    "attn_resolutions": attn_resolutions_list,
                    "dropout": float(diffusers_config.get("dropout", 0.0)),
                    "mid_block_add_attention": bool(diffusers_config.get("mid_block_add_attention", False)),
                    "norm_type": str(diffusers_config.get("norm_type", "pixel")),
                    "causality_axis": str(diffusers_config.get("causality_axis", "height")),
                },
            }
        },
    }


def _as_causality_axis(axis: Optional[str]) -> CausalityAxis:
    if axis in (None, "none"):
        return CausalityAxis.NONE
    return CausalityAxis(axis)

def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> torch.nn.Module:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = torch.nn.Module()
    mid.block_1 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid.attn_1 = make_attn(channels, attn_type=attn_type, norm_type=norm_type) if add_attention else torch.nn.Identity()
    mid.block_2 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run features through the middle block."""
    features = mid.block_1(features, temb=None)
    features = mid.attn_1(features)
    return mid.block_2(features, temb=None)


class AudioEncoder(torch.nn.Module):
    """
    Encoder that compresses audio spectrograms into latent representations.
    The encoder uses a series of downsampling blocks with residual connections,
    attention mechanisms, and configurable causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        attn_type: AttentionType = AttentionType.VANILLA,
        mid_block_add_attention: bool = True,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: int = 64,
        **_ignore_kwargs,
    ) -> None:
        """
        Initialize the Encoder.
        Args:
            Arguments are configuration parameters, loaded from the audio VAE checkpoint config
            (audio_vae.model.params.ddconfig):
            ch: Base number of feature channels used in the first convolution layer.
            ch_mult: Multiplicative factors for the number of channels at each resolution level.
            num_res_blocks: Number of residual blocks to use at each resolution level.
            attn_resolutions: Spatial resolutions (e.g., in time/frequency) at which to apply attention.
            resolution: Input spatial resolution of the spectrogram (height, width).
            z_channels: Number of channels in the latent representation.
            norm_type: Normalization layer type to use within the network (e.g., group, batch).
            causality_axis: Axis along which convolutions should be causal (e.g., time axis).
            sample_rate: Audio sample rate in Hz for the input signals.
            mel_hop_length: Hop length used when computing the mel spectrogram.
            n_fft: FFT size used to compute the spectrogram.
            mel_bins: Number of mel-frequency bins in the input spectrogram.
            in_channels: Number of channels in the input spectrogram tensor.
            double_z: If True, predict both mean and log-variance (doubling latent channels).
            is_causal: If True, use causal convolutions suitable for streaming setups.
            dropout: Dropout probability used in residual and mid blocks.
            attn_type: Type of attention mechanism to use in attention blocks.
            resamp_with_conv: If True, perform resolution changes using strided convolutions.
            mid_block_add_attention: If True, add an attention block in the mid-level of the encoder.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.double_z = double_z
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        # downsampling
        self.conv_in = make_conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.non_linearity = torch.nn.SiLU()

        self.down, block_in = build_downsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
        )

        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrogram into latent representations.
        Args:
            spectrogram: Input spectrogram of shape (batch, channels, time, frequency)
        Returns:
            Encoded latent representation of shape (batch, channels, frames, mel_bins)
        """
        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self._finalize_output(h)

        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage.block[block_idx](h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != self.num_resolutions - 1:
                h = stage.downsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_out(h)
        h = self.non_linearity(h)
        return self.conv_out(h)

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        """
        Normalize encoder latents using per-channel statistics.
        When the encoder is configured with ``double_z=True``, the final
        convolution produces twice the number of latent channels, typically
        interpreted as two concatenated tensors along the channel dimension
        (e.g., mean and variance or other auxiliary parameters).
        This method intentionally uses only the first half of the channels
        (the "mean" component) as input to the patchifier and normalization
        logic. The remaining channels are left unchanged by this method and
        are expected to be consumed elsewhere in the VAE pipeline.
        If ``double_z=False``, the encoder output already contains only the
        mean latents and the chunking operation simply returns that tensor.
        """
        means = torch.chunk(latent_output, 2, dim=1)[0]
        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[1],
            frames=means.shape[2],
            mel_bins=means.shape[3],
        )
        latent_patched = self.patchifier.patchify(means)
        latent_normalized = self.normalize_latents(latent_patched)
        return self.patchifier.unpatchify(latent_normalized, latent_shape)


class AudioDecoder(torch.nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.
    The decoder mirrors the encoder structure with configurable channel multipliers,
    attention resolutions, and causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        resolution: int,
        z_channels: int,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = None,
    ) -> None:
        """
        Initialize the Decoder.
        Args:
            Arguments are configuration parameters, loaded from the audio VAE checkpoint config
            (audio_vae.model.params.ddconfig):
            - ch, out_ch, ch_mult, num_res_blocks, attn_resolutions
            - resolution, z_channels
            - norm_type, causality_axis
        """
        super().__init__()

        # Internal behavioural defaults that are not driven by the checkpoint.
        resamp_with_conv = True
        attn_type = AttentionType.VANILLA

        # Per-channel statistics for denormalizing latents
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_ch = out_ch
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )
        self.non_linearity = torch.nn.SiLU()
        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )
        self.up, final_block_channels = build_upsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back to audio spectrograms.
        Args:
            sample: Encoded latent representation of shape (batch, channels, frames, mel_bins)
        Returns:
            Reconstructed audio spectrogram of shape (batch, channels, time, frequency)
        """
        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(self, sample: torch.Tensor) -> tuple[torch.Tensor, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.denormalize_latents(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """
        Adjust output shape to match target dimensions for variable-length audio.
        This function handles the common case where decoded audio spectrograms need to be
        resized to match a specific target shape.
        Args:
            decoded_output: Tensor of shape (batch, channels, time, frequency)
            target_shape: AudioLatentShape describing (batch, channels, time, mel bins)
        Returns:
            Tensor adjusted to match target_shape exactly
        """
        # Current output shape: (batch, channels, time, frequency)
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Step 1: Crop first to avoid exceeding target dimensions
        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        # Step 2: Calculate padding needed for time and frequency dimensions
        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        # Step 3: Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
            # For audio: pad_left/right = frequency, pad_top/bottom = time
            padding = (
                0,
                max(freq_padding_needed, 0),  # frequency padding (left, right)
                0,
                max(time_padding_needed, 0),  # time padding (top, bottom)
            )
            decoded_output = F.pad(decoded_output, padding)

        # Step 4: Final safety crop to ensure exact target shape
        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def _run_upsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                h = block(h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != 0 and hasattr(stage, "upsample"):
                h = stage.upsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.non_linearity(h)
        h = self.conv_out(h)
        return torch.tanh(h) if self.tanh_out else h



class AutoencoderKLLTX2Audio(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    LTX2 audio VAE for encoding and decoding audio latent representations.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 2,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Tuple[int, ...]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        norm_type: str = "pixel",
        causality_axis: Optional[str] = "height",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = 64,
        double_z: bool = True,
    ) -> None:
        super().__init__()

        supported_causality_axes = {"none", "width", "height", "width-compatibility"}
        if causality_axis not in supported_causality_axes:
            raise ValueError(f"{causality_axis=} is not valid. Supported values: {supported_causality_axes}")

        nested_cfg = convert_diffusers_ltx2_audio_vae_config_to_preprocessing_model_config(
            {
                "base_channels": base_channels,
                "output_channels": output_channels,
                "ch_mult": ch_mult,
                "num_res_blocks": num_res_blocks,
                "attn_resolutions": attn_resolutions,
                "in_channels": in_channels,
                "resolution": resolution,
                "latent_channels": latent_channels,
                "norm_type": norm_type,
                "causality_axis": causality_axis,
                "dropout": dropout,
                "mid_block_add_attention": mid_block_add_attention,
                "sample_rate": sample_rate,
                "mel_hop_length": mel_hop_length,
                "is_causal": is_causal,
                "mel_bins": mel_bins,
                "double_z": double_z,
            }
        )

        ddconfig = nested_cfg["model"]["params"]["ddconfig"]
        stft_cfg = nested_cfg["preprocessing"]["stft"]
        model_params = nested_cfg["model"]["params"]

        axis_enum = _as_causality_axis(ddconfig.get("causality_axis"))

        norm_enum = NormType(ddconfig.get("norm_type", "pixel"))
        attn_res_set: Set[int] = set(ddconfig.get("attn_resolutions") or [])

        # Construct encoder/decoder from the ddconfig (checkpoint-style schema).
        self.encoder = AudioEncoder(
            ch=int(ddconfig.get("ch", 128)),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=int(ddconfig.get("num_res_blocks", 2)),
            attn_resolutions=attn_res_set,
            dropout=float(ddconfig.get("dropout", 0.0)),
            resamp_with_conv=True,
            in_channels=int(ddconfig.get("in_channels", 2)),
            resolution=int(ddconfig.get("resolution", 256)),
            z_channels=int(ddconfig.get("z_channels", latent_channels)),
            double_z=bool(ddconfig.get("double_z", True)),
            mid_block_add_attention=bool(ddconfig.get("mid_block_add_attention", False)),
            norm_type=norm_enum,
            causality_axis=axis_enum,
            sample_rate=int(model_params.get("sampling_rate", sample_rate)),
            mel_hop_length=int(stft_cfg.get("hop_length", mel_hop_length)),
            n_fft=int(stft_cfg.get("filter_length", 1024)),
            is_causal=bool(stft_cfg.get("causal", is_causal)),
            mel_bins=int(ddconfig.get("mel_bins", mel_bins if mel_bins is not None else 64)),
        )

        self.decoder = AudioDecoder(
            ch=int(ddconfig.get("ch", 128)),
            out_ch=int(ddconfig.get("out_ch", output_channels)),
            ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=int(ddconfig.get("num_res_blocks", 2)),
            attn_resolutions=attn_res_set,
            resolution=int(ddconfig.get("resolution", 256)),
            z_channels=int(ddconfig.get("z_channels", latent_channels)),
            norm_type=norm_enum,
            causality_axis=axis_enum,
            dropout=float(ddconfig.get("dropout", 0.0)),
            mid_block_add_attention=bool(ddconfig.get("mid_block_add_attention", False)),
            sample_rate=int(model_params.get("sampling_rate", sample_rate)),
            mel_hop_length=int(stft_cfg.get("hop_length", mel_hop_length)),
            is_causal=bool(stft_cfg.get("causal", is_causal)),
            mel_bins=int(ddconfig.get("mel_bins", mel_bins if mel_bins is not None else 64)),
        )
        
        
        latents_std = torch.zeros((base_channels,))
        latents_mean = torch.ones((base_channels,))
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)
        
        self.decoder.denormalize_latents = self.denormalize_latents
        self.decoder.normalize_latents = self.normalize_latents
        self.encoder.denormalize_latents = self.denormalize_latents
        self.encoder.normalize_latents = self.normalize_latents
        

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Keep parity with `AutoencoderKLLTX2Video`: forward() decodes latents.
        return self.decoder(latent)

    def encode(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.encoder(spectrogram)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def denormalize_latents(self, latents: torch.Tensor):
        latents_mean = self.latents_mean.to(latents.device, latents.dtype)
        latents_std = self.latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    def normalize_latents(self, latents: torch.Tensor):
        latents_mean = self.latents_mean.to(latents.device, latents.dtype)
        latents_std = self.latents_std.to(latents.device, latents.dtype)

        return (latents - latents_mean) / latents_std

def decode_audio(latent: torch.Tensor, audio_decoder: "AudioDecoder", vocoder: "Vocoder") -> torch.Tensor:
    """
    Decode an audio latent representation using the provided audio decoder and vocoder.
    Args:
        latent: Input audio latent tensor.
        audio_decoder: Model to decode the latent to waveform features.
        vocoder: Model to convert decoded features to audio waveform.
    Returns:
        Decoded audio as a float tensor.
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio).squeeze(0).float()
    return decoded_audio
