from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING
import torch
from typing import Protocol

if TYPE_CHECKING:
    # Imported only for type-checking to avoid circular imports at runtime.
    from src.engine.ltx22.shared.patchifiers import AudioPatchifier, VideoLatentPatchifier

VIDEO_LATENT_CHANNELS = 128

class VideoPixelShape(NamedTuple):
    """
    Shape of the tensor representing the video pixel array. Assumes BGR channel format.
    """

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.
    """

    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """
    Shape of the tensor representing video in VAE latent space.
    The latent representation is a 5D tensor with dimensions ordered as
    (batch, channels, frames, height, width). Spatial and temporal dimensions
    are downscaled relative to pixel space according to the VAE's scale factors.
    """

    batch: int
    channels: int
    frames: int
    height: int
    width: int
    
    

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([int(self.batch), int(self.channels), int(self.frames), int(self.height), int(self.width)])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)

    @staticmethod
    def from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: int = 128,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors[0] + 1
        height = shape.height // scale_factors[1]
        width = shape.width // scale_factors[2]

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    """
    Shape of audio in VAE latent space: (batch, channels, frames, mel_bins).
    mel_bins is the number of frequency bins from the mel-spectrogram encoding.
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)

        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )




@dataclass(frozen=True)
class LatentState:
    """
    State of latents during the diffusion denoising process.
    Attributes:
        latent: The current noisy latent tensor being denoised.
        denoise_mask: Mask encoding the denoising strength for each token (1 = full denoising, 0 = no denoising).
        positions: Positional indices for each latent element, used for positional embeddings.
        clean_latent: Initial state of the latent before denoising, may include conditioning latents.
    """

    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
        )


class PipelineComponents:
    """
    Container class for pipeline components used throughout the LTX pipelines.
    Attributes:
        dtype (torch.dtype): Default torch dtype for tensors in the pipeline.
        device (torch.device): Target device to place tensors and modules on.
        video_scale_factors (SpatioTemporalScaleFactors): Scale factors (T, H, W) for VAE latent space.
        video_latent_channels (int): Number of channels in the video latent representation.
        video_patchifier (VideoLatentPatchifier): Patchifier instance for video latents.
        audio_patchifier (AudioPatchifier): Patchifier instance for audio latents.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # Local import to avoid circular import:
        # `patchifiers` depends on `types`, so `types` must not import `patchifiers` at module import time.
        from src.engine.ltx22.shared.patchifiers import AudioPatchifier, VideoLatentPatchifier

        self.dtype = dtype
        self.device = device

        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.video_latent_channels = VIDEO_LATENT_CHANNELS

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)


class DenoisingFunc(Protocol):
    """
    Protocol for a denoising function used in the LTX pipeline.
    Args:
        video_state (LatentState): The current latent state for video.
        audio_state (LatentState): The current latent state for audio.
        sigmas (torch.Tensor): A 1D tensor of sigma values for each diffusion step.
        step_index (int): Index of the current denoising step.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The denoised video and audio tensors.
    """

    def __call__(
        self, video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class DenoisingLoopFunc(Protocol):
    """
    Protocol for a denoising loop function used in the LTX pipeline.
    Args:
        sigmas (torch.Tensor): A 1D tensor of sigma values for each diffusion step.
        video_state (LatentState): The current latent state for video.
        audio_state (LatentState): The current latent state for audio.
        stepper (DiffusionStepProtocol): The diffusion step protocol to use.
    Returns:
        tuple[LatentState, LatentState]: The denoised video and audio latent states.
    """

    def __call__(
        self,
        sigmas: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        stepper,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
