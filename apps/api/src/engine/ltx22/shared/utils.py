from pathlib import Path
from typing import Any, Protocol

import torch


from src.engine.ltx22.shared.protocols import DiffusionStepProtocol
from ltx_core.types import LatentState
from src.engine.ltx22.shared.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS
from src.engine.ltx22.shared.patchifiers import VideoLatentPatchifier, AudioPatchifier


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


def find_matching_file(root_path: str, pattern: str) -> Path:
    """
    Recursively search for files matching a glob pattern and return the first match.
    """
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return matches[0]






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
        stepper: DiffusionStepProtocol,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
