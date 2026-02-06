from __future__ import annotations

from typing import Optional, Union

import math
import torch
from diffusers.configuration_utils import register_to_config

from src.scheduler.scheduler import SchedulerInterface


def _time_snr_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """
    Same monotonic remap used by SD3/flow schedules:
        f(t) = (shift * t) / (1 + (shift - 1) * t)
    """
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def _comfyui_ddim_uniform_from_table(
    train_sigmas: torch.Tensor, num_inference_steps: int
) -> torch.Tensor:
    """
    ComfyUI-inspired "DDIM-uniform" sigma schedule from a train sigma table.

    The original ComfyUI implementation uses integer division to stride through a
    sigma table, which can yield a schedule length that differs slightly from the
    requested `steps`. For Apex we *guarantee* exactly `num_inference_steps`
    denoise iterations by selecting exactly that many indices from the table.

    - `train_sigmas` is assumed to be an *ascending* table (low -> high) and
      typically does not contain an explicit 0.
    - the resulting sigmas are returned as a *descending* schedule ending in 0.
    """
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")

    n = int(train_sigmas.numel())
    if n < 2:
        raise ValueError("train_sigmas must have at least 2 entries")

    # Select exactly `num_inference_steps` indices from [1, n-1] (inclusive).
    # This matches the spirit of ComfyUI's "uniform subsample from a table"
    # while keeping step count stable for pipeline logic.
    if num_inference_steps == 1:
        idx = torch.tensor([n - 1], dtype=torch.long)
    else:
        idx_f = torch.linspace(1, n - 1, num_inference_steps, dtype=torch.float64)
        idx = idx_f.round().to(dtype=torch.long)
        idx = torch.clamp(idx, 1, n - 1)

        # Ensure strictly increasing indices (avoid duplicates from rounding).
        for i in range(1, idx.numel()):
            if idx[i] <= idx[i - 1]:
                idx[i] = min(idx[i - 1] + 1, n - 1)
        # If we hit the clamp ceiling, fix backwards to keep monotonicity.
        for i in range(idx.numel() - 2, -1, -1):
            if idx[i] >= idx[i + 1]:
                idx[i] = max(idx[i + 1] - 1, 1)

    # Build schedule: prepend 0, then reverse so it starts near sigma_max.
    sigs = [0.0] + [float(train_sigmas[i]) for i in idx.tolist()]
    sigmas = torch.tensor(list(reversed(sigs)), dtype=torch.float32)
    # Ensure an explicit terminal 0 exists (ComfyUI always returns one).
    if sigmas.numel() == 0 or not math.isclose(float(sigmas[-1]), 0.0, abs_tol=1e-7):
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)], dim=0)
    return sigmas


class DDIMEulerFlowScheduler(SchedulerInterface):
    """
    ComfyUI-style "DDIM-uniform" sigma schedule + Euler stepping, for flow / rectified-flow
    style predictors (i.e. models whose update rule is:

        x_{next} = x + pred * (sigma_next - sigma)

    This mirrors ComfyUI's combination of:
    - `ddim_uniform` scheduler (subsample indices from a train sigma table)
    - Euler sampler/integrator operating over those sigmas

    Notes:
    - This scheduler uses Apex's "sigma as mixing coefficient" convention where
      `add_noise` is: x_t = (1 - sigma) * x0 + sigma * noise.
    - `shift` applies the same SD3/flow time remap used in `FlowMatchScheduler`.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 50,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)

        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        training: bool = False,  # parity with other schedulers
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        **kwargs,
    ):
        # Determine effective params for this call.
        shift_eff = float(self.shift if shift is None else shift)

        # Respect denoising_strength by shrinking the maximum sigma reached.
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(
            denoising_strength
        )

        total_timesteps = int(self.num_train_timesteps)

        # Build an ascending "train sigma table" analogous to ComfyUI's `model_sampling.sigmas`.
        # We keep it on CPU and in float64 to keep the selection stable.
        train_sigmas = torch.linspace(
            float(self.sigma_min),
            float(sigma_start),
            total_timesteps + 1,
            dtype=torch.float64,
            device="cpu",
        )

        # Apply the SD3/flow time-shift remap used throughout Apex/ComfyUI.
        if shift_eff != 1.0:
            train_sigmas = _time_snr_shift(shift_eff, train_sigmas)

        sigmas = _comfyui_ddim_uniform_from_table(train_sigmas, int(num_inference_steps))

        # Optional direction toggles (kept similar to FlowMatchScheduler / Beta57Scheduler).
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        if self.reverse_sigmas:
            sigmas = 1.0 - sigmas

        self.sigmas = sigmas
        # Match the common Diffusers/Apex pattern:
        # - `sigmas` includes a terminal value (typically 0)
        # - `timesteps` excludes the terminal sigma to avoid an extra no-op model call
        self.timesteps = sigmas[:-1] * float(self.num_train_timesteps)

        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, to_final: bool = False, **kwargs):
        """
        Euler step in sigma space:
            x_{next} = x + pred * (sigma_next - sigma)
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        elif timestep.ndim == 0:
            timestep = timestep.expand(model_output.shape[0])

        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)

        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_next = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
            sigma_next = torch.full_like(sigma, float(sigma_next))
        else:
            sigma_next = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)

        prev_sample = sample + model_output * (sigma_next - sigma)
        return (prev_sample,)

