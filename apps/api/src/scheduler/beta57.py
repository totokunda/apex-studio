from __future__ import annotations

from typing import Optional, Union

import torch
from diffusers.configuration_utils import register_to_config

from src.scheduler.scheduler import SchedulerInterface


def _beta_ppf(q, alpha: float, beta: float):
    """
    Percent point function (inverse CDF) for Beta(alpha, beta).

    Prefer SciPy when available (fast + accurate). Fall back to mpmath-based
    bisection if SciPy isn't installed.
    """
    try:
        import numpy as np
        import scipy.stats

        q_np = np.asarray(q, dtype=np.float64)
        out = scipy.stats.beta.ppf(q_np, alpha, beta)
        return out
    except Exception:
        # Fallback: use mpmath betainc inversion by bisection.
        import numpy as np
        import mpmath as mp

        q_np = np.asarray(q, dtype=np.float64)

        def inv_one(p: float) -> float:
            if p <= 0.0:
                return 0.0
            if p >= 1.0:
                return 1.0
            lo = mp.mpf("0.0")
            hi = mp.mpf("1.0")
            target = mp.mpf(p)
            a = mp.mpf(alpha)
            b = mp.mpf(beta)
            # ~60 iters gives plenty for float64-ish accuracy
            for _ in range(80):
                mid = (lo + hi) / 2
                val = mp.betainc(a, b, 0, mid, regularized=True)
                if val < target:
                    lo = mid
                else:
                    hi = mid
            return float((lo + hi) / 2)

        out = np.array([inv_one(float(p)) for p in q_np.reshape(-1)], dtype=np.float64)
        return out.reshape(q_np.shape)


class Beta57Scheduler(SchedulerInterface):
    """
    Apex-native equivalent of ComfyUI/RES4LYF's `beta57` scheduler.

    In ComfyUI, `beta57` uses a Beta(0.5, 0.7) inverse-CDF to pick indices from a
    precomputed sigma table. Here we mirror that behavior by:

    - building an internal "train sigma table" of length `num_train_timesteps + 1`
      spanning [sigma_min, sigma_start]
    - using Beta PPF to select indices into that table for `num_inference_steps`

    This scheduler operates in Apex's "sigma as mixing coefficient" convention
    where `add_noise` is: x_t = (1 - sigma) * x0 + sigma * noise.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 50,
        num_train_timesteps: int = 1000,
        alpha: float = 0.5,
        beta: float = 0.7,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.alpha = float(alpha)
        self.beta = float(beta)
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
        training: bool = False,  # kept for parity with other schedulers
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        mu: Optional[float] = None,
    ):
        # Determine effective params for this call.
        alpha_eff = float(self.alpha if alpha is None else alpha)
        beta_eff = float(self.beta if beta is None else beta)
        shift_eff = float(self.shift if shift is None else shift)

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(denoising_strength)

        # Create a "train sigma table" analogous to ComfyUI's `model_sampling.sigmas`.
        # We use an ascending table so that index `total_timesteps` corresponds to sigma_start (max noise).
        total_timesteps = int(self.num_train_timesteps)
        train_sigmas = torch.linspace(
            float(self.sigma_min),
            float(sigma_start),
            total_timesteps + 1,
            dtype=torch.float64,
            device="cpu",
        )

        # Match ComfyUI's selection:
        # ts = 1 - linspace(0, 1, steps, endpoint=False)
        # idx = round(beta.ppf(ts) * total_timesteps)
        import numpy as np

        ts = 1.0 - np.linspace(0.0, 1.0, int(num_inference_steps), endpoint=False, dtype=np.float64)
        ppf = _beta_ppf(ts, alpha_eff, beta_eff)
        idx = np.rint(ppf * float(total_timesteps)).astype(np.int64)
        idx = np.clip(idx, 0, total_timesteps)

        sigmas = train_sigmas[idx].to(dtype=torch.float32)

        # Optional direction / remapping toggles (kept similar to FlowMatchScheduler).
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])

        # Apply FlowMatch-style time shift (monotonic remap of sigma).
        # This is widely used in Apex manifests, so we support it here too.
        if shift_eff != 1.0:
            sigmas = shift_eff * sigmas / (1.0 + (shift_eff - 1.0) * sigmas)

        if self.reverse_sigmas:
            sigmas = 1.0 - sigmas

        self.sigmas = sigmas
        self.timesteps = sigmas * float(self.num_train_timesteps)

        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, to_final: bool = False, **kwargs):
        # Follows the same stepping convention as FlowMatchScheduler: x_{t-1} = x_t + pred * (sigma_{next} - sigma)
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

