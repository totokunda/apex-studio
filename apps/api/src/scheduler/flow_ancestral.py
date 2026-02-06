from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers.configuration_utils import register_to_config

from src.scheduler.scheduler import SchedulerInterface


def _time_snr_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    # Same remap used across Apex's flow schedulers.
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def _randn_like(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    # torch.randn_like supports `generator` in recent PyTorch; keep centralized.
    if generator is None:
        return torch.randn_like(x)
    return torch.randn_like(x, generator=generator)


@dataclass
class _StageState:
    # Common stored state for 2-stage samplers.
    x: torch.Tensor
    denoised: torch.Tensor
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    sigma_down: torch.Tensor
    alpha_ip1: torch.Tensor
    alpha_down: torch.Tensor
    renoise_coeff: torch.Tensor


class EulerAncestralFlowScheduler(SchedulerInterface):
    """
    Flow-compatible Euler ancestral sampler (ComfyUI `sample_euler_ancestral_RF` logic),
    implemented as a SchedulerInterface so it can be used by existing engine loops.

    Update uses the "sigma mixing coefficient" convention:
      x_t = (1 - sigma) * x0 + sigma * noise
    and assumes `model_output` is the flow/velocity derivative in sigma-space.
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
        eta: float = 1.0,
        s_noise: float = 1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)

        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        training: bool = False,  # parity
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        eta: Optional[float] = None,
        s_noise: Optional[float] = None,
        **kwargs,
    ):
        if eta is not None:
            self.eta = float(eta)
        if s_noise is not None:
            self.s_noise = float(s_noise)

        shift_eff = float(self.shift if shift is None else shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(
            denoising_strength
        )
        # Clamp the linspace endpoint to a small positive value so the base
        # schedule never contains sigma=0.  When the engine sets sigma_min=0,
        # a zero in the base would cause 0/0 NaN in ratio computations.
        # The step function's final-step logic handles the actual jump to 0.
        sigma_end = max(float(self.sigma_min), 1e-6)
        sigmas = torch.linspace(float(sigma_start), sigma_end, int(num_inference_steps))
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        if shift_eff != 1.0:
            sigmas = _time_snr_shift(shift_eff, sigmas)
        if self.reverse_sigmas:
            sigmas = 1.0 - sigmas

        self.sigmas = sigmas.to(dtype=torch.float32)
        self.timesteps = self.sigmas * float(self.num_train_timesteps)
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        eta: Optional[float] = None,
        s_noise: Optional[float] = None,
        to_final: bool = False,
        **kwargs,
    ):
        # Allow per-step overrides (common diffusers pattern).
        eta_eff = float(self.eta if eta is None else eta)
        s_noise_eff = float(self.s_noise if s_noise is None else s_noise)

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
            sigma_ip1 = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
            sigma_ip1 = torch.full_like(sigma, float(sigma_ip1))
        else:
            sigma_ip1 = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)

        denoised = sample - sigma * model_output

        # Final denoise (per-sample).
        final_mask = (sigma_ip1 == 0) | torch.full_like(
            sigma_ip1, bool(to_final), dtype=torch.bool
        )

        # Rectified-flow ancestral step (ComfyUI *_RF).
        downstep_ratio = 1.0 + (sigma_ip1 / sigma - 1.0) * eta_eff
        sigma_down = sigma_ip1 * downstep_ratio

        alpha_ip1 = 1.0 - sigma_ip1
        alpha_down = 1.0 - sigma_down

        # renoise_coeff = sqrt(sigma_ip1^2 - sigma_down^2 * alpha_ip1^2 / alpha_down^2)
        term = sigma_ip1.pow(2) - sigma_down.pow(2) * alpha_ip1.pow(2) / alpha_down.pow(2)
        renoise_coeff = torch.sqrt(torch.clamp(term, min=0.0))

        sigma_down_i_ratio = sigma_down / sigma
        x = sigma_down_i_ratio * sample + (1.0 - sigma_down_i_ratio) * denoised

        if eta_eff > 0.0 and s_noise_eff > 0.0:
            x = (alpha_ip1 / alpha_down) * x + _randn_like(x, generator=generator) * (
                s_noise_eff * renoise_coeff
            )

        prev_sample = torch.where(final_mask, denoised, x)
        return (prev_sample,) if not return_dict else (prev_sample,)


class DPM2AncestralFlowScheduler(SchedulerInterface):
    """
    Flow-compatible DPM-Solver-2 ancestral sampler (ComfyUI `sample_dpm_2_ancestral_RF`),
    implemented with two internal stages (order=2).
    """

    order = 2

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
        eta: float = 1.0,
        s_noise: float = 1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)

        self._stage: Optional[_StageState] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

        self.set_timesteps(num_inference_steps)

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = int(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        training: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        eta: Optional[float] = None,
        s_noise: Optional[float] = None,
        **kwargs,
    ):
        if eta is not None:
            self.eta = float(eta)
        if s_noise is not None:
            self.s_noise = float(s_noise)

        shift_eff = float(self.shift if shift is None else shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(
            denoising_strength
        )
        # Clamp the linspace endpoint so the base schedule never contains
        # sigma=0.  The terminal value handles the final denoising step.
        sigma_end = max(float(self.sigma_min), 1e-6)
        base = torch.linspace(float(sigma_start), sigma_end, int(num_inference_steps))
        if self.inverse_timesteps:
            base = torch.flip(base, dims=[0])
        if shift_eff != 1.0:
            base = _time_snr_shift(shift_eff, base)
        if self.reverse_sigmas:
            base = 1.0 - base

        sigma_terminal = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        base_ext = torch.cat(
            [base.to(dtype=torch.float32), torch.tensor([sigma_terminal], dtype=torch.float32)],
            dim=0,
        )

        # Build interleaved sigmas: [sigma_i, sigma_mid_i, ..., sigma_terminal]
        sigmas: list[torch.Tensor] = []
        eta_eff = float(self.eta)
        eps = 1e-6
        for i in range(int(num_inference_steps)):
            sigma_i = base_ext[i]
            sigma_ip1 = base_ext[i + 1]
            # ComfyUI RF uses sigma_down based on eta.
            downstep_ratio = 1.0 + (sigma_ip1 / sigma_i - 1.0) * eta_eff
            sigma_down = sigma_ip1 * downstep_ratio
            # sigma_mid = geometric mean in log space between sigma_i and sigma_down.
            #
            # Important: when sigma_ip1 is terminal (0), ComfyUI takes an Euler
            # final step and does NOT evaluate the model at an extra tiny sigma.
            # To avoid instability (NaNs/black frames) in models that are not
            # well-behaved at extremely small sigmas, we set sigma_mid=sigma_i
            # for the final outer step. This makes stage-1 a no-op (dt1=0) and
            # stage-2 completes the final step deterministically.
            if float(sigma_ip1) == 0.0:
                sigma_mid = sigma_i
            else:
                sigma_mid = torch.exp(
                    torch.lerp(
                        torch.log(torch.clamp(sigma_i, min=eps)),
                        torch.log(torch.clamp(sigma_down, min=eps)),
                        0.5,
                    )
                )
            sigmas.append(sigma_i)
            sigmas.append(sigma_mid)
        sigmas.append(base_ext[-1])

        self.sigmas = torch.stack(sigmas).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
        self._stage = None
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        eta: Optional[float] = None,
        s_noise: Optional[float] = None,
        **kwargs,
    ):
        eta_eff = float(self.eta if eta is None else eta)
        s_noise_eff = float(self.s_noise if s_noise is None else s_noise)

        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        # Use an internal counter (Diffusers-style) instead of searching by timestep.
        # This makes the 2-stage schedule robust even when some timesteps repeat
        # (which can happen for certain eta values).
        if self._step_index is None:
            self._step_index = self._begin_index or 0

        idx = int(self._step_index)
        if idx >= (len(self.sigmas) - 1):
            # Already finished; no-op.
            return (sample,) if not return_dict else (sample,)

        stage1 = (idx % 2 == 0)

        if stage1:
            # Stage 1: x_mid = x + pred(sigma_i) * (sigma_mid - sigma_i)
            sigma_i = self.sigmas[idx].view(1, 1, 1, 1)
            sigma_mid = self.sigmas[idx + 1].view(1, 1, 1, 1)
            sigma_ip1 = self.sigmas[idx + 2].view(1, 1, 1, 1) if (idx + 2) < len(self.sigmas) else self.sigmas[-1].view(1, 1, 1, 1)

            sigma_i = sigma_i.to(device=model_output.device, dtype=model_output.dtype)
            sigma_mid = sigma_mid.to(device=model_output.device, dtype=model_output.dtype)
            sigma_ip1 = sigma_ip1.to(device=model_output.device, dtype=model_output.dtype)

            downstep_ratio = 1.0 + (sigma_ip1 / sigma_i - 1.0) * eta_eff
            sigma_down = sigma_ip1 * downstep_ratio
            alpha_ip1 = 1.0 - sigma_ip1
            alpha_down = 1.0 - sigma_down
            term = sigma_ip1.pow(2) - sigma_down.pow(2) * alpha_ip1.pow(2) / alpha_down.pow(2)
            renoise_coeff = torch.sqrt(torch.clamp(term, min=0.0))

            x_mid = sample + model_output * (sigma_mid - sigma_i)
            denoised = sample - sigma_i * model_output

            self._stage = _StageState(
                x=sample,
                denoised=denoised,
                sigma=sigma_i,
                sigma_next=sigma_ip1,
                sigma_down=sigma_down,
                alpha_ip1=alpha_ip1,
                alpha_down=alpha_down,
                renoise_coeff=renoise_coeff,
            )

            self._step_index += 1
            prev_sample = x_mid
            return (prev_sample,) if not return_dict else (prev_sample,)

        # Stage 2: x = x + pred(sigma_mid) * (sigma_down - sigma_i), then renoise to sigma_{i+1}
        if self._stage is None:
            # Shouldn't happen; fall back to Euler using adjacent sigmas.
            sigma_cur = self.sigmas[idx].view(1, 1, 1, 1).to(model_output.device, model_output.dtype)
            sigma_next = self.sigmas[idx + 1].view(1, 1, 1, 1).to(model_output.device, model_output.dtype)
            self._step_index += 1
            prev_sample = sample + model_output * (sigma_next - sigma_cur)
            return (prev_sample,) if not return_dict else (prev_sample,)

        st = self._stage

        # When sigma_down == 0 (final denoising step), ComfyUI takes a plain
        # Euler step to the denoised output.  Avoid the ratio formulas which
        # would produce 0/0 or unnecessary noise injection.
        if float(st.sigma_down) == 0.0:
            # d_2 = model_output (flow prediction at sigma_mid)
            # x = x_orig + d_2 * (sigma_down - sigma_i) = x_orig - sigma_i * d_2
            x_det = st.x + model_output * (st.sigma_down - st.sigma)
        else:
            x_det = st.x + model_output * (st.sigma_down - st.sigma)
            if eta_eff > 0.0 and s_noise_eff > 0.0:
                x_det = (st.alpha_ip1 / st.alpha_down) * x_det + _randn_like(
                    x_det, generator=generator
                ) * (s_noise_eff * st.renoise_coeff)

        self._stage = None
        self._step_index += 1
        prev_sample = x_det
        return (prev_sample,) if not return_dict else (prev_sample,)


class DPMpp2SAncestralFlowScheduler(SchedulerInterface):
    """
    Flow-compatible DPM-Solver++(2S) ancestral sampler (ComfyUI `sample_dpmpp_2s_ancestral_RF`),
    implemented with two internal stages (order=2).
    """

    order = 2

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
        eta: float = 1.0,
        s_noise: float = 1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)

        self._stage: Optional[_StageState] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

        self.set_timesteps(num_inference_steps)

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = int(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        training: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        eta: Optional[float] = None,
        s_noise: Optional[float] = None,
        **kwargs,
    ):
        if eta is not None:
            self.eta = float(eta)
        if s_noise is not None:
            self.s_noise = float(s_noise)

        shift_eff = float(self.shift if shift is None else shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(
            denoising_strength
        )
        # Clamp the linspace endpoint so the base schedule never contains
        # sigma=0.  The terminal value handles the final denoising step.
        sigma_end = max(float(self.sigma_min), 1e-6)
        base = torch.linspace(float(sigma_start), sigma_end, int(num_inference_steps))
        if self.inverse_timesteps:
            base = torch.flip(base, dims=[0])
        if shift_eff != 1.0:
            base = _time_snr_shift(shift_eff, base)
        if self.reverse_sigmas:
            base = 1.0 - base

        sigma_terminal = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        base_ext = torch.cat(
            [base.to(dtype=torch.float32), torch.tensor([sigma_terminal], dtype=torch.float32)],
            dim=0,
        )

        # Interleaved sigmas: [sigma_i, sigma_s_i, ..., sigma_terminal]
        sigmas: list[torch.Tensor] = []
        eta_eff = float(self.eta)
        eps = 1e-6

        def lambda_fn(s: torch.Tensor) -> torch.Tensor:
            # ((1-s)/s).log()
            return torch.log(torch.clamp((1.0 - s) / torch.clamp(s, min=eps), min=eps))

        def sigma_fn(lbda: torch.Tensor) -> torch.Tensor:
            # (exp(lambda)+1)^-1
            return 1.0 / (torch.exp(lbda) + 1.0)

        for i in range(int(num_inference_steps)):
            sigma_i = base_ext[i]
            sigma_ip1 = base_ext[i + 1]

            downstep_ratio = 1.0 + (sigma_ip1 / sigma_i - 1.0) * eta_eff
            sigma_down = sigma_ip1 * downstep_ratio

            # For the final outer step (sigma_ip1==0), avoid introducing an
            # extra evaluation at a potentially unstable tiny sigma_s; set
            # sigma_s = sigma_i so stage-1 becomes a no-op and stage-2
            # deterministically computes the final denoised sample.
            if float(sigma_ip1) == 0.0:
                sigma_s = sigma_i
            elif float(sigma_i) == 1.0:
                sigma_s = torch.tensor(0.9999, dtype=sigma_i.dtype)
            else:
                t_i = lambda_fn(sigma_i)
                t_down = lambda_fn(torch.clamp(sigma_down, min=eps))
                h = t_down - t_i
                s = t_i + 0.5 * h
                sigma_s = sigma_fn(s)

            sigmas.append(sigma_i)
            sigmas.append(sigma_s)
        sigmas.append(base_ext[-1])

        self.sigmas = torch.stack(sigmas).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
        self._stage = None
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        eta: Optional[float] = None,
        s_noise: Optional[float] = None,
        **kwargs,
    ):
        eta_eff = float(self.eta if eta is None else eta)
        s_noise_eff = float(self.s_noise if s_noise is None else s_noise)

        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = self._begin_index or 0

        idx = int(self._step_index)
        if idx >= (len(self.sigmas) - 1):
            return (sample,) if not return_dict else (sample,)

        stage1 = (idx % 2 == 0)

        if stage1:
            sigma_i = self.sigmas[idx].view(1, 1, 1, 1)
            sigma_s = self.sigmas[idx + 1].view(1, 1, 1, 1)
            sigma_ip1 = self.sigmas[idx + 2].view(1, 1, 1, 1) if (idx + 2) < len(self.sigmas) else self.sigmas[-1].view(1, 1, 1, 1)

            sigma_i = sigma_i.to(device=model_output.device, dtype=model_output.dtype)
            sigma_s = sigma_s.to(device=model_output.device, dtype=model_output.dtype)
            sigma_ip1 = sigma_ip1.to(device=model_output.device, dtype=model_output.dtype)

            downstep_ratio = 1.0 + (sigma_ip1 / sigma_i - 1.0) * eta_eff
            sigma_down = sigma_ip1 * downstep_ratio
            alpha_ip1 = 1.0 - sigma_ip1
            alpha_down = 1.0 - sigma_down
            term = sigma_ip1.pow(2) - sigma_down.pow(2) * alpha_ip1.pow(2) / alpha_down.pow(2)
            renoise_coeff = torch.sqrt(torch.clamp(term, min=0.0))

            denoised = sample - sigma_i * model_output
            sigma_s_i_ratio = sigma_s / sigma_i
            u = sigma_s_i_ratio * sample + (1.0 - sigma_s_i_ratio) * denoised

            self._stage = _StageState(
                x=sample,
                denoised=denoised,
                sigma=sigma_i,
                sigma_next=sigma_ip1,
                sigma_down=sigma_down,
                alpha_ip1=alpha_ip1,
                alpha_down=alpha_down,
                renoise_coeff=renoise_coeff,
            )

            self._step_index += 1
            prev_sample = u
            return (prev_sample,) if not return_dict else (prev_sample,)

        if self._stage is None:
            sigma_cur = self.sigmas[idx].view(1, 1, 1, 1).to(model_output.device, model_output.dtype)
            sigma_next = self.sigmas[idx + 1].view(1, 1, 1, 1).to(model_output.device, model_output.dtype)
            self._step_index += 1
            prev_sample = sample + model_output * (sigma_next - sigma_cur)
            return (prev_sample,) if not return_dict else (prev_sample,)

        st = self._stage
        sigma_s = self.sigmas[idx].view(1, 1, 1, 1).to(device=model_output.device, dtype=model_output.dtype)
        D_i = sample - sigma_s * model_output

        # When sigma_down == 0 (final denoising step), ComfyUI uses
        # a plain Euler step to the denoised output.  Skip the ratio
        # formulas which would produce 0/0 or unnecessary noise.
        if float(st.sigma_down) == 0.0:
            sigma_down_i_ratio = st.sigma_down / st.sigma
            x_det = sigma_down_i_ratio * st.x + (1.0 - sigma_down_i_ratio) * D_i
        else:
            sigma_down_i_ratio = st.sigma_down / st.sigma
            x_det = sigma_down_i_ratio * st.x + (1.0 - sigma_down_i_ratio) * D_i

            if eta_eff > 0.0 and s_noise_eff > 0.0:
                x_det = (st.alpha_ip1 / st.alpha_down) * x_det + _randn_like(
                    x_det, generator=generator
                ) * (s_noise_eff * st.renoise_coeff)

        self._stage = None
        self._step_index += 1
        prev_sample = x_det
        return (prev_sample,) if not return_dict else (prev_sample,)



class DDPMFlowScheduler(SchedulerInterface):
    """
    DDPM (Denoising Diffusion Probabilistic Model) sampler for flow-matching.

    Implements the classic DDPM reverse process using the Karras alpha_cumprod
    mapping (alpha_cumprod = 1 / (sigma^2 + 1)) applied to flow-matching sigmas,
    matching ComfyUI's ``sample_ddpm`` behavior.

    Single model evaluation per step with ancestral noise injection.

    Equivalent to ComfyUI's ``sample_ddpm``.
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
        self._step_index: Optional[int] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        training: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        **kwargs,
    ):
        shift_eff = float(self.shift if shift is None else shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(
            denoising_strength
        )
        sigma_end = max(float(self.sigma_min), 1e-6)
        sigmas = torch.linspace(float(sigma_start), sigma_end, int(num_inference_steps))
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        if shift_eff != 1.0:
            sigmas = _time_snr_shift(shift_eff, sigmas)
        if self.reverse_sigmas:
            sigmas = 1.0 - sigmas

        # Append terminal sigma
        sigma_terminal = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        self.sigmas = torch.cat(
            [sigmas.to(dtype=torch.float32),
             torch.tensor([sigma_terminal], dtype=torch.float32)],
            dim=0,
        )
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        # ── Flow-matching decomposition ─────────────────────────────
        # sample  = (1 - σ) * x0 + σ * ε
        # model_output = velocity  v = ε − x0
        denoised = sample - sigma * model_output          # x0 prediction

        # Terminal step → return clean prediction directly
        if float(sigma_next) <= 0:
            self._step_index += 1
            return (denoised,)

        # Recover the noise component: ε = v + x0
        eps = model_output + denoised

        # ── DDPM noise mixing in flow-matching space ────────────────
        # The Karras ᾱ = 1/(σ²+1) mapping yields a closed-form
        # posterior noise-retention coefficient:
        #     c  = σ_next / σ          (original noise retained)
        #     √(1 − c²)               (fresh noise injected)
        c = sigma_next / sigma
        noise = _randn_like(sample, generator=generator)
        noise_mix = c * eps + (1.0 - c ** 2).sqrt() * noise

        # Construct next sample: x_{t-1} = (1 − σ_next)*x0 + σ_next*noise_mix
        prev_sample = (1.0 - sigma_next) * denoised + sigma_next * noise_mix

        self._step_index += 1
        return (prev_sample,)
