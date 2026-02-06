"""
Deterministic flow-matching samplers adapted from ComfyUI's k-diffusion.

All schedulers in this module follow the flow-matching convention:

    x_t = (1 - sigma) * x0 + sigma * noise

where model_output is the flow prediction: v = noise - x0,
and the denoised x0 prediction is: denoised = sample - sigma * model_output.

The Karras ODE derivative d = (x - denoised) / sigma = model_output.

Implemented samplers:
    1. EulerFlowScheduler          – 1st-order Euler
    2. HeunFlowScheduler           – 2nd-order Heun (trapezoidal)
    3. LMSFlowScheduler            – Linear Multi-Step (Adams-Bashforth, order 1-4)
    4. DPM2FlowScheduler           – DPM-Solver-2 (midpoint method)
    5. DPMpp2MFlowScheduler        – DPM-Solver++(2M) (multi-step, denoised-space)
    6. ResMultistepFlowScheduler   – RES multistep (exponential integrator, 2nd-order)
"""
from __future__ import annotations

from typing import Optional, Union

import math
import numpy as np
import torch
from scipy import integrate
from diffusers.configuration_utils import register_to_config

from src.scheduler.scheduler import SchedulerInterface


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _time_snr_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """SD3 / flow time-shift remap: f(t) = (shift*t) / (1 + (shift-1)*t)."""
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def _build_base_sigmas(
    num_inference_steps: int,
    sigma_max: float,
    sigma_min: float,
    shift: float,
    inverse_timesteps: bool,
    reverse_sigmas: bool,
    denoising_strength: float = 1.0,
) -> torch.Tensor:
    """Build the base sigma schedule (N entries, no terminal)."""
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    sigma_end = max(sigma_min, 1e-6)
    sigmas = torch.linspace(float(sigma_start), float(sigma_end), int(num_inference_steps))
    if inverse_timesteps:
        sigmas = torch.flip(sigmas, dims=[0])
    if shift != 1.0:
        sigmas = _time_snr_shift(shift, sigmas)
    if reverse_sigmas:
        sigmas = 1.0 - sigmas
    return sigmas.to(dtype=torch.float32)


def _build_base_sigmas_logsnr(
    num_inference_steps: int,
    sigma_max: float,
    sigma_min: float,
    shift: float,
    inverse_timesteps: bool,
    reverse_sigmas: bool,
    denoising_strength: float = 1.0,
) -> torch.Tensor:
    """Build a sigma schedule with uniform spacing in flow-matching logSNR.

    This avoids the massive first-step logSNR jump that occurs with sigma-linear
    schedules, making SDE noise injection well-behaved at every step.
    """
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    sigma_end = max(sigma_min, 1e-6)
    # Apply shift to the endpoints
    if shift != 1.0:
        sigma_start = float(_time_snr_shift(shift, torch.tensor(sigma_start)))
        sigma_end = float(_time_snr_shift(shift, torch.tensor(sigma_end)))
    # Clamp to avoid logSNR singularity at sigma=1 and sigma=0
    sigma_start = min(sigma_start, 0.9999)
    sigma_end = max(sigma_end, 1e-6)
    # Build schedule uniform in logSNR = log((1-sigma)/sigma)
    eps = 1e-8
    lam_start = math.log(max((1.0 - sigma_start) / max(sigma_start, eps), eps))
    lam_end = math.log(max((1.0 - sigma_end) / max(sigma_end, eps), eps))
    lambdas = torch.linspace(float(lam_start), float(lam_end), int(num_inference_steps))
    sigmas = 1.0 / (torch.exp(lambdas) + 1.0)  # sigmoid(-lambda)
    if inverse_timesteps:
        sigmas = torch.flip(sigmas, dims=[0])
    if reverse_sigmas:
        sigmas = 1.0 - sigmas
    return sigmas.to(dtype=torch.float32)


def _terminal_sigma(inverse_timesteps: bool, reverse_sigmas: bool) -> float:
    """Terminal sigma value: 0.0 for standard, 1.0 for inverse/reverse."""
    return 1.0 if (inverse_timesteps or reverse_sigmas) else 0.0


def _append_terminal(sigmas: torch.Tensor, terminal: float) -> torch.Tensor:
    return torch.cat([sigmas, torch.tensor([terminal], dtype=torch.float32)], dim=0)


def _is_terminal(value: float, terminal: float) -> bool:
    return math.isclose(value, terminal, abs_tol=1e-7)


def _linear_multistep_coeff(order: int, t: np.ndarray, i: int, j: int) -> float:
    """
    Adams-Bashforth coefficient via numerical quadrature.
    Integrates the j-th Lagrange basis polynomial over [t[i], t[i+1]].
    """
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 1. EulerFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class EulerFlowScheduler(SchedulerInterface):
    """
    Euler method for the flow-matching ODE.

    Single model evaluation per step:
        x_{i+1} = x_i + flow * (sigma_{i+1} - sigma_i)

    This is the simplest possible integrator and serves as the baseline.
    Equivalent to ComfyUI's ``sample_euler`` with a CONST model.
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

    # ── schedule ──────────────────────────────────────────────────────────

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
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength,
        )
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        if to_final:
            terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
            sigma_next = torch.tensor(terminal, dtype=sigma.dtype, device=model_output.device)

        prev_sample = sample + model_output * (sigma_next - sigma)
        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 2. HeunFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class HeunFlowScheduler(SchedulerInterface):
    """
    Heun's method (trapezoidal / improved Euler) for the flow-matching ODE.

    Two model evaluations per outer step:
      1) Euler predictor:  x̃ = x + d₁ · Δσ
      2) Heun corrector:   x' = x + ½(d₁ + d₂) · Δσ
    Falls back to single Euler step when stepping to terminal sigma.

    Uses interleaved timesteps so the engine's `for t in scheduler.timesteps`
    loop naturally provides one model eval per iteration.

    Equivalent to ComfyUI's ``sample_heun``.
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
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self._step_index: Optional[int] = None
        self._heun_state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    # ── schedule ──────────────────────────────────────────────────────────

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
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength,
        )
        base_ext = _append_terminal(base, terminal)

        # Build interleaved sigmas.
        # For each outer step from sigma_i → sigma_{i+1}:
        #   • If sigma_{i+1} != terminal → Heun (2 evals): [sigma_i, sigma_{i+1}]
        #   • If sigma_{i+1} == terminal → Euler (1 eval):  [sigma_i]
        # Terminal value appended at the end.
        sigmas_list: list[torch.Tensor] = []
        for i in range(num_inference_steps):
            sigmas_list.append(base_ext[i])
            if not _is_terminal(float(base_ext[i + 1]), terminal):
                sigmas_list.append(base_ext[i + 1])
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._heun_state = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, model_output, timestep, sample, **kwargs):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        if self._heun_state is None:
            # ── Stage 1: First model evaluation ──
            sigma_i = self.sigmas[idx]
            sigma_next_entry = (
                self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            )

            if _is_terminal(float(sigma_next_entry), terminal):
                # Final step → Euler only (single eval).
                prev_sample = sample + model_output * (sigma_next_entry - sigma_i)
                self._step_index += 1
                return (prev_sample,)

            # Heun stage 1: Euler predictor step to sigma_{i+1}.
            d1 = model_output
            dt = sigma_next_entry - sigma_i
            x_2 = sample + d1 * dt

            self._heun_state = {"x_orig": sample, "d1": d1, "dt": dt}
            self._step_index += 1
            return (x_2,)
        else:
            # ── Stage 2: Heun corrector ──
            d2 = model_output
            st = self._heun_state
            d_prime = (st["d1"] + d2) * 0.5
            prev_sample = st["x_orig"] + d_prime * st["dt"]

            self._heun_state = None
            self._step_index += 1
            return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LMSFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class LMSFlowScheduler(SchedulerInterface):
    """
    Linear Multi-Step (Adams-Bashforth) method for the flow-matching ODE.

    Single model evaluation per step.  Uses the history of up to ``lms_order``
    previous ODE derivatives to construct a higher-order polynomial
    extrapolation.  Falls back to lower orders for the first few steps.

    Equivalent to ComfyUI's ``sample_lms``.
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
        lms_order: int = 4,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.lms_order = int(lms_order)
        self._step_index: Optional[int] = None
        self._derivative_history: list[torch.Tensor] = []
        self.set_timesteps(num_inference_steps)

    # ── schedule ──────────────────────────────────────────────────────────

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
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength,
        )
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        # Pre-compute CPU numpy sigmas for scipy quadrature in the step fn.
        self._sigmas_cpu = self.sigmas.detach().cpu().numpy()
        self._step_index = None
        self._derivative_history = []
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, model_output, timestep, sample, **kwargs):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        # For flow matching the ODE derivative d = model_output (the flow).
        d = model_output
        self._derivative_history.append(d)
        if len(self._derivative_history) > self.lms_order:
            self._derivative_history.pop(0)

        if _is_terminal(float(sigma_next), terminal):
            # Denoising step: return x0 = sample - sigma * flow.
            denoised = sample - sigma * model_output
            prev_sample = denoised
        else:
            cur_order = min(idx + 1, self.lms_order)
            coeffs = [
                _linear_multistep_coeff(cur_order, self._sigmas_cpu, idx, j)
                for j in range(cur_order)
            ]
            prev_sample = sample + sum(
                coeff * deriv
                for coeff, deriv in zip(coeffs, reversed(self._derivative_history))
            )

        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DPM2FlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPM2FlowScheduler(SchedulerInterface):
    """
    DPM-Solver-2 (midpoint method) for the flow-matching ODE.

    Two model evaluations per outer step:
      1) Evaluate at sigma_i, take half-step to sigma_mid (geometric mean).
      2) Evaluate at sigma_mid, take full step from x_orig to sigma_{i+1}.
    Falls back to Euler for the final step to terminal.

    Uses interleaved timesteps (sigma_i, sigma_mid pairs) so the engine loop
    provides one model eval per iteration.

    Equivalent to ComfyUI's ``sample_dpm_2``.
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
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self._step_index: Optional[int] = None
        self._dpm2_state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    # ── schedule ──────────────────────────────────────────────────────────

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
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength,
        )
        base_ext = _append_terminal(base, terminal)

        eps = 1e-6
        sigmas_list: list[torch.Tensor] = []
        for i in range(num_inference_steps):
            sigma_i = base_ext[i]
            sigma_ip1 = base_ext[i + 1]
            sigmas_list.append(sigma_i)
            if not _is_terminal(float(sigma_ip1), terminal):
                # Geometric midpoint in log-sigma space.
                sigma_mid = torch.exp(
                    torch.lerp(
                        torch.log(torch.clamp(sigma_i, min=eps)),
                        torch.log(torch.clamp(sigma_ip1, min=eps)),
                        0.5,
                    )
                )
                sigmas_list.append(sigma_mid)
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._dpm2_state = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, model_output, timestep, sample, **kwargs):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        if self._dpm2_state is None:
            # ── Stage 1: Evaluate at sigma_i ──
            sigma_i = self.sigmas[idx]
            sigma_next_entry = (
                self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            )

            if _is_terminal(float(sigma_next_entry), terminal):
                # Final step → Euler.
                prev_sample = sample + model_output * (sigma_next_entry - sigma_i)
                self._step_index += 1
                return (prev_sample,)

            # DPM-2 stage 1: step to sigma_mid.
            sigma_mid = sigma_next_entry  # next entry is the precomputed midpoint
            sigma_ip1 = (
                self.sigmas[idx + 2]
                if idx + 2 < len(self.sigmas)
                else self.sigmas[-1]
            )

            d = model_output
            x_mid = sample + d * (sigma_mid - sigma_i)

            self._dpm2_state = {
                "x_orig": sample,
                "sigma_i": sigma_i,
                "sigma_ip1": sigma_ip1,
            }
            self._step_index += 1
            return (x_mid,)
        else:
            # ── Stage 2: Evaluate at sigma_mid, step to sigma_{i+1} ──
            st = self._dpm2_state
            d_2 = model_output
            prev_sample = st["x_orig"] + d_2 * (st["sigma_ip1"] - st["sigma_i"])

            self._dpm2_state = None
            self._step_index += 1
            return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 5. DPMpp2MFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMpp2MFlowScheduler(SchedulerInterface):
    """
    DPM-Solver++(2M) for the flow-matching ODE.

    Single model evaluation per step.  Uses the *denoised* (x0) estimate from
    the previous step to construct a second-order correction via a multistep
    blend.

    Uses flow-matching half-logSNR space (``lambda = log((1-sigma)/sigma)``)
    with a logSNR-uniform sigma schedule to ensure stable step-size ratios
    for the second-order correction.  This matches ComfyUI's
    ``sample_dpmpp_2m_sde`` at eta=0 (deterministic).

    Equivalent to ComfyUI's ``sample_dpmpp_2m`` / ``sample_dpmpp_2m_sde``
    (eta=0, midpoint).
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
        self._old_denoised: Optional[torch.Tensor] = None
        self._h_last: Optional[torch.Tensor] = None
        self.set_timesteps(num_inference_steps)

    # ── schedule ──────────────────────────────────────────────────────────

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
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        # Use logSNR-uniform spacing to keep step-size ratios well-behaved
        # for the second-order multistep correction.
        base = _build_base_sigmas_logsnr(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength,
        )
        self.sigmas = _append_terminal(base, terminal)
        self.sigmas = _offset_first_sigma_for_flow(self.sigmas)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._old_denoised = None
        self._h_last = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, model_output, timestep, sample, **kwargs):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        # Compute denoised: x0 = x - sigma * flow
        denoised = sample - sigma * model_output

        if _is_terminal(float(sigma_next), terminal):
            # Return denoised directly for the final step.
            prev_sample = denoised
            h = None
        else:
            # DPM++(2M) in flow-matching half-logSNR space.
            # lambda(sigma) = log((1-sigma)/sigma)
            lam_s = _flow_lambda(sigma)
            lam_t = _flow_lambda(sigma_next)
            h = lam_t - lam_s
            alpha_t = sigma_next * lam_t.exp()  # = 1 - sigma_next

            # First-order term (always applied).
            prev_sample = (sigma_next / sigma) * sample \
                + alpha_t * (-h).expm1().neg() * denoised

            if self._old_denoised is not None and self._h_last is not None:
                # Second-order multistep correction (midpoint variant).
                r = self._h_last / h
                prev_sample = prev_sample + 0.5 * alpha_t * (-h).expm1().neg() \
                    * (1.0 / r) * (denoised - self._old_denoised)

        self._old_denoised = denoised
        self._h_last = h
        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ResMultistepFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class ResMultistepFlowScheduler(SchedulerInterface):
    """
    RES (Recursive Exponential Solver) multistep for the flow-matching ODE.

    Single model evaluation per step.  Uses Euler for the first step; after
    that, a second-order exponential integrator with phi-functions is used.
    This is a deterministic (eta=0) variant.

    Uses a logSNR-uniform sigma schedule to ensure stable step-size ratios
    for the second-order phi-function coefficients, matching the approach
    used by the SDE flow-matching schedulers.

    Reference: https://arxiv.org/pdf/2308.02157
    Equivalent to ComfyUI's ``sample_res_multistep`` (eta=0, no CFG++).
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
        self._old_denoised: Optional[torch.Tensor] = None
        self._old_sigma_down: Optional[torch.Tensor] = None
        self.set_timesteps(num_inference_steps)

    # ── schedule ──────────────────────────────────────────────────────────

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
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        # Use logSNR-uniform spacing to keep step-size ratios well-behaved
        # for the second-order phi-function coefficients.
        base = _build_base_sigmas_logsnr(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength,
        )
        self.sigmas = _append_terminal(base, terminal)
        self.sigmas = _offset_first_sigma_for_flow(self.sigmas)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._old_denoised = None
        self._old_sigma_down = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, model_output, timestep, sample, **kwargs):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]
        # Deterministic (eta=0): sigma_down = sigma_next, sigma_up = 0
        sigma_down = sigma_next

        # Compute denoised: x0 = x - sigma * flow
        denoised = sample - sigma * model_output

        if _is_terminal(float(sigma_down), terminal) or self._old_denoised is None:
            # Euler step:  x' = x + flow * (sigma_next - sigma)
            # which is equivalent to:  x' = x + d * dt
            # When sigma_next == 0 this yields x' = denoised.
            d = model_output
            prev_sample = sample + d * (sigma_down - sigma)
        else:
            # Second-order RES multistep (exponential integrator)
            # using log-sigma space: t = -log(sigma), matching ComfyUI.
            eps = 1e-8
            t_fn = lambda s: torch.clamp(s, min=eps).log().neg()
            sigma_fn_local = lambda t: t.neg().exp()

            phi1_fn = lambda neg_h: torch.expm1(neg_h) / neg_h
            phi2_fn = lambda neg_h: (phi1_fn(neg_h) - 1.0) / neg_h

            sigma_prev = self.sigmas[max(idx - 1, 0)]

            t = t_fn(sigma)
            t_old = t_fn(self._old_sigma_down)
            t_next = t_fn(sigma_down)
            t_prev = t_fn(sigma_prev)

            h = t_next - t
            c2 = (t_prev - t_old) / h

            neg_h = -h
            phi1_val = phi1_fn(neg_h)
            phi2_val = phi2_fn(neg_h)

            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            # scale = exp(-h) = sigma_down / sigma
            scale = sigma_fn_local(h)
            prev_sample = scale * sample + h * (b1 * denoised + b2 * self._old_denoised)

        self._old_denoised = denoised
        self._old_sigma_down = sigma_down
        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# Additional shared helpers for logSNR-based & advanced samplers
# ══════════════════════════════════════════════════════════════════════════════

_FLOW_EPS = 1e-8


def _flow_lambda(sigma: torch.Tensor) -> torch.Tensor:
    """Half-logSNR for flow matching: log((1-sigma)/sigma)."""
    return torch.log(
        torch.clamp((1.0 - sigma) / torch.clamp(sigma, min=_FLOW_EPS), min=_FLOW_EPS)
    )


def _flow_sigma_from_lambda(lbda: torch.Tensor) -> torch.Tensor:
    """Inverse of _flow_lambda: sigma = 1/(1+exp(lambda)) = sigmoid(-lambda)."""
    return 1.0 / (torch.exp(lbda) + 1.0)


def _offset_first_sigma_for_flow(sigmas: torch.Tensor, max_sigma: float = 0.9999) -> torch.Tensor:
    """Clamp first sigma to < 1 so that logSNR is finite."""
    if float(sigmas[0]) >= 1.0:
        sigmas = sigmas.clone()
        sigmas[0] = max_sigma
    return sigmas


def _ei_h_phi_1(h: torch.Tensor) -> torch.Tensor:
    """h * phi_1(h) = expm1(h)."""
    return torch.expm1(h)


def _ei_h_phi_2(h: torch.Tensor) -> torch.Tensor:
    """h * phi_2(h) = (expm1(h) - h) / h."""
    return (torch.expm1(h) - h) / h


# ── DEIS 'rhoab' coefficient helpers ─────────────────────────────────────────

def _deis_integral_2(a, b, start, end, c):
    """Analytical order-2 Lagrange-basis integral for DEIS."""
    coeff = (end ** 3 - start ** 3) / 3 - (end ** 2 - start ** 2) * (a + b) / 2 + (end - start) * a * b
    return coeff / ((c - a) * (c - b))


def _deis_integral_3(a, b, c, start, end, d):
    """Analytical order-3 Lagrange-basis integral for DEIS."""
    coeff = (
        (end ** 4 - start ** 4) / 4
        - (end ** 3 - start ** 3) * (a + b + c) / 3
        + (end ** 2 - start ** 2) * (a * b + a * c + b * c) / 2
        - (end - start) * a * b * c
    )
    return coeff / ((d - a) * (d - b) * (d - c))


def _deis_rhoab_coeff_list(sigmas: torch.Tensor, max_order: int = 3):
    """Pre-compute DEIS 'rhoab' (analytical) polynomial coefficients."""
    C = []
    for i in range(len(sigmas) - 1):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        order = min(i, max_order)
        if order == 0:
            C.append([])
        else:
            prev_t = sigmas[torch.tensor([i - k for k in range(order + 1)])]
            if order == 1:
                coeff_cur = ((t_next - prev_t[1]) ** 2 - (t_cur - prev_t[1]) ** 2) / (2 * (t_cur - prev_t[1]))
                coeff_prev1 = (t_next - t_cur) ** 2 / (2 * (prev_t[1] - t_cur))
                C.append([coeff_cur, coeff_prev1])
            elif order == 2:
                coeff_cur = _deis_integral_2(prev_t[1], prev_t[2], t_cur, t_next, t_cur)
                coeff_prev1 = _deis_integral_2(t_cur, prev_t[2], t_cur, t_next, prev_t[1])
                coeff_prev2 = _deis_integral_2(t_cur, prev_t[1], t_cur, t_next, prev_t[2])
                C.append([coeff_cur, coeff_prev1, coeff_prev2])
            elif order >= 3:
                coeff_cur = _deis_integral_3(prev_t[1], prev_t[2], prev_t[3], t_cur, t_next, t_cur)
                coeff_prev1 = _deis_integral_3(t_cur, prev_t[2], prev_t[3], t_cur, t_next, prev_t[1])
                coeff_prev2 = _deis_integral_3(t_cur, prev_t[1], prev_t[3], t_cur, t_next, prev_t[2])
                coeff_prev3 = _deis_integral_3(t_cur, prev_t[1], prev_t[2], t_cur, t_next, prev_t[3])
                C.append([coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3])
    return C


# ── SA-Solver coefficient helpers (inlined, tau_t=0 deterministic) ───────────

def _sa_compute_exponential_coeffs(
    s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float = 0.0,
) -> torch.Tensor:
    """Compute exponential integrator coefficients for SA-Solver."""
    tau_mul = 1.0 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    product_terms = t ** p - s ** p * (-tau_mul * h).exp()
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()
    return recursive_coeff_mat @ product_terms


def _sa_compute_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float = 0.0,
) -> torch.Tensor:
    """Compute b_i coefficients for SA-Solver predictor/corrector."""
    num_ts = curr_lambdas.shape[0]
    exp_coeffs = _sa_compute_exponential_coeffs(lambda_s, lambda_t, num_ts, tau_t)
    vander_T = torch.vander(curr_lambdas, num_ts, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vander_T, exp_coeffs)
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEISFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DEISFlowScheduler(SchedulerInterface):
    """
    DEIS (Diffusion Exponential Integrator Sampler) for the flow-matching ODE.

    Single eval per step with derivative history and precomputed polynomial
    coefficients ('rhoab' analytical mode).  Order ramps up to ``deis_order``.

    Equivalent to ComfyUI's ``sample_deis`` (rhoab mode).
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False, deis_order=3):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.deis_order = int(deis_order)
        self._step_index: Optional[int] = None
        self._buffer: list[torch.Tensor] = []
        self._coeff_list: list = []
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._coeff_list = _deis_rhoab_coeff_list(self.sigmas, self.deis_order)
        self._step_index = None
        self._buffer = []
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        d_cur = model_output  # ODE derivative = flow
        # DEIS rhoab: coeff_list[idx] is empty for Euler steps
        coeffs = self._coeff_list[idx] if idx < len(self._coeff_list) else []
        # Cap order by available history + 1 (current d_cur counts as 1)
        use_order = min(len(coeffs), len(self._buffer) + 1)
        if _is_terminal(float(sigma_next), terminal) or float(sigma_next) <= 0:
            use_order = 0

        if use_order == 0:
            prev_sample = sample + (sigma_next - sigma) * d_cur
        else:
            prev_sample = sample + coeffs[0] * d_cur
            for k in range(1, use_order):
                prev_sample = prev_sample + coeffs[k] * self._buffer[-k]

        if len(self._buffer) == self.deis_order - 1:
            for k in range(self.deis_order - 2):
                self._buffer[k] = self._buffer[k + 1]
            self._buffer[-1] = d_cur.detach()
        else:
            self._buffer.append(d_cur.detach())

        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 8. IPNDMFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class IPNDMFlowScheduler(SchedulerInterface):
    """
    Improved Pseudo Numerical Diffusion Model (iPNDM) for flow-matching ODE.

    Single eval per step with fixed Adams-Bashforth coefficients and derivative
    history.  Order ramps up to ``max_order``.

    Equivalent to ComfyUI's ``sample_ipndm``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False, max_order=4):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.max_order = int(max_order)
        self._step_index: Optional[int] = None
        self._buffer: list[torch.Tensor] = []
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._buffer = []
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        d = model_output
        o = min(self.max_order, idx + 1)
        dt = sigma_next - sigma

        if _is_terminal(float(sigma_next), terminal):
            prev_sample = sample - sigma * model_output  # denoised
        elif o == 1:
            prev_sample = sample + dt * d
        elif o == 2:
            prev_sample = sample + dt * (3 * d - self._buffer[-1]) / 2
        elif o == 3:
            prev_sample = sample + dt * (23 * d - 16 * self._buffer[-1] + 5 * self._buffer[-2]) / 12
        else:
            prev_sample = sample + dt * (55 * d - 59 * self._buffer[-1] + 37 * self._buffer[-2] - 9 * self._buffer[-3]) / 24

        if len(self._buffer) == self.max_order - 1:
            for k in range(self.max_order - 2):
                self._buffer[k] = self._buffer[k + 1]
            self._buffer[-1] = d
        else:
            self._buffer.append(d)

        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 9. IPNDMVFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class IPNDMVFlowScheduler(SchedulerInterface):
    """
    iPNDM-v (variable step-size) for the flow-matching ODE.

    Like iPNDM but uses actual step sizes to compute coefficients, which is
    more accurate when the sigma schedule is non-uniform.

    Equivalent to ComfyUI's ``sample_ipndm_v``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False, max_order=4):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.max_order = int(max_order)
        self._step_index: Optional[int] = None
        self._buffer: list[torch.Tensor] = []
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._buffer = []
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        d = model_output
        o = min(self.max_order, idx + 1)
        dt = sigma_next - sigma

        if _is_terminal(float(sigma_next), terminal):
            prev_sample = sample - sigma * model_output
        elif o == 1:
            prev_sample = sample + dt * d
        elif o == 2:
            h_n = float(sigma_next - sigma)
            h_n_1 = float(sigma - self.sigmas[idx - 1])
            c1 = (2 + (h_n / h_n_1)) / 2
            c2 = -(h_n / h_n_1) / 2
            prev_sample = sample + dt * (c1 * d + c2 * self._buffer[-1])
        elif o == 3:
            h_n = float(sigma_next - sigma)
            h_n_1 = float(sigma - self.sigmas[idx - 1])
            h_n_2 = float(self.sigmas[idx - 1] - self.sigmas[idx - 2])
            tmp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            c1 = (2 + (h_n / h_n_1)) / 2 + tmp
            c2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * tmp
            c3 = tmp * h_n_1 / h_n_2
            prev_sample = sample + dt * (c1 * d + c2 * self._buffer[-1] + c3 * self._buffer[-2])
        else:
            h_n = float(sigma_next - sigma)
            h_n_1 = float(sigma - self.sigmas[idx - 1])
            h_n_2 = float(self.sigmas[idx - 1] - self.sigmas[idx - 2])
            h_n_3 = float(self.sigmas[idx - 2] - self.sigmas[idx - 3])
            tmp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            tmp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            c1 = (2 + (h_n / h_n_1)) / 2 + tmp1 + tmp2
            c2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * tmp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * tmp2
            c3 = tmp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * tmp2
            c4 = -tmp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            prev_sample = sample + dt * (c1 * d + c2 * self._buffer[-1] + c3 * self._buffer[-2] + c4 * self._buffer[-3])

        if len(self._buffer) == self.max_order - 1:
            for k in range(self.max_order - 2):
                self._buffer[k] = self._buffer[k + 1]
            self._buffer[-1] = d.detach()
        else:
            self._buffer.append(d.detach())

        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 10. GradientEstimationFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class GradientEstimationFlowScheduler(SchedulerInterface):
    """
    Gradient-estimation sampler for the flow-matching ODE.

    Euler step + gradient-estimation correction from step 2 onwards.
    Paper: https://openreview.net/pdf?id=o2ND9v0CeK

    Equivalent to ComfyUI's ``sample_gradient_estimation`` (no CFG++).
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False, ge_gamma=2.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.ge_gamma = float(ge_gamma)
        self._step_index: Optional[int] = None
        self._old_d: Optional[torch.Tensor] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._old_d = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        d = model_output
        dt = sigma_next - sigma

        if _is_terminal(float(sigma_next), terminal):
            prev_sample = sample - sigma * model_output
        else:
            prev_sample = sample + d * dt
            if idx >= 1 and self._old_d is not None:
                d_bar = (self.ge_gamma - 1) * (d - self._old_d)
                prev_sample = prev_sample + d_bar * dt

        self._old_d = d
        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Seeds2FlowScheduler  (deterministic, eta=0)
# ══════════════════════════════════════════════════════════════════════════════

class Seeds2FlowScheduler(SchedulerInterface):
    """
    SEEDS-2 deterministic (eta=0) exponential integrator for flow-matching.

    Two model evals per step using flow-matching logSNR space.
    Uses ``phi_2`` solver type by default for higher accuracy.

    Equivalent to ComfyUI's ``sample_seeds_2`` with eta=0.
    """

    order = 2

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False, r=0.5):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.r = float(r)
        self._step_index: Optional[int] = None
        self._stage_state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        base_ext = _append_terminal(base, terminal)
        base_ext = _offset_first_sigma_for_flow(base_ext)

        r = self.r
        sigmas_list: list[torch.Tensor] = []
        for i in range(num_inference_steps):
            sigma_i = base_ext[i]
            sigma_ip1 = base_ext[i + 1]
            sigmas_list.append(sigma_i)
            if not _is_terminal(float(sigma_ip1), terminal):
                lam_s = _flow_lambda(sigma_i)
                lam_t = _flow_lambda(sigma_ip1)
                lam_s1 = torch.lerp(lam_s, lam_t, r)
                sigma_s1 = _flow_sigma_from_lambda(lam_s1)
                sigmas_list.append(sigma_s1)
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._stage_state = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        if self._stage_state is None:
            # ── Stage 1 ──
            sigma_i = self.sigmas[idx]
            sigma_next_entry = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            if _is_terminal(float(sigma_next_entry), terminal):
                prev_sample = sample - sigma_i * model_output
                self._step_index += 1
                return (prev_sample,)

            sigma_s1 = sigma_next_entry
            sigma_ip1 = self.sigmas[idx + 2] if idx + 2 < len(self.sigmas) else self.sigmas[-1]

            denoised = sample - sigma_i * model_output
            lam_s = _flow_lambda(sigma_i)
            lam_t = _flow_lambda(sigma_ip1)
            h = lam_t - lam_s
            r = self.r
            lam_s1 = torch.lerp(lam_s, lam_t, r)
            alpha_s1 = sigma_s1 * lam_s1.exp()

            x_2 = (sigma_s1 / sigma_i) * sample - alpha_s1 * _ei_h_phi_1(-r * h) * denoised

            self._stage_state = {
                'x_orig': sample, 'denoised': denoised, 'sigma_i': sigma_i,
                'sigma_ip1': sigma_ip1, 'h': h, 'lam_s': lam_s, 'lam_t': lam_t,
            }
            self._step_index += 1
            return (x_2,)
        else:
            # ── Stage 2 ──
            st = self._stage_state
            sigma_s1 = self.sigmas[idx]
            denoised_2 = sample - sigma_s1 * model_output

            h = st['h']
            r = self.r
            alpha_t = st['sigma_ip1'] * st['lam_t'].exp()

            b2 = _ei_h_phi_2(-h) / r
            b1 = _ei_h_phi_1(-h) - b2
            prev_sample = (st['sigma_ip1'] / st['sigma_i']) * st['x_orig'] - alpha_t * (b1 * st['denoised'] + b2 * denoised_2)

            self._stage_state = None
            self._step_index += 1
            return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 12. ExpHeun2X0FlowScheduler  (deterministic)
# ══════════════════════════════════════════════════════════════════════════════

class ExpHeun2X0FlowScheduler(Seeds2FlowScheduler):
    """
    Deterministic exponential Heun 2nd-order method in data-prediction (x0)
    and logSNR time.  Special case of SEEDS-2 with r=1.0.

    Equivalent to ComfyUI's ``sample_exp_heun_2_x0``.
    """

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift, sigma_max=sigma_max, sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps, reverse_sigmas=reverse_sigmas,
            r=1.0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 13. Seeds3FlowScheduler  (deterministic, eta=0)
# ══════════════════════════════════════════════════════════════════════════════

class Seeds3FlowScheduler(SchedulerInterface):
    """
    SEEDS-3 deterministic (eta=0) exponential integrator for flow-matching.

    Three model evals per step using flow-matching logSNR space.

    Equivalent to ComfyUI's ``sample_seeds_3`` with eta=0.
    """

    order = 3

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 r_1=1.0 / 3, r_2=2.0 / 3):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.r_1 = float(r_1)
        self.r_2 = float(r_2)
        self._step_index: Optional[int] = None
        self._stage: int = 0
        self._state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        base_ext = _append_terminal(base, terminal)
        base_ext = _offset_first_sigma_for_flow(base_ext)

        r_1, r_2 = self.r_1, self.r_2
        sigmas_list: list[torch.Tensor] = []
        for i in range(num_inference_steps):
            si = base_ext[i]
            sip1 = base_ext[i + 1]
            sigmas_list.append(si)
            if not _is_terminal(float(sip1), terminal):
                ls = _flow_lambda(si)
                lt = _flow_lambda(sip1)
                sigmas_list.append(_flow_sigma_from_lambda(torch.lerp(ls, lt, r_1)))
                sigmas_list.append(_flow_sigma_from_lambda(torch.lerp(ls, lt, r_2)))
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._stage = 0
        self._state = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        if self._stage == 0:
            # ── Stage 1: eval at sigma_i ──
            sigma_i = self.sigmas[idx]
            nxt = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            if _is_terminal(float(nxt), terminal):
                prev = sample - sigma_i * model_output
                self._step_index += 1
                return (prev,)

            sigma_s1 = self.sigmas[idx + 1]
            sigma_s2 = self.sigmas[idx + 2] if idx + 2 < len(self.sigmas) else self.sigmas[-1]
            sigma_ip1 = self.sigmas[idx + 3] if idx + 3 < len(self.sigmas) else self.sigmas[-1]

            denoised = sample - sigma_i * model_output
            ls = _flow_lambda(sigma_i)
            lt = _flow_lambda(sigma_ip1)
            h = lt - ls
            r_1 = self.r_1
            ls1 = torch.lerp(ls, lt, r_1)
            alpha_s1 = sigma_s1 * ls1.exp()

            x_2 = (sigma_s1 / sigma_i) * sample - alpha_s1 * _ei_h_phi_1(-r_1 * h) * denoised

            self._state = {
                'x_orig': sample, 'denoised': denoised, 'sigma_i': sigma_i,
                'sigma_ip1': sigma_ip1, 'h': h, 'ls': ls, 'lt': lt,
            }
            self._stage = 1
            self._step_index += 1
            return (x_2,)

        elif self._stage == 1:
            # ── Stage 2: eval at sigma_s1 ──
            st = self._state
            sigma_s1 = self.sigmas[idx]
            sigma_s2 = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            denoised_2 = sample - sigma_s1 * model_output

            h = st['h']
            r_1, r_2 = self.r_1, self.r_2
            ls2 = torch.lerp(st['ls'], st['lt'], r_2)
            alpha_s2 = sigma_s2 * ls2.exp()

            a3_2 = r_2 / r_1 * _ei_h_phi_2(-r_2 * h)
            a3_1 = _ei_h_phi_1(-r_2 * h) - a3_2
            x_3 = (sigma_s2 / st['sigma_i']) * st['x_orig'] - alpha_s2 * (a3_1 * st['denoised'] + a3_2 * denoised_2)

            st['denoised_2'] = denoised_2
            self._stage = 2
            self._step_index += 1
            return (x_3,)

        else:
            # ── Stage 3: eval at sigma_s2 ──
            st = self._state
            sigma_s2 = self.sigmas[idx]
            denoised_3 = sample - sigma_s2 * model_output

            h = st['h']
            r_2 = self.r_2
            alpha_t = st['sigma_ip1'] * st['lt'].exp()

            b3 = _ei_h_phi_2(-h) / r_2
            b1 = _ei_h_phi_1(-h) - b3
            prev = (st['sigma_ip1'] / st['sigma_i']) * st['x_orig'] - alpha_t * (b1 * st['denoised'] + b3 * denoised_3)

            self._state = None
            self._stage = 0
            self._step_index += 1
            return (prev,)


# ══════════════════════════════════════════════════════════════════════════════
# 14. SASolverFlowScheduler  (deterministic, tau=0)
# ══════════════════════════════════════════════════════════════════════════════

class SASolverFlowScheduler(SchedulerInterface):
    """
    Deterministic SA-Solver (Stochastic Adams Solver, tau=0) for flow-matching.

    Single model eval per step.  Predictor-corrector Adams method in logSNR
    space with configurable predictor and corrector orders.

    Equivalent to ComfyUI's ``sample_sa_solver`` with a tau_func that always
    returns 0.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 predictor_order=3, corrector_order=4):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.predictor_order = int(predictor_order)
        self.corrector_order = int(corrector_order)
        self._step_index: Optional[int] = None
        self._pred_list: list[torch.Tensor] = []
        self._x: Optional[torch.Tensor] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.sigmas = _offset_first_sigma_for_flow(self.sigmas)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._lambdas = _flow_lambda(self.sigmas[:-1])
        lam_terminal = _flow_lambda(torch.tensor(max(terminal, 1e-8)))
        self._lambdas_full = torch.cat([self._lambdas, lam_terminal.unsqueeze(0)])
        self._step_index = None
        self._pred_list = []
        self._x = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
            self._lambdas = self._lambdas.to(device)
            self._lambdas_full = self._lambdas_full.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        self._lambdas = self._lambdas.to(model_output.device)
        self._lambdas_full = self._lambdas_full.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]
        max_used = max(self.predictor_order, self.corrector_order)
        lower_end = _is_terminal(float(self.sigmas[-1]), terminal) and float(self.sigmas[-1]) == 0.0

        denoised = sample - sigma * model_output
        self._pred_list.append(denoised)
        self._pred_list = self._pred_list[-max_used:]

        pred_order = min(self.predictor_order, len(self._pred_list))
        corr_order = 0 if idx == 0 else min(self.corrector_order, len(self._pred_list))
        if lower_end:
            pred_order = min(pred_order, len(self.sigmas) - 2 - idx)
            corr_order = min(corr_order, len(self.sigmas) - 1 - idx)

        # ── Corrector ──
        if corr_order == 0:
            x = sample  # x = x_pred
        else:
            lams = self._lambdas_full
            curr_lams = lams[idx - corr_order + 1: idx + 1]
            b = _sa_compute_b_coeffs(sigma, curr_lams, lams[idx - 1], lams[idx], 0.0)
            pred_mat = torch.stack(self._pred_list[-corr_order:], dim=1)
            corr_res = torch.tensordot(pred_mat, b.to(pred_mat.device), dims=([1], [0]))
            x = (sigma / self.sigmas[idx - 1]) * (self._x if self._x is not None else sample) + corr_res

        # ── Predictor ──
        if _is_terminal(float(sigma_next), terminal):
            x_pred = denoised
        else:
            lams = self._lambdas_full
            curr_lams = lams[idx - pred_order + 1: idx + 1]
            b = _sa_compute_b_coeffs(sigma_next, curr_lams, lams[idx], lams[idx + 1], 0.0)
            pred_mat = torch.stack(self._pred_list[-pred_order:], dim=1)
            pred_res = torch.tensordot(pred_mat, b.to(pred_mat.device), dims=([1], [0]))
            x_pred = (sigma_next / sigma) * x + pred_res

        self._x = x
        self._step_index += 1
        return (x_pred,)


# ══════════════════════════════════════════════════════════════════════════════
# 15. SASolverPECEFlowScheduler  (deterministic, tau=0, PECE mode)
# ══════════════════════════════════════════════════════════════════════════════

class SASolverPECEFlowScheduler(SchedulerInterface):
    """
    Deterministic SA-Solver PECE (Predict-Evaluate-Correct-Evaluate) for
    flow-matching.

    Two model evals per step (except step 0) at the same sigma: one on the
    predicted state and one on the corrected state.  Uses interleaved timesteps.

    Equivalent to ComfyUI's ``sample_sa_solver_pece`` with tau=0.
    """

    order = 2

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 predictor_order=3, corrector_order=4):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.predictor_order = int(predictor_order)
        self.corrector_order = int(corrector_order)
        self._step_index: Optional[int] = None
        self._pred_list: list[torch.Tensor] = []
        self._x: Optional[torch.Tensor] = None
        self._pece_phase: bool = False
        self._corrected_x: Optional[torch.Tensor] = None
        self._outer_idx: int = 0
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        base_sigmas = _append_terminal(base, terminal)
        base_sigmas = _offset_first_sigma_for_flow(base_sigmas)

        # Interleave: [sigma_0, sigma_1, sigma_1, sigma_2, sigma_2, ..., terminal]
        sigmas_list: list[torch.Tensor] = [base_sigmas[0]]
        for i in range(1, num_inference_steps):
            sigmas_list.append(base_sigmas[i])
            sigmas_list.append(base_sigmas[i])  # repeated for PECE correction
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        # Store base sigmas & lambdas for coefficient computation
        self._base_sigmas = base_sigmas
        self._base_lambdas = _flow_lambda(base_sigmas[:-1])
        lam_t = _flow_lambda(torch.tensor(max(terminal, 1e-8)))
        self._base_lambdas_full = torch.cat([self._base_lambdas, lam_t.unsqueeze(0)])
        self._step_index = None
        self._pred_list = []
        self._x = None
        self._pece_phase = False
        self._corrected_x = None
        self._outer_idx = 0
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
            self._base_sigmas = self._base_sigmas.to(device)
            self._base_lambdas = self._base_lambdas.to(device)
            self._base_lambdas_full = self._base_lambdas_full.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        self._base_sigmas = self._base_sigmas.to(model_output.device)
        self._base_lambdas_full = self._base_lambdas_full.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        oi = self._outer_idx
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        bs = self._base_sigmas
        lams = self._base_lambdas_full
        max_used = max(self.predictor_order, self.corrector_order)
        lower_end = float(bs[-1]) == 0.0
        sigma = bs[oi]
        sigma_next = bs[min(oi + 1, len(bs) - 1)]

        if not self._pece_phase:
            # ── Main evaluation (on x_pred) ──
            denoised = sample - sigma * model_output
            self._pred_list.append(denoised)
            self._pred_list = self._pred_list[-max_used:]

            pred_order = min(self.predictor_order, len(self._pred_list))
            corr_order = 0 if oi == 0 else min(self.corrector_order, len(self._pred_list))
            if lower_end:
                pred_order = min(pred_order, len(bs) - 2 - oi)
                corr_order = min(corr_order, len(bs) - 1 - oi)

            # Corrector
            if corr_order == 0:
                x = sample
            else:
                curr_lams = lams[oi - corr_order + 1: oi + 1]
                b = _sa_compute_b_coeffs(sigma, curr_lams, lams[oi - 1], lams[oi], 0.0)
                pred_mat = torch.stack(self._pred_list[-corr_order:], dim=1)
                corr_res = torch.tensordot(pred_mat, b.to(pred_mat.device), dims=([1], [0]))
                x = (sigma / bs[oi - 1]) * (self._x if self._x is not None else sample) + corr_res

            # If corrector was applied, do PECE (return corrected x for re-eval)
            if corr_order > 0:
                self._corrected_x = x
                self._pece_phase = True
                self._step_index += 1
                return (x,)  # engine will evaluate model on corrected x at same sigma

            # No correction (step 0): go straight to predictor
            if _is_terminal(float(sigma_next), terminal):
                x_pred = denoised
            else:
                curr_lams = lams[oi - pred_order + 1: oi + 1]
                b = _sa_compute_b_coeffs(sigma_next, curr_lams, lams[oi], lams[oi + 1], 0.0)
                pred_mat = torch.stack(self._pred_list[-pred_order:], dim=1)
                pred_res = torch.tensordot(pred_mat, b.to(pred_mat.device), dims=([1], [0]))
                x_pred = (sigma_next / sigma) * x + pred_res

            self._x = x
            self._outer_idx += 1
            self._step_index += 1
            return (x_pred,)
        else:
            # ── PECE re-evaluation (on corrected x) ──
            denoised = sample - sigma * model_output
            self._pred_list[-1] = denoised  # update last prediction

            x = self._corrected_x
            self._pece_phase = False
            self._corrected_x = None

            pred_order = min(self.predictor_order, len(self._pred_list))
            if lower_end:
                pred_order = min(pred_order, len(bs) - 2 - oi)

            if _is_terminal(float(sigma_next), terminal):
                x_pred = denoised
            else:
                curr_lams = lams[oi - pred_order + 1: oi + 1]
                b = _sa_compute_b_coeffs(sigma_next, curr_lams, lams[oi], lams[oi + 1], 0.0)
                pred_mat = torch.stack(self._pred_list[-pred_order:], dim=1)
                pred_res = torch.tensordot(pred_mat, b.to(pred_mat.device), dims=([1], [0]))
                x_pred = (sigma_next / sigma) * x + pred_res

            self._x = x
            self._outer_idx += 1
            self._step_index += 1
            return (x_pred,)


# ══════════════════════════════════════════════════════════════════════════════
# 16. HeunPP2FlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class HeunPP2FlowScheduler(SchedulerInterface):
    """
    Heun++ (heunpp2) for flow-matching.

    Uses up to 3 model evaluations per outer step with a weighted derivative
    combination based on sigma ratios.  Falls back to weighted Heun (2 evals)
    for the second-to-last step and Euler (1 eval) for the last step.

    Equivalent to ComfyUI's ``sample_heunpp2``.
    """

    order = 2

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self._step_index: Optional[int] = None
        self._stage: int = 0
        self._state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        base_ext = _append_terminal(base, terminal)
        num_steps = num_inference_steps

        sigmas_list: list[torch.Tensor] = []
        for i in range(num_steps):
            sigma_i = base_ext[i]
            sigma_ip1 = base_ext[i + 1]
            sigmas_list.append(sigma_i)

            if _is_terminal(float(sigma_ip1), terminal):
                # Last step: Euler (1 eval)
                pass
            elif (i + 2 <= num_steps
                  and _is_terminal(float(base_ext[i + 2]), terminal)):
                # Second-to-last: weighted Heun (2 evals)
                sigmas_list.append(sigma_ip1)
            else:
                # Full HeunPP2 (3 evals)
                sigmas_list.append(sigma_ip1)
                if i + 2 <= num_steps:
                    sigmas_list.append(base_ext[i + 2])

        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))
        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._sigma_0 = float(base_ext[0])
        self._step_index = None
        self._stage = 0
        self._state = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        if self._stage == 0:
            sigma_i = self.sigmas[idx]
            d1 = model_output
            nxt = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]

            if _is_terminal(float(nxt), terminal):
                # Last step: Euler
                prev_sample = sample + d1 * (nxt - sigma_i)
                self._step_index += 1
                return (prev_sample,)

            sigma_ip1 = nxt
            dt = sigma_ip1 - sigma_i
            x_2 = sample + d1 * dt

            # Determine mode from the entry after sigma_ip1
            nxt2 = self.sigmas[idx + 2] if idx + 2 < len(self.sigmas) else self.sigmas[-1]
            if _is_terminal(float(nxt2), terminal):
                self._state = {'x': sample, 'd1': d1, 'dt': dt,
                               's1': sigma_ip1, 'mode': 'heun'}
            else:
                self._state = {'x': sample, 'd1': d1, 'dt': dt,
                               's1': sigma_ip1, 's2': nxt2, 'mode': 'pp2'}
            self._stage = 1
            self._step_index += 1
            return (x_2,)

        elif self._stage == 1:
            st = self._state
            d2 = model_output

            if st['mode'] == 'heun':
                w = 2.0 * self._sigma_0
                w2 = float(st['s1']) / w if w > 0 else 0.5
                w1 = 1.0 - w2
                d_prime = st['d1'] * w1 + d2 * w2
                prev_sample = st['x'] + d_prime * st['dt']

                self._state = None
                self._stage = 0
                self._step_index += 1
                return (prev_sample,)
            else:
                # HeunPP2: predict to sigma_{i+2} for 3rd eval
                dt_2 = st['s2'] - st['s1']
                x_3 = sample + d2 * dt_2
                st['d2'] = d2
                self._stage = 2
                self._step_index += 1
                return (x_3,)

        else:  # stage == 2
            st = self._state
            d3 = model_output
            w = 3.0 * self._sigma_0
            w2 = float(st['s1']) / w if w > 0 else 1.0 / 3
            w3 = float(st['s2']) / w if w > 0 else 1.0 / 3
            w1 = 1.0 - w2 - w3
            d_prime = st['d1'] * w1 + st['d2'] * w2 + d3 * w3
            prev_sample = st['x'] + d_prime * st['dt']

            self._state = None
            self._stage = 0
            self._step_index += 1
            return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 17. DPMFastFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMFastFlowScheduler(SchedulerInterface):
    """
    DPM-Solver-Fast for flow-matching.

    Uses a uniform grid in ``t = -log(sigma)`` space with mixed-order (1, 2, 3)
    DPM-Solver steps.  The total number of model evaluations equals
    ``num_inference_steps``.

    Equivalent to ComfyUI's ``sample_dpm_fast``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self._step_index: Optional[int] = None
        self._interval_idx: int = 0
        self._interval_state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(denoising_strength)
        sigma_end = max(self.sigma_min, 1e-6)
        if shift_eff != 1.0:
            sigma_start = float(_time_snr_shift(shift_eff, torch.tensor(sigma_start)))
            sigma_end = float(_time_snr_shift(shift_eff, torch.tensor(sigma_end)))
        sigma_start = min(sigma_start, 0.9999)
        sigma_end = max(sigma_end, 1e-6)

        t_start = -math.log(sigma_start)
        t_end = -math.log(sigma_end)

        n = max(num_inference_steps, 1)
        m = n // 3 + 1
        ts = torch.linspace(t_start, t_end, m + 1, dtype=torch.float64)

        if n % 3 == 0:
            orders = [3] * max(m - 2, 0) + ([2, 1] if m >= 2 else [min(n, 2)] if m == 1 else [])
        else:
            orders = [3] * max(m - 1, 0) + [n % 3]

        # Build interleaved sigmas and interval metadata
        intervals: list[dict] = []
        sigmas_list: list[torch.Tensor] = []
        for iv in range(len(orders)):
            t_i = ts[iv]
            t_ip1 = ts[iv + 1]
            order = orders[iv]
            intervals.append({'order': order, 't': t_i.float(), 't_next': t_ip1.float()})
            sigmas_list.append((-t_i).exp().float())
            h = t_ip1 - t_i
            if order >= 2:
                r1 = 0.5 if order == 2 else 1.0 / 3
                s1 = t_i + r1 * h
                sigmas_list.append((-s1).exp().float())
            if order >= 3:
                r2 = 2.0 / 3
                s2 = t_i + r2 * h
                sigmas_list.append((-s2).exp().float())

        sigmas_list.append(torch.tensor(0.0, dtype=torch.float32))
        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._intervals = intervals
        self._interval_idx = 0
        self._interval_state = None
        self._step_index = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0

        if self._interval_state is None:
            # ── New interval ──
            iv = self._intervals[self._interval_idx]
            order = iv['order']
            t = iv['t'].to(model_output.device)
            t_next = iv['t_next'].to(model_output.device)
            h = t_next - t
            eps = model_output

            if order == 1:
                sigma_next = (-t_next).exp()
                x = sample - sigma_next * h.expm1() * eps
                self._interval_idx += 1
                self._step_index += 1
                return (x,)

            # Order 2 or 3: midpoint eval needed
            r1 = 0.5 if order == 2 else 1.0 / 3
            s1 = t + r1 * h
            sigma_s1 = (-s1).exp()
            u1 = sample - sigma_s1 * (r1 * h).expm1() * eps

            self._interval_state = {
                'x': sample, 'eps': eps, 't': t, 't_next': t_next,
                'h': h, 'order': order, 'stage': 1,
            }
            self._step_index += 1
            return (u1,)
        else:
            st = self._interval_state
            t = st['t']
            t_next = st['t_next']
            h = st['h']
            eps = st['eps']
            sigma_next = (-t_next).exp()

            if st['order'] == 2:
                # DPM-2 correction
                eps_r1 = model_output
                r1 = 0.5
                x = st['x'] - sigma_next * h.expm1() * eps \
                    - sigma_next / (2 * r1) * h.expm1() * (eps_r1 - eps)
                self._interval_state = None
                self._interval_idx += 1
                self._step_index += 1
                return (x,)

            elif st['stage'] == 1:
                # DPM-3 second eval
                eps_r1 = model_output
                r1 = 1.0 / 3
                r2 = 2.0 / 3
                s2 = t + r2 * h
                sigma_s2 = (-s2).exp()
                u2 = st['x'] - sigma_s2 * (r2 * h).expm1() * eps \
                    - sigma_s2 * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
                st['eps_r1'] = eps_r1
                st['stage'] = 2
                self._step_index += 1
                return (u2,)

            else:
                # DPM-3 final correction
                eps_r2 = model_output
                r2 = 2.0 / 3
                x = st['x'] - sigma_next * h.expm1() * eps \
                    - sigma_next / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
                self._interval_state = None
                self._interval_idx += 1
                self._step_index += 1
                return (x,)


# ══════════════════════════════════════════════════════════════════════════════
# 18. DPMAdaptiveFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMAdaptiveFlowScheduler(SchedulerInterface):
    """
    DPM-Solver-3 for flow-matching with logSNR-uniform schedule.

    Uses order-3 DPM-Solver steps with a logSNR-uniform sigma schedule that
    approximates the optimal step sizes an adaptive solver would choose.
    Falls back to lower-order steps for remaining evaluations.

    Equivalent to a fixed-step approximation of ComfyUI's ``sample_dpm_adaptive``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self._step_index: Optional[int] = None
        self._interval_idx: int = 0
        self._interval_state: Optional[dict] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * float(denoising_strength)
        sigma_end = max(self.sigma_min, 1e-6)
        if shift_eff != 1.0:
            sigma_start = float(_time_snr_shift(shift_eff, torch.tensor(sigma_start)))
            sigma_end = float(_time_snr_shift(shift_eff, torch.tensor(sigma_end)))
        sigma_start = min(sigma_start, 0.9999)
        sigma_end = max(sigma_end, 1e-6)

        # LogSNR-uniform grid (better for flow matching than log-sigma-uniform)
        eps_v = 1e-8
        lam_start = math.log(max((1.0 - sigma_start) / max(sigma_start, eps_v), eps_v))
        lam_end = math.log(max((1.0 - sigma_end) / max(sigma_end, eps_v), eps_v))

        n = max(num_inference_steps, 1)
        m = n // 3 + 1
        lambdas = torch.linspace(lam_start, lam_end, m + 1, dtype=torch.float64)
        sigmas_grid = 1.0 / (1.0 + lambdas.exp())
        ts = (-sigmas_grid.clamp(min=1e-10).log()).float()

        if n % 3 == 0:
            orders = [3] * max(m - 2, 0) + ([2, 1] if m >= 2 else [min(n, 2)] if m == 1 else [])
        else:
            orders = [3] * max(m - 1, 0) + [n % 3]

        intervals: list[dict] = []
        sigmas_list: list[torch.Tensor] = []
        for iv in range(len(orders)):
            t_i = ts[iv]
            t_ip1 = ts[iv + 1]
            order = orders[iv]
            intervals.append({'order': order, 't': t_i, 't_next': t_ip1})
            sigmas_list.append((-t_i).exp())
            h = t_ip1 - t_i
            if order >= 2:
                r1 = 0.5 if order == 2 else 1.0 / 3
                s1 = t_i + r1 * h
                sigmas_list.append((-s1).exp())
            if order >= 3:
                r2 = 2.0 / 3
                s2 = t_i + r2 * h
                sigmas_list.append((-s2).exp())

        sigmas_list.append(torch.tensor(0.0, dtype=torch.float32))
        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._intervals = intervals
        self._interval_idx = 0
        self._interval_state = None
        self._step_index = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, **kw):
        # Shares the same DPM-Solver step logic as DPMFastFlowScheduler
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0

        if self._interval_state is None:
            iv = self._intervals[self._interval_idx]
            order = iv['order']
            t = iv['t'].to(model_output.device)
            t_next = iv['t_next'].to(model_output.device)
            h = t_next - t
            eps = model_output

            if order == 1:
                sigma_next = (-t_next).exp()
                x = sample - sigma_next * h.expm1() * eps
                self._interval_idx += 1
                self._step_index += 1
                return (x,)

            r1 = 0.5 if order == 2 else 1.0 / 3
            s1 = t + r1 * h
            sigma_s1 = (-s1).exp()
            u1 = sample - sigma_s1 * (r1 * h).expm1() * eps
            self._interval_state = {
                'x': sample, 'eps': eps, 't': t, 't_next': t_next,
                'h': h, 'order': order, 'stage': 1,
            }
            self._step_index += 1
            return (u1,)
        else:
            st = self._interval_state
            t, t_next, h, eps = st['t'], st['t_next'], st['h'], st['eps']
            sigma_next = (-t_next).exp()

            if st['order'] == 2:
                eps_r1 = model_output
                r1 = 0.5
                x = st['x'] - sigma_next * h.expm1() * eps \
                    - sigma_next / (2 * r1) * h.expm1() * (eps_r1 - eps)
                self._interval_state = None
                self._interval_idx += 1
                self._step_index += 1
                return (x,)
            elif st['stage'] == 1:
                eps_r1 = model_output
                r1, r2 = 1.0 / 3, 2.0 / 3
                s2 = t + r2 * h
                sigma_s2 = (-s2).exp()
                u2 = st['x'] - sigma_s2 * (r2 * h).expm1() * eps \
                    - sigma_s2 * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
                st['eps_r1'] = eps_r1
                st['stage'] = 2
                self._step_index += 1
                return (u2,)
            else:
                eps_r2 = model_output
                r2 = 2.0 / 3
                x = st['x'] - sigma_next * h.expm1() * eps \
                    - sigma_next / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
                self._interval_state = None
                self._interval_idx += 1
                self._step_index += 1
                return (x,)


# ══════════════════════════════════════════════════════════════════════════════
# 19. LCMFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class LCMFlowScheduler(SchedulerInterface):
    """
    Latent Consistency Model (LCM) sampler for flow-matching.

    Each step jumps directly to the denoised prediction, then re-noises to the
    next sigma level using the flow-matching noise scaling:

        x_next = (1 - sigma_next) * denoised + sigma_next * noise

    Single model evaluation per step.

    Equivalent to ComfyUI's ``sample_lcm``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self._step_index: Optional[int] = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, generator=None, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)

        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]

        # Denoised prediction
        denoised = sample - sigma * model_output

        # Jump to denoised
        x = denoised

        # Re-noise if not at terminal
        if not _is_terminal(float(sigma_next), terminal):
            if generator is not None:
                noise = torch.randn(x.shape, generator=generator,
                                    device=x.device, dtype=x.dtype)
            else:
                noise = torch.randn_like(x)
            # Flow-matching noise scaling
            x = (1.0 - sigma_next) * denoised + sigma_next * noise

        self._step_index += 1
        return (x,)
