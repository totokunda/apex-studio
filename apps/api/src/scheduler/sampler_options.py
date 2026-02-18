"""
Sampler options and noise mode utilities ported from RES4LYF/samplers_extensions.py.

This module provides:
    - NoiseMode enum with all SDE noise scaling modes
    - compute_noise_eta_ratio() for computing effective eta based on noise mode
    - SamplerSDEOptions for SDE noise configuration
    - StepSizeOptions for step-size overshoot
    - DetailBoostOptions for noise-scaling detail boost
    - SigmaScalingOptions for sigma schedule scaling (d_noise / "lying")
    - MomentumOptions for momentum acceleration
    - ImplicitStepOptions for implicit (Newton-iteration) stepping
    - CycleOptions for unsample/resample cycling
    - SamplerOptions aggregating all option types into one object

Noise mode reference (from RES4LYF rk_noise_sampler_beta.py):
    - hard:        eta_ratio = eta  (most aggressive)
    - exp:         eta_ratio = sqrt(1 - exp(-2 * eta * h))
    - soft:        eta_ratio = 1 - (1 - eta) + eta * (sigma_next / sigma)
    - softer:      eta_ratio = 1 - sqrt(1 - eta^2 * (sigma^2 - sigma_next^2) / sigma^2)
    - soft-linear: eta_ratio = 1 - eta * (sigma_next - sigma)
    - sinusoidal:  eta_ratio = eta * sin(pi * sigma_next / sigma_max)^2
    - eps:         eta_ratio = eta * sqrt((sigma_next/sigma)^2 * (sigma^2 - sigma_next^2))
    - lorentzian:  eta_ratio = eta, sigma_base = sqrt(1 - 1/(sigma_next^2 + 1))
    - hard_var:    variance-aware hard mode
    - vpsde:       VP-SDE formulation
    - er4:         exponential rescaling mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

import torch
from torch import Tensor


# ==============================================================================
# Noise Mode enum + computation
# ==============================================================================

class NoiseMode(str, Enum):
    """SDE noise scaling modes, mirroring RES4LYF NOISE_MODE_NAMES."""
    NONE = "none"
    HARD = "hard"
    LORENTZIAN = "lorentzian"
    SOFT = "soft"
    SOFT_LINEAR = "soft-linear"
    SOFTER = "softer"
    EPS = "eps"
    SINUSOIDAL = "sinusoidal"
    EXP = "exp"
    VPSDE = "vpsde"
    ER4 = "er4"
    HARD_VAR = "hard_var"


NOISE_MODE_NAMES: List[str] = [m.value for m in NoiseMode]


def compute_noise_eta_ratio(
    noise_mode: str,
    sigma: Tensor,
    sigma_next: Tensor,
    eta: float,
    sigma_max: float = 1.0,
) -> Tuple[Optional[Tensor], Tensor]:
    """
    Compute the effective eta_ratio and sigma_base for a given noise mode.

    Returns:
        (eta_ratio, sigma_base) where eta_ratio may be None for modes that
        directly compute sigma_up/sigma_down (vpsde, er4).
    """
    sigma_base = sigma_next.clone() if isinstance(sigma_next, Tensor) else torch.tensor(sigma_next)
    sigmax = torch.tensor(sigma_max, dtype=sigma.dtype, device=sigma.device)
    _eps = 1e-8

    if noise_mode == "none" or eta == 0.0:
        return torch.tensor(0.0, dtype=sigma.dtype, device=sigma.device), sigma_base

    if noise_mode == "hard":
        eta_ratio = torch.tensor(eta, dtype=sigma.dtype, device=sigma.device)

    elif noise_mode == "exp":
        h = -(sigma_next / torch.clamp(sigma, min=_eps)).log()
        eta_ratio = (1 - (-2 * eta * h).exp()) ** 0.5

    elif noise_mode == "soft":
        eta_ratio = 1 - (1 - eta) + eta * (sigma_next / torch.clamp(sigma, min=_eps))

    elif noise_mode == "softer":
        inner = 1 - (eta ** 2 * (sigma ** 2 - sigma_next ** 2)) / torch.clamp(sigma ** 2, min=_eps)
        eta_ratio = 1 - torch.sqrt(torch.clamp(inner, min=0.0))

    elif noise_mode == "soft-linear":
        eta_ratio = 1 - eta * (sigma_next - sigma)

    elif noise_mode == "sinusoidal":
        eta_ratio = eta * torch.sin(torch.pi * sigma_next / torch.clamp(sigmax, min=_eps)) ** 2

    elif noise_mode == "eps":
        ratio_sq = (sigma_next / torch.clamp(sigma, min=_eps)) ** 2
        diff_sq = sigma ** 2 - sigma_next ** 2
        eta_ratio = eta * torch.sqrt(torch.clamp(ratio_sq * diff_sq, min=0.0))

    elif noise_mode == "lorentzian":
        eta_ratio = torch.tensor(eta, dtype=sigma.dtype, device=sigma.device)
        alpha = 1.0 / (sigma_next ** 2 + 1)
        sigma_base = ((1 - alpha) ** 0.5).to(sigma.dtype)

    elif noise_mode == "hard_var":
        sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
        if sigma_next > sigma_var:
            eta_ratio = torch.tensor(0.0, dtype=sigma.dtype, device=sigma.device)
            sigma_base = sigma_next
        else:
            eta_ratio = torch.tensor(eta, dtype=sigma.dtype, device=sigma.device)
            sigma_base = torch.sqrt((sigma - sigma_next).abs() + 1e-10)

    elif noise_mode == "vpsde":
        return None, sigma_base

    elif noise_mode == "er4":
        return None, sigma_base

    else:
        eta_ratio = torch.tensor(eta, dtype=sigma.dtype, device=sigma.device)

    return eta_ratio, sigma_base


def compute_sde_noise_amounts(
    sigma: Tensor,
    sigma_next: Tensor,
    eta: float = 0.5,
    noise_mode: str = "hard",
    s_noise: float = 1.0,
    sigma_max: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute sigma_down and sigma_up for SDE noise injection in flow-matching.

    Uses the variance-preserving (CONST) model convention where:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        sigma_up = sqrt(sigma_next^2 - sigma_down^2 * alpha_ratio^2)

    Returns:
        (sigma_down, sigma_up, alpha_ratio)
    """
    _eps = 1e-8

    if eta == 0.0 or noise_mode == "none":
        return sigma_next.clone(), torch.zeros_like(sigma_next), torch.ones_like(sigma_next)

    eta_ratio, sigma_base = compute_noise_eta_ratio(noise_mode, sigma, sigma_next, eta, sigma_max)

    if eta_ratio is None:
        if noise_mode == "vpsde":
            dt = sigma - sigma_next
            sigma_up = eta * sigma * dt ** 0.5
            alpha_ratio = 1 - dt * (eta ** 2 / 4) * (1 + sigma)
            sigma_down = sigma_next - (eta / 4) * sigma * (1 - sigma) * (sigma - sigma_next)
            return sigma_down, sigma_up * s_noise, alpha_ratio
        elif noise_mode == "er4":
            noise_scaler = lambda s: s * ((s ** eta).exp() + 10.0)
            alpha_ratio = noise_scaler(sigma_next) / noise_scaler(sigma)
            sigma_up = (sigma_next ** 2 - sigma ** 2 * alpha_ratio ** 2) ** 0.5
            return sigma_next.clone(), sigma_up * s_noise, alpha_ratio
        else:
            return sigma_next.clone(), torch.zeros_like(sigma_next), torch.ones_like(sigma_next)

    sigma_up = sigma_base * eta_ratio
    sigma_down_sq = sigma_next ** 2 - sigma_up ** 2
    sigma_down = torch.sqrt(torch.clamp(sigma_down_sq, min=0.0))
    alpha_ip1 = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    alpha_ratio = alpha_ip1 / torch.clamp(alpha_down, min=_eps)
    sigma_up = sigma_up * s_noise

    sigma_up = torch.nan_to_num(sigma_up, 0.0)
    sigma_down = torch.nan_to_num(sigma_down, float(sigma_next))
    alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

    return sigma_down, sigma_up, alpha_ratio


# ==============================================================================
# Option dataclasses mirroring RES4LYF samplers_extensions.py nodes
# ==============================================================================

@dataclass
class SamplerSDEOptions:
    """SDE noise configuration (ClownOptions_SDE_Beta)."""
    noise_type_sde: str = "gaussian"
    noise_type_sde_substep: str = "gaussian"
    noise_mode_sde: str = "hard"
    noise_mode_sde_substep: str = "hard"
    eta: float = 0.5
    eta_substep: float = 0.5
    noise_seed_sde: int = -1
    etas: Optional[Tensor] = None
    etas_substep: Optional[Tensor] = None


@dataclass
class StepSizeOptions:
    """Step-size overshoot configuration (ClownOptions_StepSize_Beta)."""
    overshoot_mode: str = "hard"
    overshoot_mode_substep: str = "hard"
    overshoot: float = 0.0
    overshoot_substep: float = 0.0


@dataclass
class DetailBoostOptions:
    """Detail boost / noise scaling configuration (ClownOptions_DetailBoost_Beta)."""
    noise_scaling_weight: float = 0.0
    noise_scaling_type: str = "model"
    noise_scaling_mode: str = "hard"
    noise_scaling_eta: float = 0.5
    noise_scaling_cycles: int = 1
    noise_scaling_weights: Optional[Tensor] = None
    noise_scaling_etas: Optional[Tensor] = None
    noise_boost_step: float = 0.0
    noise_boost_substep: float = 0.0
    noise_boost_normalize: bool = True
    start_step: int = 0
    end_step: int = -1


DETAIL_BOOST_METHODS = [
    "sampler", "sampler_normal", "sampler_substep",
    "sampler_substep_normal", "model", "model_alpha",
]


@dataclass
class SigmaScalingOptions:
    """Sigma schedule scaling configuration (ClownOptions_SigmaScaling_Beta)."""
    noise_anchor: float = 1.0
    s_noise: float = 1.0
    s_noise_substep: float = 1.0
    d_noise: float = 1.0
    d_noise_start_step: int = 0
    d_noise_inv: float = 1.0
    d_noise_inv_start_step: int = 1
    s_noises: Optional[Tensor] = None
    s_noises_substep: Optional[Tensor] = None


@dataclass
class MomentumOptions:
    """Momentum acceleration configuration (ClownOptions_Momentum_Beta)."""
    momentum: float = 0.0


@dataclass
class ImplicitStepOptions:
    """Implicit stepping configuration (ClownOptions_ImplicitSteps_Beta)."""
    implicit_type: str = "bongmath"
    implicit_type_substeps: str = "bongmath"
    implicit_steps: int = 0
    implicit_substeps: int = 0


IMPLICIT_TYPE_NAMES = [
    "use_explicit", "predictor-corrector", "bongmath",
    "gauss-legendre", "radau_iia", "lobatto_iiic",
]


@dataclass
class CycleOptions:
    """Unsample/resample cycle configuration (ClownOptions_Cycles_Beta)."""
    cycles: float = 0.0
    unsample_eta: float = 0.5
    eta_decay_scale: float = 1.0
    unsample_cfg: float = 1.0
    unsampler_name: str = "none"

    @property
    def rebounds(self) -> int:
        return int(self.cycles * 2)


@dataclass
class SwapSamplerOptions:
    """Sampler swapping configuration (ClownOptions_SwapSampler_Beta)."""
    rk_swap_type: str = "res_3m"
    rk_swap_threshold: float = 0.0
    rk_swap_step: int = 30
    rk_swap_print: bool = False


@dataclass
class SDEMaskOptions:
    """Spatial SDE noise mask configuration (ClownOptions_SDE_Mask_Beta)."""
    sde_mask: Optional[Tensor] = None
    mask_max: float = 1.0
    mask_min: float = 0.0
    invert_mask: bool = False


@dataclass
class TileOptions:
    """Tiling configuration (ClownOptions_Tile_Beta)."""
    tile_sizes: List[Tuple[int, int]] = field(default_factory=list)

    def add_tile(self, height: int = 1024, width: int = 1024):
        self.tile_sizes.append((height, width))


@dataclass
class AutomationOptions:
    """Per-step automation configuration (ClownOptions_Automation_Beta)."""
    etas: Optional[Tensor] = None
    etas_substep: Optional[Tensor] = None
    s_noises: Optional[Tensor] = None
    s_noises_substep: Optional[Tensor] = None
    epsilon_scales: Optional[Tensor] = None
    frame_weights: Optional[Tensor] = None


@dataclass
class InitNoiseOptions:
    """Initial noise configuration (SharkOptions_Beta)."""
    noise_type_init: str = "gaussian"
    noise_init_stdev: float = 1.0
    denoise_alt: float = 1.0
    channelwise_cfg: bool = False


# ==============================================================================
# Aggregated SamplerOptions
# ==============================================================================

@dataclass
class SamplerOptions:
    """
    Aggregated sampler options combining all configuration types from
    RES4LYF samplers_extensions.py into a single object.
    """
    sde: SamplerSDEOptions = field(default_factory=SamplerSDEOptions)
    step_size: StepSizeOptions = field(default_factory=StepSizeOptions)
    detail_boost: DetailBoostOptions = field(default_factory=DetailBoostOptions)
    sigma_scaling: SigmaScalingOptions = field(default_factory=SigmaScalingOptions)
    momentum: MomentumOptions = field(default_factory=MomentumOptions)
    implicit: ImplicitStepOptions = field(default_factory=ImplicitStepOptions)
    cycles: CycleOptions = field(default_factory=CycleOptions)
    swap_sampler: SwapSamplerOptions = field(default_factory=SwapSamplerOptions)
    sde_mask: SDEMaskOptions = field(default_factory=SDEMaskOptions)
    tile: TileOptions = field(default_factory=TileOptions)
    automation: AutomationOptions = field(default_factory=AutomationOptions)
    init_noise: InitNoiseOptions = field(default_factory=InitNoiseOptions)
    extra_options: str = ""
    start_at_step: int = 0

    @classmethod
    def from_dict(cls, options: dict) -> "SamplerOptions":
        """Build SamplerOptions from a flat options dict (RES4LYF convention)."""
        opts = cls()
        opts.sde.noise_type_sde = options.get("noise_type_sde", opts.sde.noise_type_sde)
        opts.sde.noise_type_sde_substep = options.get("noise_type_sde_substep", opts.sde.noise_type_sde_substep)
        opts.sde.noise_mode_sde = options.get("noise_mode_sde", opts.sde.noise_mode_sde)
        opts.sde.noise_mode_sde_substep = options.get("noise_mode_sde_substep", opts.sde.noise_mode_sde_substep)
        opts.sde.eta = options.get("eta", opts.sde.eta)
        opts.sde.eta_substep = options.get("eta_substep", opts.sde.eta_substep)
        opts.sde.noise_seed_sde = options.get("noise_seed_sde", opts.sde.noise_seed_sde)
        opts.sde.etas = options.get("etas", opts.sde.etas)
        opts.sde.etas_substep = options.get("etas_substep", opts.sde.etas_substep)
        opts.step_size.overshoot_mode = options.get("overshoot_mode", opts.step_size.overshoot_mode)
        opts.step_size.overshoot_mode_substep = options.get("overshoot_mode_substep", opts.step_size.overshoot_mode_substep)
        opts.step_size.overshoot = options.get("overshoot", opts.step_size.overshoot)
        opts.step_size.overshoot_substep = options.get("overshoot_substep", opts.step_size.overshoot_substep)
        opts.detail_boost.noise_scaling_weight = options.get("noise_scaling_weight", opts.detail_boost.noise_scaling_weight)
        opts.detail_boost.noise_scaling_type = options.get("noise_scaling_type", opts.detail_boost.noise_scaling_type)
        opts.detail_boost.noise_scaling_mode = options.get("noise_scaling_mode", opts.detail_boost.noise_scaling_mode)
        opts.detail_boost.noise_scaling_eta = options.get("noise_scaling_eta", opts.detail_boost.noise_scaling_eta)
        opts.detail_boost.noise_scaling_weights = options.get("noise_scaling_weights", opts.detail_boost.noise_scaling_weights)
        opts.detail_boost.noise_scaling_etas = options.get("noise_scaling_etas", opts.detail_boost.noise_scaling_etas)
        opts.detail_boost.noise_boost_step = options.get("noise_boost_step", opts.detail_boost.noise_boost_step)
        opts.detail_boost.noise_boost_substep = options.get("noise_boost_substep", opts.detail_boost.noise_boost_substep)
        opts.detail_boost.noise_boost_normalize = options.get("noise_boost_normalize", opts.detail_boost.noise_boost_normalize)
        opts.sigma_scaling.noise_anchor = options.get("noise_anchor", opts.sigma_scaling.noise_anchor)
        opts.sigma_scaling.s_noise = options.get("s_noise", opts.sigma_scaling.s_noise)
        opts.sigma_scaling.s_noise_substep = options.get("s_noise_substep", opts.sigma_scaling.s_noise_substep)
        opts.sigma_scaling.d_noise = options.get("d_noise", opts.sigma_scaling.d_noise)
        opts.sigma_scaling.d_noise_start_step = options.get("d_noise_start_step", opts.sigma_scaling.d_noise_start_step)
        opts.sigma_scaling.d_noise_inv = options.get("d_noise_inv", opts.sigma_scaling.d_noise_inv)
        opts.sigma_scaling.d_noise_inv_start_step = options.get("d_noise_inv_start_step", opts.sigma_scaling.d_noise_inv_start_step)
        opts.sigma_scaling.s_noises = options.get("s_noises", opts.sigma_scaling.s_noises)
        opts.sigma_scaling.s_noises_substep = options.get("s_noises_substep", opts.sigma_scaling.s_noises_substep)
        opts.momentum.momentum = options.get("momentum", opts.momentum.momentum)
        opts.implicit.implicit_type = options.get("implicit_type", opts.implicit.implicit_type)
        opts.implicit.implicit_type_substeps = options.get("implicit_type_substeps", opts.implicit.implicit_type_substeps)
        opts.implicit.implicit_steps = options.get("implicit_steps", opts.implicit.implicit_steps)
        opts.implicit.implicit_substeps = options.get("implicit_substeps", opts.implicit.implicit_substeps)
        rebounds = options.get("rebounds", 0)
        opts.cycles.cycles = rebounds / 2.0 if rebounds else options.get("cycles", opts.cycles.cycles)
        opts.cycles.unsample_eta = options.get("unsample_eta", opts.cycles.unsample_eta)
        opts.cycles.eta_decay_scale = options.get("eta_decay_scale", opts.cycles.eta_decay_scale)
        opts.cycles.unsample_cfg = options.get("unsample_cfg", opts.cycles.unsample_cfg)
        opts.cycles.unsampler_name = options.get("unsampler_name", opts.cycles.unsampler_name)
        opts.swap_sampler.rk_swap_type = options.get("rk_swap_type", opts.swap_sampler.rk_swap_type)
        opts.swap_sampler.rk_swap_threshold = options.get("rk_swap_threshold", opts.swap_sampler.rk_swap_threshold)
        opts.swap_sampler.rk_swap_step = options.get("rk_swap_step", opts.swap_sampler.rk_swap_step)
        opts.swap_sampler.rk_swap_print = options.get("rk_swap_print", opts.swap_sampler.rk_swap_print)
        opts.sde_mask.sde_mask = options.get("sde_mask", opts.sde_mask.sde_mask)
        opts.tile.tile_sizes = options.get("tile_sizes", [])
        opts.init_noise.noise_type_init = options.get("noise_type_init", opts.init_noise.noise_type_init)
        opts.init_noise.noise_init_stdev = options.get("noise_init_stdev", opts.init_noise.noise_init_stdev)
        opts.init_noise.denoise_alt = options.get("denoise_alt", opts.init_noise.denoise_alt)
        opts.init_noise.channelwise_cfg = options.get("channelwise_cfg", opts.init_noise.channelwise_cfg)
        opts.extra_options = options.get("extra_options", opts.extra_options)
        opts.start_at_step = options.get("start_at_step", opts.start_at_step)
        return opts

    def to_dict(self) -> dict:
        """Flatten all options into a single dict (RES4LYF convention)."""
        d: dict = {}
        d["noise_type_sde"] = self.sde.noise_type_sde
        d["noise_type_sde_substep"] = self.sde.noise_type_sde_substep
        d["noise_mode_sde"] = self.sde.noise_mode_sde
        d["noise_mode_sde_substep"] = self.sde.noise_mode_sde_substep
        d["eta"] = self.sde.eta
        d["eta_substep"] = self.sde.eta_substep
        d["noise_seed_sde"] = self.sde.noise_seed_sde
        d["overshoot_mode"] = self.step_size.overshoot_mode
        d["overshoot"] = self.step_size.overshoot
        d["noise_scaling_weight"] = self.detail_boost.noise_scaling_weight
        d["noise_scaling_type"] = self.detail_boost.noise_scaling_type
        d["noise_scaling_mode"] = self.detail_boost.noise_scaling_mode
        d["noise_anchor"] = self.sigma_scaling.noise_anchor
        d["s_noise"] = self.sigma_scaling.s_noise
        d["d_noise"] = self.sigma_scaling.d_noise
        d["d_noise_start_step"] = self.sigma_scaling.d_noise_start_step
        d["d_noise_inv"] = self.sigma_scaling.d_noise_inv
        d["d_noise_inv_start_step"] = self.sigma_scaling.d_noise_inv_start_step
        d["momentum"] = self.momentum.momentum
        d["implicit_type"] = self.implicit.implicit_type
        d["implicit_steps"] = self.implicit.implicit_steps
        d["rebounds"] = self.cycles.rebounds
        d["unsample_eta"] = self.cycles.unsample_eta
        d["rk_swap_type"] = self.swap_sampler.rk_swap_type
        d["extra_options"] = self.extra_options
        d["start_at_step"] = self.start_at_step
        return d


# ==============================================================================
# Utility functions for applying options during sampling
# ==============================================================================

def apply_sigma_scaling(
    sigma: Tensor,
    step: int,
    d_noise: float = 1.0,
    d_noise_start_step: int = 0,
    d_noise_inv: float = 1.0,
    d_noise_inv_start_step: int = 1,
) -> Tensor:
    """Apply sigma schedule scaling ("lying") from SigmaScalingOptions."""
    scaled = sigma.clone() if isinstance(sigma, Tensor) else torch.tensor(sigma)
    if d_noise != 1.0 and step >= d_noise_start_step:
        scaled = scaled * d_noise
    if d_noise_inv != 1.0 and step >= d_noise_inv_start_step:
        scaled = scaled * d_noise_inv
    return scaled


def apply_momentum(
    prev_direction: Optional[Tensor],
    current_direction: Tensor,
    momentum: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Apply momentum acceleration to the step direction.
    Returns (adjusted_direction, updated_prev_direction)."""
    if momentum == 0.0 or prev_direction is None:
        return current_direction, current_direction.clone()
    adjusted = current_direction + momentum * (current_direction - prev_direction)
    return adjusted, current_direction.clone()


def apply_overshoot(
    sigma: Tensor,
    sigma_next: Tensor,
    overshoot: float = 0.0,
    overshoot_mode: str = "hard",
    sigma_max: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Apply step-size overshoot from StepSizeOptions.
    Returns (effective_sigma_next, rescale_factor)."""
    if overshoot == 0.0:
        return sigma_next, torch.ones_like(sigma)
    _eps = 1e-8
    h = sigma_next - sigma
    eta_ratio, _ = compute_noise_eta_ratio(overshoot_mode, sigma, sigma_next, overshoot, sigma_max)
    if eta_ratio is None:
        eta_ratio = torch.tensor(overshoot, dtype=sigma.dtype, device=sigma.device)
    h_overshoot = h * (1.0 + eta_ratio.abs())
    effective_sigma_next = sigma + h_overshoot
    effective_sigma_next = torch.clamp(effective_sigma_next, min=0.0, max=1.0)
    actual_h = effective_sigma_next - sigma
    if actual_h.abs() > _eps:
        rescale = h / actual_h
    else:
        rescale = torch.ones_like(sigma)
    return effective_sigma_next, rescale


def get_per_step_eta(
    step: int,
    base_eta: float,
    etas_schedule: Optional[Tensor] = None,
) -> float:
    """Get the eta value for a specific step, using per-step schedule if provided."""
    if etas_schedule is not None and step < len(etas_schedule):
        return float(etas_schedule[step])
    return base_eta


def get_per_step_s_noise(
    step: int,
    base_s_noise: float,
    s_noises_schedule: Optional[Tensor] = None,
) -> float:
    """Get the s_noise value for a specific step, using per-step schedule if provided."""
    if s_noises_schedule is not None and step < len(s_noises_schedule):
        return float(s_noises_schedule[step])
    return base_s_noise
