
# ══════════════════════════════════════════════════════════════════════════════
# SDE noise helpers  (BrownianTree via torchsde for correlated SDE noise)
# ══════════════════════════════════════════════════════════════════════════════

import torchsde  # noqa: E402  — deferred import kept near usage
from diffusers.configuration_utils import register_to_config
from src.scheduler.scheduler import SchedulerInterface
from typing import Optional
import torch
from src.scheduler.flow_deterministic import (
    _terminal_sigma, _build_base_sigmas, _build_base_sigmas_logsnr, _append_terminal,
    _offset_first_sigma_for_flow, _is_terminal, _flow_lambda, _flow_sigma_from_lambda,
    _ei_h_phi_1, _ei_h_phi_2,
)

class _BatchedBrownianTree:
    """Thin wrapper around torchsde.BrownianTree that supports batched seeds."""

    def __init__(self, x: torch.Tensor, t0, t1, seed=None, cpu: bool = True):
        self.cpu_tree = cpu
        t0, t1, self.sign = self._sort(t0, t1)
        w0 = torch.zeros_like(x)
        self.batched = False
        if seed is None:
            seed = (torch.randint(0, 2 ** 63 - 1, ()).item(),)
        elif isinstance(seed, (tuple, list)):
            if len(seed) != x.shape[0]:
                raise ValueError("seed list length must match batch size")
            self.batched = True
            w0 = w0[0]
        else:
            seed = (seed,)
        if self.cpu_tree:
            t0, w0, t1 = t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu()
        self.trees = tuple(
            torchsde.BrownianTree(t0, w0, t1, entropy=s) for s in seed
        )

    @staticmethod
    def _sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self._sort(t0, t1)
        device, dtype = t0.device, t0.dtype
        if self.cpu_tree:
            t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
        w = torch.stack([tree(t0, t1) for tree in self.trees]).to(
            device=device, dtype=dtype
        ) * (self.sign * sign)
        return w if self.batched else w[0]


class _BrownianTreeNoiseSampler:
    """Noise sampler backed by a BrownianTree for correlated SDE noise."""

    def __init__(self, x, sigma_min, sigma_max, seed=None, cpu=True):
        self.tree = _BatchedBrownianTree(
            x,
            torch.as_tensor(sigma_min),
            torch.as_tensor(sigma_max),
            seed,
            cpu=cpu,
        )

    def __call__(self, sigma, sigma_next):
        t0 = torch.as_tensor(sigma)
        t1 = torch.as_tensor(sigma_next)
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


def _make_noise_sampler(x: torch.Tensor, sigmas: torch.Tensor, seed: Optional[int] = None):
    """Create a BrownianTree noise sampler spanning the full sigma range."""
    sigma_min = float(sigmas[sigmas > 0].min())
    sigma_max = float(sigmas.max())
    return _BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)


def _get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Compute sigma_down and sigma_up for ancestral noise injection."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5,
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


# ══════════════════════════════════════════════════════════════════════════════
# 16. DPMppSDEFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMppSDEFlowScheduler(SchedulerInterface):
    """
    DPM-Solver++ (stochastic) for flow-matching ODE/SDE.

    Two model evaluations per step.  Uses half-logSNR space with ancestral
    noise injection controlled by ``eta`` and ``s_noise``.

    Equivalent to ComfyUI ``sample_dpmpp_sde`` / ``sample_dpmpp_sde_gpu``.
    """

    order = 2

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 eta=0.25, s_noise=1.0, r=0.5):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)
        self.r = float(r)
        self._step_index: Optional[int] = None
        self._stage_state: Optional[dict] = None
        self._noise_sampler = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas_logsnr(
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
                lam_s1 = lam_s + r * (lam_t - lam_s)
                sigmas_list.append(_flow_sigma_from_lambda(lam_s1))
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._stage_state = None
        self._noise_sampler = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, seed=None, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        if self._noise_sampler is None:
            self._noise_sampler = _make_noise_sampler(sample, self.sigmas, seed=seed)
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        eta, s_noise, r = self.eta, self.s_noise, self.r
        fac = 1.0 / (2.0 * r)

        if self._stage_state is None:
            # ── Stage 1: eval at sigma_i ──
            sigma_i = self.sigmas[idx]
            nxt = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            if _is_terminal(float(nxt), terminal):
                self._step_index += 1
                return (sample - sigma_i * model_output,)

            sigma_s1 = nxt
            sigma_ip1 = self.sigmas[idx + 2] if idx + 2 < len(self.sigmas) else self.sigmas[-1]
            denoised = sample - sigma_i * model_output
            lam_s = _flow_lambda(sigma_i)
            lam_t = _flow_lambda(sigma_ip1)
            lam_s1 = lam_s + r * (lam_t - lam_s)
            alpha_s = sigma_i * lam_s.exp()
            alpha_s1 = sigma_s1 * lam_s1.exp()
            alpha_t = sigma_ip1 * lam_t.exp()

            sd, su = _get_ancestral_step(lam_s.neg().exp(), lam_s1.neg().exp(), eta)
            lam_s1_ = sd.log().neg() if sd > 0 else lam_s1
            h_ = lam_s1_ - lam_s
            x_2 = (alpha_s1 / alpha_s) * (-h_).exp() * sample - alpha_s1 * (-h_).expm1() * denoised
            if eta > 0 and s_noise > 0:
                x_2 = x_2 + alpha_s1 * self._noise_sampler(sigma_i, sigma_s1) * s_noise * su

            self._stage_state = {
                'x': sample, 'den': denoised, 'si': sigma_i, 'sip1': sigma_ip1,
                'ls': lam_s, 'lt': lam_t, 'as': alpha_s, 'at': alpha_t,
            }
            self._step_index += 1
            return (x_2,)
        else:
            # ── Stage 2: eval at sigma_s1 ──
            st = self._stage_state
            den2 = sample - self.sigmas[idx] * model_output
            den_d = (1.0 - fac) * st['den'] + fac * den2

            sd, su = _get_ancestral_step(st['ls'].neg().exp(), st['lt'].neg().exp(), eta)
            lam_t_ = sd.log().neg() if sd > 0 else st['lt']
            h_ = lam_t_ - st['ls']
            x = (st['at'] / st['as']) * (-h_).exp() * st['x'] - st['at'] * (-h_).expm1() * den_d
            if eta > 0 and s_noise > 0:
                x = x + st['at'] * self._noise_sampler(st['si'], st['sip1']) * s_noise * su

            self._stage_state = None
            self._step_index += 1
            return (x,)


# ══════════════════════════════════════════════════════════════════════════════
# 17. DPMpp2MSDEFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMpp2MSDEFlowScheduler(SchedulerInterface):
    """
    DPM-Solver++(2M) SDE for flow-matching.

    Single eval per step with denoised history.  Uses flow-matching logSNR
    with stochastic noise injection.  ``solver_type`` selects 'midpoint' or
    'heun' second-order correction.

    Equivalent to ComfyUI ``sample_dpmpp_2m_sde`` / ``_gpu``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 eta=0.25, s_noise=1.0, solver_type='midpoint'):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)
        self.solver_type = str(solver_type)
        self._step_index: Optional[int] = None
        self._old_denoised: Optional[torch.Tensor] = None
        self._h_last: Optional[torch.Tensor] = None
        self._noise_sampler = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas_logsnr(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.sigmas = _offset_first_sigma_for_flow(self.sigmas)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._old_denoised = None
        self._h_last = None
        self._noise_sampler = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, seed=None, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        if self._noise_sampler is None:
            self._noise_sampler = _make_noise_sampler(sample, self.sigmas, seed=seed)
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        eta, s_noise = self.eta, self.s_noise
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]
        denoised = sample - sigma * model_output

        if _is_terminal(float(sigma_next), terminal):
            prev_sample = denoised
            h = None
        else:
            lam_s, lam_t = _flow_lambda(sigma), _flow_lambda(sigma_next)
            h = lam_t - lam_s
            h_eta = h * (eta + 1.0)
            alpha_t = sigma_next * lam_t.exp()

            prev_sample = sigma_next / sigma * (-h * eta).exp() * sample \
                + alpha_t * (-h_eta).expm1().neg() * denoised

            if self._old_denoised is not None and self._h_last is not None:
                r = self._h_last / h
                if self.solver_type == 'heun':
                    prev_sample = prev_sample + alpha_t * (
                        (-h_eta).expm1().neg() / (-h_eta) + 1
                    ) * (1.0 / r) * (denoised - self._old_denoised)
                else:
                    prev_sample = prev_sample + 0.5 * alpha_t * (
                        (-h_eta).expm1().neg()
                    ) * (1.0 / r) * (denoised - self._old_denoised)

            if eta > 0 and s_noise > 0:
                prev_sample = prev_sample + self._noise_sampler(sigma, sigma_next) \
                    * sigma_next * (-2.0 * h * eta).expm1().neg().sqrt() * s_noise

        self._old_denoised = denoised
        self._h_last = h
        self._step_index += 1
        return (prev_sample,)


# ══════════════════════════════════════════════════════════════════════════════
# 18. DPMpp2MSDEHeunFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMpp2MSDEHeunFlowScheduler(DPMpp2MSDEFlowScheduler):
    """
    DPM-Solver++(2M) SDE Heun for flow-matching.
    Same as DPMpp2MSDEFlowScheduler with ``solver_type='heun'``.

    Equivalent to ComfyUI ``sample_dpmpp_2m_sde_heun`` / ``_gpu``.
    """

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 eta=0.25, s_noise=1.0):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift, sigma_max=sigma_max, sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps, reverse_sigmas=reverse_sigmas,
            eta=eta, s_noise=s_noise, solver_type='heun',
        )


# ══════════════════════════════════════════════════════════════════════════════
# 19. DPMpp3MSDEFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

class DPMpp3MSDEFlowScheduler(SchedulerInterface):
    """
    DPM-Solver++(3M) SDE for flow-matching.

    Single eval per step with two levels of denoised history for up to
    3rd-order correction.  Uses flow-matching logSNR with stochastic noise.

    Equivalent to ComfyUI ``sample_dpmpp_3m_sde`` / ``_gpu``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 eta=0.25, s_noise=1.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)
        self._step_index: Optional[int] = None
        self._denoised_1: Optional[torch.Tensor] = None
        self._denoised_2: Optional[torch.Tensor] = None
        self._h_1: Optional[torch.Tensor] = None
        self._h_2: Optional[torch.Tensor] = None
        self._noise_sampler = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas_logsnr(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.sigmas = _offset_first_sigma_for_flow(self.sigmas)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._denoised_1 = None
        self._denoised_2 = None
        self._h_1 = None
        self._h_2 = None
        self._noise_sampler = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, seed=None, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        if self._noise_sampler is None:
            self._noise_sampler = _make_noise_sampler(sample, self.sigmas, seed=seed)
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        eta, s_noise = self.eta, self.s_noise
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]
        denoised = sample - sigma * model_output

        if _is_terminal(float(sigma_next), terminal):
            prev_sample = denoised
            h = None
        else:
            lam_s, lam_t = _flow_lambda(sigma), _flow_lambda(sigma_next)
            h = lam_t - lam_s
            h_eta = h * (eta + 1.0)
            alpha_t = sigma_next * lam_t.exp()

            prev_sample = sigma_next / sigma * (-h * eta).exp() * sample \
                + alpha_t * (-h_eta).expm1().neg() * denoised

            if self._h_2 is not None:
                r0 = self._h_1 / h
                r1 = self._h_2 / h
                d1_0 = (denoised - self._denoised_1) / r0
                d1_1 = (self._denoised_1 - self._denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                prev_sample = prev_sample + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif self._h_1 is not None:
                r = self._h_1 / h
                d = (denoised - self._denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                prev_sample = prev_sample + (alpha_t * phi_2) * d

            if eta > 0 and s_noise > 0:
                prev_sample = prev_sample + self._noise_sampler(sigma, sigma_next) \
                    * sigma_next * (-2.0 * h * eta).expm1().neg().sqrt() * s_noise

        self._denoised_1, self._denoised_2 = denoised, self._denoised_1
        self._h_1, self._h_2 = h, self._h_1
        self._step_index += 1
        return (prev_sample,)



# ══════════════════════════════════════════════════════════════════════════════
# 20. ExpHeun2X0SDEFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

def _default_noise_sampler(x, seed=None):
    """Simple independent Gaussian noise sampler (not BrownianTree)."""
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
    else:
        gen = None
    def sampler(sigma, sigma_next):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout,
                           device=x.device, generator=gen)
    return sampler


class ExpHeun2X0SDEFlowScheduler(SchedulerInterface):
    """
    Stochastic exponential Heun 2nd-order method (SDE) in data-prediction (x0)
    and logSNR time for flow-matching.

    Special case of SEEDS-2 with r=1.0 and eta>0.  Two model evaluations per
    step with SDE noise injection.

    Equivalent to ComfyUI's ``sample_exp_heun_2_x0_sde``.
    """

    order = 2

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 eta=1.0, s_noise=1.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.eta = float(eta)
        self.s_noise = float(s_noise)
        self._step_index: Optional[int] = None
        self._stage_state: Optional[dict] = None
        self._noise_sampler = None
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

        # r=1.0 means the intermediate sigma equals sigma_{i+1}
        sigmas_list: list[torch.Tensor] = []
        for i in range(num_inference_steps):
            sigma_i = base_ext[i]
            sigma_ip1 = base_ext[i + 1]
            sigmas_list.append(sigma_i)
            if not _is_terminal(float(sigma_ip1), terminal):
                sigmas_list.append(sigma_ip1)  # r=1.0 → sigma_s1 = sigma_ip1
        sigmas_list.append(torch.tensor(terminal, dtype=torch.float32))

        self.sigmas = torch.stack(sigmas_list).to(dtype=torch.float32)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None
        self._stage_state = None
        self._noise_sampler = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample, seed=None, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        if self._noise_sampler is None:
            self._noise_sampler = _default_noise_sampler(sample, seed=seed)
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        eta, s_noise = self.eta, self.s_noise
        r = 1.0

        if self._stage_state is None:
            # ── Stage 1: eval at sigma_i ──
            sigma_i = self.sigmas[idx]
            nxt = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else self.sigmas[-1]
            if _is_terminal(float(nxt), terminal):
                self._step_index += 1
                return (sample - sigma_i * model_output,)

            sigma_s1 = nxt  # r=1.0 → sigma_s1 = sigma_ip1
            sigma_ip1 = self.sigmas[idx + 2] if idx + 2 < len(self.sigmas) else self.sigmas[-1]
            denoised = sample - sigma_i * model_output

            lam_s = _flow_lambda(sigma_i)
            lam_t = _flow_lambda(sigma_ip1)
            h = lam_t - lam_s
            h_eta = h * (eta + 1.0)
            lam_s1 = torch.lerp(lam_s, lam_t, r)
            alpha_s1 = sigma_s1 * lam_s1.exp()

            # Stochastic step to sigma_s1
            x_2 = (sigma_s1 / sigma_i) * (-r * h * eta).exp() * sample \
                - alpha_s1 * _ei_h_phi_1(-r * h_eta) * denoised

            # SDE noise injection
            sde_noise = None
            if eta > 0 and s_noise > 0:
                sde_noise = (-2.0 * r * h * eta).expm1().neg().sqrt() \
                    * self._noise_sampler(sigma_i, sigma_s1)
                x_2 = x_2 + sde_noise * sigma_s1 * s_noise

            self._stage_state = {
                'x_orig': sample, 'denoised': denoised, 'sigma_i': sigma_i,
                'sigma_ip1': sigma_ip1, 'h': h, 'h_eta': h_eta,
                'lam_s': lam_s, 'lam_t': lam_t, 'sde_noise': sde_noise,
            }
            self._step_index += 1
            return (x_2,)
        else:
            # ── Stage 2: eval at sigma_s1 ──
            st = self._stage_state
            sigma_s1 = self.sigmas[idx]
            denoised_2 = sample - sigma_s1 * model_output
            h, h_eta, r = st['h'], st['h_eta'], 1.0
            alpha_t = st['sigma_ip1'] * st['lam_t'].exp()

            # phi_2 solver type (higher accuracy)
            b2 = _ei_h_phi_2(-h_eta) / r
            b1 = _ei_h_phi_1(-h_eta) - b2
            x = (st['sigma_ip1'] / st['sigma_i']) * (-h * eta).exp() * st['x_orig'] \
                - alpha_t * (b1 * st['denoised'] + b2 * denoised_2)

            # SDE noise (for r=1.0, second segment factor is 0 → only first noise)
            if eta > 0 and s_noise > 0 and st['sde_noise'] is not None:
                x = x + st['sde_noise'] * st['sigma_ip1'] * s_noise

            self._stage_state = None
            self._step_index += 1
            return (x,)


# ══════════════════════════════════════════════════════════════════════════════
# 21. ERSDEFlowScheduler
# ══════════════════════════════════════════════════════════════════════════════

def _er_sde_noise_scaler(x: torch.Tensor) -> torch.Tensor:
    """Default noise scaler for ER-SDE solver: x * (exp(x^0.3) + 10)."""
    return x * ((x ** 0.3).exp() + 10.0)


class ERSDEFlowScheduler(SchedulerInterface):
    """
    Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3) for flow-matching.

    Single model evaluation per step with up to 3 stages of history-based
    corrections and stochastic noise injection.

    arXiv: https://arxiv.org/abs/2309.06169

    Equivalent to ComfyUI's ``sample_er_sde``.
    """

    order = 1

    @register_to_config
    def __init__(self, num_inference_steps=50, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, reverse_sigmas=False,
                 s_noise=1.0, max_stage=3):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.inverse_timesteps = bool(inverse_timesteps)
        self.reverse_sigmas = bool(reverse_sigmas)
        self.s_noise = float(s_noise)
        self.max_stage = int(max_stage)
        self._step_index: Optional[int] = None
        self._old_denoised: Optional[torch.Tensor] = None
        self._old_denoised_d: Optional[torch.Tensor] = None
        self._noise_sampler = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0,
                      training=False, device=None, shift=None, **kw):
        shift_eff = float(self.shift if shift is None else shift)
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        base = _build_base_sigmas_logsnr(
            num_inference_steps, self.sigma_max, self.sigma_min,
            shift_eff, self.inverse_timesteps, self.reverse_sigmas, denoising_strength)
        self.sigmas = _append_terminal(base, terminal)
        self.sigmas = _offset_first_sigma_for_flow(self.sigmas)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)

        # Precompute er_lambdas: er_lambda = sigma / (1 - sigma) for flow matching
        _eps = 1e-8
        self._er_lambdas = self.sigmas / (1.0 - self.sigmas).clamp(min=_eps)

        self._step_index = None
        self._old_denoised = None
        self._old_denoised_d = None
        self._noise_sampler = None
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
            self._er_lambdas = self._er_lambdas.to(device)

    def step(self, model_output, timestep, sample, seed=None, **kw):
        self.sigmas = self.sigmas.to(model_output.device)
        self._er_lambdas = self._er_lambdas.to(model_output.device)
        if self._step_index is None:
            self._step_index = 0
        if self._noise_sampler is None:
            self._noise_sampler = _default_noise_sampler(sample, seed=seed)
        idx = self._step_index
        terminal = _terminal_sigma(self.inverse_timesteps, self.reverse_sigmas)
        s_noise = self.s_noise
        num_points = 200.0

        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[min(idx + 1, len(self.sigmas) - 1)]
        denoised = sample - sigma * model_output

        if _is_terminal(float(sigma_next), terminal):
            prev_sample = denoised
        else:
            stage_used = min(self.max_stage, idx + 1)
            er_lam_s = self._er_lambdas[idx]
            er_lam_t = self._er_lambdas[min(idx + 1, len(self._er_lambdas) - 1)]
            alpha_s = sigma / er_lam_s  # = 1 - sigma for flow matching
            alpha_t = sigma_next / er_lam_t
            r_alpha = alpha_t / alpha_s
            r = _er_sde_noise_scaler(er_lam_t) / _er_sde_noise_scaler(er_lam_s)

            # Stage 1: Euler
            prev_sample = r_alpha * r * sample + alpha_t * (1.0 - r) * denoised

            if stage_used >= 2:
                dt = er_lam_t - er_lam_s
                point_idx = torch.arange(0, num_points, dtype=torch.float32,
                                         device=model_output.device)
                lam_step = -dt / num_points
                lam_pos = er_lam_t + point_idx * lam_step
                scaled_pos = _er_sde_noise_scaler(lam_pos)

                # Stage 2
                s_val = torch.sum(1.0 / scaled_pos) * lam_step
                er_lam_prev = self._er_lambdas[max(idx - 1, 0)]
                denoised_d = (denoised - self._old_denoised) / (er_lam_s - er_lam_prev)
                prev_sample = prev_sample + alpha_t * (dt + s_val * _er_sde_noise_scaler(er_lam_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lam_pos - er_lam_s) / scaled_pos) * lam_step
                    er_lam_prev2 = self._er_lambdas[max(idx - 2, 0)]
                    denoised_u = (denoised_d - self._old_denoised_d) / ((er_lam_s - er_lam_prev2) / 2.0)
                    prev_sample = prev_sample + alpha_t * ((dt ** 2) / 2.0 + s_u * _er_sde_noise_scaler(er_lam_t)) * denoised_u
                self._old_denoised_d = denoised_d

            if s_noise > 0:
                noise = self._noise_sampler(sigma, sigma_next)
                noise_coeff = (er_lam_t ** 2 - er_lam_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                prev_sample = prev_sample + alpha_t * noise * s_noise * noise_coeff

        self._old_denoised = denoised
        self._step_index += 1
        return (prev_sample,)
