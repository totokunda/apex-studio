import functools
import inspect
import math
from typing import Optional

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from src.utils.dtype import supports_double


def _ensure_step_output_consistency(step_fn):
    """
    Decorator that guarantees a scheduler ``step`` (or ``step_from_to``)
    function's output tensors have the **same dtype and shape** as the
    input ``model_output`` tensor.

    Supports every return convention used across our schedulers:
      - raw ``torch.Tensor``
      - tuple of tensors, e.g. ``(prev_sample,)``
      - dataclass / named-tuple with a ``.prev_sample`` attribute
    """
    # Determine the positional index of 'model_output' in the wrapped
    # function's signature so we can locate it whether it is passed
    # positionally or as a keyword argument.
    sig = inspect.signature(step_fn)
    param_names = list(sig.parameters.keys())
    try:
        _mo_pos = param_names.index("model_output")
    except ValueError:
        _mo_pos = 1  # default: first parameter after 'self'
    # In the wrapper *args, index 0 corresponds to param_names[1]
    # (because 'self' is consumed by the method binding).
    _mo_args_idx = _mo_pos - 1

    @functools.wraps(step_fn)
    def wrapper(self, *args, **kwargs):
        # --- resolve model_output ------------------------------------------------
        model_output = kwargs.get("model_output")
        if model_output is None and len(args) > _mo_args_idx >= 0:
            model_output = args[_mo_args_idx]

        if not isinstance(model_output, torch.Tensor):
            return step_fn(self, *args, **kwargs)

        ref_dtype = model_output.dtype
        ref_shape = model_output.shape

        # --- run the real step ----------------------------------------------------
        result = step_fn(self, *args, **kwargs)

        # --- enforce dtype + shape on every returned tensor ----------------------
        ref_numel = ref_shape.numel()

        def _fix(t):
            if not isinstance(t, torch.Tensor):
                return t
            if t.shape != ref_shape:
                # Sigma broadcasting (e.g. reshape(-1,1,1,1) on 3-D inputs)
                # can introduce extra singleton dims.  If the element count
                # matches we can cheaply reshape back; otherwise it is a
                # genuine bug.
                if t.numel() == ref_numel:
                    t = t.view(ref_shape)
                else:
                    raise ValueError(
                        f"{type(self).__name__}.step output shape {t.shape} "
                        f"(numel={t.numel()}) does not match model_output "
                        f"shape {ref_shape} (numel={ref_numel})"
                    )
            return t.to(dtype=ref_dtype) if t.dtype != ref_dtype else t

        if isinstance(result, torch.Tensor):
            return _fix(result)
        if isinstance(result, tuple):
            return tuple(_fix(x) for x in result)
        # dataclass / NamedTuple with .prev_sample
        if hasattr(result, "prev_sample") and isinstance(
            getattr(result, "prev_sample", None), torch.Tensor
        ):
            result.prev_sample = _fix(result.prev_sample)
            if hasattr(result, "pred_original_sample") and isinstance(
                getattr(result, "pred_original_sample", None), torch.Tensor
            ):
                result.pred_original_sample = _fix(result.pred_original_sample)
            return result
        return result

    return wrapper


class SchedulerInterface(SchedulerMixin, ConfigMixin):
    """
    Base class for diffusion noise schedule.
    """

    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method_name in ("step", "step_from_to"):
            if method_name in cls.__dict__:
                original = cls.__dict__[method_name]
                if callable(original):
                    setattr(
                        cls,
                        method_name,
                        _ensure_step_output_consistency(original),
                    )

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def convert_x0_to_noise(
        self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        
        Convert the diffusion network's x0 prediction to noise predidction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        tensor_to_double = lambda x: (
            x.double() if supports_double(x.device) else x.to(torch.float32)
        )
        x0, xt, alphas_cumprod = map(
            lambda x: tensor_to_double(x.to(x0.device)), [x0, xt, self.alphas_cumprod]
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 predidction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        tensor_to_double = lambda x: (
            x.double() if supports_double(x.device) else x.to(torch.float32)
        )
        
        noise, xt, alphas_cumprod = map(
            lambda x: tensor_to_double(x.to(noise.device)),
            [noise, xt, self.alphas_cumprod],
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 predidction.
        velocity: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        tensor_to_double = lambda x: (
            x.double() if supports_double(x.device) else x.to(torch.float32)
        )
        velocity, xt, alphas_cumprod = map(
            lambda x: tensor_to_double(x.to(velocity.device)),
            [velocity, xt, self.alphas_cumprod],
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t**0.5) * xt - (beta_prod_t**0.5) * velocity
        return x0_pred.to(original_dtype)

    def convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype

        tensor_to_double = lambda x: (
            x.double() if supports_double(x.device) else x.to(torch.float32)
        )
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: tensor_to_double(x.to(flow_pred.device)),
            [flow_pred, xt, self.sigmas, self.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def convert_x0_to_flow_pred(
        self, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        tensor_to_double = lambda x: (
            x.double() if supports_double(x.device) else x.to(torch.float32)
        )
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: tensor_to_double(x.to(x0_pred.device)),
            [x0_pred, xt, self.sigmas, self.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample
