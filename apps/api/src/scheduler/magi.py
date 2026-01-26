import torch
from src.scheduler.scheduler import SchedulerInterface
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from typing import Optional, Union, Tuple


@dataclass
class MagiSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class MagiScheduler(SchedulerInterface):
    def __init__(
        self,
        num_inference_steps=64,
        num_train_timesteps=1000,
        shift=3.0,
        scheduler_type="sd3",
        shortcut_mode="16,16,8",
        clean_t: float = 0.9999,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.scheduler_type = scheduler_type
        self.time_interval = None
        self.timesteps = None
        self.denoise_step_per_stage = None
        self.chunk_width = None
        self.set_timesteps(num_inference_steps, shortcut_mode=shortcut_mode)
        self.clean_t = clean_t
        self.device = None

    def set_scheduler_params(self, chunk_width: int, denoise_step_per_stage: int):
        self.chunk_width = chunk_width
        self.denoise_step_per_stage = denoise_step_per_stage

    def set_timesteps(
        self,
        num_inference_steps=100,
        device=None,
        shortcut_mode="16,16,8",
    ):

        if device is not None:
            self.device = device

        if num_inference_steps == 12:
            base_t = torch.linspace(0, 1, 4 + 1, device=device) / 4
            accu_num = torch.linspace(0, 1, 4 + 1, device=device)
            if shortcut_mode == "16,16,8":
                base_t = base_t[:3]
            else:
                base_t = torch.cat([base_t[:1], base_t[2:4]], dim=0)
            t = torch.cat([base_t + accu for accu in accu_num], dim=0)[
                : (num_inference_steps + 1)
            ]
        else:
            t = torch.linspace(0, 1, num_inference_steps + 1, device=device)

        t_schedule_func = self.scheduler_type or "sd3"

        if t_schedule_func == "sd3":

            def t_resolution_transform(x, shift=3.0):
                # sd3: with a **reverse** time-schedule (0: clean, 1: noise)
                # ours (0: noise, 1: clean)
                # https://github.com/Stability-AI/sd3-ref/blob/master/sd3_impls.py#L33
                assert shift >= 1.0, "shift should >=1"
                shift_inv = 1.0 / shift
                return shift_inv * x / (1 + (shift_inv - 1) * x)

            t = t**2
            shift = self.shift
            t = t_resolution_transform(t, shift)
        elif t_schedule_func == "square":
            t = t**2
        elif t_schedule_func == "piecewise":

            def t_transform(x):
                mask = x < 0.875
                x[mask] = x[mask] * (0.5 / 0.875)
                x[~mask] = 0.5 + (x[~mask] - 0.875) * (0.5 / (1 - 0.875))
                return x

            t = t_transform(t)
        self.timesteps = t
        return t

    def step(
        self, sample, model_output, t_start, t_end, i, return_dict=False, **kwargs
    ):
        t_before = self.get_timestep(t_start, t_end, i)
        t_after = self.get_timestep(t_start, t_end, i + 1)
        delta_t = t_after - t_before
        N, C, T, H, W = sample.shape
        sample = sample.reshape(N, C, -1, self.chunk_width, H, W)
        model_output = model_output.reshape(N, C, -1, self.chunk_width, H, W)
        assert sample.size(2) == delta_t.size(0)
        delta_t = delta_t.reshape(1, 1, -1, 1, 1, 1)
        sample = sample + model_output * delta_t
        sample = sample.reshape(N, C, T, H, W)
        if return_dict:
            return MagiSchedulerOutput(prev_sample=sample)
        else:
            return (sample,)

    def get_timestep(
        self,
        start: int,
        end: int,
        denoise_idx: int,
        has_clean_t: bool = False,
        denoise_step_per_stage: int = None,
    ) -> torch.Tensor:
        """Const Method"""
        if denoise_step_per_stage is None:
            denoise_step_per_stage = self.denoise_step_per_stage
        else:
            self.denoise_step_per_stage = denoise_step_per_stage

        assert denoise_step_per_stage is not None, "denoise_step_per_stage must be set"
        t_index = []
        for i in range(start, end):
            t_index.append(i * denoise_step_per_stage + denoise_idx)
        t_index.reverse()
        timestep = self.timesteps[t_index]
        if has_clean_t:
            ones = torch.ones(1, device=timestep.device) * self.clean_t
            timestep = torch.cat([ones, timestep], 0)
        return timestep

    def set_time_interval(
        self, num_steps: int, device: torch.device, shortcut_mode: str = "16,16,8"
    ):
        """Set time interval"""
        base_time_interval = torch.ones(num_steps, device=device)
        if num_steps % 3 == 0:
            repeat_times = num_steps // 3
            if shortcut_mode == "16,16,8":
                base_time_interval = torch.tensor(
                    [1, 1, 2] * repeat_times, device=device
                )
            else:
                base_time_interval = torch.tensor(
                    [2, 1, 1] * repeat_times, device=device
                )
        self.time_interval = base_time_interval
        return base_time_interval
