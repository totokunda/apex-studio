import torch
from typing import Optional, Tuple, List, Dict
import random
import math


class BaseSchedule:
    def __init__(
        self,
        context_decay_steps: int = 1000,
        context_decay_max: float = 0.5,
        context_decay_min: float = 0.10,
    ):
        self.timesteps = 0
        self.context_decay_steps = context_decay_steps
        self.context_decay_max = context_decay_max
        self.context_decay_min = context_decay_min
        self._percent_no_context = self.context_decay_max
        self.is_training = True

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def __call__(
        self,
        past_latents: Optional[torch.Tensor] = None,
        past_indices: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        future_indices: Optional[torch.Tensor] = None,
        total_frames: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        raise NotImplementedError

    def create_target_latent_list(
        self, latents: torch.Tensor, target_indices: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        raise NotImplementedError

    @property
    def tail_handling_mode(self) -> str:
        return "delete"

    @property
    def tail_at_start(self) -> bool:
        return False

    @property
    def tail_factor(self) -> int:
        return 4

    @property
    def num_indices(self) -> int:
        raise NotImplementedError

    def num_sections(self, total_frames: int) -> int:
        raise NotImplementedError

    def get_train_inputs(self, latent: torch.Tensor) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        raise NotImplementedError

    def get_inference_inputs(
        self, latent: torch.Tensor, denoising_mask: torch.BoolTensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        raise NotImplementedError

    def decay_no_context(self):
        self.timesteps += 1
        if self.timesteps < self.context_decay_steps:
            # Apply cosine decay from max to min
            progress = self.timesteps / self.context_decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decay_range = self.context_decay_max - self.context_decay_min
            self.percent_no_context = (
                self.context_decay_min + cosine_decay * decay_range
            )
        else:
            self.percent_no_context = self.context_decay_min

    @property
    def percent_no_context(self):
        return self._percent_no_context

    @percent_no_context.setter
    def percent_no_context(self, value):
        self._percent_no_context = value

    def set_percent_no_context_timestep(self, timestep: int):
        # set the percent based on the timestep
        self.timesteps = timestep - 1
        self.decay_no_context()


##########################
class Schedule_F2K1_G9_F1K1F2K2F16K4F32K8(BaseSchedule):
    def __init__(
        self,
        context_decay_steps: int = 1000,
        context_decay_max: float = 0.5,
        context_decay_min: float = 0.10,
    ):
        super().__init__(context_decay_steps, context_decay_max, context_decay_min)

    @property
    def num_indices(self):
        return 9

    def __call__(
        self,
        past_latents: Optional[torch.Tensor] = None,
        past_indices: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        future_indices: Optional[torch.Tensor] = None,
        total_frames: Optional[int] = None,
    ):
        past_context = []
        future_context = []

        if past_latents is not None and past_indices is not None:
            past_context.append(
                (
                    past_latents[:, :, past_indices, :, :],
                    past_indices,
                    1,
                )
            )

        if future_latents is not None and future_indices is not None:
            future_indices_dict = self.split_tensor_future(future_indices)
            max_future_index = future_indices[-1]
            for key in future_indices_dict:
                if future_indices_dict[key] is not None:
                    compression_factor = key.split("_")[1]
                    indices = future_indices_dict[key]
                    future_context.append(
                        (
                            future_latents[:, :, indices - max_future_index, :, :],
                            indices,
                            compression_factor,
                        )
                    )

        return past_context + future_context

    def get_train_inputs(self, latent: torch.Tensor):
        B, C, T, H, W = latent.shape

        if random.random() < self.percent_no_context:
            target_indices = torch.arange(2, device=latent.device, dtype=torch.long)
            target_latents = latent[:, :, target_indices, :, :]
            past_indices = None
            future_indices = None
            past_latents = None
            future_latents = None
        else:
            min_rand_index = max(2, T - self.num_indices) + 1
            target_index = torch.randint(2, min_rand_index, (1,)).item()
            max_target_index = min(target_index + self.num_indices, T)

            target_indices = torch.arange(
                target_index, max_target_index, device=latent.device, dtype=torch.long
            )

            target_latents = latent[:, :, target_indices, :, :]

            future_indices = torch.arange(
                max_target_index, T, device=latent.device, dtype=torch.long
            )

            past_indices = torch.arange(2, device=latent.device, dtype=torch.long)

            if past_indices is None or past_indices.shape[0] == 0:
                past_latents = None
                past_indices = None
            else:
                past_latents = latent[:, :, past_indices, :, :]

            if future_indices is None or future_indices.shape[0] == 0:
                future_latents = None
                future_indices = None
            else:
                future_latents = latent[:, :, future_indices, :, :]

        return (
            past_latents,
            past_indices,
            future_latents,
            future_indices,
            target_latents,
            target_indices,
        )

    def get_inference_inputs(
        self,
        latent: torch.Tensor,
        denoising_mask: torch.BoolTensor,
        reverse=False,
        **kwargs,
    ):
        B, C, T, H, W = latent.shape

        latent_indices = torch.arange(0, T, device=latent.device)

        if denoising_mask.all():
            return None, None, None, None, None, None

        if not denoising_mask.any():
            target_indices = latent_indices[:2]
            seeds = kwargs.get("seeds", [None]) or [None]
            generators = [
                torch.Generator(device=latent.device) for _ in range(len(seeds))
            ]
            targets = []
            for i, seed in enumerate(seeds):
                if seed is not None:
                    generators[i].manual_seed(seed)
                targets.append(
                    torch.randn(
                        latent[0:1, :, target_indices, :, :].size(),
                        device=latent.device,
                        dtype=latent.dtype,
                        generator=generators[i],
                    )
                )
            target_latents = torch.cat(targets, dim=0)
            past_indices = None
            future_indices = None
            past_latents = None
            future_latents = None

        else:
            end_index = T - 1

            while denoising_mask[end_index]:
                end_index -= 1

            end_index += 1
            start_index = max(end_index - self.num_indices, 2)
            target_indices = latent_indices[start_index:end_index]
            target_latents = latent[:, :, target_indices, :, :]

            future_indices = latent_indices[end_index:]
            future_latents = latent[:, :, future_indices, :, :]

            if future_indices.shape[0] == 0:
                future_latents = None
                future_indices = None

            past_indices = latent_indices[:2]
            past_latents = latent[:, :, past_indices, :, :]

        return (
            past_latents,
            past_indices,
            future_latents,
            future_indices,
            target_latents,
            target_indices,
        )

    def num_sections(self, total_frames: int) -> int:
        return math.ceil((total_frames - 2) / self.num_indices) + 1

    def _split_tensor(
        self,
        frames: torch.Tensor,
        reverse: bool = False,
        limits: List[int] = [1, 2, 16, 32],
        keys: List[str] = ["F_1", "F_2", "F_4", "F_8"],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Generic splitter for both â€œpastâ€ and â€œfutureâ€ modes.
        Splits a 1D tensor (length â‰¤ 20) into up to four segments:
          â€“ F_1: 1 frame
          â€“ F_2: next 2 frames
          â€“ F_4: next up to 16 frames
          â€“ F_8: any remaining frames
        If reverse=True, iterates from the end (for â€œpastâ€). Otherwise, from the start (for â€œfutureâ€).
        Within each split, frames are ordered in ascendingâ€time order.
        """
        # Maximum counts per bucket (F_1, F_2, F_4, â€œall elseâ€)

        splits: Dict[str, Optional[torch.Tensor]] = {k: None for k in keys}
        counts = {k: 0 for k in keys}

        idx_iter = (
            range(frames.shape[0] - 1, -1, -1) if reverse else range(frames.shape[0])
        )

        for i in idx_iter:
            chunk = frames[i : i + 1].clone()  # shape = [1]
            for limit, key in zip(limits, keys):
                if counts[key] < limit:
                    if splits[key] is None:
                        splits[key] = chunk
                    else:
                        if reverse:
                            # Prepend to build ascending order when iterating backward
                            splits[key] = torch.cat([chunk, splits[key]], dim=0)
                        else:
                            # Append when iterating forward
                            splits[key] = torch.cat([splits[key], chunk], dim=0)
                    counts[key] += 1
                    break

        return splits

    def split_tensor_past(
        self,
        frames: torch.Tensor,
        limits: List[int] = [1, 2, 16, 32],
        keys: List[str] = ["F_1", "F_2", "F_4", "F_8"],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Split a 1D tensor (len â‰¤ 20) into F_1, F_2, F_4, F_8 â€œpastâ€ buckets.
        Always includes the last frame as F_1, then the two before as F_2, next up to sixteen as F_4, and the remainder as F_8.
        """
        return self._split_tensor(frames, reverse=True, limits=limits, keys=keys)

    def split_tensor_future(
        self,
        frames: torch.Tensor,
        limits: List[int] = [1, 2, 16, 32],
        keys: List[str] = ["F_1", "F_2", "F_4", "F_8"],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Split a 1D tensor (len â‰¤ 20) into F_1, F_2, F_4, F_8 â€œfutureâ€ buckets.
        Always includes the first frame as F_1, then the next two as F_2, next up to sixteen as F_4, and the remainder as F_8.
        """
        return self._split_tensor(frames, reverse=False, limits=limits, keys=keys)


class Schedule_F2K1_G9_F1K1F2K2F16K4(Schedule_F2K1_G9_F1K1F2K2F16K4F32K8):
    def __init__(
        self,
        context_decay_steps: int = 1000,
        context_decay_max: float = 0.5,
        context_decay_min: float = 0.10,
    ):
        super().__init__(context_decay_steps, context_decay_max, context_decay_min)

    def __call__(
        self,
        past_latents: Optional[torch.Tensor] = None,
        past_indices: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        future_indices: Optional[torch.Tensor] = None,
        total_frames: Optional[int] = None,
    ):
        past_context = []
        future_context = []

        if past_latents is not None and past_indices is not None:
            past_context.append(
                (
                    past_latents[:, :, past_indices, :, :],
                    past_indices,
                    1,
                )
            )

        if future_latents is not None and future_indices is not None:
            future_indices_dict = self.split_tensor_future(
                future_indices, limits=[1, 2, 16], keys=["F_1", "F_2", "F_4"]
            )
            max_future_index = future_indices[-1]
            for key in future_indices_dict:
                if future_indices_dict[key] is not None:
                    compression_factor = key.split("_")[1]
                    indices = future_indices_dict[key]
                    future_context.append(
                        (
                            future_latents[:, :, indices - max_future_index, :, :],
                            indices,
                            compression_factor,
                        )
                    )
        else:
            return past_context

        context = (past_context + future_context)[::-1]
        # swap the second last and last elements
        context[-2], context[-1] = context[-1], context[-2]
        return context
