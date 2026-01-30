import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class MultiModalGuiderParams:
    """
    Parameters for multimodal guidance, mirroring LTX-2's `MultiModalGuiderParams`.
    """

    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    # Which transformer blocks to perturb for STG:
    # - None => perturb all blocks
    # - []   => disable STG perturbations (even if stg_scale != 0)
    stg_blocks: Optional[Sequence[int]] = None
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    skip_step: int = 0


@dataclass(frozen=True)
class MultiModalGuider:
    """
    Multi-modal guider.

    Applies combined guidance:
      pred = cond
           + (cfg_scale - 1) * (cond - uncond_text)
           + stg_scale       * (cond - uncond_perturbed)
           + (modality_scale - 1) * (cond - uncond_modality)

    and optional rescaling to match `cond` std.
    """

    params: MultiModalGuiderParams

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
    ) -> torch.Tensor:
        pred = (
            cond
            + (float(self.params.cfg_scale) - 1.0) * (cond - uncond_text)
            + float(self.params.stg_scale) * (cond - uncond_perturbed)
            + (float(self.params.modality_scale) - 1.0) * (cond - uncond_modality)
        )

        if float(self.params.rescale_scale) != 0.0:
            # Match LTX-2's guider: scalar std ratio across the whole tensor.
            # Add eps to avoid division by zero in pathological cases.
            pred_std = pred.std()
            cond_std = cond.std()
            factor = cond_std / (pred_std + 1e-12)
            factor = float(self.params.rescale_scale) * factor + (1.0 - float(self.params.rescale_scale))
            pred = pred * factor

        return pred

    def do_unconditional_generation(self) -> bool:
        return not math.isclose(float(self.params.cfg_scale), 1.0)

    def do_perturbed_generation(self) -> bool:
        if math.isclose(float(self.params.stg_scale), 0.0):
            return False
        # README semantics: stg_blocks=[] disables STG.
        if self.params.stg_blocks is not None and len(self.params.stg_blocks) == 0:
            return False
        return True

    def do_isolated_modality_generation(self) -> bool:
        return not math.isclose(float(self.params.modality_scale), 1.0)

    def should_skip_step(self, step: int) -> bool:
        if int(self.params.skip_step) == 0:
            return False
        return step % (int(self.params.skip_step) + 1) != 0

