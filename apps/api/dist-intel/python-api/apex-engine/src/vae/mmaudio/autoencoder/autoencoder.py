from typing import Literal
import torch
import torch.nn as nn

from .vae import VAE, get_my_vae
from .distributions import DiagonalGaussianDistribution
from ..bigvgan import BigVGAN


class AutoEncoderModule(nn.Module):
    def __init__(
        self,
        *,
        mode: Literal["16k"],
        need_vae_encoder: bool = True,
    ):
        """
        Lightweight container for the MMAudio VAE + vocoder stack.
        Checkpoint paths are optional so this can be used in a diffusers-style
        workflow where weights may be loaded externally (e.g. via `from_pretrained`)
        instead of being hard-wired into the constructor.
        """
        super().__init__()

        # --- VAE ---
        self.vae: VAE = get_my_vae(mode).eval()
        self.vocoder = BigVGAN().eval()
        self.weight_norm_removed = False

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    def remove_weight_norm(self):
        if not self.weight_norm_removed:
            self.vae.remove_weight_norm()
            self.vocoder.remove_weight_norm()
            self.weight_norm_removed = True

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
