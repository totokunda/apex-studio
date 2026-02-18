from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vae.mmaudio.autoencoder import AutoEncoderModule
from diffusers.models.autoencoders.vae import (
    DiagonalGaussianDistribution,
    DecoderOutput,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from .mel_encoder import get_mel_converter
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config



class AutoencoderMMAudio(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        *,
        mode: Literal["16k", "44k"] = "16k",
        vocoder_config_path: str = None,
        vocoder_config_id: str = None,
        vocoder_config: dict = None,
        need_vae_encoder: bool = True,
        **kwargs,
    ):
        """
        Diffusers-style audio autoencoder wrapper around the custom VAE + BigVGAN stack.

        Instead of taking `model_path` / `extra_model_paths` directly in the constructor, this
        follows the usual `ModelMixin` pattern where paths are stored in the config (via
        `register_to_config`) and can be provided either when instantiating or via
        `from_pretrained` / `from_config`.
        """
        super().__init__()

        # Resolve vocoder_config_id to local path if provided
        if vocoder_config_id and not vocoder_config_path:
            from src.config_registry import resolve_config_path as _resolve
            try:
                vocoder_config_path = _resolve(vocoder_config_id)
            except FileNotFoundError:
                pass

        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(mode=mode, vocoder_config_path=vocoder_config_path, vocoder_config=vocoder_config, need_vae_encoder=need_vae_encoder)

    def compile(self):
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, "VAE is not loaded"
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.vocode(mel)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        """
        Conventional diffusers-style `decode`: latent -> waveform.
        """
        assert self.tod is not None, "VAE is not loaded"
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            mel_decoded = self.tod.decode(z)
            audio = self.tod.vocode(mel_decoded)
            if return_dict:
                return DecoderOutput(sample=audio)
            else:
                return (audio,)

    @torch.no_grad()
    def encode(self, audio, return_dict: bool = False):
        with torch.amp.autocast("cuda", dtype=self.dtype):
            dist = self.encode_audio(audio)
            if return_dict:
                return AutoencoderKLOutput(latent_dist=dist)
            else:
                return (dist.mean,)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Pass through
        return latents

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Pass through
        return latents
