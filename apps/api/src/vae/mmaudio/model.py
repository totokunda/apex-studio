from typing import Literal, Optional, List

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


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class AutoencoderMMAudio(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        *,
        mode: Literal["16k"] = "16k",
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

        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(mode=mode, need_vae_encoder=need_vae_encoder)

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
