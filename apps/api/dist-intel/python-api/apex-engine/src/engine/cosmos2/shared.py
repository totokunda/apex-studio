import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as F
from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor


class Cosmos2Shared(BaseEngine):
    """Base class for Cosmos engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 4
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 8
        )

        self.num_channels_latents = getattr(self.vae, "config", {}).get("z_dim", 16)

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        sigmas = kwargs.get("sigmas")
        guidance_scale = kwargs.get("guidance_scale")
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        num_inference_steps = kwargs.get("num_inference_steps")
        cond_mask = kwargs.get("cond_mask")
        uncond_mask = kwargs.get("uncond_mask")
        conditioning_latents = kwargs.get("conditioning_latents")
        unconditioning_latents = kwargs.get("unconditioning_latents")
        padding_mask = kwargs.get("padding_mask")
        t_conditioning = kwargs.get("t_conditioning")
        cond_indicator = kwargs.get("cond_indicator")
        transformer_dtype = kwargs.get("transformer_dtype", torch.bfloat16)
        fps = kwargs.get("fps", 16)
        prompt_embeds = kwargs.get("prompt_embeds")
        uncond_indicator = kwargs.get("uncond_indicator", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        render_on_step = kwargs.get("render_on_step", False)

        with self._progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                current_sigma = sigmas[i].to(latents.device)
                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]

                cond_latent = latents * c_in
                cond_latent = (
                    cond_indicator * conditioning_latents
                    + (1 - cond_indicator) * cond_latent
                )
                cond_latent = cond_latent.to(transformer_dtype)

                cond_timestep = (
                    cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                )
                cond_timestep = cond_timestep.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=cond_latent,
                    timestep=cond_timestep,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(
                    transformer_dtype
                )
                noise_pred = (
                    cond_indicator * conditioning_latents
                    + (1 - cond_indicator) * noise_pred
                )

                if use_cfg_guidance:
                    uncond_latent = latents * c_in
                    uncond_latent = (
                        uncond_indicator * unconditioning_latents
                        + (1 - uncond_indicator) * uncond_latent
                    )
                    uncond_latent = uncond_latent.to(transformer_dtype)
                    uncond_timestep = (
                        uncond_indicator * t_conditioning
                        + (1 - uncond_indicator) * timestep
                    )
                    uncond_timestep = uncond_timestep.to(transformer_dtype)

                    noise_pred_uncond = self.transformer(
                        hidden_states=uncond_latent,
                        timestep=uncond_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        fps=fps,
                        condition_mask=uncond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = (
                        c_skip * latents + c_out * noise_pred_uncond.float()
                    ).to(transformer_dtype)
                    noise_pred_uncond = (
                        uncond_indicator * unconditioning_latents
                        + (1 - uncond_indicator) * noise_pred_uncond
                    )

                    noise_pred = noise_pred + guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )

                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                pbar.update(1)

        return latents
