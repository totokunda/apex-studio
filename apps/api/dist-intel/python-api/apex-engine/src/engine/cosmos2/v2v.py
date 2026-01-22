from .shared import Cosmos2Shared
from typing import Dict, Any, Callable, List, Union, Optional
import torch
import numpy as np
from PIL import Image
from loguru import logger


# Referencing from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2VideoToWorldPipeline.prepare_latents
class Cosmos2V2VEngine(Cosmos2Shared):
    """Cosmos Video-to-Video Engine Implementation"""

    def run(
        self,
        video: Union[List[Image.Image], str, List[str], np.ndarray, torch.Tensor],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 704,
        width: int = 1280,
        duration: int | str = 93,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        sigma_conditioning: float = 0.0001,
        render_on_step_callback: Callable = None,
        render_on_step: bool = False,
        offload: bool = True,
        disable_guardrail: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        transformer_dtype = self.component_dtypes["transformer"]

        if self.text_encoder is None:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        ).to(transformer_dtype)

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            ).to(transformer_dtype)
        else:
            negative_prompt_embeds = None

        if offload:
            self._offload("text_encoder")

        if self.scheduler is None:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        sigmas_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps, sigmas=sigmas
        )
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            self.logger.info(
                "Replacing the last sigma (which is zero) with the minimum sigma value"
            )
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]
        sigmas = self.scheduler.sigmas

        if self.transformer is None:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        loaded_video = self._load_video(video, fps=fps)
        video = self.video_processor.preprocess_video(loaded_video, height, width)
        num_cond_frames = video.size(2)
        num_frames = self._parse_num_frames(duration, fps)

        if num_cond_frames >= num_frames:
            # Take the last `num_frames` frames for conditioning
            num_cond_latent_frames = (
                num_frames - 1
            ) // self.vae_scale_factor_temporal + 1
            video = video[:, :, -num_frames:]
        else:
            num_cond_latent_frames = (
                num_cond_frames - 1
            ) // self.vae_scale_factor_temporal + 1
            num_padding_frames = num_frames - num_cond_frames
            last_frame = video[:, :, -1:]
            padding = last_frame.repeat(1, 1, num_padding_frames, 1, 1)
            video = torch.cat([video, padding], dim=2)

        conditioning_latents = self.vae_encode(
            video,
            offload=offload,
            sample_mode="sample",
            sample_generator=generator,
            dtype=torch.float32,
        )

        batch_size = prompt_embeds.shape[0]

        latents = self._get_latents(
            height=height,
            width=width,
            duration=num_frames,
            fps=fps,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            generator=generator,
            dtype=torch.float32,
        )

        latents = latents * self.scheduler.config.sigma_max
        batch_size, _, num_latent_frames, latent_height, latent_width = latents.shape
        padding_shape = (batch_size, 1, num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        uncond_indicator = None
        uncond_mask = None

        if use_cfg_guidance:
            uncond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = (
                uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding
            )

        cond_mask = cond_mask.to(transformer_dtype)
        if use_cfg_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        sigma_conditioning = torch.tensor(
            sigma_conditioning, dtype=torch.float32, device=self.device
        )
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # 6. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            cond_mask=cond_mask,
            uncond_mask=uncond_mask,
            conditioning_latents=conditioning_latents,
            unconditioning_latents=unconditioning_latents,
            prompt_embeds=prompt_embeds,
            fps=fps,
            negative_prompt_embeds=negative_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            padding_mask=padding_mask,
            t_conditioning=t_conditioning,
            cond_indicator=cond_indicator,
            uncond_indicator=uncond_indicator,
            transformer_dtype=transformer_dtype,
            render_on_step_callback=render_on_step_callback,
            render_on_step=render_on_step,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            if "cosmos.guardrail" not in self.postprocessors and not disable_guardrail:
                self.load_postprocessor_by_type("cosmos.guardrail")

            guardrail = self.postprocessors.get("cosmos.guardrail")

            if guardrail is not None and not disable_guardrail:
                self.to_device(guardrail)
                video = guardrail(video)
            else:
                self.logger.warning(
                    "Using cosmos without the guardrail postprocessor may violates the NVIDIA Open Model License Agreement."
                )

            if guardrail is not None and offload:
                self._offload("guardrail")

            return self._tensor_to_frames(video)
