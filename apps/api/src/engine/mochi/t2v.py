import torch
from typing import Dict, Any, Callable, List, Union, Optional
import numpy as np
from src.engine.base_engine import BaseEngine


def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return np.array([1.0])
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return np.array(sigma_schedule)


class MochiT2VEngine(BaseEngine):
    """Mochi Text-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, model_type=ModelType.T2V, **kwargs)
        self.vae_scale_factor_spatial = (
            self.vae.config.get("scaling_factor", 8) if self.vae else 8
        )
        self.vae_scale_factor_temporal = 6  # Mochi specific

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.num_channels_latents = (
            self.transformer.config.in_channels if self.transformer else 12
        )

    def vae_decode(
        self, latents: torch.Tensor, offload: bool = False, dtype: torch.dtype = None
    ):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        # unscale/denormalize the latents
        has_latents_mean = (
            hasattr(self.vae.config, "latents_mean")
            and self.vae.config.latents_mean is not None
        )
        has_latents_std = (
            hasattr(self.vae.config, "latents_std")
            and self.vae.config.latents_std is not None
        )
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.num_channels_latents, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.num_channels_latents, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = (
                latents * latents_std / self.vae.config.scaling_factor + latents_mean
            )
        else:
            latents = latents / self.vae.config.scaling_factor

        video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

        if offload:
            self._offload("vae")

        return video.to(dtype=dtype)

    def run(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 848,
        duration: int = 97,
        num_inference_steps: int = 64,
        guidance_scale: float = 4.5,
        num_videos: int = 1,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_latents: bool = False,
        offload: bool = True,
        render_on_step: bool = False,
        render_on_step_callback: Optional[Callable] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 256,
        threshold_noise: float = 0.025,
        fps: int = 30,
        **kwargs,
    ):
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        transformer_dtype = self.component_dtypes.get("transformer")

        prompt_embeds, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            return_attention_mask=True,
        )

        batch_size = prompt_embeds.shape[0]

        use_cfg_guidance = guidance_scale > 1.0
        if use_cfg_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos,
                    max_sequence_length=max_sequence_length,
                    return_attention_mask=True,
                )
            )
            prompt_embeds = (
                torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                .to(self.device)
                .to(transformer_dtype)
            )
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            ).to(self.device)
        else:
            prompt_embeds = prompt_embeds.to(self.device).to(transformer_dtype)
            prompt_attention_mask = prompt_attention_mask.to(self.device)

        if offload:
            self._offload("text_encoder")

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        num_frames = self._parse_num_frames(duration, fps=fps)
        latent_num_frames = int((num_frames - 1) // self.vae_scale_factor_temporal + 1)

        if latents is None:
            latents = self._get_latents(
                height=height,
                width=width,
                duration=latent_num_frames,
                batch_size=batch_size,
                num_channels_latents=self.num_channels_latents,
                dtype=torch.float32,
                generator=generator,
                seed=seed,
                parse_frames=False,
            )

        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)

        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising Mochi T2V"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if use_cfg_guidance:
                    latent_model_input = torch.cat([latents] * 2).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = (
                    t.expand(latent_model_input.shape[0])
                    .to(self.device)
                    .to(transformer_dtype)
                )

                if hasattr(self.transformer, "cache_context"):
                    cache_context = self.transformer.cache_context("cond_uncond")
                else:
                    cache_context = nullcontext()

                with cache_context:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        encoder_attention_mask=prompt_attention_mask,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred.to(torch.float32)

                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = scheduler.step(
                    noise_pred, t, latents.to(torch.float32), return_dict=False
                )[0].to(transformer_dtype)

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    pbar.update(1)

        self.logger.info("Denoising completed.")

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents

        video = self.vae_decode(latents, offload=offload, dtype=prompt_embeds.dtype)
        video = self._tensor_to_frames(video, output_type=output_type)
        return video
