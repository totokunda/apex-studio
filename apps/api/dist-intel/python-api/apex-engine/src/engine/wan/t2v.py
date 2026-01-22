import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared


class WanT2VEngine(WanShared):
    """WAN Text-to-Video Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        high_noise_guidance_scale: float = 1.0,
        low_noise_guidance_scale: float = 1.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        progress_callback: Callable | None = None,
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        render_on_step_interval: int = 1,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = 0.875,
        expand_timesteps: bool = False,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor = None,
        enhance_kwargs: Dict[str, Any] = {},
        chunking_profile: str = "none",
        rope_on_cpu: bool = False,
        **kwargs,
    ):

        if (
            high_noise_guidance_scale is not None
            and low_noise_guidance_scale is not None
        ):
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]
            safe_emit_progress(
                progress_callback, 0.01, "Using high/low-noise guidance scales"
            )

        safe_emit_progress(progress_callback, 0.0, "Starting text-to-video pipeline")
        if guidance_scale is not None and isinstance(guidance_scale, list):
            use_cfg_guidance = (
                negative_prompt is not None
                and guidance_scale[0] > 1.0
                and guidance_scale[1] > 1.0
            )
        else:
            use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0

        if num_inference_steps <= 8:
            render_on_step = False

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.02, "Loading text encoder")
            self.load_component_by_type("text_encoder")

        safe_emit_progress(progress_callback, 0.03, "Moving text encoder to device")
        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Encoding prompt")

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            safe_emit_progress(progress_callback, 0.11, "Encoding negative prompt")
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None

        safe_emit_progress(
            progress_callback,
            0.13,
            (
                "Prepared negative prompt"
                if negative_prompt_embeds is not None
                else "Skipped negative prompt"
            ),
        )

        if offload:
            safe_emit_progress(progress_callback, 0.14, "Offloading text encoder")
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Text encoder offloaded")

        transformer_dtype = self.component_dtypes["transformer"]
        safe_emit_progress(progress_callback, 0.16, "Moving embeddings to device")
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.17, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.18, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.19, "Moving scheduler to device")
        self.to_device(self.scheduler)
        scheduler = self.scheduler

        safe_emit_progress(progress_callback, 0.195, "Configuring scheduler timesteps")
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        safe_emit_progress(progress_callback, 0.198, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(progress_callback, 0.20, "Scheduler and timesteps prepared")

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )

        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

        safe_emit_progress(progress_callback, 0.26, "Initializing latent noise")
        
        if expand_timesteps:
            height = (height // 32) * 32
            width = (width // 32) * 32

        latents = self._get_latents(
            height,
            width,
            duration,
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        safe_emit_progress(progress_callback, 0.30, "Initialized latent noise")

        if boundary_ratio is not None:
            safe_emit_progress(progress_callback, 0.32, "Computing boundary timestep")
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        # Set preview context for per-step rendering on the main engine when available
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoise (CFG: {'on' if use_cfg_guidance else 'off'})",
        )
        mask = torch.ones(latents.shape, dtype=torch.float32, device=self.device)

        latents = self.denoise(
            expand_timesteps=expand_timesteps,
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            mask=mask,
            latents=latents,
            latent_condition=None,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
                rope_on_cpu=rope_on_cpu,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    enhance_kwargs=enhance_kwargs,
                    rope_on_cpu=rope_on_cpu,
                )
                if negative_prompt_embeds is not None
                else None
            ),
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=use_cfg_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            render_on_step_interval=render_on_step_interval,
            denoise_progress_callback=denoise_progress_callback,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            ip_image=ip_image,
            chunking_profile=chunking_profile,
        )

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            safe_emit_progress(progress_callback, 0.94, "Decoding latents to video")
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents to video")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(
                progress_callback, 1.0, "Completed text-to-video pipeline"
            )
            return postprocessed_video
