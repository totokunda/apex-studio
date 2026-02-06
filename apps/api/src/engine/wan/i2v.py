import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.types import InputImage


class WanI2VEngine(WanShared):
    """WAN Image-to-Video Engine Implementation"""

    def run(
        self,
        image: InputImage,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        duration: int | str = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        high_noise_guidance_scale: float = 1.0,
        low_noise_guidance_scale: float = 1.0,
        progress_callback: Callable = None,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor = None,
        enhance_kwargs: Dict[str, Any] = {},
        chunking_profile: str = "none",
        rope_on_cpu: bool = False,
        **kwargs,
    ):
        
        if expand_timesteps:
            fps = 24
            
        if (
            high_noise_guidance_scale is not None
            and low_noise_guidance_scale is not None
        ):
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]
            safe_emit_progress(
                progress_callback, 0.01, "Using high/low-noise guidance scales"
            )

        safe_emit_progress(progress_callback, 0.0, "Starting image-to-video pipeline")
        if guidance_scale is not None and isinstance(guidance_scale, list):
            use_cfg_guidance = (
                negative_prompt is not None
                and guidance_scale[0] > 1.0
                and guidance_scale[1] > 1.0
            )
        else:
            use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
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
            0.14,
            (
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Text encoder offloaded")

        loaded_image = self._load_image(image)

        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image,
            max_area=height * width,
            mod_value=32 if expand_timesteps else 16,
        )

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        transformer_dtype = self.component_dtypes["transformer"]

        if boundary_ratio is None and not expand_timesteps:
            image_embeds = self.helpers["clip"](
                loaded_image, hidden_states_layer=-2
            ).to(self.device, dtype=transformer_dtype)
        else:
            image_embeds = None

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload and boundary_ratio is None and not expand_timesteps:
            self._offload("clip")

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(
            progress_callback, 0.20, "Scheduler ready and timesteps computed"
        )

        num_frames = self._parse_num_frames(duration, fps)

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

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

        safe_emit_progress(progress_callback, 0.3, "Initialized latent noise")

        if preprocessed_image.ndim == 4:
            preprocessed_image = preprocessed_image.unsqueeze(2)

        video_condition = torch.cat(
            [
                preprocessed_image,
                preprocessed_image.new_zeros(
                    preprocessed_image.shape[0],
                    preprocessed_image.shape[1],
                    num_frames - 1,
                    height,
                    width,
                ),
            ],
            dim=2,
        )

        latent_condition = self.vae_encode(
            video_condition,
            offload=offload,
            dtype=latents.dtype,
            normalize_latents_dtype=latents.dtype,
        )

        batch_size, _, num_latent_frames, latent_height, latent_width = latents.shape

        if expand_timesteps:
            first_frame_mask = torch.ones(
                1,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=latents.dtype,
                device=latents.device,
            )
            first_frame_mask[:, :, 0] = 0
        else:
            mask_lat_size = torch.ones(
                batch_size,
                1,
                num_frames,
                latent_height,
                latent_width,
                device=self.device,
            )

            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
            )

            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(
                batch_size,
                -1,
                self.vae_scale_factor_temporal,
                latent_height,
                latent_width,
            )

            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latents.device)

            latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(progress_callback, 0.45, "Starting denoise phase")

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=latent_condition,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
                rope_on_cpu=rope_on_cpu,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
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
            denoise_progress_callback=denoise_progress_callback,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            expand_timesteps=expand_timesteps,
            first_frame_mask=first_frame_mask,
            ip_image=ip_image,
            chunking_profile=chunking_profile,
        )

        if offload:
            self._offload("transformer")
            self._offload("high_noise_transformer")
            self._offload("low_noise_transformer")
            
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents to video")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(
                progress_callback, 1.0, "Completed image-to-video pipeline"
            )
            return postprocessed_video
