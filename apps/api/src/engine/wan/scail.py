import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.types import InputVideo, InputImage
import torch.nn.functional as F


class WanSCAILEngine(WanShared):
    """WAN SCAIL Engine Implementation"""

    def run(
        self,
        image: InputImage,
        pose_video: InputVideo,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        use_video_duration: bool = True,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
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
        enhance_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting image-to-video pipeline")
        num_frames = self._parse_num_frames(duration, fps)

        use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.02, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.03, "Text encoder loaded")

        safe_emit_progress(progress_callback, 0.04, "Moving text encoder to device")
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
            0.14,
            (
                "Prepared negative prompt"
                if negative_prompt_embeds is not None
                else "Skipped negative prompt"
            ),
        )

        if offload:
            safe_emit_progress(progress_callback, 0.15, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 0.16, "Text encoder offloaded")

        safe_emit_progress(
            progress_callback, 0.17, "Loading inputs (image + pose video)"
        )
        loaded_image = self._load_image(image)
        pose_video = self._load_video(
            pose_video,
            fps=fps,
            num_frames=num_frames if use_video_duration else None,
        )
        if use_video_duration:
            num_frames = len(pose_video)
        original_height, original_width = height, width
        safe_emit_progress(progress_callback, 0.18, "Preprocessing pose video")
        for idx, frame in enumerate(pose_video):
            frame, height, width = self._aspect_ratio_resize(
                frame, max_area=original_height * original_width, mod_value=32
            )
            frame, height, width = self._center_crop_resize(frame, height, width)
        pose_video = self.video_processor.preprocess_video(
            pose_video, height=height, width=width
        ).to(self.device, dtype=torch.float32)
        pose_video = F.interpolate(
            pose_video.squeeze(0),
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(0)

        safe_emit_progress(progress_callback, 0.19, "Preprocessing reference image")
        loaded_image, _, _ = self._aspect_ratio_resize(
            loaded_image, max_area=original_height * original_width, mod_value=32
        )

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        transformer_dtype = self.component_dtypes["transformer"]

        safe_emit_progress(progress_callback, 0.20, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.21, "Encoding image with CLIP")
        image_embeds = self.helpers["clip"](loaded_image, hidden_states_layer=-2).to(
            self.device, dtype=transformer_dtype
        )

        safe_emit_progress(progress_callback, 0.22, "Moving embeddings to device")
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload and boundary_ratio is None and not expand_timesteps:
            safe_emit_progress(progress_callback, 0.23, "Offloading CLIP")
            self._offload("clip")

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.24, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.25, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.26, "Moving scheduler to device")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        safe_emit_progress(progress_callback, 0.27, "Configuring scheduler timesteps")
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        safe_emit_progress(progress_callback, 0.28, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(progress_callback, 0.30, "Scheduler and timesteps prepared")

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

        safe_emit_progress(progress_callback, 0.32, "Initializing latent noise")
        latents = self._get_latents(
            height,
            width,
            num_frames,
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        safe_emit_progress(progress_callback, 0.36, "Initialized latent noise")

        if preprocessed_image.ndim == 4:
            preprocessed_image = preprocessed_image.unsqueeze(2)

        safe_emit_progress(progress_callback, 0.38, "Encoding reference image (VAE)")
        reference_latents = self.vae_encode(
            preprocessed_image,
            offload=offload,
            dtype=latents.dtype,
            normalize_latents_dtype=latents.dtype,
            offload_type="cpu",
        )

        safe_emit_progress(progress_callback, 0.42, "Encoding pose video (VAE)")
        pose_latents = self.vae_encode(
            pose_video,
            offload=offload,
            dtype=latents.dtype,
            normalize_latents_dtype=latents.dtype,
            offload_type="discard",
        )

        batch_size, _, num_latent_frames, latent_height, latent_width = latents.shape

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoising (CFG: {'on' if use_cfg_guidance else 'off'})",
        )
        transformer_config = self.load_config_by_type("transformer")
        max_seq_len = (
            num_latent_frames
            * latent_height
            * latent_width
            // (
                transformer_config.get("patch_size", (1, 2, 2))[1]
                * transformer_config.get("patch_size", (1, 2, 2))[2]
            )
        )

        with torch.autocast(device_type=self.device.type, dtype=transformer_dtype):
            latents = self.denoise(
                timesteps=timesteps,
                latents=latents,
                transformer_kwargs=dict(
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_clip=image_embeds,
                    encoder_hidden_states_pose=pose_latents,
                    encoder_hidden_states_reference=reference_latents,
                    attention_kwargs=attention_kwargs,
                    seq_len=max_seq_len,
                    enhance_kwargs=enhance_kwargs,
                ),
                unconditional_transformer_kwargs=(
                    dict(
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_clip=image_embeds,
                        encoder_hidden_states_pose=pose_latents,
                        encoder_hidden_states_reference=reference_latents,
                        seq_len=max_seq_len,
                        attention_kwargs=attention_kwargs,
                        enhance_kwargs=enhance_kwargs,
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
            )

        if offload:
            safe_emit_progress(progress_callback, 0.91, "Offloading transformer")
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            safe_emit_progress(progress_callback, 0.94, "Decoding latents to video")
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(
                progress_callback, 1.0, "Completed image-to-video pipeline"
            )
            return postprocessed_video
