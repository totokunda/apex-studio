import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torch.nn.functional as F
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared


class WanVaceEngine(WanShared):
    """WAN VACE (Video Acceleration) Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        video: (
            Union[List[Image.Image], List[str], str, np.ndarray, torch.Tensor] | None
        ) = None,
        reference_images: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        mask: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
        height: int = 480,
        width: int = 832,
        duration: int | str | None = 81,
        fps: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int | None = None,
        num_videos: int = 1,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        progress_callback: Callable | None = None,
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        return_latents: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor = None,
        enhance_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        use_cfg_guidance = guidance_scale > 1.0 and negative_prompt is not None

        safe_emit_progress(
            progress_callback, 0.0, "Starting VACE video generation pipeline"
        )

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.02, "Loading text encoder")
            self.load_component_by_type("text_encoder")

        safe_emit_progress(progress_callback, 0.03, "Moving text encoder to device")
        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.05, "Encoding prompt")

        num_frames = self._parse_num_frames(duration, fps=fps)

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

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.16, "Loading transformer")
            self.load_component_by_type("transformer")

        pt, ph, pw = self.transformer.config.patch_size
        safe_emit_progress(progress_callback, 0.17, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.18, "Transformer ready")
        transformer_dtype = self.component_dtypes["transformer"]

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.19, "Loading scheduler")
            self.load_component_by_type("scheduler")

        safe_emit_progress(progress_callback, 0.21, "Moving scheduler to device")
        self.to_device(self.scheduler)
        scheduler = self.scheduler
        safe_emit_progress(progress_callback, 0.22, "Configuring scheduler timesteps")
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )
        safe_emit_progress(progress_callback, 0.23, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(
            progress_callback, 0.24, "Scheduler and timesteps prepared"
        )

        if mask:
            safe_emit_progress(progress_callback, 0.25, "Loading mask")
            loaded_mask = self._load_video(mask, fps=fps, num_frames=num_frames)

        safe_emit_progress(progress_callback, 0.26, "Preparing conditioning scales")
        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(
                self.transformer.config.vace_layers
            )
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(
                device=self.device, dtype=transformer_dtype
            )

        if video is not None:
            safe_emit_progress(progress_callback, 0.28, "Loading input video")
            loaded_video = self._load_video(video, fps=fps, num_frames=num_frames)

            max_area = height * width
            loaded_video = [
                self._aspect_ratio_resize(frame, max_area)[0] for frame in loaded_video
            ]
            video_height, video_width = loaded_video[0].height, loaded_video[0].width

            safe_emit_progress(progress_callback, 0.30, "Preprocessing input video")
            preprocessed_video = self.video_processor.preprocess_video(
                loaded_video,
                height=video_height,
                width=video_width,
            )

            height, width = video_height, video_width
        else:
            preprocessed_video = torch.zeros(
                batch_size,
                3,
                num_frames,
                height,
                width,
                device=self.device,
                dtype=torch.float32,
            )

        if not mask:
            preprocessed_mask = torch.ones_like(preprocessed_video)
        else:
            safe_emit_progress(progress_callback, 0.32, "Preprocessing mask")
            preprocessed_mask = self.video_processor.preprocess_video(
                loaded_mask, video_height, video_width
            )
            preprocessed_mask = torch.clamp((preprocessed_mask + 1) / 2, min=0, max=1)

        safe_emit_progress(progress_callback, 0.34, "Preparing reference images")
        if reference_images is None or not isinstance(reference_images, (list, tuple)):
            if reference_images is not None:
                reference_images = self._load_image(reference_images)
            reference_images = [
                [reference_images] for _ in range(preprocessed_video.shape[0])
            ]

        elif isinstance(reference_images, (list, tuple)) and isinstance(
            next(iter(reference_images)), list
        ):
            reference_images = [
                [
                    self._load_image(image)
                    for image in reference_images_batch
                    if image is not None
                ]
                for reference_images_batch in reference_images
            ]
        elif isinstance(reference_images, (list, tuple)):
            reference_images = [
                self._load_image(image)
                for image in reference_images
                if image is not None
            ]
            reference_images = [
                reference_images if reference_images else None
                for _ in range(preprocessed_video.shape[0])
            ]

        assert reference_images is not None, "reference_images must be provided"
        assert isinstance(reference_images, list), "reference_images must be a list"
        assert (
            len(reference_images) == preprocessed_video.shape[0]
        ), "reference_images must be a list of the same length as the video"

        reference_images_preprocessed = []
        for i, reference_images_batch in enumerate(reference_images):
            preprocessed_images = []
            for j, image in enumerate(reference_images_batch):
                if image is None:
                    continue
                image = self.video_processor.preprocess(image, None, None)
                img_height, img_width = image.shape[-2:]
                scale = min(height / img_height, width / img_width)
                new_height, new_width = int(img_height * scale), int(img_width * scale)
                resized_image = torch.nn.functional.interpolate(
                    image,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(
                    0
                )  # [C, H, W]
                top = (height - new_height) // 2
                left = (width - new_width) // 2
                canvas = torch.ones(
                    3, height, width, device=self.device, dtype=torch.float32
                )
                canvas[:, top : top + new_height, left : left + new_width] = (
                    resized_image
                )
                preprocessed_images.append(canvas)
            reference_images_preprocessed.append(preprocessed_images)

        num_reference_images = len(reference_images_preprocessed[0])

        mask = torch.where(preprocessed_mask > 0.5, 1.0, 0.0)
        inactive = preprocessed_video * (1 - mask)
        reactive = preprocessed_video * mask

        safe_emit_progress(progress_callback, 0.36, "Encoding inactive/reactive regions (VAE)")
        inactive = self.vae_encode(
            inactive,
            offload=offload,
            dtype=torch.float32,
            normalize_latents_dtype=torch.float32,
        )
        reactive = self.vae_encode(
            reactive,
            offload=offload,
            dtype=torch.float32,
            normalize_latents_dtype=torch.float32,
        )

        latents = torch.cat([inactive, reactive], dim=1)

        safe_emit_progress(progress_callback, 0.40, "Encoding reference images (VAE)")
        latent_list = []
        for latent, reference_images_batch in zip(
            latents, reference_images_preprocessed
        ):
            for reference_image in reference_images_batch:
                assert reference_image.ndim == 3
                reference_image = reference_image[
                    None, :, None, :, :
                ]  # [1, C, 1, H, W]
                reference_latent = self.vae_encode(
                    reference_image,
                    offload=offload,
                    dtype=torch.float32,
                    normalize_latents_dtype=torch.float32,
                )
                reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
                reference_latent = torch.cat(
                    [reference_latent, torch.zeros_like(reference_latent)], dim=0
                )
                latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
            latent_list.append(latent)

        mask_list = []

        for mask_, reference_images_batch in zip(
            preprocessed_mask, reference_images_preprocessed
        ):
            num_channels, _num_frames, height, width = mask_.shape
            new_num_frames = (
                _num_frames + self.vae_scale_factor_temporal - 1
            ) // self.vae_scale_factor_temporal

            new_height = height // (self.vae_scale_factor_spatial * ph) * ph
            new_width = width // (self.vae_scale_factor_spatial * ph) * ph
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                _num_frames,
                new_height,
                self.vae_scale_factor_spatial,
                new_width,
                self.vae_scale_factor_spatial,
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(
                0, 1
            )  # [8x8, num_frames, new_height, new_width]
            mask_ = F.interpolate(
                mask_.unsqueeze(0),
                size=(new_num_frames, new_height, new_width),
                mode="nearest-exact",
            ).squeeze(0)
            num_ref_images = len(reference_images_batch)
            if num_ref_images > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)

        safe_emit_progress(progress_callback, 0.42, "Preparing conditioning latents/masks")
        conditioning_latents = torch.stack(latent_list, dim=0).to(self.device)
        conditioning_masks = torch.stack(mask_list, dim=0).to(self.device)

        conditioning_latents = torch.cat(
            [conditioning_latents, conditioning_masks], dim=1
        )
        conditioning_latents = conditioning_latents.to(transformer_dtype)
        prompt_embeds = prompt_embeds.to(transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        safe_emit_progress(progress_callback, 0.44, "Initializing latent noise")
        latents = self._get_latents(
            height,
            width,
            num_frames + num_reference_images * self.vae_scale_factor_temporal,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        if latents.shape[2] != conditioning_latents.shape[2]:
            self.logger.warning(
                "The number of frames in the conditioning latents does not match the number of frames to be generated. Generation quality may be affected."
            )

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoise (CFG: {'on' if use_cfg_guidance else 'off'})",
        )

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=conditioning_latents,
                control_hidden_states_scale=conditioning_scale,
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    control_hidden_states=conditioning_latents,
                    control_hidden_states_scale=conditioning_scale,
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
            ip_image=ip_image,
            num_reference_images=num_reference_images,
        )

        if offload:
            safe_emit_progress(progress_callback, 0.91, "Offloading transformer")
            self._offload("transformer")

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            latents = latents[:, :, num_reference_images:]
            safe_emit_progress(progress_callback, 0.94, "Decoding latents to video")
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents to video")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(
                progress_callback, 1.0, "Completed VACE video generation pipeline"
            )
            return postprocessed_video
