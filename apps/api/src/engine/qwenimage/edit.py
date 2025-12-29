import torch
import math
from typing import Dict, Any, Callable, List
from .shared import QwenImageShared
import numpy as np
from PIL import Image
from src.utils.progress import safe_emit_progress, make_mapped_progress
from loguru import logger


class QwenImageEditEngine(QwenImageShared):
    """QwenImage Edit Engine Implementation"""

    def run(
        self,
        image: Image.Image | np.ndarray | torch.Tensor | str | None = None,
        prompt: List[str] | str | None = None,
        negative_prompt: List[str] | str = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float | None = None,
        true_cfg_scale: float = 4.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting edit pipeline")
        safe_emit_progress(progress_callback, 0.02, "Validating inputs")

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.03, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.04, "Text encoder loaded")

        safe_emit_progress(progress_callback, 0.045, "Moving text encoder to device")
        self.to_device(self.text_encoder)

        if image is None or prompt is None:
            raise ValueError("Image and prompt are required")

        safe_emit_progress(progress_callback, 0.05, "Loading image and resizing")
        batch_size = (len(prompt) if isinstance(prompt, list) else 1) * num_images
        loaded_image = self._load_image(image)
        max_area = height * width if height is not None and width is not None else None
        loaded_image, calculated_height, calculated_width = self._aspect_ratio_resize(
            loaded_image, max_area=max_area, mod_value=32
        )
        safe_emit_progress(progress_callback, 0.10, "Image loaded and resized")

        safe_emit_progress(progress_callback, 0.12, "Preprocessing image for VAE")
        preprocessed_image = self.image_processor.preprocess(loaded_image).unsqueeze(2)
        if height is None:
            height = calculated_height
        if width is None:
            width = calculated_width
        safe_emit_progress(
            progress_callback, 0.15, "Resolved target dimensions and preprocessed image"
        )

        # get dtype
        dtype = self.component_dtypes["text_encoder"]
        safe_emit_progress(progress_callback, 0.16, "Encoding prompt")
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            image=loaded_image,
            device=self.device,
            num_images_per_prompt=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
            dtype=dtype,
        )

        safe_emit_progress(progress_callback, 0.20, "Encoded prompt")

        if negative_prompt is not None and use_cfg_guidance:
            safe_emit_progress(progress_callback, 0.205, "Encoding negative prompt")
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                image=loaded_image,
                device=self.device,
                num_images_per_prompt=num_images,
                text_encoder_kwargs=text_encoder_kwargs,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
        safe_emit_progress(
            progress_callback,
            0.23,
            (
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            safe_emit_progress(progress_callback, 0.235, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 0.25, "Text encoder offloaded")

        transformer_dtype = self.component_dtypes["transformer"]
        safe_emit_progress(progress_callback, 0.26, "Moving embeddings to device")
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        # prepare guidance embeddings (if any) after moving to device
        image_latents = self.vae_encode(preprocessed_image, offload=offload)
        image_latents = torch.cat([image_latents] * batch_size, dim=0)
        image_latent_height, image_latent_width = image_latents.shape[3:]
        image_latents = self._pack_latents(
            image_latents,
            batch_size,
            self.num_channels_latents,
            image_latent_height,
            image_latent_width,
        )
        safe_emit_progress(progress_callback, 0.30, "Prepared image latents")

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.305, "Loading transformer")
            self.load_component_by_type("transformer")

        safe_emit_progress(progress_callback, 0.31, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.32, "Transformer ready")

        if negative_prompt_embeds is not None:
            safe_emit_progress(progress_callback, 0.33, "Preparing negative embeddings for transformer")
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)
        safe_emit_progress(
            progress_callback,
            0.34,
            f"Guidance embeddings prepared (CFG: {'on' if (negative_prompt_embeds is not None and use_cfg_guidance) else 'off'})",
        )

        safe_emit_progress(progress_callback, 0.35, "Initializing latent noise")
        latents = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            seed=seed,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.38, "Initialized latent noise")

        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                ),
                (
                    1,
                    calculated_height // self.vae_scale_factor // 2,
                    calculated_width // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size
        safe_emit_progress(progress_callback, 0.40, "Prepared shape metadata")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.41, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.42, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.43, "Moving scheduler to device")
        self.to_device(self.scheduler)

        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        safe_emit_progress(progress_callback, 0.45, "Scheduler prepared")

        safe_emit_progress(progress_callback, 0.47, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )
        safe_emit_progress(
            progress_callback, 0.50, "Timesteps computed; starting denoise"
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is not None:
            safe_emit_progress(progress_callback, 0.485, "Preparing guidance embeddings")
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )

        self.scheduler.set_begin_index(0)

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        # Set preview context for per-step rendering on the main engine (denoise runs there)
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        safe_emit_progress(
            progress_callback,
            0.50,
            f"Starting denoising (CFG: {'on' if (negative_prompt_embeds is not None and use_cfg_guidance) else 'off'})",
        )
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            guidance=guidance,
            true_cfg_scale=true_cfg_scale,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            txt_seq_lens=txt_seq_lens,
            negative_txt_seq_lens=negative_txt_seq_lens,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            use_cfg_guidance=use_cfg_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            denoise_progress_callback=denoise_progress_callback,
            render_on_step_interval=render_on_step_interval,
        )

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            safe_emit_progress(progress_callback, 0.93, "Offloading transformer")
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.96, "Decoding latents")
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        tensor_image = self.vae_decode(latents, offload=offload)[:, :, 0]
        image = self._tensor_to_frame(tensor_image)
        safe_emit_progress(progress_callback, 1.0, "Completed edit pipeline")
        return image
