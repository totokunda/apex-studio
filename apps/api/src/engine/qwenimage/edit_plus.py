import torch
from typing import Dict, Any, Callable, List
from .shared import QwenImageShared
import numpy as np
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.types.media import InputImage
import math
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


class QwenImageEditPlusEngine(QwenImageShared):
    """QwenImage Edit Plus Engine Implementation"""
    
    @staticmethod
    def calculate_dimensions(target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio

        width = round(width / 32) * 32
        height = round(height / 32) * 32

        return width, height
    
    def _prepare_image_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        seed,
        generator=None,
        latents=None,
        offload=True,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        orig_height = height
        orig_width = width
        height = 2 * (int(orig_height) // (self.vae_scale_factor * 2))
        width = 2 * (int(orig_width) // (self.vae_scale_factor * 2))

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for idx, image in enumerate(images):
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.num_channels_latents:
                    image_latents = self.vae_encode(
                        image, sample_mode="mode", offload=offload, offload_type="cpu" if idx != len(images) - 1 else "discard"
                    )
                else:
                    image_latents = image
                if (
                    batch_size > image_latents.shape[0]
                    and batch_size % image_latents.shape[0] == 0
                ):
                    # expand init_latents for batch_size
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat(
                        [image_latents] * additional_image_per_prompt, dim=0
                    )
                elif (
                    batch_size > image_latents.shape[0]
                    and batch_size % image_latents.shape[0] != 0
                ):
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents,
                    batch_size,
                    num_channels_latents,
                    image_latent_height,
                    image_latent_width,
                )
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if latents is None:
            latents = self._get_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=orig_height,
                width=orig_width,
                dtype=dtype,
                device=device,
                generator=generator,
                seed=seed,
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents


    def run(
        self,
        image_list: List[InputImage] | None = None,
        prompt: List[str] | str | None = None,
        negative_prompt: List[str] | str = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float | None = None,
        true_cfg_scale: float = 4.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        latents: torch.Tensor | None = None,
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        if image_list is None:
            raise ValueError("At least one image is required")

        images = image_list

        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None

        if len(images) == 0:
            raise ValueError("At least one image is required")

        safe_emit_progress(progress_callback, 0.0, "Starting edit plus pipeline")
        safe_emit_progress(progress_callback, 0.02, "Loading images")
        for idx, image in enumerate(images):
            images[idx] = self._load_image(image)

        _, calculated_width, calculated_height = self._aspect_ratio_resize(
            images[-1].copy(), max_area=1024 * 1024, mod_value=32
        )
        height = height or calculated_height
        width = width or calculated_width

        safe_emit_progress(progress_callback, 0.05, "Loading text encoder")

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        safe_emit_progress(progress_callback, 0.06, "Moving text encoder to device")
        self.to_device(self.text_encoder)

        batch_size = (len(prompt) if isinstance(prompt, list) else 1) * num_images
        safe_emit_progress(progress_callback, 0.07, "Resizing and preprocessing images")
        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in images:
            img = self._load_image(img)
            image_width, image_height = img.size
            condition_width, condition_height = self.calculate_dimensions(
                CONDITION_IMAGE_SIZE, image_width / image_height
            )
            vae_width, vae_height = self.calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            condition_image_sizes.append((condition_width, condition_height))
            vae_image_sizes.append((vae_width, vae_height))
            condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
            vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        safe_emit_progress(progress_callback, 0.10, "Images loaded and resized")

        safe_emit_progress(
            progress_callback,
            0.15,
            "Resolved target dimensions and preprocessed images",
        )

        # get dtype
        dtype = self.component_dtypes["text_encoder"]
        safe_emit_progress(progress_callback, 0.16, "Encoding prompt")
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            image=condition_images,
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
                image=condition_images,
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
            del self.text_encoder
        safe_emit_progress(progress_callback, 0.25, "Text encoder offloaded")

        transformer_dtype = self.component_dtypes["transformer"]
        safe_emit_progress(progress_callback, 0.26, "Moving embeddings to device")
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        if negative_prompt_embeds is not None:
            safe_emit_progress(progress_callback, 0.27, "Preparing negative embeddings for transformer")
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)

        safe_emit_progress(progress_callback, 0.275, "Loading transformer config")
        transformer_config = self.load_config_by_type("transformer")

        num_channels_latents = transformer_config.in_channels // 4
        safe_emit_progress(progress_callback, 0.28, "Preparing latent noise and image latents")
        latents, image_latents = self._prepare_image_latents(
            vae_images,
            batch_size * num_images,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            seed,
            generator,
            latents,
            offload,
        )

        safe_emit_progress(
            progress_callback, 0.30, "Initialized latent noise and image latents"
        )

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.31, "Loading transformer")
            self.load_component_by_type("transformer")
        safe_emit_progress(progress_callback, 0.32, "Moving transformer to device")
        self.to_device(self.transformer)

        safe_emit_progress(progress_callback, 0.325, "Transformer ready")

        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                ),
                *[
                    (
                        1,
                        vae_height // self.vae_scale_factor // 2,
                        vae_width // self.vae_scale_factor // 2,
                    )
                    for vae_width, vae_height in vae_image_sizes
                ],
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
            safe_emit_progress(progress_callback, 0.475, "Preparing guidance embeddings")
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        safe_emit_progress(
            progress_callback,
            0.49,
            f"Guidance embeddings prepared (CFG: {'on' if (negative_prompt is not None and use_cfg_guidance) else 'off'})",
        )

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
            f"Starting denoising (CFG: {'on' if (negative_prompt is not None and use_cfg_guidance) else 'off'})",
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

    