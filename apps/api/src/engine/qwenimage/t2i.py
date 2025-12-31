import torch
from typing import Dict, Any, Callable, List
from .shared import QwenImageShared
import numpy as np
from src.utils.progress import safe_emit_progress, make_mapped_progress
import time
from loguru import logger


class QwenImageT2IEngine(QwenImageShared):
    """QwenImage Text-to-Image Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 1.0,
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
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None
        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")
        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.01, "Loading text encoder")
            self.load_component_by_type("text_encoder")

        safe_emit_progress(progress_callback, 0.03, "Moving text encoder to device")
        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        batch_size = (
            num_images * len(prompt) if isinstance(prompt, list) else num_images
        )

        prompt_encode_cb = make_mapped_progress(progress_callback, 0.06, 0.14)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            num_images_per_prompt=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
            device=self.device,
            dtype=self.component_dtypes.get("text_encoder"),
            progress_callback=prompt_encode_cb,
        )
        safe_emit_progress(progress_callback, 0.15, "Encoded prompt")

        if negative_prompt is not None and use_cfg_guidance:
            neg_prompt_encode_cb = make_mapped_progress(progress_callback, 0.14, 0.18)
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                num_images_per_prompt=num_images,
                text_encoder_kwargs=text_encoder_kwargs,
                device=self.device,
                dtype=self.component_dtypes.get("text_encoder"),
                progress_callback=neg_prompt_encode_cb,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
        safe_emit_progress(
            progress_callback,
            0.18,
            (
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            safe_emit_progress(progress_callback, 0.19, "Offloading text encoder")
            del self.text_encoder
        safe_emit_progress(progress_callback, 0.20, "Text encoder offloaded")

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.21, "Loading transformer")
            self.load_component_by_type("transformer")
            safe_emit_progress(progress_callback, 0.23, "Transformer loaded")
        safe_emit_progress(progress_callback, 0.24, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.25, "Transformer ready")

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)
        safe_emit_progress(progress_callback, 0.28, "Guidance embeddings prepared")

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
        safe_emit_progress(progress_callback, 0.32, "Initialized latent noise")

        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ] * num_images
        safe_emit_progress(progress_callback, 0.34, "Prepared shape metadata")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.33, "Loading scheduler")
            self.load_component_by_type("scheduler")

        safe_emit_progress(progress_callback, 0.34, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.35, "Moving scheduler to device")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.36, "Scheduler ready")

        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )
        safe_emit_progress(progress_callback, 0.40, "Timesteps computed")

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # handle guidance
        if self.transformer.config.guidance_embeds:
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
        safe_emit_progress(
            denoise_progress_callback,
            0.0,
            f"Starting denoise (CFG: {'on' if use_cfg_guidance else 'off'})",
        )

        # Set preview context for per-step rendering on the main engine (denoise runs there)
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            guidance=guidance,
            true_cfg_scale=true_cfg_scale,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            txt_seq_lens=txt_seq_lens,
            negative_txt_seq_lens=negative_txt_seq_lens,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            use_cfg_guidance=use_cfg_guidance,
            denoise_progress_callback=denoise_progress_callback,
            render_on_step_interval=render_on_step_interval,
        )

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            del self.transformer
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.95, "Unpacking latents")
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        safe_emit_progress(progress_callback, 0.97, "Decoding latents")
        tensor_image = self.vae_decode(latents, offload=offload)[:, :, 0]
        image = self._tensor_to_frame(tensor_image)
        safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
        return image
