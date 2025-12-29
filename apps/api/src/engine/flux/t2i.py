import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import FluxShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from loguru import logger
import psutil


class FluxT2IEngine(FluxShared):
    """Flux Text-to-Image Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels_latents = (
            self.transformer.config.out_channels // 4 if self.transformer else 16
        )

    def run(
        self,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 3.5,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        text_encoder_2_kwargs: Dict[str, Any] = {},
        joint_attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        sigmas: List[float] | None = None,
        timesteps: List[int] | None = None,
        ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting t2i pipeline")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None

        safe_emit_progress(progress_callback, 0.02, "Encoding prompts")
        (
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            negative_text_ids,
        ) = self.encode_prompt(
            prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            use_cfg_guidance,
            offload,
            num_images,
            text_encoder_kwargs,
            text_encoder_2_kwargs,
            progress_callback=make_mapped_progress(progress_callback, 0.02, 0.20),
        )

        safe_emit_progress(progress_callback, 0.20, "Encoded prompts")


        if offload:
            safe_emit_progress(progress_callback, 0.21, "Offloading text encoder 2")
            del self.text_encoder_2
            safe_emit_progress(progress_callback, 0.22, "Text encoder 2 offloaded")


        transformer_dtype = self.component_dtypes.get("transformer", None)
        batch_size = prompt_embeds.shape[0]

        safe_emit_progress(progress_callback, 0.24, "Initializing latent noise")
        latents, latent_ids = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.38, "Initialized latent noise")

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.39, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.40, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.41, "Moving scheduler to device")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.42, "Scheduler on device")

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        safe_emit_progress(progress_callback, 0.45, "Scheduler prepared")
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            mu=mu,
        )
        safe_emit_progress(progress_callback, 0.48, "Timesteps computed")

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        if not hasattr(self, "transformer") or not self.transformer:
            safe_emit_progress(progress_callback, 0.485, "Loading transformer")
            self.load_component_by_type("transformer")

        safe_emit_progress(progress_callback, 0.49, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.495, "Transformer ready")

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None
            and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        joint_attention_kwargs = joint_attention_kwargs or {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            safe_emit_progress(progress_callback, 0.498, "Preparing IP adapter embeddings")
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                batch_size,
            )
        if (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            safe_emit_progress(
                progress_callback, 0.499, "Preparing negative IP adapter embeddings"
            )
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                self.device,
                batch_size,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
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
            guidance=guidance,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            latent_ids=latent_ids,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            true_cfg_scale=true_cfg_scale,
            denoise_progress_callback=denoise_progress_callback,
        )
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.95, "Unpacking latents")
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        safe_emit_progress(progress_callback, 0.97, "Decoding latents")
        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        safe_emit_progress(progress_callback, 1.0, "Completed t2i pipeline")
        return image
