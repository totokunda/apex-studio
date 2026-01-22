import torch
from typing import Dict, Any, Callable, List
import numpy as np
from diffusers.schedulers import UniPCMultistepScheduler
from .shared import HidreamShared
import math
from src.utils.progress import safe_emit_progress, make_mapped_progress


class HidreamT2IEngine(HidreamShared):
    """Hidream Text-to-Image Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        prompt_3: List[str] | str = None,
        prompt_4: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        negative_prompt_3: List[str] | str = None,
        negative_prompt_4: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        text_encoder_2_kwargs: Dict[str, Any] = {},
        text_encoder_3_kwargs: Dict[str, Any] = {},
        joint_attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        sigmas: List[float] | None = None,
        timesteps: List[int] | None = None,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        batch_size = (
            num_images * len(prompt) if isinstance(prompt, list) else num_images
        )

        use_cfg_guidance = guidance_scale >= 1.0

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(
            height * scale // division * division
        )

        safe_emit_progress(progress_callback, 0.10, "Encoding prompts")
        (
            prompt_embeds,
            negative_prompt_embeds,
            llama_prompt_embeds,
            llama_negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            text_encoder_kwargs=text_encoder_kwargs,
            text_encoder_2_kwargs=text_encoder_2_kwargs,
            text_encoder_3_kwargs=text_encoder_3_kwargs,
            num_images=num_images,
            use_cfg_guidance=use_cfg_guidance,
            offload=offload,
        )

        transformer_dtype = self.component_dtypes.get("transformer", None)

        if use_cfg_guidance:
            llama_prompt_embeds = torch.cat(
                [
                    llama_negative_prompt_embeds.to(self.device),
                    llama_prompt_embeds.to(self.device),
                ],
                dim=1,
            ).to(transformer_dtype)
            prompt_embeds = torch.cat(
                [negative_prompt_embeds.to(self.device), prompt_embeds.to(self.device)],
                dim=0,
            ).to(transformer_dtype)
            pooled_prompt_embeds = torch.cat(
                [
                    negative_pooled_prompt_embeds.to(self.device),
                    pooled_prompt_embeds.to(self.device),
                ],
                dim=0,
            ).to(transformer_dtype)

        safe_emit_progress(progress_callback, 0.20, "Preparing latents")
        latents = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
        )

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.30, "Scheduler ready")

        if not hasattr(self, "transformer") or not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.40, "Transformer ready")

        if not isinstance(self.scheduler, UniPCMultistepScheduler):
            sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                if sigmas is None
                else sigmas
            )
            mu = self.calculate_shift(
                self.transformer.max_seq,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        else:
            mu = None
            sigmas = None

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            mu=mu,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(progress_callback, 0.45, "Preparing denoise")

        # Set preview context for per-step rendering on the main engine (denoise runs there)
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        safe_emit_progress(progress_callback, 0.50, "Starting denoise")

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if use_cfg_guidance else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    encoder_hidden_states_t5=prompt_embeds,
                    encoder_hidden_states_llama3=llama_prompt_embeds,
                    pooled_embeds=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(
                            min((i + 1) / total_steps, 1.0),
                            f"Denoising step {i + 1}/{total_steps}",
                        )
                    except Exception:
                        pass

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.94, "Decoding image")
        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
        return image
