import torch
from typing import Dict, Any, Callable, List
import numpy as np
from diffusers.schedulers import UniPCMultistepScheduler
from .shared import HidreamShared
from src.utils.progress import safe_emit_progress, make_mapped_progress


class HidreamEditEngine(HidreamShared):
    """Hidream Edit Engine Implementation"""

    def run(
        self,
        image: torch.Tensor | str | np.ndarray,
        prompt: List[str] | str,
        prompt_2: List[str] | str = None,
        prompt_3: List[str] | str = None,
        prompt_4: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        negative_prompt_3: List[str] | str = None,
        negative_prompt_4: List[str] | str = None,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
        image_guidance_scale: float = 2.0,
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
        clip_cfg_norm: bool = True,
        refine_strength: float = 0.0,
        resize_to: int = 1024,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting edit pipeline")

        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")
        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        use_cfg_guidance = guidance_scale > 1.0
        batch_size = (
            num_images * len(prompt) if isinstance(prompt, list) else num_images
        )

        (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
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
        safe_emit_progress(progress_callback, 0.20, "Encoded prompts")

        if "Target Image Description:" in prompt:
            target_prompt = prompt.split("Target Image Description:")[1].strip()
            (
                target_prompt_embeds_t5,
                target_negative_prompt_embeds_t5,
                target_prompt_embeds_llama3,
                target_negative_prompt_embeds_llama3,
                target_pooled_prompt_embeds,
                target_negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=target_prompt,
                prompt_2=None,
                prompt_3=None,
                prompt_4=None,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                negative_prompt_3=None,
                negative_prompt_4=None,
                use_cfg_guidance=use_cfg_guidance,
                num_images=num_images,
                text_encoder_kwargs=text_encoder_kwargs,
                text_encoder_2_kwargs=text_encoder_2_kwargs,
                text_encoder_3_kwargs=text_encoder_3_kwargs,
                offload=offload,
            )
        else:
            target_prompt_embeds_t5 = prompt_embeds_t5
            target_negative_prompt_embeds_t5 = negative_prompt_embeds_t5
            target_prompt_embeds_llama3 = prompt_embeds_llama3
            target_negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3
            target_pooled_prompt_embeds = pooled_prompt_embeds
            target_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        safe_emit_progress(progress_callback, 0.25, "Prepared target prompt embeddings")

        transformer_dtype = self.component_dtypes.get("transformer", None)

        if use_cfg_guidance:
            if clip_cfg_norm:
                prompt_embeds_t5 = torch.cat(
                    [prompt_embeds_t5, negative_prompt_embeds_t5, prompt_embeds_t5],
                    dim=0,
                )
                prompt_embeds_llama3 = torch.cat(
                    [
                        prompt_embeds_llama3,
                        negative_prompt_embeds_llama3,
                        prompt_embeds_llama3,
                    ],
                    dim=1,
                )
                pooled_prompt_embeds = torch.cat(
                    [
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        pooled_prompt_embeds,
                    ],
                    dim=0,
                )
            else:
                prompt_embeds_t5 = torch.cat(
                    [
                        negative_prompt_embeds_t5,
                        negative_prompt_embeds_t5,
                        prompt_embeds_t5,
                    ],
                    dim=0,
                )
                prompt_embeds_llama3 = torch.cat(
                    [
                        negative_prompt_embeds_llama3,
                        negative_prompt_embeds_llama3,
                        prompt_embeds_llama3,
                    ],
                    dim=1,
                )
                pooled_prompt_embeds = torch.cat(
                    [
                        negative_pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        pooled_prompt_embeds,
                    ],
                    dim=0,
                )

            target_prompt_embeds_t5 = torch.cat(
                [target_negative_prompt_embeds_t5, target_prompt_embeds_t5], dim=0
            )
            target_prompt_embeds_llama3 = torch.cat(
                [target_negative_prompt_embeds_llama3, target_prompt_embeds_llama3],
                dim=1,
            )
            target_pooled_prompt_embeds = torch.cat(
                [target_negative_pooled_prompt_embeds, target_pooled_prompt_embeds],
                dim=0,
            )

        safe_emit_progress(progress_callback, 0.28, "Applied guidance packing")

        image = self._load_image(image)
        image = self.resize_image(image, image_size=resize_to)
        image = self.image_processor.preprocess(image)
        safe_emit_progress(progress_callback, 0.30, "Loaded and preprocessed image")

        image_latents = self.vae_encode(image, offload=offload)
        latent_height, latent_width = image_latents.shape[2:]
        height = latent_height * self.vae_scale_factor
        width = latent_width * self.vae_scale_factor

        if image_latents.shape[0] != batch_size:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt)
        else:
            image_latents = torch.cat([image_latents])

        if use_cfg_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat(
                [uncond_image_latents, image_latents, image_latents], dim=0
            )
        safe_emit_progress(progress_callback, 0.35, "Prepared image latents")

        latents = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.40, "Initialized latent noise")

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        max_seq = 8192
        if not isinstance(self.scheduler, UniPCMultistepScheduler):
            sigmas = (
                np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                if sigmas is None
                else sigmas
            )
            mu = self.calculate_shift(
                max_seq,
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
        safe_emit_progress(
            progress_callback, 0.50, "Timesteps computed; starting denoise"
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        if not hasattr(self, "transformer") or not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.52, "Transformer ready")

        refine_stage = False

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        # Provide preview decode context for per-step rendering

        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        self.transformer.max_seq = 8192

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # === STAGE DETERMINATION ===
                # Check if we need to switch from editing stage to refining stage
                if i == int(num_inference_steps * (1.0 - refine_strength)):
                    refine_stage = True

                # === INPUT PREPARATION ===
                if refine_stage:
                    # Refining stage: Use target prompts and simpler input (no image conditioning)
                    latent_model_input_with_condition = (
                        torch.cat([latents] * 2) if use_cfg_guidance else latents
                    )
                    current_prompt_embeds_t5 = target_prompt_embeds_t5
                    current_prompt_embeds_llama3 = target_prompt_embeds_llama3
                    current_pooled_prompt_embeds = target_pooled_prompt_embeds
                else:
                    # Editing stage: Use original prompts and include image conditioning
                    latent_model_input = (
                        torch.cat([latents] * 3) if use_cfg_guidance else latents
                    )
                    latent_model_input_with_condition = torch.cat(
                        [latent_model_input, image_latents], dim=-1
                    )
                    current_prompt_embeds_t5 = prompt_embeds_t5
                    current_prompt_embeds_llama3 = prompt_embeds_llama3
                    current_pooled_prompt_embeds = pooled_prompt_embeds

                # === TRANSFORMER SELECTION ===
                # Choose which transformer to use for this step

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input_with_condition.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input_with_condition,
                    timesteps=timestep,
                    encoder_hidden_states_t5=current_prompt_embeds_t5,
                    encoder_hidden_states_llama3=current_prompt_embeds_llama3,
                    pooled_embeds=current_pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                # perform guidance
                noise_pred = -1.0 * noise_pred[..., : latents.shape[-1]]
                if use_cfg_guidance:
                    if refine_stage:
                        uncond, full_cond = noise_pred.chunk(2)
                        noise_pred = uncond + guidance_scale * (full_cond - uncond)
                    else:
                        if clip_cfg_norm:
                            uncond, image_cond, full_cond = noise_pred.chunk(3)
                            pred_text_ = image_cond + guidance_scale * (
                                full_cond - image_cond
                            )
                            norm_full_cond = torch.norm(full_cond, dim=1, keepdim=True)
                            norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
                            scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(
                                min=0.0, max=1.0
                            )
                            pred_text = pred_text_ * scale
                            noise_pred = uncond + image_guidance_scale * (
                                pred_text - uncond
                            )
                        else:
                            uncond, image_cond, full_cond = noise_pred.chunk(3)
                            noise_pred = (
                                uncond
                                + image_guidance_scale * (image_cond - uncond)
                                + guidance_scale * (full_cond - image_cond)
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

        if offload:
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        safe_emit_progress(progress_callback, 1.0, "Completed edit pipeline")
        return image
