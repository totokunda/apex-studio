import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from loguru import logger
from src.engine.hunyuanvideo15.shared import HunyuanVideo15Shared
from src.types.media import InputImage
from src.helpers.hunyuanvideo15.cache import CacheHelper
from src.utils.progress import safe_emit_progress, make_mapped_progress
import torch


class HunyuanVideo15I2VEngine(HunyuanVideo15Shared):
    """HunyuanVideo 1.5 Text/Image-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)

    def prepare_cond_latents_and_mask(
        self,
        latents: torch.Tensor,
        image: InputImage,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Prepare conditional latents and mask for t2v generation.

        Args:
            latents: Main latents tensor (B, C, F, H, W)

        Returns:
            tuple: (cond_latents_concat, mask_concat) - both are zero tensors for t2v
        """

        batch, channels, frames, height, width = latents.shape

        image_latents = self._get_image_latents(
            image=image,
            height=height,
            width=width,
            device=device,
        )

        latent_condition = image_latents.repeat(batch_size, 1, frames, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        latent_condition = latent_condition.to(device=device, dtype=dtype)

        latent_mask = torch.zeros(
            batch, 1, frames, height, width, dtype=dtype, device=device
        )
        latent_mask[:, :, 0, :, :] = 1.0

        return latent_condition, latent_mask

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def run(
        self,
        image: InputImage,
        height: int = 480,
        width: int = 832,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        duration: int | str = 121,
        fps: int = 24,
        num_inference_steps: int = 50,
        chunking_profile: str = "none",
        sigmas: List[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        seed: int = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        offload: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        return_latents: bool = False,
        guidance_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        # --- Video VAE tiling / framewise controls (UI-configurable) ---
        use_light_vae: bool = False,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting image-to-video pipeline")
        num_frames = self._parse_num_frames(duration, fps)
        safe_emit_progress(progress_callback, 0.02, "Loading and resizing input image")
        image = self._load_image(image)
        image, height, width = self._aspect_ratio_resize(
            image, mod_value=32, max_area=height * width
        )
        image = self.video_processor.resize(
            image, height=height, width=width, resize_mode="crop"
        )
        device = self.device

        self._get_image_latents(
            image=image,
            height=height,
            width=width,
            device=device,
        )

        self.load_component_by_type("vae")
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        transformer_dtype = self.component_dtypes.get("transformer")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode image
        safe_emit_progress(progress_callback, 0.06, "Encoding image")
        image_embeds = self.encode_image(
            image=image,
            batch_size=batch_size * num_videos_per_prompt,
            device=device,
            dtype=transformer_dtype,
        )

        # 4. Encode input prompt
        safe_emit_progress(progress_callback, 0.10, "Encoding prompt")
        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = (
            self.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=transformer_dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                prompt_embeds_2=prompt_embeds_2,
                prompt_embeds_mask_2=prompt_embeds_mask_2,
                offload=offload,
            )
        )

        do_classifier_free_guidance = guidance_scale > 1.0 and (
            negative_prompt is not None or negative_prompt_embeds is not None
        )

        if (
            do_classifier_free_guidance
            and negative_prompt is not None
            and negative_prompt_embeds is None
        ):
            safe_emit_progress(
                progress_callback, 0.14, "Encoding negative prompt (CFG)"
            )
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                device=device,
                dtype=transformer_dtype,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            )

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.18, "Loading scheduler")
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        safe_emit_progress(progress_callback, 0.20, "Preparing timesteps")
        sigmas = (
            np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            if sigmas is None
            else sigmas
        )
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler, num_inference_steps, sigmas=sigmas
        )

        # 6. Prepare latent variables
        safe_emit_progress(progress_callback, 0.25, "Preparing latents")
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=transformer_dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        cond_latents_concat, mask_concat = self.prepare_cond_latents_and_mask(
            latents=latents,
            image=image,
            batch_size=batch_size * num_videos_per_prompt,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=device,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.30, "Loading transformer")
            self.load_component_by_name("transformer")

        self.to_device(self.transformer)
        if chunking_profile != "none" and hasattr(
            self.transformer, "set_chunking_profile"
        ):
            self.transformer.set_chunking_profile(chunking_profile)

        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoise (CFG: {'on' if do_classifier_free_guidance else 'off'})",
        )
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat(
                    [latents, cond_latents_concat, mask_concat], dim=1
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(
                    latent_model_input.dtype
                )

                if self.transformer.config.use_meanflow:
                    if i == len(timesteps) - 1:
                        timestep_r = torch.tensor([0.0], device=device)
                    else:
                        timestep_r = timesteps[i + 1]
                    timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype)
                else:
                    timestep_r = None

                # Manual classifier-free guidance (CFG)
                cond_kwargs = {
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_attention_mask": prompt_embeds_mask,
                    "encoder_hidden_states_2": prompt_embeds_2,
                    "encoder_attention_mask_2": prompt_embeds_mask_2,
                }

                if do_classifier_free_guidance:
                    if negative_prompt_embeds is None:
                        raise ValueError(
                            "CFG requested (guidance_scale > 1.0) but no negative prompt / negative prompt embeds were provided."
                        )

                    uncond_kwargs = {
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_attention_mask": negative_prompt_embeds_mask,
                        "encoder_hidden_states_2": negative_prompt_embeds_2,
                        "encoder_attention_mask_2": negative_prompt_embeds_mask_2,
                    }

                    with self.transformer.cache_context("pred_uncond"):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            image_embeds=image_embeds,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            **uncond_kwargs,
                        )[0]

                    with self.transformer.cache_context("pred_cond"):
                        noise_pred_text = self.transformer(
                            hidden_states=latent_model_input,
                            image_embeds=image_embeds,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0]

                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if guidance_rescale > 0.0:
                        # Rescale CFG to reduce overexposure (see: https://arxiv.org/pdf/2305.08891.pdf, Sec. 3.4)
                        std_text = noise_pred_text.std(
                            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
                        )
                        std_cfg = noise_pred.std(
                            dim=list(range(1, noise_pred.ndim)), keepdim=True
                        )
                        noise_pred_rescaled = noise_pred * (std_text / std_cfg)
                        noise_pred = (
                            guidance_rescale * noise_pred_rescaled
                            + (1 - guidance_rescale) * noise_pred
                        )
                else:
                    with self.transformer.cache_context("pred_cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            image_embeds=image_embeds,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    denoise_progress_callback(
                        float(i + 1) / float(max(num_inference_steps, 1)),
                        f"Denoising step {i + 1}/{num_inference_steps}",
                    )

        self._current_timestep = None
        if offload:
            self._offload("transformer")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            if not self.vae:
                self.load_component_by_type("vae")
            self.vae.enable_tiling(use_light_vae=use_light_vae)
            self.to_device(self.vae)
            safe_emit_progress(
                progress_callback, 0.95, "Decoding latents to video with light VAE"
            )
            safe_emit_progress(progress_callback, 0.95, "Decoding latents to video")
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(
                progress_callback, 1.0, "Completed image-to-video pipeline"
            )
            return postprocessed_video
