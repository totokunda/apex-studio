import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from diffusers.models.embeddings import get_3d_rotary_pos_embed
import inspect
import torch.nn.functional as F
from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor
from diffusers.schedulers import CogVideoXDPMScheduler
from contextlib import nullcontext
from src.utils.type import EnumType
import math


class CogVideoShared(BaseEngine):
    """Base class for CogVideo engine implementations containing common functionality"""

    def __init__(
        self,
        yaml_path: str,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor_temporal = (
            getattr(self.vae, "config", {}).get("temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            2
            ** (
                len(
                    getattr(self.vae, "config", {}).get("block_out_channels", [1, 1, 1])
                )
                - 1
            )
            if getattr(self, "vae", None)
            else 8
        )
        self.vae_scaling_factor_image = (
            getattr(self.vae, "config", {}).get("scaling_factor", None) or 0.7
            if getattr(self, "vae", None)
            else 0.7
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Encode prompts using T5 text encoder"""
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if isinstance(prompt, str):
            prompt = [prompt]

        prompt_embeds = self.text_encoder.encode(
            prompt,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            use_attention_mask=False,
            pad_with_zero=False,
            dtype=dtype,
        )

        # Handle negative prompt
        negative_prompt_embeds = None
        if negative_prompt is not None:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                max_sequence_length=max_sequence_length,
                pad_to_max_length=True,
                num_videos_per_prompt=num_videos_per_prompt,
                use_attention_mask=False,
                pad_with_zero=False,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        transformer_config: Dict[str, Any] = None,
    ):
        """Prepare rotary positional embeddings for CogVideoX"""
        if transformer_config is None:
            transformer_config = self.load_config_by_type("transformer")

        grid_height = height // (
            self.vae_scale_factor_spatial * transformer_config.get("patch_size", 16)
        )
        grid_width = width // (
            self.vae_scale_factor_spatial * transformer_config.get("patch_size", 16)
        )

        p = transformer_config.get("patch_size", 16)
        p_t = transformer_config.get("patch_size_t", None)

        base_size_width = transformer_config.get("sample_width", 1024) // p
        base_size_height = transformer_config.get("sample_height", 1024) // p

        if p_t is None:
            # CogVideoX 1.0
            from thirdparty.diffusers.src.diffusers.pipelines.cogvideo.pipeline_cogvideox import (
                get_resize_crop_region_for_grid,
            )

            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.get("attention_head_dim", 128),
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.get("attention_head_dim", 128),
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    def _get_v2v_timesteps(
        self, num_inference_steps: int, timesteps: List[int], strength: float
    ):
        """Get timesteps for video-to-video generation based on strength"""
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def _prepare_v2v_latents(
        self,
        video: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        timestep: Optional[torch.Tensor] = None,
    ):
        """Prepare latents for video-to-video generation"""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1

        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # Encode video to latents using vae_encode
        if isinstance(generator, list):
            init_latents = [
                self.vae_encode(
                    video[i].unsqueeze(0),
                    sample_mode="mode",
                    sample_generator=generator[i],
                    dtype=dtype,
                )
                for i in range(batch_size)
            ]
        else:
            init_latents = [
                self.vae_encode(
                    vid.unsqueeze(0),
                    sample_mode="mode",
                    sample_generator=generator,
                    dtype=dtype,
                )
                for vid in video
            ]

        init_latents = torch.cat(init_latents, dim=0)
        # Add noise to the initial latents
        from diffusers.utils.torch_utils import randn_tensor

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.add_noise(init_latents, noise, timestep)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _retrieve_latents(
        self,
        encoder_output: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        sample_mode: str = "sample",
    ):
        """Retrieve latents from encoder output"""
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def _add_noise_to_reference_video(self, image, ratio=None):
        if ratio is None:
            sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(
                image.device
            )
            sigma = torch.exp(sigma).to(image.dtype)
        else:
            sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio

        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image_noise = torch.where(image == -1, torch.zeros_like(image), image_noise)
        image = image + image_noise
        return image

    def _resize_mask(self, mask, latent, process_first_frame_only=True):
        latent_size = latent.size()
        batch_size, channels, num_frames, height, width = mask.shape

        if process_first_frame_only:
            target_size = list(latent_size[2:])
            target_size[0] = 1
            first_frame_resized = F.interpolate(
                mask[:, :, 0:1, :, :],
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

            target_size = list(latent_size[2:])
            target_size[0] = target_size[0] - 1
            if target_size[0] != 0:
                remaining_frames_resized = F.interpolate(
                    mask[:, :, 1:, :, :],
                    size=target_size,
                    mode="trilinear",
                    align_corners=False,
                )
                resized_mask = torch.cat(
                    [first_frame_resized, remaining_frames_resized], dim=2
                )
            else:
                resized_mask = first_frame_resized
        else:
            target_size = list(latent_size[2:])
            resized_mask = F.interpolate(
                mask, size=target_size, mode="trilinear", align_corners=False
            )
        return resized_mask

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        """Unified denoising loop for all CogVideo modes"""
        latents = kwargs.get("latents", None)
        timesteps = kwargs.get("timesteps", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 6.0)
        use_dynamic_cfg = kwargs.get("use_dynamic_cfg", False)
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", False)
        noise_pred_kwargs = kwargs.get("noise_pred_kwargs", {})
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        extra_step_kwargs = kwargs.get("extra_step_kwargs", {})

        # Mode-specific inputs
        image_latents = kwargs.get("image_latents", None)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * scheduler.order, 0
        )

        with self._progress_bar(total=num_inference_steps, desc=f"Denoising") as pbar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                # Expand latents if doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t
                ).to(transformer_dtype)

                # Mode-specific input preparation
                if image_latents is not None:
                    # Concatenate with image latents for I2V
                    latent_image_input = (
                        torch.cat([image_latents] * 2)
                        if do_classifier_free_guidance
                        else image_latents
                    )
                    latent_model_input = torch.cat(
                        [latent_model_input, latent_image_input], dim=2
                    ).to(transformer_dtype)

                # Broadcast timestep to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                if hasattr(self.transformer, "cache_context"):
                    cache_context = self.transformer.cache_context(
                        "cond_uncond" if do_classifier_free_guidance else None
                    )
                else:
                    cache_context = nullcontext()

                with cache_context:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **noise_pred_kwargs,
                    )[0]

                noise_pred = noise_pred.float()

                # Perform guidance
                if use_dynamic_cfg:
                    # Dynamic CFG scaling based on timestep
                    dynamic_guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - t.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2
                    )
                else:
                    dynamic_guidance_scale = guidance_scale

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + dynamic_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Scheduler step - handle different scheduler types
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )

                latents = latents.to(transformer_dtype)

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
                ):
                    pbar.update(1)

        self.logger.info(f"Denoising completed.")
