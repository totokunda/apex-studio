from .shared import LongCatShared
from typing import Union, List, Optional, Dict, Any, Literal
import torch
from src.types import InputVideo
import numpy as np


class LongCatVCEngine(LongCatShared):
    """LongCat Video-to-Video Engine Implementation"""

    def run(
        self,
        video: InputVideo,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=True,
        offload=True,
        **kwargs,
    ):
        """Video-to-video generation following LongCatPipeline"""

        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.transformer is not None and self.transformer.cp_split_hw is not None:
            scale_factor_spatial *= max(self.transformer.cp_split_hw)
        video = self._load_video(video)
        height, width = self.get_condition_shape(
            video, resolution, scale_factor_spatial=scale_factor_spatial
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            self.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        dit_dtype = self.component_dtypes["transformer"]

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            dtype=dit_dtype,
            device=device,
        )

        if offload:
            self._offload("text_encoder")

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # 4. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)

        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
        )

        if enhance_hf:
            tail_uniform_start = 500
            tail_uniform_end = 0
            num_tail_uniform_steps = 10
            timesteps_uniform_tail = list(
                np.linspace(
                    tail_uniform_start,
                    tail_uniform_end,
                    num_tail_uniform_steps,
                    dtype=np.float32,
                    endpoint=(tail_uniform_end != 0),
                )
            )
            timesteps_uniform_tail = [
                torch.tensor(t, device=device).unsqueeze(0)
                for t in timesteps_uniform_tail
            ]
            filtered_timesteps = [
                timestep.unsqueeze(0)
                for timestep in timesteps
                if timestep > tail_uniform_start
            ]
            timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat(
                [timesteps / 1000, torch.zeros(1, device=timesteps.device)]
            )

        video = self.video_processor.preprocess_video(video, height=height, width=width)
        video = video.to(device=device, dtype=prompt_embeds.dtype)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            dtype=dit_dtype,
            device=device,
            generator=generator,
            latents=latents,
            offload=offload,
        )

        num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
        if use_kv_cache:
            cond_latents = latents[:, :, :num_cond_latents]
            self._cache_clean_latents(
                cond_latents,
                max_sequence_length,
                offload_kv_cache=offload_kv_cache,
                device=self.device,
                dtype=dit_dtype,
            )
            kv_cache_dict = self._get_kv_cache_dict()
            latents = latents[:, :, num_cond_latents:]
        else:
            kv_cache_dict = {}

        with self._progress_bar(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :num_cond_latents] = 0

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=num_cond_latents,
                    kv_cache_dict=kv_cache_dict,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    # Calculate the optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    # Reshape for broadcasting
                    st_star = st_star.view(B, 1, 1, 1)
                    # print(f'step i: {i} --> scale: {st_star}')

                    noise_pred = noise_pred_uncond * st_star + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond * st_star
                    )

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                if use_kv_cache:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]
                else:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(
                        noise_pred[:, :, num_cond_latents:],
                        t,
                        latents[:, :, num_cond_latents:],
                        return_dict=False,
                    )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

            if use_kv_cache:
                latents = torch.cat([cond_latents, latents], dim=2)

            self._current_timestep = None

        if offload:
            self._offload("transformer")

        if not return_latents:
            output_video = self.vae_decode(latents, offload=offload)
            output_video = self._tensor_to_frames(output_video)
        else:
            output_video = latents

        return output_video
