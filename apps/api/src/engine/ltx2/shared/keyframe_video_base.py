from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import copy

from src.engine.ltx2.ti2v import LTX2TI2VEngine
from src.helpers.ltx2.upsampler import upsample_video
from diffusers.utils.torch_utils import randn_tensor
from src.types import InputImage, InputVideo

from src.engine.ltx2.shared.keyframe_engine import LTX2KeyframeConditioningMixin


class LTX2KeyframeVideoBaseEngine(LTX2TI2VEngine, LTX2KeyframeConditioningMixin):
    """
    Shared base for LTX2 video engines that use *keyframe-token append* conditioning.

    This mirrors the ltx-core `VideoConditionByKeyframeIndex` idea:
    we append extra latent tokens (and their coords) for each conditioning image/video,
    and use a denoise mask to control how strongly they are enforced.
    """

    @torch.inference_mode()
    def run(  # noqa: PLR0913
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # Keyframe images (conditioning)
        images: Optional[Union[InputImage, List[InputImage]]] = None,
        image_strengths: Optional[Union[float, List[float]]] = None,
        image_pixel_frame_indices: Optional[Union[int, List[int]]] = None,
        # Optional keyframe video conditioning (IC-LoRA style)
        conditioning_video: Optional[InputVideo] = None,
        conditioning_video_strength: float = 1.0,
        conditioning_video_pixel_frame_index: int = 0,
        # Generation / diffusion controls
        height: int = 512,
        width: int = 768,
        duration: Union[str, int] = 121,
        fps: float = 25.0,
        num_inference_steps: int = 40,
        timesteps: List[int] = None,
        use_distilled_stage_1: bool = False,
        use_distilled_stage_2: bool = False,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 1024,
        offload: bool = True,
        return_latents: bool = False,
        upsample: bool = True,
    ):
        num_frames = self._parse_num_frames(duration, fps)

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        target_height = height
        target_width = width

        # Stage-1 runs at half resolution when upsampling is enabled.
        if upsample:
            height = target_height // 2
            width = target_width // 2

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        # 3. Prepare text embeddings
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
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        connectors = self.helpers["connectors"]
        # Defensive: after OOM/offload, connectors may be on CPU between runs.
        self.to_device(connectors, device=device)
        additive_attention_mask = (
            1 - prompt_attention_mask.to(prompt_embeds.dtype)
        ) * -1000000.0
        (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_attention_mask,
        ) = connectors(prompt_embeds, additive_attention_mask, additive_mask=True)

        # 4. Prepare latent variables / shapes
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width

        transformer_config = self.load_config_by_type("transformer")
        num_channels_latents = transformer_config.in_channels

        # 4a. Preprocess conditioning images (keyframes)
        cond_items: List[torch.Tensor] = []
        cond_strengths: List[float] = []
        cond_pixel_indices: List[int] = []

        if images is not None:
            if not isinstance(images, list):
                images = [images]

            # Load images so we can safely inspect size and preprocess
            pil_images = [self._load_image(img) for img in images]

            # Choose conditioning spatial size once (based on first keyframe image)
            max_area = height * width
            first_img, cond_h, cond_w = self._aspect_ratio_resize(
                pil_images[0], max_area=max_area, mod_value=32
            )
            height, width = cond_h, cond_w

            cond_items.append(
                self.video_processor.preprocess(first_img, height=height, width=width)
            )
            for img in pil_images[1:]:
                cond_items.append(
                    self.video_processor.preprocess(img, height=height, width=width)
                )

            # Normalize strengths/indices to list form for the keyframe conditionings
            if image_strengths is None:
                cond_strengths = [1.0] * len(cond_items)
            elif isinstance(image_strengths, (int, float)):
                cond_strengths = [float(image_strengths)] * len(cond_items)
            else:
                if len(image_strengths) != len(cond_items):
                    raise ValueError(
                        f"`image_strengths` length {len(image_strengths)} must match number of images {len(cond_items)}."
                    )
                cond_strengths = [float(s) for s in image_strengths]

            if image_pixel_frame_indices is None:
                cond_pixel_indices = [0] * len(cond_items)
            elif isinstance(image_pixel_frame_indices, int):
                cond_pixel_indices = [int(image_pixel_frame_indices)] * len(cond_items)
            else:
                if len(image_pixel_frame_indices) != len(cond_items):
                    raise ValueError(
                        f"`image_pixel_frame_indices` length {len(image_pixel_frame_indices)} must match number of images {len(cond_items)}."
                    )
                cond_pixel_indices = [int(i) for i in image_pixel_frame_indices]

        # 4b. Optional conditioning video (IC-LoRA style): append as keyframe tokens too
        if conditioning_video is not None:
            frames = self._load_video(
                conditioning_video, fps=int(fps), num_frames=num_frames
            )
            # Resize to the same conditioning spatial size (height/width computed above if images provided,
            # otherwise use generation height/width).
            video_tensors = []
            for fr in frames:
                fr = self._load_image(fr)  # ensure RGB
                video_tensors.append(
                    self.video_processor.preprocess(fr, height=height, width=width)
                )
            # [F, 3, H, W] -> [1, 3, F, H, W]
            vid = torch.cat(video_tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
            cond_items.append(vid)
            cond_strengths.append(float(conditioning_video_strength))
            cond_pixel_indices.append(int(conditioning_video_pixel_frame_index))

        # 4c. Build base video latents and append keyframe tokens
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        latents, denoise_mask, video_coords, clean_latents, base_token_count = (
            self._prepare_keyframe_conditioned_video_latents(
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                pixel_num_frames=num_frames,
                pixel_height=height,
                pixel_width=width,
                fps=fps,
                dtype=torch.float32,
                device=device,
                generator=generator,
                base_latents=latents,
                cond_latent_inputs=cond_items,
                cond_strengths=cond_strengths if len(cond_strengths) > 0 else None,
                cond_pixel_frame_indices=(
                    cond_pixel_indices if len(cond_pixel_indices) > 0 else None
                ),
                offload=offload,
            )
        )

        # 4d. Prepare audio latents
        num_mel_bins = (
            self.audio_vae.config.mel_bins
            if getattr(self, "audio_vae", None) is not None
            else 64
        )
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        num_channels_latents_audio = (
            self.audio_vae.config.latent_channels
            if getattr(self, "audio_vae", None) is not None
            else 8
        )
        audio_latents, audio_num_frames, _, _ = self.prepare_audio_latents(
            audio=None,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            num_mel_bins=num_mel_bins,
            num_frames=num_frames,
            frame_rate=fps,
            sampling_rate=self.audio_sampling_rate,
            hop_length=self.audio_hop_length,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        # 5. Prepare timesteps
        if use_distilled_stage_1:
            sigmas = self.distilled_stage_1_sigma_values
            num_inference_steps = len(sigmas)
        elif use_distilled_stage_2:
            sigmas = self.distilled_stage_2_sigma_values
            num_inference_steps = len(sigmas)
        else:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = self.calculate_shift(
            video_sequence_length,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # For now, duplicate the scheduler for use with the audio latents
        audio_scheduler = copy.deepcopy(self.scheduler)
        _, _ = self._get_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare micro-conditions (coords)
        # NOTE: we already built video_coords for the full token sequence (base + conditioning tokens).
        # Build audio coords for audio latents.
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device, fps=fps
        )

        # If CFG is enabled, expand coords and denoise mask across the doubled batch for the transformer call.
        if self.do_classifier_free_guidance:
            video_coords_model = torch.cat([video_coords, video_coords], dim=0)
            denoise_mask_model = torch.cat([denoise_mask, denoise_mask], dim=0)
            audio_coords_model = torch.cat([audio_coords, audio_coords], dim=0)
        else:
            video_coords_model = video_coords
            denoise_mask_model = denoise_mask
            audio_coords_model = audio_coords

        # 7. Denoising loop
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                audio_latent_model_input = (
                    torch.cat([audio_latents] * 2)
                    if self.do_classifier_free_guidance
                    else audio_latents
                )
                audio_latent_model_input = audio_latent_model_input.to(
                    prompt_embeds.dtype
                )

                timestep = t.expand(latent_model_input.shape[0])
                # Masked timesteps: tokens with denoise_mask=0 get timestep 0.
                video_timestep = (
                    timestep.unsqueeze(-1) * denoise_mask_model
                    if denoise_mask_model is not None
                    else timestep
                )

                with self.transformer.cache_context("cond_uncond"):
                    noise_pred_video, noise_pred_audio = self.transformer(
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latent_model_input,
                        encoder_hidden_states=connector_prompt_embeds,
                        audio_encoder_hidden_states=connector_audio_prompt_embeds,
                        timestep=video_timestep,
                        audio_timestep=timestep,
                        encoder_attention_mask=connector_attention_mask,
                        audio_encoder_attention_mask=connector_attention_mask,
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        fps=fps,
                        audio_num_frames=audio_num_frames,
                        video_coords=video_coords_model,
                        audio_coords=audio_coords_model,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )

                noise_pred_video = noise_pred_video.float()
                noise_pred_audio = noise_pred_audio.float()

                if self.do_classifier_free_guidance:
                    noise_pred_video_uncond, noise_pred_video_text = (
                        noise_pred_video.chunk(2)
                    )
                    noise_pred_video = noise_pred_video_uncond + self.guidance_scale * (
                        noise_pred_video_text - noise_pred_video_uncond
                    )

                    noise_pred_audio_uncond, noise_pred_audio_text = (
                        noise_pred_audio.chunk(2)
                    )
                    noise_pred_audio = noise_pred_audio_uncond + self.guidance_scale * (
                        noise_pred_audio_text - noise_pred_audio_uncond
                    )

                    if self.guidance_rescale > 0:
                        noise_pred_video = self.rescale_noise_cfg(
                            noise_pred_video,
                            noise_pred_video_text,
                            guidance_rescale=self.guidance_rescale,
                        )
                        noise_pred_audio = self.rescale_noise_cfg(
                            noise_pred_audio,
                            noise_pred_audio_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                latents = self.scheduler.step(
                    noise_pred_video, t, latents, return_dict=False
                )[0]
                audio_latents = audio_scheduler.step(
                    noise_pred_audio, t, audio_latents, return_dict=False
                )[0]

                # Re-impose conditioning based on denoise mask (closer to ltx-core `post_process_latent` behavior).
                if denoise_mask is not None:
                    latents = (
                        latents * denoise_mask.unsqueeze(-1)
                        + clean_latents.float() * (1 - denoise_mask.unsqueeze(-1))
                    ).to(latents.dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 8. Upsample stage (spatial) then re-run stage-2 refinement
        if upsample:
            # Strip appended conditioning tokens before upsampling (upsampler expects only base video latent).
            base_latents_tokens = latents[:, :base_token_count]

            # Unpack tokens -> latent grid, upsample, then pack back to tokens.
            base_grid = self._unpack_latents(
                base_latents_tokens,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

            upsampler = self.helpers["latent_upsampler"]
            self.to_device(upsampler)
            if not getattr(self, "video_vae", None):
                self.load_component_by_name("video_vae")
            self.to_device(self.video_vae)

            up_grid = upsample_video(base_grid, self.video_vae, upsampler)

            if offload:
                self._offload("upsampler")
                self._offload("video_vae")

            up_tokens = self._pack_latents(
                up_grid,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

            video, audio = self.run(
                prompt=prompt,
                negative_prompt=negative_prompt,
                images=images,
                image_strengths=image_strengths,
                image_pixel_frame_indices=image_pixel_frame_indices,
                conditioning_video=conditioning_video,
                conditioning_video_strength=conditioning_video_strength,
                conditioning_video_pixel_frame_index=conditioning_video_pixel_frame_index,
                height=target_height,
                width=target_width,
                duration=duration,
                fps=fps,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                latents=up_tokens,
                audio_latents=audio_latents,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                offload=offload,
                return_latents=return_latents,
                upsample=False,
                guidance_scale=1.0,
                guidance_rescale=0.0,
                use_distilled_stage_2=True,
            )
            return video, audio

        # 9. Return latents (base only) if requested
        if return_latents:
            return (latents[:, :base_token_count], audio_latents)

        # 10. Decode video latents (base only)
        base_latents_tokens = latents[:, :base_token_count]
        base_grid = self._unpack_latents(
            base_latents_tokens,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        self.video_vae.enable_tiling()

        base_grid = self.video_vae.denormalize_latents(base_grid)
        base_grid = base_grid.to(prompt_embeds.dtype)

        if not self.vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = randn_tensor(
                base_grid.shape,
                generator=generator,
                device=device,
                dtype=base_grid.dtype,
            )
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size
            timestep = torch.tensor(
                decode_timestep, device=device, dtype=base_grid.dtype
            )
            decode_noise_scale = torch.tensor(
                decode_noise_scale, device=device, dtype=base_grid.dtype
            )[:, None, None, None, None]
            base_grid = (
                1 - decode_noise_scale
            ) * base_grid + decode_noise_scale * noise

        base_grid = base_grid.to(self.vae.dtype)
        video = self.vae.decode(base_grid, timestep, return_dict=False)[0]
        video = self._tensor_to_frames(video)

        if offload:
            self._offload("video_vae")

        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
        self.to_device(self.audio_vae)

        audio_latents = audio_latents.to(self.audio_vae.dtype)
        audio_latents = self.audio_vae.denormalize_latents(audio_latents)
        audio_latents = self._unpack_audio_latents(
            audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins
        )
        generated_mel_spectrograms = self.audio_vae.decode(
            audio_latents, return_dict=False
        )[0]

        if offload:
            self._offload("audio_vae")

        vocoder = self.helpers["vocoder"]
        audio = vocoder(generated_mel_spectrograms)

        if offload:
            self._offload("vocoder")

        return video, audio
