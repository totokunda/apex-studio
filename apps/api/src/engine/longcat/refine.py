from .shared import LongCatShared
from typing import Union, List, Optional, Dict, Any
import torch
from src.types import InputImage, InputVideo
import numpy as np
import math
from torch.nn import functional as F
from src.utils.cache import empty_cache


class LongCatRefineEngine(LongCatShared):
    """LongCat Refine Engine Implementation"""

    def run(
        self,
        image: InputImage = None,
        video: InputVideo = None,
        prompt: Union[str, List[str]] = None,
        stage1_video: Optional[str] = None,
        num_cond_frames: int = 0,
        num_inference_steps: int = 50,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        t_thresh=0.5,
        spatial_refine_only=False,
        offload=True,
        **kwargs,
    ):
        scale_factor_spatial = self.vae_scale_factor_spatial * 2 * 4

        if self.transformer is not None and self.transformer.cp_split_hw is not None:
            scale_factor_spatial *= max(self.transformer.cp_split_hw)

        height, width = self.get_condition_shape(
            stage1_video, "720p", scale_factor_spatial=scale_factor_spatial
        )
        setattr(self, "_guidance_scale", 1.0)

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

        (prompt_embeds, prompt_attention_mask, _, _) = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            dtype=dit_dtype,
            device=device,
        )

        if offload:
            self._offload("transformer")

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        self.apply_refinement_lora()

        # 4. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        sigmas = self.get_timesteps_sigmas(num_inference_steps)
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
        )

        if t_thresh:
            t_thresh_tensor = torch.tensor(
                t_thresh * 1000, dtype=timesteps.dtype, device=timesteps.device
            )
            timesteps = torch.cat(
                [t_thresh_tensor.unsqueeze(0), timesteps[timesteps < t_thresh_tensor]]
            )
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat(
                [timesteps / 1000, torch.zeros(1, device=timesteps.device)]
            )

        # 5. Prepare latent variables
        num_frame = len(stage1_video)
        new_frame_size = num_frame if spatial_refine_only else 2 * num_frame
        stage1_video = torch.from_numpy(np.array(stage1_video)).permute(0, 3, 1, 2)
        stage1_video = stage1_video.to(device=device, dtype=prompt_embeds.dtype)

        video_DOWN = F.interpolate(
            stage1_video, size=(height, width), mode="bilinear", align_corners=True
        )
        video_DOWN = video_DOWN.permute(1, 0, 2, 3).unsqueeze(
            0
        )  # [frame, C, H, W] -> [1, C, frame, H, W]
        video_DOWN = video_DOWN / 255.0

        video_UP = F.interpolate(
            video_DOWN,
            size=(new_frame_size, height, width),
            mode="trilinear",
            align_corners=True,
        )  # [B, C, frame, H, W]
        video_UP = video_UP * 2 - 1

        # do padding
        bsa_latent_granularity = 4
        num_noise_frames = video_UP.shape[2] - num_cond_frames

        num_cond_latents = 0
        num_cond_frames_added = 0
        if num_cond_frames > 0:
            num_cond_latents = 1 + math.ceil(
                (num_cond_frames - 1) / self.vae_scale_factor_temporal
            )
            num_cond_latents = (
                math.ceil(num_cond_latents / bsa_latent_granularity)
                * bsa_latent_granularity
            )
            num_cond_frames_added = (
                1
                + (num_cond_latents - 1) * self.vae_scale_factor_temporal
                - num_cond_frames
            )
            num_cond_frames = num_cond_frames + num_cond_frames_added

        num_noise_latents = math.ceil(num_noise_frames / self.vae_scale_factor_temporal)
        num_noise_latents = (
            math.ceil(num_noise_latents / bsa_latent_granularity)
            * bsa_latent_granularity
        )
        num_noise_frames_added = (
            num_noise_latents * self.vae_scale_factor_temporal - num_noise_frames
        )

        pad_front = video_UP[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
        pad_back = video_UP[:, :, -1:].repeat(1, 1, num_noise_frames_added, 1, 1)
        video_UP = torch.cat([pad_front, video_UP, pad_back], dim=2)

        latent_up = self.vae_encode(video_UP, offload=offload)
        latent_up = (1 - t_thresh) * latent_up + t_thresh * torch.randn_like(
            latent_up
        ).contiguous()
        del video_DOWN, video_UP, stage1_video
        empty_cache()

        num_channels_latents = self.transformer.config.in_channels

        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=prompt_embeds.dtype)
        if video is not None:
            video = self.video_processor.preprocess_video(
                video, height=height, width=width
            )
            video = video.to(device=device, dtype=prompt_embeds.dtype)

        latents = self.prepare_latents(
            image=image,
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_cond_frames=num_cond_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latent_up,
            num_cond_frames_added=num_cond_frames_added,
        )

        with self._progress_bar(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :num_cond_latents] = 0

                noise_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    num_cond_latents=num_cond_latents,
                )

                noise_pred = noise_pred_cond

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents[:, :, num_cond_latents:] = self.scheduler.step(
                    noise_pred[:, :, num_cond_latents:],
                    t,
                    latents[:, :, num_cond_latents:],
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if offload:
            self._offload("transformer")

        if not return_latents:
            output_video = self.vae_decode(latents, offload=offload)
            output_video = self._tensor_to_frames(output_video)
            for i in range(len(output_video)):
                output_video[i] = output_video[i][
                    num_cond_frames_added : new_frame_size + num_cond_frames_added
                ]
        else:
            output_video = latents
        return output_video
