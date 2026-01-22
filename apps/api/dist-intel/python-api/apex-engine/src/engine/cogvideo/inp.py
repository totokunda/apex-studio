import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import CogVideoShared
import torch.nn.functional as F
from einops import rearrange


class CogVideoInpEngine(CogVideoShared):
    """CogVideo Fun Engine Implementation"""

    def _prepare_mask_latents(
        self, masked_image, noise_aug_strength, transformer_config
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if masked_image is not None:
            if transformer_config.get("add_noise_in_inpaint_model", False):
                masked_image = self._add_noise_to_reference_video(
                    masked_image, ratio=noise_aug_strength
                )
            masked_image = masked_image.to(device=self.device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim=0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor

        return masked_image_latents

    def run(
        self,
        prompt: Union[List[str], str],
        video: Union[List[Image.Image], torch.Tensor],
        mask_video: Union[List[Image.Image], torch.Tensor] = None,
        negative_prompt: Union[List[str], str] = "",
        height: int | None = 480,
        width: int | None = 832,
        num_inference_steps: int = 50,
        duration: int = 49,
        fps: int = 8,
        num_videos: int = 1,
        seed: int = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        max_sequence_length: int = 226,
        eta: float = 0.0,
        strength: float = 1.0,
        noise_aug_strength: float = 0.0563,
        **kwargs,
    ):
        """Control video generation following CogVideoXFunControlPipeline"""

        # 1. Process control video
        video = self._load_video(video, fps=fps)

        if height is None or width is None:
            height, width = video[0].height, video[0].width

        video = self.video_processor.preprocess_video(video, height=height, width=width)

        video = video.to(device=self.device)
        video_length = video.shape[2]
        video_length = (
            int(
                (video_length - 1)
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
            )
            + 1
            if video_length != 1
            else 1
        )

        is_strength_max = strength == 1.0

        if mask_video is not None:
            mask_video = self._load_video(mask_video)
            mask_video = self.video_processor.preprocess_video(
                mask_video, height=height, width=width
            )

        num_frames = self._parse_num_frames(duration=duration, fps=fps)
        transformer_config = self.load_config_by_type("transformer")

        local_latent_length = (video_length - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = transformer_config.get("patch_size_t", None)
        additional_frames = 0

        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video_length > num_frames:
            self.logger.warning(
                "The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. "
            )
            video_length = num_frames
            video = video[:, :, :video_length]
            mask_video = mask_video[:, :, :video_length]
        else:
            num_frames = video_length

        # 2. Encode prompts
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            **text_encoder_kwargs,
        )

        if offload:
            self._offload("text_encoder")

        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device)

        # 8. Prepare guidance
        do_classifier_free_guidance = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None
        )

        batch_size = prompt_embeds.shape[0]

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            strength=strength,
        )

        # 6. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, check that latent frames are divisible by patch_size_t
        patch_size_t = transformer_config.get("patch_size_t", None)
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            raise ValueError(
                f"The number of latent frames must be divisible by `{patch_size_t=}` but the given video "
                f"contains {latent_frames=}, which is not divisible."
            )

        if not self.vae:
            self.load_component_by_type("vae")

        self.to_device(self.vae)

        # Prepare control video latents using vae_encode
        if video is not None:
            video = video.to(device=self.device, dtype=self.vae.dtype)
            bs = 1
            new_video = []
            for i in range(0, video.shape[0], bs):
                video_bs = video[i : i + bs]
                video_bs = self.vae.encode(video_bs)[0]
                video_bs = video_bs.mode()
                new_video.append(video_bs)
            new_video = torch.cat(new_video, dim=0)
            new_video = new_video * self.vae.config.scaling_factor
            video_latents = new_video.repeat(
                batch_size // new_video.shape[0], 1, 1, 1, 1
            )
            video_latents = video_latents.to(
                device=self.device, dtype=transformer_dtype
            )
            video_latents = rearrange(video_latents, "b c f h w -> b f c h w")
        else:
            video_latents = None

        num_channels_latents = self.vae.config.latent_channels

        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latents = self._get_latents(
            height,
            width,
            latent_num_frames,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=transformer_dtype,
            parse_frames=False,
            order="BFC",
        )

        latent_timestep = timesteps[:1].repeat(batch_size)
        if not is_strength_max:
            latents = self.scheduler.add_noise(video_latents, latents, latent_timestep)
        else:
            latents = latents * self.scheduler.init_noise_sigma

        prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
        if negative_prompt is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=transformer_dtype)

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height,
            width,
            latents.size(1),
            self.device,
            transformer_config=transformer_config,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        num_channels_transformer = transformer_config.get("in_channels", 32)

        if mask_video is not None:
            if (mask_video == 1).all():
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(
                    latents.device, latents.dtype
                )
                masked_video_latents = torch.zeros_like(latents).to(
                    latents.device, latents.dtype
                )

                mask_input = (
                    torch.cat([mask_latents] * 2)
                    if do_classifier_free_guidance
                    else mask_latents
                )
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2)
                    if do_classifier_free_guidance
                    else masked_video_latents
                )
                inpaint_latents = torch.cat(
                    [mask_input, masked_video_latents_input], dim=2
                ).to(latents.dtype)
            else:
                # Prepare mask latent variables
                mask_condition = mask_video.to(torch.float32)

                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(
                        mask_condition, [1, 3, 1, 1, 1]
                    ).to(video)
                    masked_video = (
                        video * (mask_condition_tile < 0.5)
                        + torch.ones_like(video) * (mask_condition_tile > 0.5) * -1
                    )

                    masked_video_latents = self._prepare_mask_latents(
                        masked_video,
                        noise_aug_strength=noise_aug_strength,
                        transformer_config=transformer_config,
                    )

                    mask_latents = self._resize_mask(
                        1 - mask_condition, masked_video_latents
                    )

                    mask_latents = (
                        mask_latents.to(masked_video_latents.device)
                        * self.vae.config.scaling_factor
                    )

                    mask = torch.tile(
                        mask_condition, [1, num_channels_latents, 1, 1, 1]
                    )
                    mask = F.interpolate(
                        mask,
                        size=latents.size()[-3:],
                        mode="trilinear",
                        align_corners=True,
                    ).to(latents.device, latents.dtype)

                    mask_input = (
                        torch.cat([mask_latents] * 2)
                        if do_classifier_free_guidance
                        else mask_latents
                    )
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2)
                        if do_classifier_free_guidance
                        else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(
                        masked_video_latents_input, "b c f h w -> b f c h w"
                    )

                    inpaint_latents = torch.cat(
                        [mask_input, masked_video_latents_input], dim=2
                    ).to(latents.dtype)
                else:
                    mask = torch.tile(
                        mask_condition, [1, num_channels_latents, 1, 1, 1]
                    )
                    mask = F.interpolate(
                        mask,
                        size=latents.size()[-3:],
                        mode="trilinear",
                        align_corners=True,
                    ).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")

                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                mask = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(
                    latents.device, latents.dtype
                )

                mask_input = (
                    torch.cat([mask] * 2) if do_classifier_free_guidance else mask
                )
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2)
                    if do_classifier_free_guidance
                    else masked_video_latents
                )
                inpaint_latents = torch.cat(
                    [mask_input, masked_video_latents_input], dim=1
                ).to(latents.dtype)
            else:
                mask = torch.zeros_like(video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(
                    mask, size=latents.size()[-3:], mode="trilinear", align_corners=True
                ).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")

                inpaint_latents = None

        if offload:
            self._offload("vae")

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        noise_pred_kwargs = dict(
            inpaint_latents=inpaint_latents,
            encoder_hidden_states=prompt_embeds,
            image_rotary_emb=image_rotary_emb,
        )

        # 9. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=noise_pred_kwargs,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
            transformer_dtype=transformer_dtype,
            extra_step_kwargs=self.prepare_extra_step_kwargs(generator, eta),
            **kwargs,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
