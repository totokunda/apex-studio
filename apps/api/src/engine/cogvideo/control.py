import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .shared import CogVideoShared
import torch.nn.functional as F
from einops import rearrange


class CogVideoControlEngine(CogVideoShared):
    """CogVideo Fun Engine Implementation"""

    def _prepare_control_latents(
        self,
        mask: Optional[torch.Tensor] = None,
        masked_image: Optional[torch.Tensor] = None,
    ):
        """Prepare control latents for control video generation"""
        if mask is not None:
            masks = []
            for i in range(mask.size(0)):
                current_mask = mask[i].unsqueeze(0)
                current_mask = self.vae_encode(current_mask, sample_mode="mode")
                masks.append(current_mask)
            mask = torch.cat(masks, dim=0)

        if masked_image is not None:
            mask_pixel_values = []
            for i in range(masked_image.size(0)):
                mask_pixel_value = masked_image[i].unsqueeze(0)
                mask_pixel_value = self.vae_encode(mask_pixel_value, sample_mode="mode")
                mask_pixel_values.append(mask_pixel_value)
            masked_image_latents = torch.cat(mask_pixel_values, dim=0)
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def run(
        self,
        prompt: Union[List[str], str],
        control_video: Union[List[Image.Image], torch.Tensor],
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
        **kwargs,
    ):
        """Control video generation following CogVideoXFunControlPipeline"""

        # 1. Process control video
        control_video = self._load_video(control_video, fps=fps)

        if height is None or width is None:
            height, width = control_video[0].height, control_video[0].width

        control_video = self.video_processor.preprocess_video(
            control_video, height=height, width=width
        )

        control_video = control_video.to(device=self.device)
        video_length = control_video.shape[2]

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
            control_video = control_video[:, :, :video_length]

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

        batch_size = num_videos

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
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

        # Prepare control video latents using vae_encode
        if control_video is not None:
            control_video = control_video.to(
                device=self.device, dtype=prompt_embeds.dtype
            )
            control_video_latents = self._prepare_control_latents(
                mask=None,
                masked_image=control_video,
            )[1]
        else:
            control_video_latents = None

        num_channels_latents = self.vae.config.latent_channels

        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        batch_size = prompt_embeds.shape[0]
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

        # Scale initial noise by scheduler's init_noise_sigma
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

        if offload:
            self._offload("vae")

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        latent_control_input = (
            torch.cat([control_video_latents] * 2)
            if do_classifier_free_guidance
            else control_video_latents
        )

        noise_pred_kwargs = dict(
            encoder_hidden_states=prompt_embeds,
            image_rotary_emb=image_rotary_emb,
            control_latents=latent_control_input,
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
