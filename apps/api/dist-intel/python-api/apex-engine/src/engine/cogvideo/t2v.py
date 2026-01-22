import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import CogVideoShared


class CogVideoT2VEngine(CogVideoShared):
    """CogVideo Text-to-Video Engine Implementation"""

    def run(
        self,
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = "",
        height: int = 480,
        width: int = 720,
        duration: int = 49,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 8,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        max_sequence_length: int = 226,
        sigmas: List[float] = None,
        eta: float = 0.0,
        **kwargs,
    ):
        """Text-to-video generation following CogVideoXPipeline"""

        # 1. Encode prompts
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            **text_encoder_kwargs,
        )

        if offload:
            self._offload("text_encoder")

        # 2. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 3. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        num_frames = self._parse_num_frames(duration, fps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, pad latent frames to be divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        num_channels_latents = self.transformer.config.get("in_channels", 16)
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

        # 6. Prepare rotary embeddings
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), self.device
            )
            if self.transformer.config.get("use_rotary_positional_embeddings", False)
            else None
        )

        # 7. Prepare guidance
        do_classifier_free_guidance = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 8. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
            additional_frames=additional_frames,
            transformer_dtype=transformer_dtype,
            extra_step_kwargs=self.prepare_extra_step_kwargs(generator, eta),
            **kwargs,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents
        else:
            # Discard any padding frames that were added for CogVideoX 1.5
            if additional_frames > 0:
                latents = latents[:, additional_frames:]
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
