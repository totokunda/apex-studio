import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import CogVideoShared


class CogVideoI2VEngine(CogVideoShared):
    """CogVideo Image-to-Video Engine Implementation"""

    def run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = "",
        height: int = 480,
        width: int = 720,
        duration: int | str = 49,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
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
        """Image-to-video generation following CogVideoXImageToVideoPipeline"""

        # 1. Process input image
        loaded_image = self._load_image(image)

        # Preprocess image
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

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

        transformer_config = self.load_config_by_type("transformer")

        # 3. Load transformer

        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        # 6. Prepare latents
        num_frames = self._parse_num_frames(duration)
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, pad latent frames to be divisible by patch_size_t
        patch_size_t = transformer_config.get("patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        # Encode image to latents
        image_tensor = image_tensor.to(dtype=prompt_embeds.dtype, device=self.device)
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)  # Add temporal dimension

        # Use VAE to encode image
        if isinstance(generator, list):
            image_latents = [
                self.vae_encode(
                    image_tensor_unsqueezed[i].unsqueeze(0),
                    sample_mode="sample",
                    sample_generator=generator[i],
                    dtype=prompt_embeds.dtype,
                )
                for i in range(image_tensor_unsqueezed.shape[0])
            ]
        else:
            image_latents = [
                self.vae_encode(
                    img.unsqueeze(0),
                    sample_mode="sample",
                    sample_generator=generator,
                    dtype=prompt_embeds.dtype,
                )
                for img in image_tensor_unsqueezed
            ]

        image_latents = torch.cat(image_latents, dim=0)

        # Create padding for remaining frames
        padding_shape = (
            num_videos,
            latent_frames - 1,
            transformer_config.get("in_channels", 16) // 2,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(
            padding_shape, device=self.device, dtype=prompt_embeds.dtype
        )

        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Handle CogVideoX 1.5 padding
        if patch_size_t is not None:
            first_frame = image_latents[:, : image_latents.size(1) % patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=1)

        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # Prepare noise latents
        latent_channels = transformer_config.get("in_channels", 16) // 2
        batch_size = prompt_embeds.shape[0]
        latents = self._get_latents(
            height,
            width,
            latent_num_frames,
            batch_size=batch_size,
            num_channels_latents=latent_channels,
            seed=seed,
            generator=generator,
            dtype=transformer_dtype,
            parse_frames=False,
            order="BFC",
        )

        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height,
            width,
            latents.size(1),
            self.device,
            transformer_config=transformer_config,
        )

        # 8. Prepare ofs embeddings (for CogVideoX 1.5)
        ofs_emb = None
        if transformer_config.get("ofs_embed_dim", None) is not None:
            ofs_emb = latents.new_full((1,), fill_value=2.0)

        # 9. Prepare guidance
        do_classifier_free_guidance = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        # 10. Denoising loop
        latents = self.denoise(
            latents=latents,
            image_latents=image_latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                ofs=ofs_emb,
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
