import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import HunyuanVideoShared


class HunyuanI2VEngine(HunyuanVideoShared):
    """Hunyuan Image-to-Video Engine Implementation"""

    def run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        prompt_2: Union[List[str], str] = None,
        negative_prompt: Union[List[str], str] = None,
        negative_prompt_2: Union[List[str], str] = None,
        height: int = 720,
        width: int = 1280,
        duration: int = 61,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 16,
        guidance_scale: float = 1.0,
        true_guidance_scale: float = 1.0,
        use_true_cfg_guidance: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        timesteps_as_indices: bool = True,
        max_sequence_length: int = 256,
        sigmas: List[float] = None,
        image_embed_interleave: Optional[int] = None,
        image_condition_type: Optional[str] = None,
        **kwargs,
    ):
        """Image-to-video generation following HunyuanVideoImageToVideoPipeline"""

        # 1. Process input image
        loaded_image = self._load_image(image)

        # Preprocess image
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

        # 4. Prepare image latents
        if self.transformer is not None:
            image_condition_type = getattr(
                self.transformer.config, "image_condition_type", "token_replace"
            )
        else:
            image_condition_type = (
                "token_replace"
                if image_condition_type is None
                else image_condition_type
            )

        image_embed_interleave = (
            image_embed_interleave
            if image_embed_interleave is not None
            else (
                2
                if image_condition_type == "latent_concat"
                else 4 if image_condition_type == "token_replace" else 1
            )
        )

        # 2. Encode prompts with image context
        (
            pooled_prompt_embeds,
            prompt_embeds,
            prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            image=image,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            image_embed_interleave=image_embed_interleave,
            **text_encoder_kwargs,
        )

        if negative_prompt is not None:
            (
                negative_pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                num_videos_per_prompt=num_videos,
                max_sequence_length=max_sequence_length,
                **text_encoder_kwargs,
            )

        if offload:
            self._offload("text_encoder")
            if self.llama_text_encoder is not None:
                self._offload("llama_text_encoder")

        # 3. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )

        if negative_prompt is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                self.device, dtype=transformer_dtype
            )

        if image_condition_type == "latent_concat":
            num_channels_latents = (
                getattr(self.transformer.config, "in_channels", 16) - 1
            ) // 2
        elif image_condition_type == "token_replace":
            num_channels_latents = getattr(self.transformer.config, "in_channels", 16)

        # Encode image to latents
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)  # Add temporal dimension
        image_latents = self.vae_encode(
            image_tensor_unsqueezed,
            offload=offload,
            sample_mode="mode",
            normalize_latents_dtype=torch.float32,
            dtype=torch.float32,
        )

        # Repeat for all frames
        num_frames = self._parse_num_frames(duration, fps)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        image_latents = image_latents.repeat(1, 1, num_latent_frames, 1, 1)

        batch_size = prompt_embeds.shape[0]

        # 5. Prepare latents
        latents = self._get_latents(
            height=height,
            width=width,
            duration=num_latent_frames,
            fps=fps,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=torch.float32,
            parse_frames=False,
        )

        # Mix latents with image latents
        t = torch.tensor([0.999]).to(device=self.device)
        latents = latents * t + image_latents * (1 - t)

        if image_condition_type == "token_replace":
            image_latents = image_latents[:, :, :1]

        # Create mask for image conditioning
        if image_condition_type == "latent_concat":
            image_latents[:, :, 1:] = 0
            mask = image_latents.new_ones(
                image_latents.shape[0], 1, *image_latents.shape[2:]
            )
            mask[:, :, 1:] = 0
        else:
            mask = None

        # 6. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 7. Prepare timesteps
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self._get_timesteps(
            sigmas=sigmas,
            scheduler=self.scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        # 8. Prepare guidance
        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = (
                torch.tensor(
                    [guidance_scale] * latents.shape[0],
                    dtype=transformer_dtype,
                    device=self.device,
                )
                * 1000.0
            )

        use_true_cfg_guidance = (
            true_guidance_scale > 1.0 and negative_prompt is not None
        )

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            true_guidance_scale=true_guidance_scale,
            use_true_cfg_guidance=use_true_cfg_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds.to(self.device),
                encoder_attention_mask=prompt_attention_mask.to(self.device).to(
                    transformer_dtype
                ),
                pooled_projections=pooled_prompt_embeds.to(self.device),
                guidance=guidance.to(self.device),
                attention_kwargs=attention_kwargs,
            ),
            unconditional_noise_pred_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds.to(self.device),
                    encoder_attention_mask=negative_prompt_attention_mask.to(
                        self.device
                    ).to(transformer_dtype),
                    pooled_projections=negative_pooled_prompt_embeds.to(self.device),
                    guidance=guidance.to(self.device),
                    attention_kwargs=attention_kwargs,
                )
                if (negative_prompt is not None and true_guidance_scale > 1.0)
                else None
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            image_condition_type=image_condition_type,
            image_latents=image_latents,
            transformer_dtype=transformer_dtype,
            mask=mask,
            **kwargs,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            if image_condition_type == "latent_concat":
                return latents[:, :, 1:, :, :]
            else:
                return latents
        else:
            if image_condition_type == "latent_concat":
                video_latents = latents[:, :, 4:, :, :]  # Skip first few frames
            else:
                video_latents = latents

            video = self.vae_decode(video_latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
