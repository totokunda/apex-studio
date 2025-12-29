import torch
from typing import Dict, Any, Callable, List, Union
import numpy as np
from .shared import HunyuanVideoShared


class HunyuanT2VEngine(HunyuanVideoShared):
    """Hunyuan Text-to-Video Engine Implementation"""

    def run(
        self,
        prompt: Union[List[str], str],
        prompt_2: Union[List[str], str, None] = None,
        negative_prompt: Union[List[str], str, None] = None,
        negative_prompt_2: Union[List[str], str, None] = None,
        height: int = 720,
        width: int = 1280,
        duration: str | int = 10,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 6.0,
        true_guidance_scale: float = 1.0,
        use_true_cfg_guidance: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        max_sequence_length: int = 256,
        sigmas: List[float] | np.ndarray | None = None,
        **kwargs,
    ):
        """Text-to-video generation following HunyuanVideoPipeline"""

        # 1. Encode prompts
        (
            pooled_prompt_embeds,
            prompt_embeds,
            prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
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

        # 2. Load transformer
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

        # 3. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 4. Prepare timesteps
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=self.scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
        )

        batch_size = prompt_embeds.shape[0]

        # 5. Prepare latents
        num_channels_latents = getattr(self.transformer.config, "in_channels", 16)
        latents = self._get_latents(
            height,
            width,
            duration,
            fps,
            batch_size,
            num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=torch.float32,
        )

        # 6. Prepare guidance
        guidance = (
            torch.tensor(
                [guidance_scale] * latents.shape[0],
                device=self.device,
                dtype=transformer_dtype,
            )
            # * 1000.0
        )

        guidance = guidance * 1000.0

        use_true_cfg_guidance = (
            true_guidance_scale > 1.0 and negative_prompt is not None
        )

        # 7. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            true_guidance_scale=true_guidance_scale,
            use_true_cfg_guidance=use_true_cfg_guidance,
            num_inference_steps=num_inference_steps,
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
            transformer_dtype=transformer_dtype,
            **kwargs,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents
        else:
            if self.vae is None:
                self.load_component_by_type("vae")
                self.to_device(self.vae)

            video = self.vae_decode(latents, offload=offload)
            video = self._tensor_to_frames(video)
            return video
