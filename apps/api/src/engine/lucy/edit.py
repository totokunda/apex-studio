import torch
from typing import Dict, Any, Callable, List, Union, Optional, Tuple
from PIL import Image
import numpy as np
from src.engine.wan.shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.utils.cache import empty_cache
import torchvision.transforms.functional as F


class LucyEditEngine(WanShared):
    """Lucy Edit Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels_latents = 48
        self.vae_scale_factor_spatial = 16
        self.vae_scale_factor_temporal = 4

    def run(
        self,
        video: Union[List[Image.Image], torch.Tensor],
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        duration: int | str = 81,
        guidance_scale: float = 5.0,
        fps: int = 24,
        num_videos: int = 1,
        guidance_scale_2: Optional[float] = None,
        boundary_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 1,
        render_on_step: bool = False,
        progress_callback: Callable = None,
        denoise_progress_callback: Callable = None,
        offload: bool = True,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting edit pipeline")

        # make sure height and width are divisible by 32
        height = height // 32 * 32
        width = width // 32 * 32

        # 1. Check inputs and setup defaults
        if self.config.get("boundary_ratio") is not None and boundary_ratio is None:
            boundary_ratio = self.config.get("boundary_ratio")

        if boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        # 2. Encode prompt
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            num_videos=num_videos,
            progress_callback=progress_callback,
            text_encoder_kwargs=text_encoder_kwargs,
            offload=offload,
        )

        transformer_dtype = self.component_dtypes.get("transformer", torch.float32)
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latents and condition latents
        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        video = self._load_video(video, fps=fps)

        for i in range(len(video)):
            video[i] = video[i].resize((width, height))

        # Preprocess video
        video_tensor = self.video_processor.preprocess_video(
            video, height=height, width=width
        )
        video_tensor = video_tensor.to(self.device, dtype=torch.float32)
        num_frames = self._parse_num_frames(duration, fps)

        if num_frames < video_tensor.shape[2]:
            video_tensor = video_tensor[:, :, :num_frames, :, :]
        elif num_frames > video_tensor.shape[2]:
            if video_tensor.shape[2] % 4 == 0:
                num_frames = (video_tensor.shape[2] // 4) * 4 - 3
            else:
                num_frames = (video_tensor.shape[2] // 4) * 4 + 1
            video_tensor = video_tensor[:, :, :num_frames, :, :]

        # Prepare noise latents
        latents = self._get_latents(
            height,
            width,
            num_frames=num_frames,
            duration=duration,
            fps=fps,
            num_channels_latents=self.num_channels_latents,
            batch_size=len(prompt),
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        safe_emit_progress(progress_callback, 0.35, "Initialized latent noise")

        # Prepare condition latents
        condition_latents_list = []
        for i in range(video_tensor.shape[0]):
            vid = video_tensor[i].unsqueeze(0)
            cond = self.vae_encode(vid, offload=False)
            condition_latents_list.append(cond)

        condition_latents = torch.cat(condition_latents_list, dim=0).to(torch.float32)
        if offload:
            self._offload("vae")

        # 5. Boundary timestep
        if boundary_ratio is not None:
            boundary_timestep = (
                boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        else:
            boundary_timestep = None

        # 6. Denoise
        mapped_denoise_progress = make_mapped_progress(progress_callback, 0.40, 0.92)
        denoise_progress_callback = denoise_progress_callback or mapped_denoise_progress
        safe_emit_progress(progress_callback, 0.40, "Starting denoising")

        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        expand_timesteps = self.config.get("expand_timesteps", False)

        final_guidance_scale = guidance_scale
        if guidance_scale_2 is not None:
            final_guidance_scale = [guidance_scale, guidance_scale_2]

        # Use custom denoise loop to support Lucy specific logic
        latents = self.lucy_denoise(
            timesteps=timesteps,
            latents=latents,
            latent_condition=condition_latents,
            scheduler=self.scheduler,
            guidance_scale=final_guidance_scale,
            boundary_timestep=boundary_timestep,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            attention_kwargs=attention_kwargs,
            transformer_dtype=transformer_dtype,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            render_on_step_interval=render_on_step_interval,
            denoise_progress_callback=denoise_progress_callback,
            expand_timesteps=expand_timesteps,
        )

        safe_emit_progress(progress_callback, 0.94, "Denoising complete")

        # 7. Decode
        if offload:
            if getattr(self, "transformer", None):
                self._offload("transformer")
            if getattr(self, "transformer_2", None):
                self._offload("transformer_2")

        safe_emit_progress(progress_callback, 0.96, "Transformers offloaded")

        # Denormalize
        video = self.vae_decode(latents, offload=offload)
        video = self._tensor_to_frames(video)

        safe_emit_progress(progress_callback, 1.0, "Completed edit pipeline")
        return video

    def lucy_denoise(
        self,
        timesteps: List[int],
        latents: torch.Tensor,
        latent_condition: torch.Tensor,
        scheduler: Any,
        guidance_scale: Union[float, List[float]],
        boundary_timestep: Optional[float],
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        attention_kwargs: Dict[str, Any],
        transformer_dtype: torch.dtype,
        render_on_step: bool,
        render_on_step_callback: Callable,
        render_on_step_interval: int,
        denoise_progress_callback: Callable,
        expand_timesteps: bool = False,
    ) -> torch.Tensor:

        total_steps = len(timesteps)
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")

        mask = torch.ones(latents.shape, dtype=torch.float32, device=self.device)

        with self._progress_bar(len(timesteps), desc=f"Sampling Lucy") as pbar:
            for i, t in enumerate(timesteps):

                if boundary_timestep is None or t >= boundary_timestep:
                    # High noise stage
                    if hasattr(self, "transformer_2") and getattr(
                        self, "transformer_2", None
                    ):
                        self._offload("transformer_2")
                        empty_cache()

                    if not getattr(self, "transformer", None):
                        self.load_component_by_type("transformer")
                    self.to_device(self.transformer)

                    current_model = self.transformer
                    current_guidance_scale = (
                        guidance_scale[0]
                        if isinstance(guidance_scale, list)
                        else guidance_scale
                    )
                else:
                    # Low noise stage
                    if getattr(self, "transformer", None):
                        self._offload("transformer")
                        empty_cache()

                    if not getattr(self, "transformer_2", None):
                        # Check if transformer_2 is loaded, if not load it
                        if not hasattr(self, "transformer_2"):
                            # If not in attributes, maybe load by name "transformer_2"
                            # The config should have it.
                            self.load_component_by_name("transformer_2")
                        self.to_device(self.transformer_2)

                    current_model = self.transformer_2
                    current_guidance_scale = (
                        guidance_scale[1]
                        if isinstance(guidance_scale, list)
                        else guidance_scale
                    )

                latent_model_input = torch.cat([latents, latent_condition], dim=1).to(
                    transformer_dtype
                )

                if expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if negative_prompt_embeds is not None:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (
                        noise_pred - noise_uncond
                    )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                pbar.update(1)
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps),
                    f"Denoising step {i + 1}/{total_steps}",
                )

        return latents
