from typing import Any, Callable, Dict, List, Optional, Union
import torch
from .shared import Kandinsky5Shared


class Kandinsky5T2VEngine(Kandinsky5Shared):
    """Kandinsky 5.0 text-to-video engine implemented from the diffusers pipeline."""

    @torch.no_grad()
    def run(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds_qwen: Optional[torch.Tensor] = None,
        prompt_embeds_clip: Optional[torch.Tensor] = None,
        negative_prompt_embeds_qwen: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clip: Optional[torch.Tensor] = None,
        prompt_cu_seqlens: Optional[torch.Tensor] = None,
        negative_prompt_cu_seqlens: Optional[torch.Tensor] = None,
        render_on_step_interval: int = 3,
        return_latents: bool = False,
        output_type: Optional[str] = "pil",
        progress_callback: Callable | None = None,
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        max_sequence_length: int = 512,
        seed: Optional[int] = None,
        **kwargs,
    ):

        # make sure height and width are divisible by 64
        height = height // 64 * 64
        width = width // 64 * 64

        # Sequential component loading to limit concurrent GPU memory
        if getattr(self, "text_encoder", None) is None:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        if getattr(self, "text_encoder_2", None) is None:
            self.load_component_by_name("text_encoder_2")
        self.to_device(self.text_encoder_2)

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
        self._interrupt = False

        device = self.device
        dtype = (self.component_dtypes or {}).get("transformer")

        if generator is None and seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_qwen.shape[0]

        if prompt_embeds_qwen is None:
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = (
                self.encode_prompt(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
            )

        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

            if isinstance(negative_prompt, str):
                negative_prompt = (
                    [negative_prompt] * len(prompt)
                    if prompt is not None
                    else [negative_prompt]
                )
            elif len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt` must have same length as `prompt`. Got {len(negative_prompt)} vs {len(prompt)}."
                )

            if negative_prompt_embeds_qwen is None:
                (
                    negative_prompt_embeds_qwen,
                    negative_prompt_embeds_clip,
                    negative_prompt_cu_seqlens,
                ) = self.encode_prompt(
                    prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        # Offload text encoders before heavy steps
        self._offload("text_encoder")
        self._offload("text_encoder_2")

        if getattr(self, "scheduler", None) is None:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Load transformer only when needed
        if getattr(self, "transformer", None) is None:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        num_channels_latents = self.transformer.config.in_visual_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            dtype,
            device,
            generator,
            latents,
        )

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        visual_rope_pos = [
            torch.arange(num_latent_frames, device=device),
            torch.arange(height // self.vae_scale_factor_spatial // 2, device=device),
            torch.arange(width // self.vae_scale_factor_spatial // 2, device=device),
        ]

        text_rope_pos = torch.arange(
            prompt_cu_seqlens.diff().max().item(), device=device
        )
        negative_text_rope_pos = (
            torch.arange(negative_prompt_cu_seqlens.diff().max().item(), device=device)
            if negative_prompt_cu_seqlens is not None
            else None
        )

        sparse_params = self.get_sparse_params(latents, device)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        component = self.get_component_by_type("transformer")
        extra_kwargs = component.get("extra_kwargs", {})
        if (
            "attention_backend" in extra_kwargs
            and extra_kwargs["attention_backend"] == "flex"
        ):
            self.transformer.set_attention_backend("flex")
            self.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising Kandinsky 5.0"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.unsqueeze(0).repeat(batch_size * num_videos_per_prompt)

                pred_velocity = self.transformer(
                    hidden_states=latents.to(dtype),
                    encoder_hidden_states=prompt_embeds_qwen.to(dtype),
                    pooled_projections=prompt_embeds_clip.to(dtype),
                    timestep=timestep.to(dtype),
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=self._get_scale_factor(height, width),
                    sparse_params=sparse_params,
                    return_dict=True,
                ).sample

                if (
                    self.do_classifier_free_guidance
                    and negative_prompt_embeds_qwen is not None
                ):

                    uncond_pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype),
                        encoder_hidden_states=negative_prompt_embeds_qwen.to(dtype),
                        pooled_projections=negative_prompt_embeds_clip.to(dtype),
                        timestep=timestep.to(dtype),
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=negative_text_rope_pos,
                        scale_factor=self._get_scale_factor(height, width),
                        sparse_params=sparse_params,
                        return_dict=True,
                    ).sample

                    pred_velocity = uncond_pred_velocity + guidance_scale * (
                        pred_velocity - uncond_pred_velocity
                    )

                latents[:, :, :, :, :num_channels_latents] = self.scheduler.step(
                    pred_velocity,
                    t,
                    latents[:, :, :, :, :num_channels_latents],
                    return_dict=False,
                )[0]

                if progress_callback is not None:
                    progress_callback(
                        min((i + 1) / num_inference_steps, 1.0),
                        f"Denoising step {i + 1}/{num_inference_steps}",
                    )
                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    pbar.update()

        latents = latents[:, :, :, :, :num_channels_latents]

        # Offload transformer after denoising
        self._offload("transformer")

        if return_latents:
            return latents
        else:
            video = latents.reshape(
                batch_size,
                num_videos_per_prompt,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                num_channels_latents,
            )
            video = video.permute(0, 1, 5, 2, 3, 4)
            video = video.reshape(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            video = self.vae_decode(video, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)

            return postprocessed_video
