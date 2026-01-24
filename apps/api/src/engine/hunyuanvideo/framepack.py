import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import math
from .shared import HunyuanVideoShared


class HunyuanFramepackEngine(HunyuanVideoShared):
    """Hunyuan Framepack Engine Implementation"""

    def _soft_append(
        self, history: torch.Tensor, current: torch.Tensor, overlap: int = 0
    ):
        """Soft append with blending for framepack generation"""
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert (
            history.shape[2] >= overlap
        ), f"Current length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert (
            current.shape[2] >= overlap
        ), f"History length ({current.shape[2]}) must be >= overlap ({overlap})"

        weights = torch.linspace(
            1, 0, overlap, dtype=history.dtype, device=history.device
        ).view(1, 1, -1, 1, 1)
        blended = (
            weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        )
        output = torch.cat(
            [history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2
        )

        return output.to(history)

    def run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        last_image: Optional[
            Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor]
        ] = None,
        prompt_2: Union[List[str], str] = None,
        negative_prompt: Union[List[str], str] = None,
        negative_prompt_2: Union[List[str], str] = None,
        height: int = 480,
        width: int = 832,
        duration: str | int = 129,
        latent_window_size: int = 9,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 30,
        guidance_scale: float = 6.0,
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
        exp_max: float = 7.0,
        sigmas: List[float] = None,
        sampling_type: str = "inverted_anti_drifting",
        **kwargs,
    ):
        """Framepack generation following HunyuanVideoFramepackPipeline"""

        # 1. Process input images
        loaded_image = self._load_image(image)
        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

        loaded_image = loaded_image.resize((width, height))

        last_image_tensor = None
        if last_image is not None:
            loaded_last_image = self._load_image(last_image)
            loaded_last_image, height, width = self._aspect_ratio_resize(
                loaded_last_image, max_area=height * width
            )

            last_image_tensor = self.video_processor.preprocess(
                loaded_last_image, height, width
            ).to(self.device)

            loaded_last_image = loaded_last_image.resize((width, height))

        # 2. Encode prompts
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

        batch_size = prompt_embeds.shape[0]

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

        clip_image_encoder = self.helpers["clip"]
        self.to_device(clip_image_encoder)

        image_embeds = clip_image_encoder(loaded_image, hidden_states_layer=-1).to(
            self.device
        )

        if last_image_tensor is not None:
            last_image_embeds = clip_image_encoder(
                loaded_last_image, hidden_states_layer=-1
            ).to(self.device)
            # Blend embeddings as in the original implementation
            image_embeds = (image_embeds + last_image_embeds) / 2

        if offload:
            self._offload("text_encoder")
            if self.llama_text_encoder is not None:
                self._offload("llama_text_encoder")

        # 4. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        prompt_attention_mask = prompt_attention_mask.to(
            self.device, dtype=transformer_dtype
        )
        image_embeds = image_embeds.to(self.device, dtype=transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                self.device, dtype=transformer_dtype
            )
        # 5. Prepare image latents
        num_channels_latents = getattr(self.transformer.config, "in_channels", 16)

        # Encode first image
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)
        image_latents = self.vae_encode(
            image_tensor_unsqueezed,
            offload=offload,
            sample_mode="sample",
            dtype=torch.float32,
        )

        # Encode last image if provided
        last_image_latents = None
        if last_image_tensor is not None:
            last_image_tensor_unsqueezed = last_image_tensor.unsqueeze(2)
            last_image_latents = self.vae_encode(
                last_image_tensor_unsqueezed,
                offload=offload,
                sample_mode="sample",
                dtype=torch.float32,
            )

        # 6. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 7. Framepack generation loop
        num_frames = self._parse_num_frames(duration, fps)
        window_num_frames = (
            latent_window_size - 1
        ) * self.vae_scale_factor_temporal + 1
        num_latent_sections = max(
            1, (num_frames + window_num_frames - 1) // window_num_frames
        )
        history_video = None
        total_generated_latent_frames = 0

        # Initialize history based on sampling type
        if sampling_type == "inverted_anti_drifting":
            history_sizes = [1, 2, 16]
        else:  # vanilla
            history_sizes = [16, 2, 1]

        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            math.ceil(height / self.vae_scale_factor_spatial),
            math.ceil(width / self.vae_scale_factor_spatial),
            device=self.device,
            dtype=torch.float32,
        )

        if sampling_type == "vanilla":
            history_latents = torch.cat([history_latents, image_latents], dim=2)
            total_generated_latent_frames += 1

        # 8. Guidance preparation
        guidance = (
            torch.tensor(
                [guidance_scale] * batch_size,
                dtype=transformer_dtype,
                device=self.device,
            )
            * 1000.0
        )

        use_true_cfg_guidance = (
            true_guidance_scale > 1.0 and negative_prompt is not None
        )

        # 9. Generation loop for each section
        with self._progress_bar(
            num_latent_sections, desc="Generating sections"
        ) as pbar:
            for k in range(num_latent_sections):
                # Prepare latents for this section
                latents = self._get_latents(
                    height=height,
                    width=width,
                    duration=window_num_frames,
                    fps=fps,
                    batch_size=batch_size,
                    num_channels_latents=num_channels_latents,
                    seed=seed,
                    generator=generator,
                    dtype=torch.float32,
                )

                # Prepare timesteps with dynamic shift
                if sigmas is None:
                    sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

                # Calculate shift based on sequence length (from framepack implementation)
                image_seq_len = (
                    latents.shape[2]
                    * latents.shape[3]
                    * latents.shape[4]
                    / getattr(self.transformer.config, "patch_size", 2) ** 2
                )
                mu = self._calculate_shift(
                    image_seq_len,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 4096),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15),
                )
                mu = min(mu, math.log(exp_max))

                input_timesteps, num_inference_steps = self._get_timesteps(
                    scheduler=self.scheduler,
                    timesteps=timesteps,
                    timesteps_as_indices=timesteps_as_indices,
                    num_inference_steps=num_inference_steps,
                    mu=mu,
                    sigmas=sigmas,
                )

                # Prepare history latents for this section
                if sampling_type == "inverted_anti_drifting":
                    latent_paddings = list(reversed(range(num_latent_sections)))
                    if num_latent_sections > 4:
                        latent_paddings = [3] + [2] * (num_latent_sections - 3) + [1, 0]

                    is_first_section = k == 0
                    is_last_section = k == num_latent_sections - 1
                    latent_padding_size = latent_paddings[k] * latent_window_size

                    indices = torch.arange(
                        0,
                        sum(
                            [1, latent_padding_size, latent_window_size, *history_sizes]
                        ),
                    )

                    (
                        indices_prefix,
                        indices_padding,
                        indices_latents,
                        indices_latents_history_1x,
                        indices_latents_history_2x,
                        indices_latents_history_4x,
                    ) = indices.split(
                        [1, latent_padding_size, latent_window_size, *history_sizes],
                        dim=0,
                    )

                    indices_latents_clean = torch.cat(
                        [indices_prefix, indices_latents_history_1x], dim=0
                    )

                    latents_prefix = image_latents
                    latents_history_1x, latents_history_2x, latents_history_4x = (
                        history_latents[:, :, : sum(history_sizes)].split(
                            history_sizes, dim=2
                        )
                    )

                    if last_image_latents is not None and is_first_section:
                        latents_history_1x = last_image_latents

                    latents_clean = torch.cat(
                        [latents_prefix, latents_history_1x], dim=2
                    )

                else:  # vanilla
                    indices = torch.arange(
                        0, sum([1, *history_sizes, latent_window_size])
                    )
                    (
                        indices_prefix,
                        indices_latents_history_4x,
                        indices_latents_history_2x,
                        indices_latents_history_1x,
                        indices_latents,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)

                    indices_latents_clean = torch.cat(
                        [indices_prefix, indices_latents_history_1x], dim=0
                    )

                    latents_prefix = image_latents
                    latents_history_4x, latents_history_2x, latents_history_1x = (
                        history_latents[:, :, -sum(history_sizes) :].split(
                            history_sizes, dim=2
                        )
                    )

                    latents_clean = torch.cat(
                        [latents_prefix, latents_history_1x], dim=2
                    )

                latents = self.denoise(
                    latents=latents,
                    timesteps=input_timesteps,
                    scheduler=self.scheduler,
                    true_guidance_scale=true_guidance_scale,
                    use_true_cfg_guidance=use_true_cfg_guidance,
                    noise_pred_kwargs=dict(
                        indices_latents=indices_latents,
                        latents_clean=latents_clean.to(transformer_dtype),
                        indices_latents_clean=indices_latents_clean,
                        image_embeds=image_embeds.to(transformer_dtype),
                        latents_history_2x=latents_history_2x.to(transformer_dtype),
                        indices_latents_history_2x=indices_latents_history_2x,
                        latents_history_4x=latents_history_4x.to(transformer_dtype),
                        indices_latents_history_4x=indices_latents_history_4x,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance,
                        attention_kwargs=attention_kwargs,
                    ),
                    unconditional_noise_pred_kwargs=(
                        dict(
                            indices_latents=indices_latents,
                            latents_clean=latents_clean.to(transformer_dtype),
                            indices_latents_clean=indices_latents_clean,
                            image_embeds=image_embeds.to(transformer_dtype),
                            latents_history_2x=latents_history_2x.to(transformer_dtype),
                            indices_latents_history_2x=indices_latents_history_2x,
                            latents_history_4x=latents_history_4x.to(transformer_dtype),
                            indices_latents_history_4x=indices_latents_history_4x,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_attention_mask=negative_prompt_attention_mask,
                            pooled_projections=negative_pooled_prompt_embeds,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                        )
                        if (use_true_cfg_guidance and negative_prompt is not None)
                        else None
                    ),
                    render_on_step=render_on_step,
                    render_on_step_callback=render_on_step_callback,
                    transformer_dtype=transformer_dtype,
                    num_inference_steps=num_inference_steps,
                    num_latent_sections=num_latent_sections,
                    **kwargs,
                )

                # Update history
                if sampling_type == "inverted_anti_drifting":
                    if is_last_section:
                        latents = torch.cat([image_latents, latents], dim=2)
                    total_generated_latent_frames += latents.shape[2]
                    history_latents = torch.cat([latents, history_latents], dim=2)
                    real_history_latents = history_latents[
                        :, :, :total_generated_latent_frames
                    ]
                    section_latent_frames = (
                        (latent_window_size * 2 + 1)
                        if is_last_section
                        else (latent_window_size * 2)
                    )
                    index_slice = (
                        slice(None),
                        slice(None),
                        slice(0, section_latent_frames),
                    )

                else:  # vanilla
                    total_generated_latent_frames += latents.shape[2]
                    history_latents = torch.cat([history_latents, latents], dim=2)
                    real_history_latents = history_latents[
                        :, :, -total_generated_latent_frames:
                    ]
                    section_latent_frames = latent_window_size * 2
                    index_slice = (
                        slice(None),
                        slice(None),
                        slice(-section_latent_frames, None),
                    )

                if history_video is None:
                    if not return_latents:
                        current_latents = real_history_latents
                        history_video = self.vae_decode(
                            current_latents, offload=offload
                        )
                    else:
                        history_video = [real_history_latents]
                else:
                    if not return_latents:
                        overlapped_frames = (
                            latent_window_size - 1
                        ) * self.vae_scale_factor_temporal + 1
                        current_latents = real_history_latents[index_slice]
                        current_video = self.vae_decode(
                            current_latents, offload=offload
                        )

                        if sampling_type == "inverted_anti_drifting":
                            history_video = self._soft_append(
                                current_video, history_video, overlapped_frames
                            )
                        else:  # vanilla
                            history_video = self._soft_append(
                                history_video, current_video, overlapped_frames
                            )
                    else:
                        history_video.append(real_history_latents)

                pbar.update(1)

        if offload:
            self._offload("transformer")

        if return_latents:
            return history_video
        else:
            # Ensure proper frame count
            generated_frames = history_video.size(2)
            generated_frames = (
                generated_frames - 1
            ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            history_video = history_video[:, :, :generated_frames]
            postprocessed_video = self._tensor_to_frames(history_video)
            return postprocessed_video
