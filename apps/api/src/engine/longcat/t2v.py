from .shared import LongCatShared
from typing import Union, List, Optional, Dict, Any
import torch


class LongCatT2VEngine(LongCatShared):
    """LongCat Text-to-Video Engine Implementation"""

    def _get_refine_engine(self):
        # Lazy accessor for the optional refine sub-engine
        return self.sub_engines.get("refine")

    def run(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        duration: str | int = 93,
        fps: int = 15,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        offload: bool = True,
        enable_refine: bool = False,
        refine_num_inference_steps: int = 50,
        **kwargs,
    ):
        """Text-to-video generation following LongCatPipeline.

        If ``refine=True`` and ``return_latents=False``, a second 720p refinement
        stage is run using the `refine` sub-engine, mirroring the official demo.
        """

        num_frames = self._parse_num_frames(duration, fps)

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.transformer is not None and self.transformer.cp_split_hw is not None:
            scale_factor_spatial *= max(self.transformer.cp_split_hw)

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
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode input prompt
        dit_dtype = self.component_dtypes["transformer"]

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            dtype=dit_dtype,
            device=device,
        )

        if offload:
            self._offload("text_encoder")

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        if use_distill:
            self.apply_distill_lora()

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # 4. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)

        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        with self._progress_bar(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    # Calculate the optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    # Reshape for broadcasting
                    st_star = st_star.view(B, 1, 1, 1)

                    noise_pred = noise_pred_uncond * st_star + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond * st_star
                    )

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents

        # Decode 480p (stage 1) video
        output_video = self.vae_decode(latents, offload=offload)
        output_video = self._tensor_to_frames(output_video)

        # Optional 720p refinement using the refine sub-engine
        if enable_refine:
            self.offload_engine(self)
            refine_engine = self._get_refine_engine()
            if refine_engine is None:
                self.logger.warning(
                    "[LongCatT2VEngine] `refine=True` was requested but no `refine` sub-engine "
                    "is configured. Returning base 480p video."
                )
                return output_video

            refined_batch = []
            # output_video is List[Video]; mirror interactive engine behaviour
            num_samples = len(output_video)

            for idx in range(num_samples):
                stage1_video = output_video[idx]

                # Resolve a single prompt string for this sample
                if isinstance(prompt, str) or prompt is None:
                    cur_prompt = prompt
                elif isinstance(prompt, list) and len(prompt) > idx:
                    cur_prompt = prompt[idx]
                else:
                    cur_prompt = None

                refine_output = refine_engine.run(
                    image=None,
                    video=None,
                    prompt=cur_prompt,
                    stage1_video=stage1_video,
                    num_cond_frames=0,
                    num_inference_steps=refine_num_inference_steps,
                    num_videos_per_prompt=1,
                    generator=generator,
                    spatial_refine_only=False,
                    offload=offload,
                )

                # Engines return [OutputVideo]; we operate on the single sample
                refined_batch.append(refine_output[0])

            # Free refine engine components from device memory
            self.offload_engine(refine_engine)
            output_video = refined_batch

        return output_video
