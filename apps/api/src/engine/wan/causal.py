import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared


class WanCausalEngine(WanShared):
    """WAN Causal Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        image: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        fps: int = 16,
        num_videos: int = 1,
        seed: int | None = None,
        use_cfg_guidance: bool = False,
        progress_callback: Callable | None = None,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step: bool = False,
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 1,
        num_frame_per_block: int = 3,
        context_noise: float = 0.0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        offload: bool = True,
        num_inference_steps: int = 4,
        timesteps: List[int] | None = None,
        encoder_cache_size: int = 512,
        generator: torch.Generator | None = None,
        timesteps_as_indices: bool = True,
        **kwargs,
    ):

        safe_emit_progress(
            progress_callback, 0.0, "Starting WAN causal text-to-video pipeline"
        )

        kv_cache1 = []

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        batch_size = prompt_embeds.shape[0]

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Text encoder offloaded")

        latent_image = None
        latent_video = None
        num_input_frames = 0

        if image is not None:
            loaded_image = self._load_image(image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )

            preprocessed_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            ).to(self.device, dtype=torch.float32)

            latent_image = self.vae_encode(
                preprocessed_image.unsqueeze(2),
                offload=offload,
                sample_mode="mode",
                normalize_latents_dtype=torch.float32,
            )
            num_input_frames = latent_image.shape[2]
            safe_emit_progress(progress_callback, 0.18, "Prepared input image")
        elif video is not None:
            loaded_video = self._load_video(video, fps=fps)
            preprocessed_video = self.video_processor.preprocess_video(
                loaded_video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            latent_video = self.vae_encode(
                preprocessed_video,
                offload=offload,
                sample_mode="mode",
                normalize_latents_dtype=torch.float32,
            )
            num_input_frames = latent_video.shape[2]
            safe_emit_progress(progress_callback, 0.18, "Prepared input video")

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        transformer_dtype = self.component_dtypes["transformer"]

        latents, generator = self._get_latents(
            height,
            width,
            duration,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
            return_generator=True,
        )

        safe_emit_progress(progress_callback, 0.25, "Initialized latent noise")

        batch_size, num_channels, latent_frames, latent_height, latent_width = (
            latents.shape
        )

        # Initialize the KV cache
        head_dim = (
            self.transformer.blocks[0].attn1.inner_dim
            // self.transformer.blocks[0].attn1.heads
        )
        heads = self.transformer.blocks[0].attn1.heads
        pt, ph, pw = self.transformer.config.patch_size
        frame_seq_length = latent_height // ph * latent_width // pw
        if local_attn_size == -1:
            kv_cache_size = (
                (latent_frames + num_input_frames) // pt
            ) * frame_seq_length
        else:
            kv_cache_size = frame_seq_length * local_attn_size

        for _ in range(len(self.transformer.blocks)):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=self.device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=self.device
                    ),
                }
            )

        crossattn_cache = []
        for _ in range(len(self.transformer.blocks)):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, encoder_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, encoder_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "is_init": False,
                }
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000,
            training=True,
            device=self.device,
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(
            progress_callback, 0.35, "Scheduler ready and timesteps computed"
        )

        num_output_frames = num_input_frames + latent_frames
        num_blocks = latent_frames // num_frame_per_block
        all_num_frames = [num_frame_per_block] * num_blocks

        current_start_frame = 0
        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, latent_height, latent_width],
            dtype=latents.dtype,
            device=self.device,
        )

        if latent_image is not None or latent_video is not None:
            safe_emit_progress(progress_callback, 0.40, "Caching input frames")
            timestep = (
                torch.ones([batch_size, 1], device=self.device, dtype=torch.int64) * 0
            )
            if latent_image is not None:
                initial_latent = latent_image
                output[:, :, :num_input_frames, :, :] = initial_latent[
                    :, :, :num_input_frames, :, :
                ]
                latent_model_input = initial_latent.to(transformer_dtype)
                assert (num_input_frames - 1) % num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // num_frame_per_block

                self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                    attention_kwargs=attention_kwargs,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
                if render_on_step:
                    if render_on_step_callback:
                        self._render_step(latent_model_input, render_on_step_callback)

                current_start_frame += 1
            else:
                assert num_input_frames % num_frame_per_block == 0
                num_input_blocks = num_input_frames // num_frame_per_block
                initial_latent = latent_video

            with self._progress_bar(
                total=num_input_blocks,
                desc="Caching Input Frames",
                disable=num_input_blocks == 0,
            ) as pbar:
                for _ in range(num_input_blocks):
                    current_ref_latents = initial_latent[
                        :,
                        :,
                        current_start_frame : current_start_frame + num_frame_per_block,
                        :,
                        :,
                    ]
                    output[
                        :,
                        :,
                        current_start_frame : current_start_frame + num_frame_per_block,
                        :,
                        :,
                    ] = current_ref_latents

                    latent_model_input = current_ref_latents.to(transformer_dtype)

                    self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep * 0,
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                        attention_kwargs=attention_kwargs,
                        local_attn_size=local_attn_size,
                        sink_size=sink_size,
                    )

                    pbar.update(1)

                    if render_on_step:
                        if render_on_step_callback:
                            self._render_step(
                                current_ref_latents, render_on_step_callback
                            )

                    current_start_frame += num_frame_per_block

        safe_emit_progress(
            progress_callback,
            0.45,
            "Finished caching input frames; starting causal generation",
        )

        # Reserve a progress span for causal denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        total_denoise_steps = (
            len(all_num_frames) * len(timesteps) if timesteps is not None else 0
        )
        current_denoise_step = 0

        safe_emit_progress(progress_callback, 0.50, "Beginning causal denoising loop")
        # Initialize preview buffer for iterative rendering; clone so cached input frames are preserved
        self._preview_output = output.clone()
        self._preview_output_offload = offload

        with self._progress_bar(
            total=len(all_num_frames) * len(timesteps), desc="Causal Generation"
        ) as pbar:
            for current_num_frames in all_num_frames:
                latent = latents[
                    :,
                    :,
                    current_start_frame
                    - num_input_frames : current_start_frame
                    + current_num_frames
                    - num_input_frames,
                    :,
                    :,
                ]
                for i, t in enumerate(timesteps):
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=self.device,
                            dtype=torch.int64,
                        )
                        * t
                    )
                    latent_model_input = latent.to(transformer_dtype)

                    if i < len(timesteps) - 1:
                        flow_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            current_start=current_start_frame * frame_seq_length,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            return_dict=False,
                            attention_kwargs=attention_kwargs,
                            local_attn_size=local_attn_size,
                            sink_size=sink_size,
                        )[0]

                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

                        pred_x0 = self.scheduler.convert_flow_pred_to_x0(
                            flow_pred.flatten(0, 1),
                            latent_model_input.permute(0, 2, 1, 3, 4).flatten(0, 1),
                            timestep.flatten(0, 1),
                        ).unflatten(0, flow_pred.shape[:2])

                        next_timestep = timesteps[i + 1]

                        latent = self.scheduler.add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=self.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, flow_pred.shape[:2])
                        latent = latent.permute(0, 2, 1, 3, 4)

                    else:
                        flow_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            current_start=current_start_frame * frame_seq_length,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            return_dict=False,
                            attention_kwargs=attention_kwargs,
                            local_attn_size=local_attn_size,
                            sink_size=sink_size,
                        )[0]
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                        pred_x0 = self.scheduler.convert_flow_pred_to_x0(
                            flow_pred.flatten(0, 1),
                            latent_model_input.permute(0, 2, 1, 3, 4).flatten(0, 1),
                            timestep.flatten(0, 1),
                        ).unflatten(0, flow_pred.shape[:2])
                        latent = pred_x0.permute(0, 2, 1, 3, 4)

                    pbar.update(1)

                    if total_denoise_steps > 0:
                        current_denoise_step += 1
                        denoise_progress_callback(
                            current_denoise_step / float(total_denoise_steps),
                            f"Causal denoising step {current_denoise_step}/{total_denoise_steps}",
                        )

                output[
                    :,
                    :,
                    current_start_frame : current_start_frame + current_num_frames,
                    :,
                    :,
                ] = latent

                if render_on_step:
                    if render_on_step_callback:
                        self._render_step(output, render_on_step_callback)

                context_timestep = torch.ones_like(timestep) * context_noise
                # Clean context cache
                self.transformer(
                    hidden_states=latent,
                    encoder_hidden_states=prompt_embeds,
                    timestep=context_timestep,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                    attention_kwargs=attention_kwargs,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    return_dict=False,
                )
                current_start_frame += current_num_frames

        safe_emit_progress(progress_callback, 0.92, "Causal denoising complete")

        if offload:
            self._offload("transformer")

        if return_latents:

            safe_emit_progress(progress_callback, 1.0, "Returning latent video")
            return output
        else:
            video = self.vae_decode(output, offload=offload)

            safe_emit_progress(progress_callback, 0.96, "Decoded latents to video")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(
                progress_callback, 1.0, "Completed WAN causal text-to-video pipeline"
            )
            return postprocessed_video
