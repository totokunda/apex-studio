import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
from .shared import WanShared
from torch.nn import functional as F
from torchvision import transforms as T
from diffusers.utils.torch_utils import randn_tensor
from src.utils.models.wan import match_and_blend_colors
import numpy as np
import math
from src.utils.progress import safe_emit_progress, make_mapped_progress


class WanMultitalkEngine(WanShared):
    """WAN MultiTalk (Audio-driven) Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        image: Union[Image.Image, str, np.ndarray, torch.Tensor] | None = None,
        video: Union[List[Image.Image], str, np.ndarray, torch.Tensor, None] = None,
        audio_paths: Optional[Dict[str, str]] = None,
        person_1_audio: Optional[str] = None,
        person_2_audio: Optional[str] = None,
        audio_type: str = "para",
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 81,
        max_num_frames: int = 1000,
        num_inference_steps: int = 40,
        num_videos: int = 1,
        seed: int | None = None,
        motion_frames: int = 25,
        fps: int = 16,
        guidance_scale: float = 5.0,
        audio_guidance_scale: float = 4.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        face_scale: float = 0.05,
        color_correction_strength: float = 1.0,
        bbox: Optional[Dict[str, List[float]]] = None,
        render_on_step_interval: int = 1,
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        Generate MultiTalk video from image, text prompt, and audio inputs.

        Args:
            prompt: Text prompt for the video
            image: Input conditioning image (path or PIL Image)
            audio_paths: Dictionary mapping person names to audio file paths
            audio_embeddings: Pre-computed audio embeddings
            audio_type: Type of audio combination ("para" or "add")
            negative_prompt: Negative text prompt
            height: Output video height
            width: Output video width
            num_frames: Number of frames to generate
            motion_frames: Number of motion frames for extended generation
            num_inference_steps: Number of diffusion steps
            num_videos: Number of videos to generate
            seed: Random seed
            fps: Frames per second
            guidance_scale: Text guidance scale
            audio_guidance_scale: Audio guidance scale
            bbox: Bounding boxes for multiple people
            shift: Timestep transform shift parameter
        """

        safe_emit_progress(progress_callback, 0.0, "Starting multitalk pipeline")
        num_frames = self._parse_num_frames(duration, fps)
        use_cfg_guidance = guidance_scale > 1.0 and negative_prompt is not None

        assert (
            image is not None or video is not None
        ), "Either image or video must be provided"

        if image is not None:
            safe_emit_progress(progress_callback, 0.02, "Loading conditioning image")
            loaded_image = self._load_image(image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width, mod_value=16
            )

            safe_emit_progress(progress_callback, 0.05, "Preprocessing conditioning image")
            cond_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            cond_image = cond_image.unsqueeze(2)

        if video is not None:
            safe_emit_progress(progress_callback, 0.02, "Loading conditioning video")
            input_video = self._load_video(video, fps=fps)
            image = input_video[0]
            safe_emit_progress(progress_callback, 0.05, "Preprocessing conditioning video")
            for idx, frame in enumerate(input_video):
                frame, height, width = self._aspect_ratio_resize(
                    frame, max_area=height * width, mod_value=16
                )
                input_video[idx] = frame

            loaded_image = input_video[0]
            input_video = self.video_processor.preprocess_video(
                input_video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            cond_image = input_video[:, :, :1, :, :]
        else:
            input_video = None

        cond_frame = None

        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()

        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if (
            audio_paths is None
            and person_1_audio is not None
            and person_2_audio is None
        ):
            audio_paths = {
                "person1": person_1_audio,
            }
        elif (
            audio_paths is None
            and person_1_audio is not None
            and person_2_audio is not None
        ):
            audio_paths = {
                "person1": person_1_audio,
                "person2": person_2_audio,
            }

        preprocessor = self.helpers["wan.multitalk"]

        safe_emit_progress(progress_callback, 0.08, "Preparing audio/masks inputs")
        processed_inputs = preprocessor(
            image=image,
            audio_paths=audio_paths,
            audio_type=audio_type,
            num_frames=num_frames,
            vae_scale=self.vae_scale_factor_temporal,
            bbox=bbox,
            face_scale=face_scale,
        )

        human_masks = processed_inputs["human_masks"]
        human_num = processed_inputs["human_num"]
        full_audio_embs = processed_inputs["audio_embeddings"]
        safe_emit_progress(progress_callback, 0.12, "Prepared audio embeddings and masks")

        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = num_frames
        cur_motion_frames_num = 1
        audio_start_idx = 0
        arrive_last_frame = False
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        self._preview_video_list = []
        self._preview_max_num_frames = max_num_frames
        self._preview_num_frames = num_frames
        self._preview_using_video_input = video is not None
        self._preview_full_audio_embs = full_audio_embs
        self._preview_miss_lengths = []
        gen_latents_list = []
        is_first_clip = True

        transformer_dtype = self.component_dtypes["transformer"]

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.14, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.15, "Text encoder loaded")

        safe_emit_progress(progress_callback, 0.16, "Moving text encoder to device")
        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.17, "Encoding prompt")
        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            safe_emit_progress(progress_callback, 0.18, "Encoding negative prompt")
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None
        safe_emit_progress(
            progress_callback,
            0.19,
            "Prepared negative prompt" if negative_prompt_embeds is not None else "Skipped negative prompt",
        )

        if offload:
            safe_emit_progress(progress_callback, 0.20, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 0.21, "Text encoder offloaded")

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.22, "Loading transformer")
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)
            safe_emit_progress(progress_callback, 0.23, "Transformer loaded")

        using_video_input = input_video is not None

        # Estimate total frame budget for monotonic progress across multiple clips
        try:
            per_human_frames = [
                int(full_audio_embs[h].shape[0]) for h in range(int(human_num))
            ]
            total_target_frames = min(int(max_num_frames), min(per_human_frames)) if per_human_frames else int(max_num_frames)
        except Exception:
            total_target_frames = int(max_num_frames)
        total_target_frames = max(1, int(total_target_frames))

        # Reserve progress spans: pre-clip prep [0.30, 0.55], denoise [0.55, 0.90], decode/post [0.90, 0.98]
        pre_clip_progress = make_mapped_progress(progress_callback, 0.30, 0.55)
        denoise_progress = make_mapped_progress(progress_callback, 0.55, 0.90)
        decode_progress = make_mapped_progress(progress_callback, 0.90, 0.98)
        safe_emit_progress(progress_callback, 0.24, "Starting clip generation")

        clip_idx = 0
        while True:
            clip_idx += 1
            clip_start = min(1.0, float(audio_start_idx) / float(total_target_frames))
            clip_end = min(1.0, float(min(audio_end_idx, total_target_frames)) / float(total_target_frames))
            # Ensure a non-zero span to avoid progress staying constant for tiny/edge clips
            if clip_end <= clip_start:
                clip_end = min(1.0, clip_start + (1.0 / float(total_target_frames)))

            clip_pre_progress = make_mapped_progress(pre_clip_progress, clip_start, clip_end)
            clip_denoise_progress = make_mapped_progress(denoise_progress, clip_start, clip_end)
            clip_decode_progress = make_mapped_progress(decode_progress, clip_start, clip_end)

            safe_emit_progress(
                clip_pre_progress,
                0.0,
                f"Preparing clip {clip_idx} (frames {audio_start_idx}:{audio_end_idx})",
            )
            audio_embs = []
            # split audio with window size
            for human_idx in range(human_num):
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(
                    center_indices, min=0, max=full_audio_embs[human_idx].shape[0] - 1
                )
                audio_emb = full_audio_embs[human_idx][center_indices][None, ...].to(
                    self.device
                )
                audio_embs.append(audio_emb)

            audio_embs = torch.cat(audio_embs, dim=0).to(transformer_dtype)
            safe_emit_progress(clip_pre_progress, 0.10, "Prepared audio window embeddings")

            latent_height, latent_width = (
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            # get mask
            safe_emit_progress(clip_pre_progress, 0.15, "Preparing masks")
            mask_lat_size = torch.ones(
                batch_size,
                1,
                num_frames,
                latent_height,
                latent_width,
                device=self.device,
            )
            # For InfiniteTalk (video input), only the first frame is preserved; for image mode, preserve cur_motion_frames_num
            if using_video_input:
                mask_lat_size[:, :, 1:] = 0
            else:
                mask_lat_size[:, :, cur_motion_frames_num:] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
            )

            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )

            mask_lat_size = mask_lat_size.view(
                batch_size,
                -1,
                self.vae_scale_factor_temporal,
                latent_height,
                latent_width,
            )

            mask_lat_size = mask_lat_size.transpose(1, 2)

            # get clip embedding
            clip_processor = self.helpers["clip"]
            safe_emit_progress(clip_pre_progress, 0.22, "Moving CLIP to device")
            self.to_device(clip_processor)

            safe_emit_progress(clip_pre_progress, 0.25, "Encoding image with CLIP")
            image_embeds = clip_processor(loaded_image, hidden_states_layer=-2, device=self.device, dtype=torch.float32).to(
                transformer_dtype
            )

            if offload:
                safe_emit_progress(clip_pre_progress, 0.28, "Offloading CLIP")
                self._offload("clip", offload_type="cpu")

            # zero padding and vae encode
            # InfiniteTalk: always condition on the current source video frame when a video is provided;
            # MultiTalk (image-only): condition on previous generated frames after the first clip
            if is_first_clip:
                base_condition = cond_image
            else:
                base_condition = cond_image if using_video_input else cond_frame

            video_condition = torch.cat(
                [
                    base_condition,
                    base_condition.new_zeros(
                        base_condition.shape[0],
                        base_condition.shape[1],
                        num_frames - base_condition.shape[2],
                        height,
                        width,
                    ),
                ],
                dim=2,
            )

            safe_emit_progress(clip_pre_progress, 0.35, "Encoding conditioning frames (VAE)")
            latent_condition = self.vae_encode(
                video_condition,
                offload=offload,
                dtype=torch.float32,
                normalize_latents_dtype=torch.float32,
            )

            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
            # For InfiniteTalk (video input), match reference behavior:
            #   - first clip: inject latents encoded from the current source frame
            #   - subsequent clips: inject latents encoded from last generated frames
            if using_video_input:
                motion_source = cond_image if is_first_clip else cond_frame
                safe_emit_progress(clip_pre_progress, 0.40, "Encoding motion reference (VAE)")
                motion_latents = self.vae_encode(
                    motion_source,
                    offload=offload,
                    dtype=torch.float32,
                    normalize_latents_dtype=torch.float32,
                )
                latent_motion_frames = motion_latents[0]
            else:
                latent_motion_frames = latent_condition[
                    :, :, :cur_motion_frames_latent_num
                ][
                    0
                ]  # C T H W
            latent_condition = torch.concat(
                [mask_lat_size.to(latent_condition), latent_condition], dim=1
            )  # B 4+C T H W

            # prepare masks
            safe_emit_progress(clip_pre_progress, 0.45, "Preparing face/body target masks")
            ref_target_masks = self.resize_and_centercrop(human_masks, (height, width))
            ref_target_masks = F.interpolate(
                ref_target_masks.unsqueeze(0),
                size=(latent_height, latent_width),
                mode="nearest",
            ).squeeze()
            ref_target_masks = ref_target_masks > 0
            ref_target_masks = ref_target_masks.float().to(self.device)

            # prepare noise
            safe_emit_progress(clip_pre_progress, 0.50, "Initializing latent noise")
            latents = randn_tensor(
                (
                    batch_size,
                    self.num_channels_latents,
                    (num_frames - 1) // 4 + 1,
                    latent_height,
                    latent_width,
                ),
                dtype=torch.float32,
                generator=generator,
                device=self.device,
            )

            if not self.scheduler:
                safe_emit_progress(clip_pre_progress, 0.52, "Loading scheduler")
                self.load_component_by_type("scheduler")
                self.to_device(self.scheduler)
                safe_emit_progress(clip_pre_progress, 0.53, "Scheduler loaded")

            scheduler = self.scheduler

            safe_emit_progress(clip_pre_progress, 0.55, "Computing timesteps")
            input_timesteps, _ = self._get_timesteps(
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                timesteps_as_indices=timesteps_as_indices,
            )

            if not is_first_clip:
                latent_motion_frames = latent_motion_frames.to(latents.dtype).to(
                    self.device
                )
                motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                add_latent = scheduler.add_noise(
                    latent_motion_frames,
                    motion_add_noise,
                    input_timesteps[0].expand(latents.shape[0]),
                )
                C, T_m, H, W = add_latent.shape
                latents[:, :C, :T_m, :H, :W] = add_latent

            total_steps = len(input_timesteps) if input_timesteps is not None else 0
            safe_emit_progress(
                clip_denoise_progress,
                0.0,
                f"Starting denoising clip {clip_idx} (CFG: {'on' if use_cfg_guidance else 'off'})",
            )
            audio_embeds = audio_embs.to(transformer_dtype).to(self.device)

            with self._progress_bar(len(input_timesteps), desc=f"Sampling MULTITALK") as pbar:
                total_steps = len(input_timesteps)
                for i, t in enumerate(input_timesteps):
                    if using_video_input:
                        latents[:, :, :cur_motion_frames_latent_num] = (
                            latent_motion_frames.unsqueeze(0)
                        )

                    latent_model_input = torch.cat(
                        [latents, latent_condition], dim=1
                    ).to(transformer_dtype)

                    timestep = t.expand(latents.shape[0])

                    noise_pred_cond = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        encoder_hidden_states_audio=audio_embeds,
                        ref_target_masks=ref_target_masks,
                        human_num=human_num,
                        return_dict=False,
                        **attention_kwargs,
                    )[0]

                    if math.isclose(guidance_scale, 1.0):
                        noise_pred_drop_audio = self.transformer(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            encoder_hidden_states_audio=torch.zeros_like(audio_embeds)[
                                -1:
                            ],
                            ref_target_masks=ref_target_masks,
                            human_num=human_num,
                            return_dict=False,
                            **attention_kwargs,
                        )[0]
                    else:
                        noise_pred_drop_text = self.transformer(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            encoder_hidden_states_audio=audio_embeds,
                            ref_target_masks=ref_target_masks,
                            human_num=human_num,
                            return_dict=False,
                            **attention_kwargs,
                        )[0]

                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            encoder_hidden_states_audio=torch.zeros_like(audio_embeds)[
                                -1:
                            ],
                            ref_target_masks=ref_target_masks,
                            human_num=human_num,
                            return_dict=False,
                            **attention_kwargs,
                        )[0]

                    if math.isclose(guidance_scale, 1.0):
                        noise_pred = noise_pred_drop_audio + audio_guidance_scale * (
                            noise_pred_cond - noise_pred_drop_audio
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_cond - noise_pred_drop_text)
                            + audio_guidance_scale
                            * (noise_pred_drop_text - noise_pred_uncond)
                        )

                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                        0
                    ]

                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(
                            latents.dtype
                        ).to(self.device)
                        motion_add_noise = torch.randn_like(
                            latent_motion_frames
                        ).contiguous()
                        noise_timestep = input_timesteps[i + 1].expand(latents.shape[0]) if i < len(input_timesteps) - 1 else torch.zeros_like(input_timesteps[-1]).expand(latents.shape[0])
                        add_latent = scheduler.add_noise(
                            latent_motion_frames, motion_add_noise, noise_timestep
                        )
                        _, T_m, _, _ = add_latent.shape
                        latents[:, :, :T_m] = add_latent

                    if using_video_input:
                        latents[:, :, :cur_motion_frames_latent_num] = (
                            latent_motion_frames.unsqueeze(0)
                        )

                    if (
                        render_on_step
                        and render_on_step_callback
                        and ((i + 1) % render_on_step_interval == 0 or i == 0)
                        and i != len(input_timesteps) - 1
                    ):
                        self._render_step(latents, render_on_step_callback)
                    pbar.update(1)
                    safe_emit_progress(
                        clip_denoise_progress,
                        float(i + 1) / float(total_steps),
                        f"Denoising step {i + 1}/{total_steps} (clip {clip_idx})",
                    )

                self.logger.info("Denoising completed.")
            safe_emit_progress(clip_denoise_progress, 1.0, f"Denoising completed (clip {clip_idx})")

            safe_emit_progress(clip_decode_progress, 0.0, f"Decoding latents (clip {clip_idx})")
            videos = self.vae_decode(latents, offload=offload)
            safe_emit_progress(clip_decode_progress, 0.6, f"Decoded latents (clip {clip_idx})")

            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                safe_emit_progress(clip_decode_progress, 0.7, f"Applying color correction (clip {clip_idx})")
                videos = match_and_blend_colors(
                    videos.float(),
                    original_color_reference.float(),
                    color_correction_strength,
                )
            # >>> END OF COLOR CORRECTION STEP <<<
            safe_emit_progress(clip_decode_progress, 1.0, f"Clip {clip_idx} complete")

            if is_first_clip:
                gen_video_list.append(videos)
                self._preview_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])
                self._preview_video_list.append(videos[:, :, cur_motion_frames_num:])
            # decide whether is done
            if arrive_last_frame:
                break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frames
            cond_frame = (
                videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            )

            if video is None:
                loaded_image = cond_frame[:, :, -1, :, :]
            else:
                # Clamp index to available frames
                next_src_idx = min(audio_start_idx, input_video.shape[2] - 1)
                loaded_image = input_video[:, :, next_src_idx, :, :]

            loaded_image = loaded_image.squeeze(0).permute(1, 2, 0)

            loaded_image = Image.fromarray(
                ((loaded_image + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            )
            # Update cond_image from source video for the next iteration in video mode
            if using_video_input:
                cond_idx = min(audio_start_idx, input_video.shape[2] - 1)
                cond_image = input_video[:, :, cond_idx : cond_idx + 1, :, :]
            audio_start_idx += num_frames - cur_motion_frames_num
            audio_end_idx = audio_start_idx + clip_length
            miss_lengths = []
            


            if audio_end_idx >= min(max_num_frames, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(human_num):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length = (
                            audio_end_idx - len(full_audio_embs[human_inx]) + 3
                        )
                        add_audio_emb = torch.flip(
                            full_audio_embs[human_inx][-1 * miss_length :], dims=[0]
                        )
                        full_audio_embs[human_inx] = torch.cat(
                            [full_audio_embs[human_inx], add_audio_emb], dim=0
                        )
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)

            if max_num_frames <= num_frames:
                break
            

        if offload:
            safe_emit_progress(progress_callback, 0.985, "Offloading transformer")
            self._offload("transformer")

        if return_latents:
            safe_emit_progress(progress_callback, 0.99, "Collecting latents")
            latents = torch.cat(gen_latents_list, dim=2).cpu()
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            # postprocess
            safe_emit_progress(progress_callback, 0.99, "Postprocessing video")
            gen_video_samples = torch.cat(gen_video_list, dim=2)[
                :, :, : int(max_num_frames)
            ]
            if max_num_frames > num_frames and sum(miss_lengths) > 0:
                # split video frames
                if using_video_input:
                    gen_video_samples = gen_video_samples[
                        :, :, : full_audio_embs[0].shape[0]
                    ]
                else:
                    gen_video_samples = gen_video_samples[:, :, : -1 * miss_lengths[0]]

            postprocessed_video = self._tensor_to_frames(gen_video_samples)
            safe_emit_progress(progress_callback, 1.0, "Completed multitalk pipeline")
            return postprocessed_video

    def _render_step(self, latents, render_on_step_callback):
        video = self.vae_decode(latents)
        gen_video_list = self._preview_video_list + [video]
        gen_video_samples = torch.cat(gen_video_list, dim=2)[
            :, :, : int(self._preview_max_num_frames)
        ]
        rendered_video = self._tensor_to_frames(gen_video_samples)
        render_on_step_callback(rendered_video[0])
