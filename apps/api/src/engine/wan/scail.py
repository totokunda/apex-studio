import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.types import InputVideo, InputImage
import torch.nn.functional as F
from tqdm import trange


class WanSCAILEngine(WanShared):
    """WAN SCAIL Engine Implementation"""

    def run(
        self,
        image: InputImage,
        pose_video: InputVideo,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        use_video_duration: bool = True,
        height: int = 480,
        width: int = 832,
        segment_frame_length: int | None = None,
        segment_overlap_frames: int = 1,
        num_inference_steps: int = 40,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        progress_callback: Callable = None,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
        chunking_profile: str = "none",
        expand_timesteps: bool = False,
        rope_on_cpu: bool = False,
        enhance_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        Set `segment_frame_length` to enable segmented generation for long videos.

        Segments overlap by `segment_overlap_frames` (dropped from later segments)
        to keep temporal alignment stable.
        """

        def _pad_video_frames(frames: List[Image.Image], num_target_frames: int):
            if len(frames) >= num_target_frames:
                return frames[:num_target_frames]
            if len(frames) == 0:
                raise ValueError("Pose video has no frames")
            idx = 0
            flip = False
            out: List[Image.Image] = []
            while len(out) < num_target_frames:
                out.append(frames[idx].copy())
                idx = idx - 1 if flip else idx + 1
                if idx == 0 or idx == len(frames) - 1:
                    flip = not flip
            return out

        safe_emit_progress(progress_callback, 0.0, "Starting image-to-video pipeline")
        num_frames = self._parse_num_frames(duration, fps)
        input_timesteps = timesteps

        use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.02, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.03, "Text encoder loaded")

        safe_emit_progress(progress_callback, 0.04, "Moving text encoder to device")
        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Encoding prompt")

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            safe_emit_progress(progress_callback, 0.11, "Encoding negative prompt")
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
            0.14,
            (
                "Prepared negative prompt"
                if negative_prompt_embeds is not None
                else "Skipped negative prompt"
            ),
        )

        if offload:
            safe_emit_progress(progress_callback, 0.15, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 0.16, "Text encoder offloaded")

        safe_emit_progress(
            progress_callback, 0.17, "Loading inputs (image + pose video)"
        )
        loaded_image = self._load_image(image)
        pose_video = self._load_video(
            pose_video,
            fps=fps,
            num_frames=None if use_video_duration else num_frames,
        )
        cond_video_frames = len(pose_video)
        if not use_video_duration:
            cond_video_frames = num_frames

        original_height, original_width = height, width
        if len(pose_video) == 0:
            raise ValueError("Pose video has no frames")
        safe_emit_progress(progress_callback, 0.18, "Inferring output resolution")
        pose0, height, width = self._aspect_ratio_resize(
            pose_video[0], max_area=original_height * original_width, mod_value=32
        )
        _, height, width = self._center_crop_resize(pose0, height, width)

        safe_emit_progress(progress_callback, 0.19, "Preprocessing reference image")
        loaded_image, _, _ = self._aspect_ratio_resize(
            loaded_image, max_area=original_height * original_width, mod_value=32
        )

        transformer_dtype = self.component_dtypes["transformer"]

        safe_emit_progress(progress_callback, 0.20, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.21, "Encoding image with CLIP")

        safe_emit_progress(progress_callback, 0.22, "Moving embeddings to device")
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload and boundary_ratio is None and not expand_timesteps:
            safe_emit_progress(progress_callback, 0.23, "Offloading CLIP")
            self._offload("clip")

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.24, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.25, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.26, "Moving scheduler to device")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        safe_emit_progress(progress_callback, 0.27, "Configuring scheduler timesteps")
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        safe_emit_progress(progress_callback, 0.28, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=input_timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        safe_emit_progress(progress_callback, 0.30, "Scheduler and timesteps prepared")

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )
        
        
        
        

        if seed is not None and generator is not None:
            self.logger.warning(
                "Both `seed` and `generator` are provided. `seed` will be ignored."
            )
        if generator is None:
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)

        def _normalize_segment_length(n: int) -> int:
            if n <= 0:
                return 1
            if n % vae_scale_factor_temporal != 1:
                self.logger.warning(
                    f"`segment_frame_length - 1` must be divisible by {vae_scale_factor_temporal}. Adjusting."
                )
                n = (n // vae_scale_factor_temporal) * vae_scale_factor_temporal + 1
            return max(int(n), 1)

        do_segment = (
            segment_frame_length is not None
            and int(segment_frame_length) > 0
            and cond_video_frames > int(segment_frame_length)
        )

        if do_segment:
            segment_frame_length = _normalize_segment_length(int(segment_frame_length))
            if segment_overlap_frames < 0:
                raise ValueError("segment_overlap_frames must be >= 0")
            if segment_overlap_frames >= segment_frame_length:
                raise ValueError(
                    "segment_overlap_frames must be < segment_frame_length"
                )

            effective_segment_length = segment_frame_length - segment_overlap_frames
            if cond_video_frames <= segment_overlap_frames:
                raise ValueError(
                    "Pose video must be longer than segment_overlap_frames"
                )
            last_segment_frames = (
                cond_video_frames - segment_overlap_frames
            ) % effective_segment_length
            num_padding_frames = (
                0
                if last_segment_frames == 0
                else (effective_segment_length - last_segment_frames)
            )
            num_target_frames = cond_video_frames + num_padding_frames
            num_segments = num_target_frames // effective_segment_length
            pose_video = _pad_video_frames(pose_video, num_target_frames)
        else:
            num_segments = 1
            num_target_frames = max(cond_video_frames, 1)
            if use_video_duration:
                remainder = (num_target_frames - 1) % vae_scale_factor_temporal
                if remainder != 0:
                    num_target_frames = num_target_frames + (
                        vae_scale_factor_temporal - remainder
                    )
                    pose_video = _pad_video_frames(pose_video, num_target_frames)
            else:
                num_target_frames = num_frames
                pose_video = _pad_video_frames(pose_video, num_target_frames)

        transformer_config = self.load_config_by_type("transformer")

        # Reserve a progress span for denoising [0.50, 0.90]
        safe_emit_progress(
            progress_callback,
            0.45,
            f"Starting denoising (CFG: {'on' if use_cfg_guidance else 'off'})",
        )

        all_out_frames: List[torch.Tensor] = []

        start = 0
        end = segment_frame_length if do_segment else num_target_frames
        clip = self.helpers["clip"]
        self.to_device(clip)
        if do_segment:
            effective_segment_length = segment_frame_length - segment_overlap_frames

        for i in trange(num_segments):
            segment_start_progress = 0.50 + (i / num_segments) * 0.40
            segment_end_progress = 0.50 + ((i + 1) / num_segments) * 0.40
            denoise_progress_callback = make_mapped_progress(
                progress_callback, segment_start_progress, segment_end_progress
            )

            if do_segment:
                segment_num_frames = segment_frame_length
                pose_frames = pose_video[start:end]
            else:
                segment_num_frames = num_target_frames
                pose_frames = pose_video

            # update image_embeds and reference_latents to new segment

            self.to_device(clip)

            image_embeds = clip(loaded_image, hidden_states_layer=-2).to(
                self.device, dtype=transformer_dtype
            )
            if image_embeds.shape[0] == 1 and batch_size > 1:
                image_embeds = image_embeds.expand(batch_size, -1, -1)

            if offload:
                self._offload("clip", offload_type="cpu")

            preprocessed_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            ).to(self.device, dtype=torch.float32)

            if preprocessed_image.ndim == 4:
                preprocessed_image = preprocessed_image.unsqueeze(2)

            reference_latents = self.vae_encode(
                preprocessed_image,
                offload=offload,
                dtype=torch.float32,
                normalize_latents_dtype=torch.float32,
                offload_type="cpu",
            )

            if reference_latents.shape[0] == 1 and batch_size > 1:
                reference_latents = reference_latents.expand(batch_size, -1, -1, -1, -1)

            timesteps_segment, num_inference_steps_segment = self._get_timesteps(
                scheduler=scheduler,
                timesteps=input_timesteps,
                timesteps_as_indices=timesteps_as_indices,
                num_inference_steps=num_inference_steps,
            )

            latents = self._get_latents(
                height,
                width,
                segment_num_frames,
                num_channels_latents=getattr(vae_config, "z_dim", 16),
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                vae_scale_factor_temporal=vae_scale_factor_temporal,
                fps=fps,
                batch_size=batch_size,
                dtype=torch.float32,
                generator=generator,
            )

            pose_tensor = self.video_processor.preprocess_video(
                pose_frames, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            pose_tensor = F.interpolate(
                pose_tensor.squeeze(0),
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)

            pose_latents = self.vae_encode(
                pose_tensor,
                offload=offload,
                dtype=latents.dtype,
                normalize_latents_dtype=latents.dtype,
                offload_type="cpu" if do_segment else "discard",
            )

            if pose_latents.shape[0] == 1 and batch_size > 1:
                pose_latents = pose_latents.expand(batch_size, -1, -1, -1, -1)

            _, _, num_latent_frames, latent_height, latent_width = latents.shape
            max_seq_len = (
                num_latent_frames
                * latent_height
                * latent_width
                // (
                    transformer_config.get("patch_size", (1, 2, 2))[1]
                    * transformer_config.get("patch_size", (1, 2, 2))[2]
                )
            )

            with torch.autocast(device_type=self.device.type, dtype=transformer_dtype):
                latents = self.denoise(
                    timesteps=timesteps_segment,
                    latents=latents,
                    transformer_kwargs=dict(
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_clip=image_embeds,
                        encoder_hidden_states_pose=pose_latents,
                        encoder_hidden_states_reference=reference_latents,
                        attention_kwargs=attention_kwargs,
                        seq_len=max_seq_len,
                        enhance_kwargs=enhance_kwargs,
                        rope_on_cpu=rope_on_cpu,
                    ),
                    unconditional_transformer_kwargs=(
                        dict(
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_clip=image_embeds,
                            encoder_hidden_states_pose=pose_latents,
                            encoder_hidden_states_reference=reference_latents,
                            seq_len=max_seq_len,
                            attention_kwargs=attention_kwargs,
                            enhance_kwargs=enhance_kwargs,
                            rope_on_cpu=rope_on_cpu,
                        )
                        if negative_prompt_embeds is not None
                        else None
                    ),
                    transformer_dtype=transformer_dtype,
                    use_cfg_guidance=use_cfg_guidance,
                    render_on_step=render_on_step,
                    render_on_step_callback=render_on_step_callback,
                    denoise_progress_callback=denoise_progress_callback,
                    scheduler=scheduler,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps_segment,
                    chunking_profile=chunking_profile,
                )
            out_frames = self.vae_decode(latents, offload=offload)
            if do_segment:
                out_frames = out_frames.detach().to("cpu")
            if do_segment and i > 0 and segment_overlap_frames > 0:
                out_frames = out_frames[:, :, segment_overlap_frames:]
            all_out_frames.append(out_frames)
            loaded_image = self._tensor_to_frames(out_frames)[0][0]

            if do_segment:
                start += effective_segment_length
                end += effective_segment_length
            else:
                break

        if offload:
            safe_emit_progress(progress_callback, 0.91, "Offloading transformer")
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        safe_emit_progress(progress_callback, 0.94, "Decoding latents to video")
        safe_emit_progress(progress_callback, 0.96, "Decoded latents")
        video = torch.cat(all_out_frames, dim=2)[:, :, :cond_video_frames]
        postprocessed_video = self._tensor_to_frames(video)
        safe_emit_progress(progress_callback, 1.0, "Completed image-to-video pipeline")
        return postprocessed_video
