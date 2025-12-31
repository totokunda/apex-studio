import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.types import InputImage
from tqdm import tqdm

class WanSVIEngine(WanShared):
    """WAN Image-to-Video Engine Implementation"""
    
    def _prepare_image_latents_pro(self, is_first_clip: bool, input_image: InputImage | List[InputImage], anchor: InputImage, width: int, height: int, num_frames: int, offload: bool = True, prev_last_latent: torch.Tensor = None, end_image: InputImage = None, num_motion_latent: int = 1, end_frame_fill=0.5, end_frame_max_strength=1.0):
        input_image = [input_image] if not isinstance(input_image, list) else input_image
        input_image = [self._load_image(img) for img in input_image]
        
        if is_first_clip: # Use first frame as anchor
            anchor = input_image[0]
        if end_image is not None:
            end_image = self._load_image(end_image)
            end_image = self.video_processor.preprocess(end_image, height=height, width=width).to(self.device, dtype=torch.float32)
            
        input_image = [self.video_processor.preprocess(img, height=height, width=width).to(self.device, dtype=torch.float32) for img in input_image]
        
        # Preprocess anchor
        anchor = self.video_processor.preprocess(anchor, height=height, width=width).to(self.device, dtype=torch.float32).unsqueeze(2)
        input_image = torch.cat(input_image, dim=1)  # Shape: [3, num_frames, H, W]
        
        if end_image is not None:
            end_image = self.video_processor.preprocess(end_image, height=height, width=width).to(self.device, dtype=torch.float32).unsqueeze(2)
            end_image_latent = self.vae_encode(
                end_image,
                offload=offload,
                dtype=torch.float32,
                normalize_latents_dtype=torch.float32,
            )[0]
        else:
            end_image_latent = None

        # Encode anchor frame to latent
        anchor_latent = self.vae_encode(
            anchor,
            offload=offload,
            dtype=torch.float32,
            normalize_latents_dtype=torch.float32,
        )[0]
        total_latents = (num_frames - 1) // 4 + 1
        
        if end_image_latent is not None:
            end_latent = end_image_latent.clone()
            end_frames = end_latent.shape[2]
            num_anchor = anchor_latent.shape[2]

            # Calculate where to start blending (last N frames of anchor)
            blend_start_idx = max(0, num_anchor - end_frames)

            # Blend the overlapping frames: gradual transition from anchor to end
            for frame_idx in range(end_frames):
                anchor_frame_idx = blend_start_idx + frame_idx
                if anchor_frame_idx < num_anchor:
                    # Blend factor increases from 0 to 1 as we go through end frames
                    blend_factor = (frame_idx + 1) / end_frames
                    anchor_latent[:, :, anchor_frame_idx] = (
                        (1 - blend_factor) * anchor_latent[:, :, anchor_frame_idx] +
                        blend_factor * end_latent[:, :, frame_idx]
                    )
        
        if is_first_clip:
            # First clip: only anchor + padding
            padding_size = total_latents - anchor_latent.shape[1]
            image_cond_latent = anchor_latent
        else:
            # Subsequent clips: anchor + motion + padding
            motion_latent = prev_last_latent.to(device=self.device)[:, -num_motion_latent:]
            padding_size = total_latents - anchor_latent.shape[1] - motion_latent.shape[1]
            image_cond_latent = torch.concat([anchor_latent, motion_latent], dim=1)
            
        padding = torch.zeros(
                anchor_latent.shape[0], padding_size, anchor_latent.shape[2], anchor_latent.shape[3],
                dtype=torch.float32, device=self.device
            )
        
        if end_image_latent is not None and padding_size > 0:
            end_latent = end_image_latent.clone()
            end_frames = end_latent.shape[2]

            # Calculate how many padding frames to blend based on end_frame_fill
            blend_frames = max(1, int(padding_size * end_frame_fill))
            blend_frames = min(blend_frames, end_frames, padding_size)
            padding_blend_start = padding_size - blend_frames

            for frame_idx in range(blend_frames):
                padding_frame_idx = padding_blend_start + frame_idx
                # Blend factor increases from 0 to strength across the blended padding frames
                blend_factor = ((frame_idx + 1) / blend_frames) * end_frame_max_strength
                padding[:, padding_frame_idx] = (
                    (1 - blend_factor) * padding[:, :, padding_frame_idx] +
                    blend_factor * end_latent[:, :, frame_idx]
                )

        # Create frame mask (1 for first frame, 0 for rest)
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, anchor_latent.shape[1]:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        y = torch.concat([image_cond_latent, padding], dim=1)

        y = torch.concat([msk, y]).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        return y

    def _prepare_image_latents(self, input_image: InputImage, width: int, height: int, num_frames: int, offload: bool = True, end_image: InputImage = None, anchor: InputImage = None,  num_motion_latent: int = 1):
        
        input_image = self._load_image(input_image)
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        
        if anchor:
            anchor = self.video_processor.preprocess(anchor, height=height, width=width).to(self.device, dtype=torch.float32)
            
        image = self.video_processor.preprocess(input_image, height=height, width=width).to(self.device, dtype=torch.float32)
        if end_image is not None:
            end_image = self.video_processor.preprocess(end_image, height=height, width=width).to(self.device, dtype=torch.float32)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            if anchor is not None:
                input_pad = anchor.transpose(0, 1).repeat(1, num_frames-1, 1, 1)  
            else:
                input_pad = torch.zeros(3, num_frames-1, height, width, device=self.device)          
            vae_input = torch.concat([image.transpose(0, 1), input_pad], dim=1)
        # Preprocess anchor
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        y = self.vae_encode(vae_input.unsqueeze(0))[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        return y

    def run(
        self,
        image: InputImage | List[InputImage],
        prompts: List[str],
        end_image: InputImage = None,
        negative_prompt: List[str] | str = None,
        duration: int | str = 81,
        num_frames_per_segment: int = 81,
        num_seconds_per_segment: float | str = 5.0,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        high_noise_guidance_scale: float = 1.0,
        low_noise_guidance_scale: float = 1.0,
        progress_callback: Callable = None,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor = None,
        enhance_kwargs: Dict[str, Any] = {},
        num_motion_latent: int = 1,
        num_overlap_frames: int = 4,
        num_motion_frame: int = 4,
        easy_cache_thresh: float = 0.00,
        easy_cache_ret_steps: int = 10,
        easy_cache_cutoff_steps: int | None = None,
        **kwargs,
    ):
        """
        Stable Infinite Video (SVI) generation with verbose progress reporting.

        Progress is reported over the full [0.0, 1.0] span:
        - [0.00, 0.20]: one-time setup (text encoding, inputs)
        - [0.20, 0.95]: per-clip generation (mapped per clip; includes denoise + decode/stitch)
        - [0.95, 1.00]: finalization / return
        """
        if (
            high_noise_guidance_scale is not None
            and low_noise_guidance_scale is not None
        ):
            guidance_scale = [high_noise_guidance_scale, low_noise_guidance_scale]
            safe_emit_progress(progress_callback, 0.01, "Using high/low-noise guidance scales")

        safe_emit_progress(progress_callback, 0.0, "Starting stable infinite video pipeline (SVI)")
        if guidance_scale is not None and isinstance(guidance_scale, list):
            use_cfg_guidance = (
                negative_prompt is not None
                and guidance_scale[0] > 1.0
                and guidance_scale[1] > 1.0
            )
        else:
            use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        safe_emit_progress(
            progress_callback,
            0.06,
            f"Encoding {len(prompts)} prompt(s) for SVI (num_videos={num_videos})",
        )
        prompt_embeds_list = []
        for p_idx, prompt in enumerate(prompts):
            prompt_embeds_list.append(
                self.text_encoder.encode(
                    prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos,
                    **text_encoder_kwargs,
                )
            )
            safe_emit_progress(
                progress_callback,
                0.06 + (0.04 * ((p_idx + 1) / max(1, len(prompts)))),
                f"Encoded prompt {p_idx + 1}/{len(prompts)}",
            )

        safe_emit_progress(progress_callback, 0.10, "All prompts encoded")

        batch_size = 1
        
        if num_seconds_per_segment is not None:
            if not isinstance(num_seconds_per_segment, str):
                num_seconds_per_segment = str(num_seconds_per_segment) + "s"
            num_frames_per_segment = self._parse_num_frames(num_seconds_per_segment, fps)

        if negative_prompt is not None and use_cfg_guidance:
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
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Text encoder offloaded")
        all_video_frames = []
        all_latents = []
        safe_emit_progress(progress_callback, 0.16, "Loading and resizing input image(s)")
        image = self._load_image(image)
        if end_image is not None:
            end_image = self._load_image(end_image)
            end_image, _, _ = self._aspect_ratio_resize(end_image, max_area=height * width, mod_value=16)
        image, height, width = self._aspect_ratio_resize(image, max_area=height * width, mod_value=16)
        current_input_image = image
        total_num_frames = self._parse_num_frames(duration, fps)
        num_clips = max(total_num_frames // num_frames_per_segment, len(prompt_embeds_list))
        safe_emit_progress(
            progress_callback,
            0.18,
            f"Prepared SVI clip plan: total_frames={total_num_frames}, "
            f"frames_per_clip={num_frames_per_segment}, num_clips={num_clips}, "
            f"resolution={width}x{height}, fps={fps}",
        )
        input_timesteps = timesteps
        is_first_clip = True
        prev_last_latent = None
        # Per-clip progress occupies the bulk of the overall progress range.
        clip_loop_progress = make_mapped_progress(progress_callback, 0.20, 0.95)
        for idx in tqdm(range(num_clips), desc="Processing clips"):
            clip_idx_1 = idx + 1
            clip_progress_start = idx / max(1, num_clips)
            clip_progress_end = (idx + 1) / max(1, num_clips)
            clip_progress_callback = make_mapped_progress(
                clip_loop_progress, clip_progress_start, clip_progress_end
            )
            safe_emit_progress(
                clip_progress_callback,
                0.00,
                f"Clip {clip_idx_1}/{num_clips}: starting (first_clip={is_first_clip})",
            )
            prompt_idx = idx % len(prompt_embeds_list)
            prompt_embeds = prompt_embeds_list[prompt_idx]

            transformer_dtype = self.component_dtypes["transformer"]

            prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    self.device, dtype=transformer_dtype
                )
            safe_emit_progress(
                clip_progress_callback,
                0.05,
                f"Clip {clip_idx_1}/{num_clips}: using prompt {prompt_idx + 1}/{len(prompt_embeds_list)} "
                f"(dtype={transformer_dtype})",
            )

            if not self.scheduler:
                self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)

            scheduler = self.scheduler

            timesteps, num_inference_steps = self._get_timesteps(
                scheduler=scheduler,
                timesteps=input_timesteps,
                timesteps_as_indices=timesteps_as_indices,
                num_inference_steps=num_inference_steps,
            )
            
            safe_emit_progress(
                clip_progress_callback,
                0.10,
                f"Clip {clip_idx_1}/{num_clips}: scheduler ready "
                f"(num_inference_steps={num_inference_steps}, timesteps={len(timesteps)})",
            )
            
            vae_config = self.load_config_by_type("vae")
            vae_scale_factor_spatial = getattr(
                vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
            )

            vae_scale_factor_temporal = getattr(
                vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
            )
            
            if seed is None:
                # generate a random seed
                generator = torch.Generator(device=self.device)
            else:
                generator = torch.Generator(device=self.device).manual_seed(seed * (idx + 1))

            noise = self._get_latents(
                height,
                width,
                duration=num_frames_per_segment,
                num_channels_latents=getattr(vae_config, "z_dim", 16),
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                vae_scale_factor_temporal=vae_scale_factor_temporal,
                fps=fps,
                batch_size=batch_size,
                dtype=torch.float32,
                generator=generator,
            )

            safe_emit_progress(
                clip_progress_callback,
                0.15,
                f"Clip {clip_idx_1}/{num_clips}: initialized latent noise "
                f"(z_dim={getattr(vae_config, 'z_dim', 16)}, "
                f"scale_spatial={vae_scale_factor_spatial}, scale_temporal={vae_scale_factor_temporal})",
            )
            if num_motion_latent > 0:
                latent_condition = self._prepare_image_latents_pro(
                    is_first_clip=is_first_clip,
                    input_image=current_input_image,
                    anchor=image,
                    width=width,
                    height=height,
                    num_frames=num_frames_per_segment,
                    offload=offload,
                    prev_last_latent=prev_last_latent,
                    num_motion_latent=num_motion_latent,
                    end_image=end_image,
                )
            else:
                latent_condition = self._prepare_image_latents(
                    input_image=current_input_image,
                    anchor=image,
                    width=width,
                    height=height,
                    num_frames=num_frames_per_segment,
                    offload=offload
                )
            safe_emit_progress(
                clip_progress_callback,
                0.22,
                f"Clip {clip_idx_1}/{num_clips}: prepared latent condition "
                f"(num_motion_latent={num_motion_latent}, "
                f"condition_shape={tuple(latent_condition.shape) if hasattr(latent_condition, 'shape') else 'unknown'})",
            )

            if boundary_ratio is not None:
                boundary_timestep = boundary_ratio * getattr(
                    self.scheduler.config, "num_train_timesteps", 1000
                )
            else:
                boundary_timestep = None

            # Reserve a progress span for denoising within this clip.
            denoise_progress_callback = make_mapped_progress(
                clip_progress_callback, 0.30, 0.85
            )
            safe_emit_progress(
                clip_progress_callback,
                0.28,
                f"Clip {clip_idx_1}/{num_clips}: starting denoise "
                f"(cfg={use_cfg_guidance}, guidance_scale={guidance_scale}, boundary_timestep={boundary_timestep})",
            )

            latents = self.denoise(
                boundary_timestep=boundary_timestep,
                timesteps=timesteps,
                latents=noise,
                latent_condition=latent_condition,
                transformer_kwargs=dict(
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    enhance_kwargs=enhance_kwargs,
                ),
                unconditional_transformer_kwargs=(
                    dict(
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        enhance_kwargs=enhance_kwargs,
                    )
                    if negative_prompt_embeds is not None
                    else None
                ),
                transformer_dtype=transformer_dtype,
                use_cfg_guidance=use_cfg_guidance,
                render_on_step=False,
                render_on_step_callback=render_on_step_callback,
                denoise_progress_callback=denoise_progress_callback,
                scheduler=scheduler,
                guidance_scale=guidance_scale,
                expand_timesteps=expand_timesteps,
                ip_image=ip_image,
                easy_cache_thresh=easy_cache_thresh,
                easy_cache_ret_steps=easy_cache_ret_steps,
                easy_cache_cutoff_steps=easy_cache_cutoff_steps,
            )
            
            safe_emit_progress(
                clip_progress_callback,
                0.86,
                f"Clip {clip_idx_1}/{num_clips}: denoise complete",
            )
            prev_last_latent = latents[0].detach()
            is_first_clip = False

            # Decode/stitch (or collect latents)
            if return_latents:
                safe_emit_progress(
                    clip_progress_callback,
                    0.90,
                    f"Clip {clip_idx_1}/{num_clips}: collecting latents (skipping decode)",
                )
                all_latents.append(latents)
                postprocessed_video = None
            else:
                safe_emit_progress(
                    clip_progress_callback,
                    0.90,
                    f"Clip {clip_idx_1}/{num_clips}: decoding latents to frames",
                )
                video = self.vae_decode(latents, offload=offload)
                postprocessed_video = self._tensor_to_frames(video)[0]
                # Seamlessly stitch this clip onto the running output using the overlap window.
                if idx == 0:
                    all_video_frames.extend(postprocessed_video)
                else:
                    all_video_frames = all_video_frames 
                    if num_motion_latent > 0:
                        all_video_frames.extend(postprocessed_video[num_overlap_frames:])
                    else:
                        all_video_frames.extend(postprocessed_video[1:])
                if render_on_step_callback is not None:
                    render_on_step_callback(all_video_frames)
                safe_emit_progress(
                    clip_progress_callback,
                    0.96,
                    f"Clip {clip_idx_1}/{num_clips}: stitched frames "
                    f"(total_frames_so_far={len(all_video_frames)})",
                )
            if num_motion_latent > 0:
                if postprocessed_video is not None:
                    current_input_image = postprocessed_video[-num_motion_frame:]
            else:
                if postprocessed_video is not None:
                    current_input_image = postprocessed_video[-1]
            safe_emit_progress(
                clip_progress_callback,
                1.00,
                f"Clip {clip_idx_1}/{num_clips}: complete",
            )

        safe_emit_progress(progress_callback, 0.95, "All clips complete")

        if return_latents:
            safe_emit_progress(progress_callback, 0.98, "Concatenating and returning latents")
            return torch.cat(all_latents, dim=2)
        else:
            safe_emit_progress(progress_callback, 0.98, "Completed stable infinite video pipeline (SVI)")
            safe_emit_progress(progress_callback, 1.0, "Returning frames")
            return [all_video_frames]
