from .shared import WanShared
from src.types import InputImage, InputAudio, InputVideo
from typing import List, Union, Optional, Dict, Any, Tuple, Callable
from torch import Tensor
from PIL import Image
import torch
from src.utils.progress import safe_emit_progress, make_mapped_progress
import numpy as np
import torch.nn.functional as F
import math
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor


class WanS2VEngine(WanShared):
    """WAN Sound-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, resample="bilinear"
        )
        self.motion_frames = 73
        self.drop_first_motion = True

    def load_pose_condition(
        self, pose_video, num_chunks, num_frames_per_chunk, height, width
    ):
        device = self.device
        dtype = self.component_dtypes["vae"]
        if pose_video is not None:
            padding_frame_num = num_chunks * num_frames_per_chunk - pose_video.shape[2]
            pose_video = pose_video.to(dtype=dtype, device=device)
            pose_video = torch.cat(
                [
                    pose_video,
                    -torch.ones(
                        [1, 3, padding_frame_num, height, width],
                        dtype=dtype,
                        device=device,
                    ),
                ],
                dim=2,
            )

            pose_video = torch.chunk(pose_video, num_chunks, dim=2)
        else:
            pose_video = [
                -torch.ones(
                    [1, 3, num_frames_per_chunk, height, width],
                    dtype=dtype,
                    device=device,
                )
            ]

        # Vectorized processing: concatenate all chunks along batch dimension
        all_poses = torch.cat(
            [torch.cat([cond[:, :, 0:1], cond], dim=2) for cond in pose_video], dim=0
        )  # Shape: [num_chunks, 3, num_frames_per_chunk+1, height, width]

        pose_condition = self.vae_encode(all_poses, sample_mode="mode")[:, :, 1:]

        return pose_condition

    def prepare_latents(
        self,
        image: InputImage,
        batch_size: int,
        latent_motion_frames: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames_per_chunk: int = 80,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        pose_video: Optional[List[Image.Image]] = None,
        init_first_frame: bool = False,
        num_chunks: int = 1,
        offload: bool = True,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]
        ],
    ]:

        num_latent_frames = (
            num_frames_per_chunk + 3 + self.motion_frames
        ) // self.vae_scale_factor_temporal - latent_motion_frames
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        dtype = self.component_dtypes["vae"]
        device = device or self.device
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        if image is not None:
            image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

            video_condition = image.to(device=device, dtype=dtype)

            if isinstance(generator, list):
                latent_condition = [
                    self.vae_encode(video_condition, offload=offload) for _ in generator
                ]
                latent_condition = torch.cat(latent_condition)
            else:
                latent_condition = self.vae_encode(video_condition, offload=offload)
                latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            motion_pixels = torch.zeros(
                [1, 3, self.motion_frames, height, width], dtype=dtype, device=device
            )
            # Get pose condition input if needed
            pose_condition = self.load_pose_condition(
                pose_video, num_chunks, num_frames_per_chunk, height, width
            )

            # Encode motion latents
            videos_last_pixels = motion_pixels.detach()
            if init_first_frame:
                self.drop_first_motion = False
                motion_pixels[:, :, -6:] = video_condition
            motion_latents = self.vae_encode(motion_pixels, offload=offload)

            return (
                latents,
                latent_condition,
                videos_last_pixels,
                motion_latents,
                pose_condition,
            )

        else:
            return latents

    @staticmethod
    def get_sample_indices(
        original_fps, total_frames, target_fps, num_sample, fixed_start=None
    ):
        required_duration = num_sample / target_fps
        required_origin_frames = int(np.ceil(required_duration * original_fps))
        if required_duration > total_frames / original_fps:
            raise ValueError("required_duration must be less than video length")

        if fixed_start is not None and fixed_start >= 0:
            start_frame = fixed_start
        else:
            max_start = total_frames - required_origin_frames
            if max_start < 0:
                raise ValueError("video length is too short")
            start_frame = np.random.randint(0, max_start + 1)
        start_time = start_frame / original_fps

        end_time = start_time + required_duration
        time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

        frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)
        return frame_indices

    @staticmethod
    def linear_interpolation(features, input_fps, output_fps, output_len=None):
        """
        Args:
            features: shape=[1, T, 512]
            input_fps: fps for audio, f_a
            output_fps: fps for video, f_m
            output_len: video length
        """
        features = features.transpose(1, 2)  # [1, 512, T]
        seq_len = features.shape[2] / float(input_fps)  # T/f_a
        output_len = int(seq_len * output_fps)  # f_m*T/f_a
        output_features = F.interpolate(
            features, size=output_len, align_corners=True, mode="linear"
        )  # [1, 512, output_len]
        return output_features.transpose(1, 2)  # [1, output_len, 512]

    def encode_audio(
        self,
        audio: InputAudio,
        sampling_rate: int,
        num_frames: int,
        fps: int = 16,
        device: Optional[torch.device] = None,
        offload: bool = True,
    ):
        device = device or self.device
        video_rate = 30
        audio_sample_m = 0

        audio = self._load_audio(audio, sampling_rate)

        input_values = self.helpers["audio_processor"](
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values

        # retrieve logits & take argmax
        res = self.helpers["audio_encoder"](
            input_values.to(device), output_hidden_states=True
        )
        feat = torch.cat(res.hidden_states)

        feat = self.linear_interpolation(feat, input_fps=50, output_fps=video_rate)

        audio_embed = feat.to(torch.float32)  # Encoding for the motion

        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        scale = video_rate / fps

        num_repeat = int(audio_frame_num / (num_frames * scale)) + 1

        bucket_num = num_repeat * num_frames
        padd_audio_num = (
            math.ceil(num_repeat * num_frames / fps * video_rate) - audio_frame_num
        )

        batch_idx = self.get_sample_indices(
            original_fps=video_rate,
            total_frames=audio_frame_num + padd_audio_num,
            target_fps=fps,
            num_sample=bucket_num,
            fixed_start=0,
        )
        batch_audio_eb = []
        audio_sample_stride = int(video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:
                chosen_idx = list(
                    range(
                        bi - audio_sample_m * audio_sample_stride,
                        bi + (audio_sample_m + 1) * audio_sample_stride,
                        audio_sample_stride,
                    )
                )
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [
                    audio_frame_num - 1 if c >= audio_frame_num else c
                    for c in chosen_idx
                ]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(
                        start_dim=-2, end_dim=-1
                    )
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = (
                    torch.zeros(
                        [audio_dim * (2 * audio_sample_m + 1)],
                        device=audio_embed.device,
                    )
                    if not return_all_layers
                    else torch.zeros(
                        [num_layers, audio_dim * (2 * audio_sample_m + 1)],
                        device=audio_embed.device,
                    )
                )
            batch_audio_eb.append(frame_audio_embed)
        audio_embed_bucket = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

        audio_embed_bucket = audio_embed_bucket.to(device)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)

        if offload:
            self._offload("audio_encoder")

        return audio_embed_bucket, num_repeat

    def run(
        self,
        prompt: List[str] | str,
        audio: InputAudio,
        image: InputImage,
        sampling_rate: int,
        negative_prompt: List[str] | str = None,
        pose_video: InputVideo = None,
        height: int = 480,
        width: int = 832,
        num_frames_per_chunk: int = 80,
        num_inference_steps: int = 40,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 4.5,
        init_first_frame: bool = False,
        return_latents: bool = False,
        progress_callback: Callable | None = None,
        denoise_progress_callback: Callable | None = None,
        render_on_step_callback: Callable = None,
        offload: bool = True,
        latents: Optional[torch.Tensor] = None,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_interval: int = 3,
        num_chunks: Optional[int] = None,
        chunking_profile: str = "none", 
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting s2v pipeline")

        use_cfg_guidance = negative_prompt is not None and guidance_scale > 1.0

        if return_latents:
            self.logger.warning("Returning latents is not supported for WanS2VEngine")
            return None

        if num_frames_per_chunk % self.vae_scale_factor_temporal != 0:
            num_frames_per_chunk = (
                num_frames_per_chunk
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
            )
            self.logger.warning(
                f"`num_frames_per_chunk` had to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number: {num_frames_per_chunk}"
            )
            safe_emit_progress(
                progress_callback,
                0.01,
                f"Adjusted num_frames_per_chunk -> {num_frames_per_chunk}",
            )
        num_frames_per_chunk = max(num_frames_per_chunk, 1)

        safe_emit_progress(progress_callback, 0.02, "Encoding prompts")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_cfg_guidance=use_cfg_guidance,
            num_videos=num_videos,
            text_encoder_kwargs=text_encoder_kwargs,
            progress_callback=progress_callback,
            offload=offload,
        )
        safe_emit_progress(progress_callback, 0.10, "Prompts encoded")

        if negative_prompt_embeds is None:
            use_cfg_guidance = False
            safe_emit_progress(
                progress_callback, 0.105, "CFG disabled (no negative prompt)"
            )

        batch_size = prompt_embeds.shape[0]
        transformer_dtype = self.component_dtypes["transformer"]

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        safe_emit_progress(progress_callback, 0.12, "Encoding audio")
        audio_embeds, num_chunks_audio = self.encode_audio(
            audio, sampling_rate, num_frames_per_chunk, fps, offload=offload
        )
        if num_chunks is None or num_chunks > num_chunks_audio:
            num_chunks = num_chunks_audio

        safe_emit_progress(
            progress_callback, 0.20, f"Audio encoded (chunks={num_chunks_audio})"
        )
        audio_embeds = audio_embeds.to(transformer_dtype)

        latent_motion_frames = (
            self.motion_frames + 3
        ) // self.vae_scale_factor_temporal

        safe_emit_progress(
            progress_callback, 0.21, "Loading and preprocessing input image"
        )
        image = self._load_image(image)
        image, height, width = self._aspect_ratio_resize(image, max_area=height * width)

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            self.device, dtype=torch.float32
        )
        safe_emit_progress(
            progress_callback, 0.24, f"Prepared image ({width}x{height})"
        )

        if pose_video is not None:
            safe_emit_progress(
                progress_callback, 0.25, "Loading and preprocessing pose video"
            )
            num_frames = num_frames_per_chunk * num_chunks
            pose_video = self._load_video(
                pose_video, num_frames=num_frames, reverse=True, fps=fps
            )
            for idx, frame in enumerate(pose_video):
                frame, _, _ = self._aspect_ratio_resize(frame, max_area=height * width)
                frame, _, _ = self._center_crop_resize(frame, height, width)
                pose_video[idx] = frame
            pose_video = self.video_processor.preprocess_video(
                pose_video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            safe_emit_progress(progress_callback, 0.29, "Pose video prepared")
        else:
            safe_emit_progress(progress_callback, 0.29, "No pose video provided")

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.30, "Loading transformer")
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        if chunking_profile != "none":
            self.transformer.set_chunking_profile(chunking_profile)
        safe_emit_progress(progress_callback, 0.32, "Transformer ready")

        video_chunks = []
        self._current_chunk = 0
        self._preview_video_chunks = []

        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            safe_emit_progress(progress_callback, 0.33, f"Seeded generator ({seed})")

        # Reserve an overall progress window for chunk processing
        chunks_progress = make_mapped_progress(progress_callback, 0.35, 0.95)
        safe_emit_progress(
            progress_callback, 0.35, f"Starting chunk processing (chunks={num_chunks})"
        )

        for r in range(num_chunks):
            self._current_chunk = r
            chunk_progress = make_mapped_progress(
                chunks_progress,
                float(r) / float(max(num_chunks, 1)),
                float(r + 1) / float(max(num_chunks, 1)),
            )
            safe_emit_progress(
                chunk_progress, 0.0, f"Chunk {r + 1}/{num_chunks}: preparing latents"
            )
            latents_outputs = self.prepare_latents(
                image if r == 0 else None,
                batch_size * num_videos,
                latent_motion_frames,
                self.num_channels_latents,
                height,
                width,
                num_frames_per_chunk,
                torch.float32,
                self.device,
                generator,
                latents if r == 0 else None,
                pose_video,
                init_first_frame,
                num_chunks,
                offload=offload,
            )

            if r == 0:
                (
                    latents,
                    condition,
                    videos_last_pixels,
                    motion_latents,
                    pose_condition,
                ) = latents_outputs
            else:
                latents = latents_outputs

            with torch.no_grad():
                left_idx = r * num_frames_per_chunk
                right_idx = r * num_frames_per_chunk + num_frames_per_chunk
                pose_latents = (
                    pose_condition[r]
                    if pose_video is not None
                    else pose_condition[0] * 0
                )
                pose_latents = pose_latents.to(
                    dtype=transformer_dtype, device=self.device
                )
                audio_embeds_input = audio_embeds[..., left_idx:right_idx]
            motion_latents_input = motion_latents.to(transformer_dtype).clone()

            if not self.scheduler:
                safe_emit_progress(chunk_progress, 0.10, "Loading scheduler")
                self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)
            # 4. Prepare timesteps by resetting scheduler in each chunk
            safe_emit_progress(chunk_progress, 0.12, "Preparing timesteps")
            timesteps, num_inference_steps = self._get_timesteps(
                scheduler=self.scheduler,
                num_inference_steps=num_inference_steps,
            )

            num_warmup_steps = (
                len(timesteps) - num_inference_steps * self.scheduler.order
            )
            self._num_timesteps = len(timesteps)
            safe_emit_progress(
                chunk_progress,
                0.15,
                f"Chunk {r + 1}/{num_chunks}: starting denoise (steps={num_inference_steps})",
            )

            # Map denoise step progress into the chunk progress window.
            local_denoise_progress_callback = (
                denoise_progress_callback
                if denoise_progress_callback is not None
                else make_mapped_progress(chunk_progress, 0.15, 0.80)
            )
            safe_emit_progress(local_denoise_progress_callback, 0.0, "Starting denoise")

            with self._progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):

                    self._current_timestep = t

                    latent_model_input = latents.to(transformer_dtype)
                    condition = condition.to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            motion_latents=motion_latents_input,
                            image_latents=condition,
                            pose_latents=pose_latents,
                            audio_embeds=audio_embeds_input,
                            motion_frames=[self.motion_frames, latent_motion_frames],
                            drop_motion_frames=self.drop_first_motion and r == 0,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if use_cfg_guidance:
                        with self.transformer.cache_context("uncond"):
                            noise_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                motion_latents=motion_latents_input,
                                image_latents=condition,
                                pose_latents=pose_latents,
                                audio_embeds=0.0 * audio_embeds_input,
                                motion_frames=[
                                    self.motion_frames,
                                    latent_motion_frames,
                                ],
                                drop_motion_frames=self.drop_first_motion and r == 0,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred = noise_uncond + guidance_scale * (
                                noise_pred - noise_uncond
                            )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]

                    if (
                        render_on_step
                        and render_on_step_callback
                        and ((i + 1) % render_on_step_interval == 0 or i == 0)
                        and i != len(timesteps) - 1
                    ):
                        total_steps = len(timesteps)
                        step_progress = (
                            min(float(i + 1) / float(total_steps), 1.0)
                            if total_steps > 0
                            else 0.0
                        )
                        safe_emit_progress(
                            local_denoise_progress_callback,
                            step_progress,
                            "Rendering preview",
                        )
                        self._render_step(latents, render_on_step_callback)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        total_steps = len(timesteps)
                        if total_steps > 0:
                            safe_emit_progress(
                                local_denoise_progress_callback,
                                min(float(i + 1) / float(total_steps), 1.0),
                                f"Denoising step {i + 1}/{total_steps}",
                            )

            safe_emit_progress(
                chunk_progress, 0.82, "Denoising complete; decoding chunk"
            )
            if not (self.drop_first_motion and r == 0):
                decode_latents = torch.cat([motion_latents, latents], dim=2)
            else:
                decode_latents = torch.cat([condition, latents], dim=2)

            video = self.vae_decode(decode_latents, offload=offload)
            video = video[:, :, -(num_frames_per_chunk):]

            if self.drop_first_motion and r == 0:
                video = video[:, :, 3:]

            num_overlap_frames = min(self.motion_frames, video.shape[2])

            videos_last_pixels = torch.cat(
                [
                    videos_last_pixels[:, :, num_overlap_frames:],
                    video[:, :, -num_overlap_frames:],
                ],
                dim=2,
            )

            # Update motion_latents for next iteration
            motion_latents = self.vae_encode(
                videos_last_pixels, sample_mode="mode", offload=offload
            )

            video_chunks.append(video)
            self._preview_video_chunks.append(video)

            safe_emit_progress(
                chunk_progress, 1.0, f"Chunk {r + 1}/{num_chunks} complete"
            )

        if offload:
            safe_emit_progress(progress_callback, 0.96, "Offloading transformer")
            self._offload("transformer")

        safe_emit_progress(
            progress_callback, 0.98, "Concatenating and postprocessing video"
        )
        video_chunks = torch.cat(video_chunks, dim=2)
        safe_emit_progress(progress_callback, 1.0, "Completed s2v pipeline")
        return self._tensor_to_frames(video_chunks)

    def _render_step(self, latents, render_on_step_callback):
        video = self.vae_decode(latents)
        total_video = torch.cat(self._preview_video_chunks + [video], dim=2)
        rendered_video = self._tensor_to_frames(total_video)
        render_on_step_callback(rendered_video[0])
