from .shared import WanShared
from src.types import InputImage, InputAudio, InputVideo, OutputVideo
from typing import List, Union, Optional, Dict, Any, Tuple, Callable
import torch
from src.utils.progress import safe_emit_progress, make_mapped_progress
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.wan.image_processor import WanAnimateImageProcessor
from copy import deepcopy


class WanAnimateEngine(WanShared):
    """WAN Animate Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial, resample="bilinear"
        )
        spatial_patch_size = (
            self.transformer.config.patch_size[-2:]
            if self.transformer is not None
            else (2, 2)
        )
        self.vae_image_processor = WanAnimateImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            spatial_patch_size=spatial_patch_size,
            resample="bilinear",
            fill_color=0,
        )
        self.video_processor_for_mask = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            do_normalize=False,
            do_convert_grayscale=True,
        )
        self.fps = 16

    def get_i2v_mask(
        self,
        batch_size: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        mask_len: int = 1,
        mask_pixel_values: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        # mask_pixel_values shape (if supplied): [B, C = 1, T, latent_h, latent_w]
        if mask_pixel_values is None:
            mask_lat_size = torch.zeros(
                batch_size,
                1,
                (latent_t - 1) * 4 + 1,
                latent_h,
                latent_w,
                dtype=dtype,
                device=device,
            )
        else:
            mask_lat_size = mask_pixel_values.clone().to(device=device, dtype=dtype)
        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        # Repeat first frame mask self.vae_scale_factor_temporal (= 4) times in the frame dimension
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_h, latent_w
        ).transpose(
            1, 2
        )  # [B, C = 1, 4 * T_lat, H_lat, W_lat] --> [B, C = 4, T_lat, H_lat, W_lat]

        return mask_lat_size

    def prepare_reference_image_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        sample_mode: str = "mode",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        offload: bool = True,
    ) -> torch.Tensor:
        # image shape: (B, C, H, W) or (B, C, T, H, W)
        dtype = dtype or self.component_dtypes["vae"]
        if image.ndim == 4:
            # Add a singleton frame dimension after the channels dimension
            image = image.unsqueeze(2)

        _, _, _, height, width = image.shape

        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        # Encode image to latents using VAE
        image = image.to(device=device, dtype=dtype)
        if isinstance(generator, list):
            # Like in prepare_latents, assume len(generator) == batch_size
            ref_image_latents = [
                self.vae_encode(
                    image, sample_generator=g, sample_mode=sample_mode, offload=offload
                )
                for g in generator
            ]
            ref_image_latents = torch.cat(ref_image_latents)
        else:
            ref_image_latents = self.vae_encode(
                image,
                sample_generator=generator,
                sample_mode=sample_mode,
                offload=offload,
            )
        # Handle the case where we supply one image and one generator, but batch_size > 1 (e.g. generating multiple
        # videos per prompt)
        if ref_image_latents.shape[0] == 1 and batch_size > 1:
            ref_image_latents = ref_image_latents.expand(batch_size, -1, -1, -1, -1)

        # Prepare I2V mask in latent space and prepend to the reference image latents along channel dim
        reference_image_mask = self.get_i2v_mask(
            batch_size, 1, latent_height, latent_width, 1, None, dtype, device
        )
        reference_image_latents = torch.cat(
            [reference_image_mask, ref_image_latents], dim=1
        )

        return reference_image_latents

    def prepare_prev_segment_cond_latents(
        self,
        prev_segment_cond_video: Optional[torch.Tensor] = None,
        background_video: Optional[torch.Tensor] = None,
        mask_video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        segment_frame_length: int = 77,
        start_frame: int = 0,
        height: int = 720,
        width: int = 1280,
        prev_segment_cond_frames: int = 1,
        task: str = "animate",
        interpolation_mode: str = "bicubic",
        sample_mode: str = "mode",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        offload: bool = True,
    ) -> torch.Tensor:
        # prev_segment_cond_video shape: (B, C, T, H, W) in pixel space if supplied
        # background_video shape: (B, C, T, H, W) (same as prev_segment_cond_video shape)
        # mask_video shape: (B, 1, T, H, W) (same as prev_segment_cond_video, but with only 1 channel)
        device = device or self.device
        dtype = dtype or self.component_dtypes["vae"]
        if prev_segment_cond_video is None:
            if task == "replace":
                prev_segment_cond_video = background_video[
                    :, :, :prev_segment_cond_frames
                ].to(dtype)
            else:
                cond_frames_shape = (
                    batch_size,
                    3,
                    prev_segment_cond_frames,
                    height,
                    width,
                )  # In pixel space

                prev_segment_cond_video = torch.zeros(
                    cond_frames_shape, dtype=dtype, device=device
                )

        data_batch_size, channels, _, segment_height, segment_width = (
            prev_segment_cond_video.shape
        )
        num_latent_frames = (
            segment_frame_length - 1
        ) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        if segment_height != height or segment_width != width:
            print(
                f"Interpolating prev segment cond video from ({segment_width}, {segment_height}) to ({width}, {height})"
            )
            # Perform a 4D (spatial) rather than a 5D (spatiotemporal) reshape, following the original code
            prev_segment_cond_video = prev_segment_cond_video.transpose(1, 2).flatten(
                0, 1
            )  # [B * T, C, H, W]
            prev_segment_cond_video = F.interpolate(
                prev_segment_cond_video, size=(height, width), mode=interpolation_mode
            )
            prev_segment_cond_video = prev_segment_cond_video.unflatten(
                0, (batch_size, -1)
            ).transpose(1, 2)

        # Fill the remaining part of the cond video segment with zeros (if animating) or the background video (if
        # replacing).
        if task == "replace":
            remaining_segment = background_video[:, :, prev_segment_cond_frames:].to(
                dtype
            )
        else:
            remaining_segment_frames = segment_frame_length - prev_segment_cond_frames
            remaining_segment = torch.zeros(
                batch_size,
                channels,
                remaining_segment_frames,
                height,
                width,
                dtype=dtype,
                device=device,
            )

        # Prepend the conditioning frames from the previous segment to the remaining segment video in the frame dim
        prev_segment_cond_video = prev_segment_cond_video.to(dtype=dtype)
        full_segment_cond_video = torch.cat(
            [prev_segment_cond_video, remaining_segment], dim=2
        )

        if isinstance(generator, list):
            if data_batch_size == len(generator):
                prev_segment_cond_latents = [
                    self.vae_encode(
                        full_segment_cond_video[i].unsqueeze(0),
                        sample_generator=g,
                        sample_mode=sample_mode,
                        offload=offload,
                    )
                    for i, g in enumerate(generator)
                ]
            elif data_batch_size == 1:
                # Like prepare_latents, assume len(generator) == batch_size
                prev_segment_cond_latents = [
                    self.vae_encode(
                        full_segment_cond_video,
                        sample_mode=sample_mode,
                        generator=g,
                        offload=offload,
                    )
                    for g in generator
                ]
            else:
                raise ValueError(
                    f"The batch size of the prev segment video should be either {len(generator)} or 1 but is"
                    f" {data_batch_size}"
                )
            prev_segment_cond_latents = torch.cat(prev_segment_cond_latents)
        else:
            prev_segment_cond_latents = self.vae_encode(
                full_segment_cond_video,
                sample_generator=generator,
                sample_mode=sample_mode,
                offload=offload,
            )

        # Prepare I2V mask
        if task == "replace":
            mask_video = 1 - mask_video
            mask_video = mask_video.permute(0, 2, 1, 3, 4)
            mask_video = mask_video.flatten(0, 1)
            mask_video = F.interpolate(
                mask_video, size=(latent_height, latent_width), mode="nearest"
            )
            mask_pixel_values = mask_video.unflatten(0, (batch_size, -1))
            mask_pixel_values = mask_pixel_values.permute(
                0, 2, 1, 3, 4
            )  # output shape: [B, C = 1, T, H_lat, W_lat]
        else:
            mask_pixel_values = None
        prev_segment_cond_mask = self.get_i2v_mask(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            mask_len=prev_segment_cond_frames if start_frame > 0 else 0,
            mask_pixel_values=mask_pixel_values,
            dtype=dtype,
            device=device,
        )

        # Prepend cond I2V mask to prev segment cond latents along channel dimension
        prev_segment_cond_latents = torch.cat(
            [prev_segment_cond_mask, prev_segment_cond_latents], dim=1
        )
        return prev_segment_cond_latents

    def encode_image(
        self,
        image: InputImage,
        offload: bool = True,
    ):
        image_processor = self.helpers["image_processor"]
        image_encoder = self.helpers["image_encoder"]
        image = image_processor(images=image, return_tensors="pt").to(self.device)
        image_embeds = image_encoder(**image, output_hidden_states=True)
        if offload:
            self._offload("image_encoder")
        return image_embeds.hidden_states[-2]

    def prepare_pose_latents(
        self,
        pose_video: torch.Tensor,
        batch_size: int = 1,
        sample_mode: str = "mode",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        offload: bool = True,
    ) -> torch.Tensor:
        # pose_video shape: (B, C, T, H, W)
        device = device or self.device
        if isinstance(generator, list):
            pose_latents = [
                self.vae_encode(
                    pose_video,
                    sample_generator=g,
                    sample_mode=sample_mode,
                    offload=offload,
                )
                for g in generator
            ]
            pose_latents = torch.cat(pose_latents)
        else:
            pose_latents = self.vae_encode(
                pose_video,
                sample_generator=generator,
                sample_mode=sample_mode,
                offload=offload,
            )
        if pose_latents.shape[0] == 1 and batch_size > 1:
            pose_latents = pose_latents.expand(batch_size, -1, -1, -1, -1)
        return pose_latents

    def pad_video_frames(self, frames: List[Any], num_target_frames: int) -> List[Any]:
        """
        Pads an array-like video `frames` to `num_target_frames` using a "reflect"-like strategy. The frame dimension
        is assumed to be the first dimension. In the 1D case, we can visualize this strategy as follows:

        pad_video_frames([1, 2, 3, 4, 5], 10) -> [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
        """
        idx = 0
        flip = False
        target_frames = []
        while len(target_frames) < num_target_frames:
            target_frames.append(deepcopy(frames[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(frames) - 1:
                flip = not flip

        return target_frames

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 77,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames + 1,
            latent_height,
            latent_width,
        )
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

        return latents

    def run(
        self,
        image: InputImage,
        pose_video: Optional[InputVideo] = None,
        face_video: Optional[InputVideo] = None,
        background_video: Optional[InputVideo] = None,
        mask_video: Optional[InputVideo] = None,
        prompt: Union[str, List[str]] = "视频中的人在做动作",
        negative_prompt: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        segment_frame_length: int = 77,
        num_inference_steps: int = 20,
        mode: str = "animate",
        prev_segment_conditioning_frames: int = 1,
        motion_encode_batch_size: Optional[int] = None,
        guidance_scale: float = 1.0,
        num_videos: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        progress_callback: Callable = None,
        render_on_step: bool = False,
        render_on_step_callback: Callable = None,
        return_latents: bool = False,
        offload: bool = True,
        render_step_interval: int = 3,
        attention_kwargs: Dict[str, Any] = None,
        chunking_profile: str = "none",
        **kwargs,
    ) -> OutputVideo:

        safe_emit_progress(progress_callback, 0.0, "Starting animate pipeline")
        if mode == "animate":
            if pose_video is None:
                raise ValueError("Pose video is required for animate mode")
            if face_video is None:
                raise ValueError("Face video is required for animate mode")
        elif mode == "replace":
            if background_video is None:
                raise ValueError("Background video is required for replace mode")
            if mask_video is None:
                raise ValueError("Mask video is required for replace mode")
        if segment_frame_length % self.vae_scale_factor_temporal != 1:
            self.logger.warning(
                f"`segment_frame_length - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the"
                f" nearest number."
            )
            segment_frame_length = (
                segment_frame_length
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        segment_frame_length = max(segment_frame_length, 1)
        use_cfg_guidance = guidance_scale > 1.0

        image = self._load_image(image)
        image, height, width = self._aspect_ratio_resize(image, max_area=height * width)

        pose_video = self._load_video(pose_video, fps=self.fps)
        face_video = self._load_video(face_video, fps=self.fps)
        if background_video is not None:
            background_video = self._load_video(background_video, fps=self.fps)
        if mask_video is not None:
            mask_video = self._load_video(mask_video, fps=self.fps)

        self._guidance_scale = guidance_scale

        device = self.device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # As we generate in segments of `segment_frame_length`, set the target frame length to be the least multiple
        # of the effective segment length greater than or equal to the length of `pose_video`.
        cond_video_frames = len(pose_video)

        effective_segment_length = (
            segment_frame_length - prev_segment_conditioning_frames
        )
        last_segment_frames = (
            cond_video_frames - prev_segment_conditioning_frames
        ) % effective_segment_length
        if last_segment_frames == 0:
            num_padding_frames = 0
        else:
            num_padding_frames = effective_segment_length - last_segment_frames
        num_target_frames = cond_video_frames + num_padding_frames
        num_segments = num_target_frames // effective_segment_length

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_cfg_guidance=use_cfg_guidance,
            num_videos=num_videos,
            max_sequence_length=512,
            offload=offload,
            progress_callback=progress_callback,
        )
        safe_emit_progress(progress_callback, 0.15, "Encoded prompt")

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Preprocess and encode the reference (character) image
        image_pixels = self.vae_image_processor.preprocess(
            image, height=height, width=width, resize_mode="fill"
        ).to(device, dtype=torch.float32)

        # Get CLIP features from the reference image
        if image_embeds is None:
            image_embeds = self.encode_image(image, offload=offload)
        safe_emit_progress(progress_callback, 0.20, "Encoded reference image")
        image_embeds = image_embeds.repeat(batch_size * num_videos, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        if self.transformer is None:
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)
        if chunking_profile != "none":
            self.transformer.set_chunking_profile(chunking_profile)
        safe_emit_progress(progress_callback, 0.25, "Transformer ready")

        # 5. Encode conditioning videos (pose, face)
        pose_video = self.pad_video_frames(pose_video, num_target_frames)
        face_video = self.pad_video_frames(face_video, num_target_frames)

        # TODO: also support np.ndarray input (e.g. from a video reader like the original implementation?)
        pose_video_width, pose_video_height = pose_video[0].size
        if pose_video_height != height or pose_video_width != width:
            self.logger.warning(
                f"Reshaping pose video from ({pose_video_width}, {pose_video_height}) to ({width}, {height})"
            )
        pose_video = self.video_processor.preprocess_video(
            pose_video, height=height, width=width
        ).to(device, dtype=torch.float32)

        face_video_width, face_video_height = face_video[0].size
        expected_face_size = self.transformer.motion_encoder_size
        # check if face_video is square
        if face_video_width != face_video_height:
            self.logger.warning(
                f"Reshaping face video from ({face_video_width}, {face_video_height}) to ({face_video_height},"
                f" {face_video_height})"
            )
            face_video = [
                frame.resize((face_video_height, face_video_height))
                for frame in face_video
            ]

        if (
            face_video_width != expected_face_size
            or face_video_height != expected_face_size
        ):
            # we will resize the face video to the expected face size
            self.logger.info(
                f"Reshaping face video from ({face_video_width}, {face_video_height}) to ({expected_face_size}, {expected_face_size})"
            )
            face_video = [
                frame.resize((expected_face_size, expected_face_size))
                for frame in face_video
            ]

        face_video = self.video_processor.preprocess_video(
            face_video, height=expected_face_size, width=expected_face_size
        ).to(device, dtype=torch.float32)

        if mode == "replace":
            background_video = self.pad_video_frames(
                background_video, num_target_frames
            )
            mask_video = self.pad_video_frames(mask_video, num_target_frames)

            background_video = self.video_processor.preprocess_video(
                background_video, height=height, width=width
            ).to(device, dtype=torch.float32)
            mask_video = self.video_processor_for_mask.preprocess_video(
                mask_video, height=height, width=width
            ).to(device, dtype=torch.float32)

        safe_emit_progress(progress_callback, 0.30, "Processed conditioning videos")

        if self.scheduler is None:
            self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)
        # 6. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
        )
        safe_emit_progress(progress_callback, 0.35, "Timesteps computed")

        # Get VAE-encoded latents of the reference (character) image
        reference_image_latents = self.prepare_reference_image_latents(
            image_pixels,
            batch_size * num_videos,
            generator=generator,
            device=device,
            offload=offload,
        )

        # 8. Loop over video inference segments
        start = 0
        end = segment_frame_length  # Data space frames, not latent frames
        all_out_frames = []
        self._preview_all_out_frames = []
        self._cond_video_frames = cond_video_frames
        out_frames = None

        safe_emit_progress(progress_callback, 0.40, "Starting generation segments")
        for i in range(num_segments):
            # Calculate progress for this segment
            segment_start_progress = 0.40 + (i / num_segments) * 0.50
            segment_end_progress = 0.40 + ((i + 1) / num_segments) * 0.50
            denoise_progress_callback = make_mapped_progress(
                progress_callback, segment_start_progress, segment_end_progress
            )

            assert start + prev_segment_conditioning_frames < cond_video_frames

            # Sample noisy latents from prior for the current inference segment
            latents = self.prepare_latents(
                batch_size * num_videos,
                num_channels_latents=self.num_channels_latents,
                height=height,
                width=width,
                num_frames=segment_frame_length,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=(
                    latents if start == 0 else None
                ),  # Only use pre-calculated latents for first segment
            )

            pose_video_segment = pose_video[:, :, start:end]
            face_video_segment = face_video[:, :, start:end]

            face_video_segment = face_video_segment.expand(
                batch_size * num_videos, -1, -1, -1, -1
            )
            face_video_segment = face_video_segment.to(dtype=transformer_dtype)

            if start > 0:
                prev_segment_cond_video = (
                    out_frames[:, :, -prev_segment_conditioning_frames:]
                    .clone()
                    .detach()
                )
            else:
                prev_segment_cond_video = None

            if mode == "replace":
                background_video_segment = background_video[:, :, start:end]
                mask_video_segment = mask_video[:, :, start:end]

                background_video_segment = background_video_segment.expand(
                    batch_size * num_videos, -1, -1, -1, -1
                )
                mask_video_segment = mask_video_segment.expand(
                    batch_size * num_videos, -1, -1, -1, -1
                )
            else:
                background_video_segment = None
                mask_video_segment = None

            pose_latents = self.prepare_pose_latents(
                pose_video_segment,
                batch_size * num_videos,
                generator=generator,
                device=device,
                offload=offload,
            )
            pose_latents = pose_latents.to(dtype=transformer_dtype)

            prev_segment_cond_latents = self.prepare_prev_segment_cond_latents(
                prev_segment_cond_video,
                background_video=background_video_segment,
                mask_video=mask_video_segment,
                batch_size=batch_size * num_videos,
                segment_frame_length=segment_frame_length,
                start_frame=start,
                height=height,
                width=width,
                prev_segment_cond_frames=prev_segment_conditioning_frames,
                task=mode,
                generator=generator,
                device=device,
                offload=offload,
            )

            # Concatenate the reference latents in the frame dimension
            reference_latents = torch.cat(
                [reference_image_latents, prev_segment_cond_latents], dim=2
            )

            # 8.1 Denoising loop
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * self.scheduler.order
            )
            self._num_timesteps = len(timesteps)


            latents = self.denoise(
                timesteps=timesteps,
                latents=latents,
                latent_condition=reference_latents,
                num_warmup_steps=num_warmup_steps,
                scheduler=self.scheduler,
                guidance_scale=guidance_scale,
                render_on_step_callback=render_on_step_callback,
                denoise_progress_callback=denoise_progress_callback,
                transformer_kwargs=dict(
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    pose_hidden_states=pose_latents,
                    face_pixel_values=face_video_segment,
                    motion_encode_batch_size=motion_encode_batch_size,
                    attention_kwargs=attention_kwargs,
                ),
                unconditional_transformer_kwargs=dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    pose_hidden_states=pose_latents,
                    face_pixel_values=face_video_segment,
                    motion_encode_batch_size=motion_encode_batch_size,
                    attention_kwargs=attention_kwargs,
                ),
                transformer_dtype=transformer_dtype,
                use_cfg_guidance=use_cfg_guidance,
                render_on_step=render_on_step,
            )

            out_frames = self.vae_decode(latents[:, :, 1:], offload=offload)
            video = self._tensor_to_frames(out_frames)
            from diffusers.utils import export_to_video

            export_to_video(video[0], "output_animate_segment.mp4", fps=16, quality=8.0)
            if start > 0:
                out_frames = out_frames[:, :, prev_segment_conditioning_frames:]
            all_out_frames.append(out_frames)
            self._preview_all_out_frames.append(out_frames)

            start += effective_segment_length
            end += effective_segment_length

            # Reset scheduler timesteps / state for next denoising loop
            timesteps, num_inference_steps = self._get_timesteps(
                scheduler=self.scheduler,
                num_inference_steps=num_inference_steps,
            )

        if offload:
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            return latents

        video = torch.cat(all_out_frames, dim=2)[:, :, :cond_video_frames]

        video = self._tensor_to_frames(video)
        safe_emit_progress(progress_callback, 1.0, "Completed animate pipeline")

        return video

    def _render_step(self, latents, render_on_step_callback):
        video = self.vae_decode(latents)
        total_video = torch.cat(self._preview_all_out_frames + [video], dim=2)[
            :, :, : self._cond_video_frames
        ]

        rendered_video = self._tensor_to_frames(total_video)
        render_on_step_callback(rendered_video[0])
