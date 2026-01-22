import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torch.nn.functional as F
from src.helpers.wan.fun_camera import Camera
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared


class WanFunEngine(WanShared):
    """WAN FUN (Controllable) Engine Implementation"""

    def run(
        self,
        reference_image: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        start_image: Union[
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
        camera_poses: Union[List[float], str, List[Camera], Camera, None] = None,
        subject_reference_images: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        mask: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        process_first_mask_frame_only: bool = False,
        prompt: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        progress_callback: Callable | None = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
        chunking_profile: str = "none",
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting FUN control pipeline")

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
            0.13,
            (
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.15, "Text encoder offloaded")

        if start_image is not None:
            loaded_image = self._load_image(start_image)

            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )
            preprocessed_image = (
                self.video_processor.preprocess(
                    loaded_image, height=height, width=width
                )
                .to(self.device, dtype=torch.float32)
                .unsqueeze(2)
            )

            start_image_latents = self._prepare_fun_control_latents(
                preprocessed_image, dtype=torch.float32, generator=generator
            )

        transformer_config = self.load_config_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        latents = self._get_latents(
            height,
            width,
            duration,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        if start_image is not None:
            start_image_latents_in = torch.zeros_like(latents)
            if start_image_latents_in.shape[2] > 1:
                start_image_latents_in[:, :, :1] = start_image_latents
        else:
            preprocessed_image = None
            start_image_latents = None
            start_image_latents_in = torch.zeros_like(latents)

        if camera_poses is not None:
            control_latents = None
            if isinstance(camera_poses, Camera):
                camera_poses = [camera_poses]
            camera_preprocessor = self.helpers["wan.fun_camera"]
            control_camera_video = camera_preprocessor(
                camera_poses, H=height, W=width, device=self.device
            )
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(
                        control_camera_video[:, :, 0:1], repeats=4, dim=2
                    ),
                    control_camera_video[:, :, 1:],
                ],
                dim=2,
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = (
                control_camera_latents.contiguous()
                .view(b, f // 4, 4, c, h, w)
                .transpose(2, 3)
            )
            control_camera_latents = (
                control_camera_latents.contiguous()
                .view(b, f // 4, c * 4, h, w)
                .transpose(1, 2)
            )

        elif video is not None:

            pt, ph, pw = transformer_config.patch_size
            loaded_video = self._load_video(video, fps=fps)
            video_height, video_width = self.video_processor.get_default_height_width(
                loaded_video[0]
            )
            base = self.vae_scale_factor_spatial * ph
            if video_height * video_width > height * width:
                scale = min(width / video_width, height / video_height)
                video_height, video_width = int(video_height * scale), int(
                    video_width * scale
                )

            if video_height % base != 0 or video_width % base != 0:
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base

            assert video_height * video_width <= height * width

            preprocessed_video = self.video_processor.preprocess_video(
                loaded_video, video_height, video_width
            )

            if mask is not None:
                batch_size, latent_num_frames, _, _, _ = latents.shape
                loaded_mask = self._load_video(mask, fps=fps)
                preprocessed_mask = self.video_processor.preprocess_video(
                    loaded_mask, video_height, video_width
                )
                preprocessed_mask = torch.clamp(
                    (preprocessed_mask + 1) / 2, min=0, max=1
                )

                if (preprocessed_mask == 0).all():
                    mask_latents = torch.tile(
                        torch.zeros_like(latents)[:, :1].to(
                            self.device, transformer_dtype
                        ),
                        [1, 4, 1, 1, 1],
                    )
                    masked_video_latents = torch.zeros_like(latents).to(
                        self.device, transformer_dtype
                    )
                else:
                    masked_video = preprocessed_video * (
                        torch.tile(preprocessed_mask, [1, 3, 1, 1, 1]) < 0.5
                    )
                    masked_video_latents = self._prepare_fun_control_latents(
                        masked_video, dtype=torch.float32, generator=generator
                    )
                    mask_condition = torch.concat(
                        [
                            torch.repeat_interleave(
                                preprocessed_mask[:, :, 0:1], repeats=4, dim=2
                            ),
                            preprocessed_mask[:, :, 1:],
                        ],
                        dim=2,
                    )
                    mask_condition = mask_condition.view(
                        batch_size, mask_condition.shape[2] // 4, 4, height, width
                    )
                    mask_condition = mask_condition.transpose(1, 2)
                    latent_size = latents.size()
                    batch_size, channels, num_frames, height, width = (
                        masked_video_latents.shape
                    )
                    inverse_mask_condition = 1 - mask_condition

                    if process_first_mask_frame_only:
                        target_size = list(latent_size[2:])
                        target_size[0] = 1
                        first_frame_resized = F.interpolate(
                            inverse_mask_condition[:, :, 0:1, :, :],
                            size=target_size,
                            mode="trilinear",
                            align_corners=False,
                        )

                        target_size = list(latent_size[2:])
                        target_size[0] = target_size[0] - 1
                        if target_size[0] != 0:
                            remaining_frames_resized = F.interpolate(
                                inverse_mask_condition[:, :, 1:, :, :],
                                size=target_size,
                                mode="trilinear",
                                align_corners=False,
                            )
                            resized_mask = torch.cat(
                                [first_frame_resized, remaining_frames_resized], dim=2
                            )
                        else:
                            resized_mask = first_frame_resized
                    else:
                        target_size = list(latent_size[2:])
                        resized_mask = F.interpolate(
                            inverse_mask_condition,
                            size=target_size,
                            mode="trilinear",
                            align_corners=False,
                        )
                    mask_latents = resized_mask

                control_camera_latents = None
                control_latents = torch.concat(
                    [mask_latents, masked_video_latents], dim=1
                )
            else:
                control_latents = self._prepare_fun_control_latents(
                    preprocessed_video, dtype=torch.float32, generator=generator
                )
                control_camera_latents = None
        else:
            control_latents = torch.zeros_like(latents)
            control_camera_latents = None

        if reference_image is not None and transformer_config.get(
            "add_ref_control", False
        ):
            loaded_image = self._load_image(reference_image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )
            preprocessed_image = (
                self.video_processor.preprocess(
                    loaded_image, height=height, width=width
                )
                .to(self.device, dtype=torch.float32)
                .unsqueeze(2)
            )

            reference_image_latents = self._prepare_fun_control_latents(
                preprocessed_image, dtype=torch.float32, generator=generator
            )
        else:
            reference_image_latents = torch.zeros_like(latents)[:, :, :1]

        if subject_reference_images is not None:
            subject_reference_image_latents = []
            for image in subject_reference_images:
                loaded_image = self._load_image(image)
                loaded_image, height, width = self._aspect_ratio_resize(
                    loaded_image, max_area=height * width
                )
                preprocessed_image = (
                    self.video_processor.preprocess(
                        loaded_image, height=height, width=width
                    )
                    .to(self.device, dtype=torch.float32)
                    .unsqueeze(2)
                )
                subject_reference_image_latent = self._prepare_fun_control_latents(
                    preprocessed_image, dtype=torch.float32, generator=generator
                )
                subject_reference_image_latents.append(subject_reference_image_latent)
            subject_reference_image_latents = torch.cat(
                subject_reference_image_latents, dim=2
            )
        else:
            subject_reference_image_latents = None

        if reference_image is not None:
            clip_image = reference_image
        elif start_image is not None:
            clip_image = start_image

        if clip_image is not None:
            loaded_image = self._load_image(clip_image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )
            image_embeds = self.helpers["clip"](
                loaded_image, hidden_states_layer=-2
            ).to(self.device, dtype=transformer_dtype)
        else:
            image_embeds = None

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload:
            self._offload("clip")

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        safe_emit_progress(
            progress_callback, 0.20, "Scheduler ready and timesteps computed"
        )

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        if control_latents is not None and start_image_latents_in is not None:
            control_latents = torch.concat(
                [
                    control_latents,
                    start_image_latents_in,
                ],
                dim=1,
            )
        elif control_latents is None and start_image_latents_in is not None:
            control_latents = start_image_latents_in

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        # Set preview context for per-step rendering on the main engine when available
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        # Reserve a progress span for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(progress_callback, 0.45, "Starting denoise phase")

        latents = self.denoise(
            expand_timesteps=expand_timesteps,
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=control_latents,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                encoder_hidden_states_camera=(
                    control_camera_latents.to(transformer_dtype)
                    if control_camera_latents is not None
                    else None
                ),
                encoder_hidden_states_full_ref=(
                    reference_image_latents.to(transformer_dtype)
                    if reference_image_latents is not None
                    else None
                ),
                encoder_hidden_states_subject_ref=(
                    subject_reference_image_latents.to(transformer_dtype)
                    if subject_reference_image_latents is not None
                    else None
                ),
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_camera=(
                        control_camera_latents.to(transformer_dtype)
                        if control_camera_latents is not None
                        else None
                    ),
                    encoder_hidden_states_full_ref=(
                        reference_image_latents.to(transformer_dtype)
                        if reference_image_latents is not None
                        else None
                    ),
                    encoder_hidden_states_subject_ref=(
                        subject_reference_image_latents.to(transformer_dtype)
                        if subject_reference_image_latents is not None
                        else None
                    ),
                    attention_kwargs=attention_kwargs,
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
            chunking_profile=chunking_profile,
        )

        if offload:
            self._offload("transformer")

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            safe_emit_progress(progress_callback, 0.96, "Decoded latents to video")
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(progress_callback, 1.0, "Completed FUN control pipeline")
            return postprocessed_video
