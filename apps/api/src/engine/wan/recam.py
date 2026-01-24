import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared
from src.utils.progress import safe_emit_progress, make_mapped_progress
from torchvision.transforms import v2
import imageio
from einops import rearrange
import torchvision


class WanRecamEngine(WanShared):
    """WAN Recam Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_frames = 81
        self.frame_interval = 1
        self.num_frames = 81
        self.height = 480
        self.width = 832

    def create_frame_process(self, height, width):
        self.height = height
        self.width = width
        self.logger.info(
            f"Creating frame process for height: {height} and width: {width}"
        )
        return v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames(
        self, video, max_num_frames, start_frame_id, interval, num_frames, frame_process
    ):
        input_frames = self._load_video(video)
        total_frames = len(input_frames)

        # If the clip is too short or doesn't have enough frames for the desired
        # stride pattern, evenly duplicate frames so we have exactly max_num_frames
        # frames to sample from.
        if (
            total_frames < max_num_frames
            or total_frames - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            original_length = total_frames
            target_length = max_num_frames

            # Evenly distribute duplicates across the sequence
            resampled = []
            for i in range(target_length):
                source_idx = int(i * original_length / target_length)
                resampled.append(input_frames[source_idx])

            input_frames = resampled
            total_frames = len(input_frames)

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = input_frames[start_frame_id + frame_id * interval]
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def run(
        self,
        source_video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        prompt: List[str] | str,
        camera_extrinsics: (
            str | np.ndarray | torch.Tensor
        ) = "assets/pose/camera_extrinsics.json",
        negative_prompt: List[str] | str = None,
        cam_type: int = 1,
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
        progress_callback: Callable = None,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting recam pipeline")

        frame_process = self.create_frame_process(height, width)
        start_frame_id = torch.randint(
            0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,)
        )[0]
        preprocessed_video = self.load_frames(
            source_video,
            self.max_num_frames,
            start_frame_id,
            self.frame_interval,
            self.num_frames,
            frame_process,
        )

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            use_attention_mask=True,
            clean_text=False,
            **text_encoder_kwargs,
        )

        batch_size = prompt_embeds.shape[0]

        use_cfg_guidance = guidance_scale > 1.0 and negative_prompt is not None

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                use_attention_mask=True,
                clean_text=False,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None

        safe_emit_progress(
            progress_callback,
            0.15,
            (
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.18, "Text encoder offloaded")

        safe_emit_progress(progress_callback, 0.22, "Loaded source video")

        if isinstance(camera_extrinsics, str):
            camera_extrinsics = self.helpers["wan.recam"](
                camera_extrinsics, num_frames=num_frames, cam_type=cam_type
            ).to(self.device)
        elif isinstance(camera_extrinsics, np.ndarray):
            camera_extrinsics = torch.from_numpy(camera_extrinsics).to(self.device)
        else:
            camera_extrinsics = camera_extrinsics.to(self.device)

        safe_emit_progress(progress_callback, 0.26, "Prepared camera extrinsics")

        safe_emit_progress(progress_callback, 0.30, "Preprocessed source video")

        source_latents = self.vae_encode(
            preprocessed_video.unsqueeze(0), offload=offload, sample_mode="mode"
        )

        safe_emit_progress(progress_callback, 0.34, "Encoded source latents")

        if not self.transformer:
            self.load_component_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        self.to_device(self.transformer)

        safe_emit_progress(progress_callback, 0.38, "Transformer ready")

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        safe_emit_progress(progress_callback, 0.42, "Scheduler ready")

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

        latent_timestep = timesteps[:1].repeat(num_videos)

        safe_emit_progress(progress_callback, 0.46, "Computed timesteps")

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

        latents = self._get_latents(
            height,
            width,
            num_frames,
            device=torch.device("cpu"),
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        ).to(self.device, dtype=transformer_dtype)

        safe_emit_progress(progress_callback, 0.50, "Initialized latent noise")

        if video is not None:
            video = self._load_video(video, fps=fps)
            preprocessed_video = self.video_processor.preprocess_video(
                video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            cond_latent = self.vae_encode(preprocessed_video, offload=offload)
            cond_latent = cond_latent[:, :, : latents.shape[2], :, :]
        else:
            cond_latent = None

        if cond_latent is not None:
            if hasattr(self.scheduler, "add_noise"):
                latents = self.scheduler.add_noise(
                    cond_latent, latents, latent_timestep
                )
            else:
                latents = self.scheduler.scale_noise(
                    latents, latent_timestep, cond_latent
                )

        total_steps = len(timesteps) if timesteps is not None else 0
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.5, 0.9)

        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        model_type_str = getattr(self, "model_type", "WAN")
        target_length = latents.shape[2]

        with self._progress_bar(
            len(timesteps), desc=f"Sampling {model_type_str}"
        ) as pbar:
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(
                    dtype=transformer_dtype, device=self.device
                )
                latent_model_input = torch.cat([latents, source_latents], dim=2).to(
                    transformer_dtype
                )

                noise_pred = self.transformer(
                    x=latent_model_input,
                    timestep=timestep,
                    context=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    cam_emb=camera_extrinsics.to(transformer_dtype),
                )

                if use_cfg_guidance and negative_prompt_embeds is not None:
                    uncond_noise_pred = self.transformer(
                        x=latent_model_input,
                        timestep=timestep,
                        context=negative_prompt_embeds,
                        cam_emb=camera_extrinsics.to(transformer_dtype),
                        attention_kwargs=attention_kwargs,
                    )
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                latents = scheduler.step(
                    noise_pred[:, :, :target_length, ...],
                    timesteps[i],
                    latent_model_input[:, :, :target_length, ...],
                    return_dict=False,
                )[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)
                pbar.update(1)
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps),
                    f"Denoising step {i + 1}/{total_steps}",
                )

            self.logger.info("Denoising completed.")

        safe_emit_progress(progress_callback, 0.92, "Denoising completed")

        if offload:
            self._offload("transformer")

        safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            safe_emit_progress(progress_callback, 1.0, "Completed recam pipeline")
            return postprocessed_video
