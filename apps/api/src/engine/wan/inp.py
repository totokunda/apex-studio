import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import torch.nn.functional as F
from src.utils.progress import safe_emit_progress, make_mapped_progress
from .shared import WanShared
from einops import rearrange


class WanInpEngine(WanShared):
    """WAN Inpainting Engine Implementation for video inpainting with masks"""

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
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
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
        progress_callback: Callable | None = None,
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = 0.875,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting inpainting pipeline")

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        safe_emit_progress(progress_callback, 0.05, "Encoded prompt")

        batch_size = prompt_embeds.shape[0]
        loaded_video = self._load_video(video, fps=fps) if video is not None else None
        min_num_frames = len(loaded_video) if loaded_video is not None else None
        if min_num_frames is not None and min_num_frames % 4 == 0:
            min_num_frames = min_num_frames - 1
        num_frames = self._parse_num_frames(duration, fps, min_num_frames)

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
            0.08,
            (
                "Prepared negative prompt embeds"
                if negative_prompt is not None and use_cfg_guidance
                else "Skipped negative prompt embeds"
            ),
        )

        if offload:
            self._offload("text_encoder")

        safe_emit_progress(progress_callback, 0.10, "Text encoder offloaded")

        transformer_config = self.load_config_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        latents = self._get_latents(
            height,
            width,
            duration,
            num_frames=num_frames,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        safe_emit_progress(progress_callback, 0.15, "Initialized latent noise")

        _mask = None
        masked_video_latents = None

        if video is not None and mask is not None:
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

            video = torch.from_numpy(
                np.array([np.array(frame) for frame in loaded_video])
            )[:num_frames]

            video = video.permute([3, 0, 1, 2]).unsqueeze(0) / 255
            video = self.video_processor.preprocess(
                rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width
            )
            video = video.to(dtype=torch.float32)

            video = rearrange(video, "(b f) c h w -> b c f h w", f=num_frames)

            batch_size, latent_num_frames, _, _, _ = latents.shape
            loaded_mask = self._load_video(mask, fps=fps)
            mask = torch.from_numpy(
                np.array([np.array(frame) for frame in loaded_mask])
            )[:num_frames]
            mask = mask.permute([3, 0, 1, 2]).unsqueeze(0) / 255
            mask = self.video_processor.preprocess(
                rearrange(mask, "b c f h w -> (b f) c h w"), height=height, width=width
            )
            mask = mask.to(dtype=torch.float32)
            mask = rearrange(mask, "(b f) c h w -> b c f h w", f=num_frames)

            mask = torch.clamp((mask + 1) / 2, min=0, max=1)

            if (mask == 0).all():
                mask_latents = torch.tile(
                    torch.zeros_like(latents)[:, :1].to(self.device, transformer_dtype),
                    [1, 4, 1, 1, 1],
                )
                masked_video_latents = torch.zeros_like(latents).to(
                    self.device, transformer_dtype
                )
                if self.vae_scale_factor_spatial >= 16:
                    _mask = (
                        torch.ones_like(latents)
                        .to(self.device, transformer_dtype)[:, :1]
                        .to(self.device, transformer_dtype)
                    )
                else:
                    _mask = None
            else:
                # print(f"\n\nmask.shape: {mask.shape}\n\n")
                # print(f"\n\nvideo.shape: {video.shape}\n\n")
                # print(f"\n\ntorch.tile(mask, [1, 3, 1, 1, 1]).shape: {torch.tile(mask, [1, 1, 1, 1, 1]).shape}\n\n")
                # exit()
                # masked_video = video * (
                #     # torch.tile(mask, [1, 3, 1, 1, 1]) < 0.5
                #     torch.tile(mask, [1, 1, 1, 1, 1]) < 0.5
                # )
                # masked_video_latents = self._prepare_fun_control_latents(
                #     masked_video, dtype=torch.float32, generator=generator
                # )

                # mask_f: float [0,1]
                mask_f = mask.float()
                if mask_f.max() > 1:
                    mask_f = mask_f / 255.0

                # collapse to 1 channel
                mask1 = mask_f[:, :1]  # [B,1,T,H,W]

                masked_video = video * (mask1 < 0.5).expand(-1, 3, -1, -1, -1)

                masked_video_latents = self._prepare_fun_control_latents(
                    masked_video, dtype=torch.float32, generator=generator
                )

                # mask_condition = torch.concat(
                #     [
                #         torch.repeat_interleave(
                #             mask[:, :, 0:1], repeats=4, dim=2
                #         ),
                #         mask[:, :, 1:],
                #     ],
                #     dim=2,
                # )
                # mask_condition = mask_condition.view(
                #     batch_size, mask_condition.shape[2] // 4, 4, height, width
                # )
                # mask_condition = mask_condition.transpose(1, 2)

                mask_condition = torch.concat(
                    [
                        torch.repeat_interleave(mask1[:, :, 0:1], repeats=4, dim=2),
                        mask1[:, :, 1:],
                    ],
                    dim=2,
                )

                mask_condition = mask_condition.view(
                    batch_size, mask_condition.shape[2] // 4, 4, height, width
                )

                mask_condition = mask_condition.transpose(1, 2)
                mask_latents = self._resize_mask(
                    1 - mask_condition, masked_video_latents, True
                ).to(self.device, transformer_dtype)

                if self.vae_scale_factor_spatial >= 16:
                    _mask = F.interpolate(
                        mask_condition[:, :1],
                        size=latents.size()[-3:],
                        mode="trilinear",
                        align_corners=True,
                    ).to(self.device, transformer_dtype)
                    if not _mask[:, :, 0, :, :].any():
                        _mask[:, :, 1:, :, :] = 1
                        latents = (1 - _mask) * masked_video_latents + _mask * latents
                else:
                    _mask = None

            control_latents = torch.concat([mask_latents, masked_video_latents], dim=1)
        else:
            control_latents = torch.zeros_like(latents)

        if reference_image is not None and transformer_config.get(
            "add_ref_conv", False
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
            )[:, :, 0]
        else:
            reference_image_latents = torch.zeros_like(latents)[:, :, 0]

        if reference_image is not None:

            clip_image = reference_image
        else:
            clip_image = None

        if clip_image is not None and self.denoise_type != "moe":
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

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                getattr(self.scheduler, "config", self.scheduler),
                "num_train_timesteps",
                1000,
            )
        else:
            boundary_timestep = None

        # Reserve a progress span for denoising [0.40, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.40, 0.90)
        safe_emit_progress(progress_callback, 0.35, "Starting denoise phase")

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=control_latents,
            mask_kwargs=dict(
                mask=_mask,
                masked_video_latents=masked_video_latents,
            ),
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                encoder_hidden_states_full_ref=(
                    reference_image_latents.to(transformer_dtype)
                    if reference_image_latents is not None
                    else None
                ),
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_full_ref=(
                        reference_image_latents.to(transformer_dtype)
                        if reference_image_latents is not None
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
            safe_emit_progress(progress_callback, 1.0, "Completed inpainting pipeline")
            return postprocessed_video
