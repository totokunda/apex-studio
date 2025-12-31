import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as F
import math
from torchvision import transforms
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Tuple
from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import VaeImageProcessor
from src.utils.progress import safe_emit_progress
from src.utils.cache import empty_cache
from .mlx import WanMLXDenoise
from torch import Tensor
import os
import gc

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


class WanShared(BaseEngine, WanMLXDenoise):
    """Base class for WAN engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 4
        )

        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 8
        )

        self.num_channels_latents = getattr(self.vae, "config", {}).get("z_dim", 16)

        self.video_processor = VideoProcessor(
            vae_scale_factor=kwargs.get(
                "vae_scale_factor", self.vae_scale_factor_spatial
            )
        )

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def _prepare_fun_control_latents(
        self, control, dtype=torch.float32, generator: torch.Generator | None = None
    ):
        """Prepare control latents for FUN implementation"""
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        bs = 1
        new_control = []
        for i in range(0, control.shape[0], bs):
            control_bs = control[i : i + bs]
            control_bs = self.vae_encode(
                control_bs, sample_generator=generator, normalize_latents_dtype=dtype
            )
            new_control.append(control_bs)
        control = torch.cat(new_control, dim=0)

        return control

    def resize_and_centercrop(self, cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size

        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = F.interpolate(
                cond_image, size=(final_h, final_w), mode="nearest"
            ).contiguous()
            # crop
            cropped_tensor = transforms.functional.center_crop(
                resized_tensor, target_size
            )
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize(
                (final_w, final_h), resample=Image.BILINEAR
            )
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = (
                torch.from_numpy(resized_image)[None, ...]
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            cropped_tensor = transforms.functional.center_crop(
                resized_tensor, target_size
            )
            cropped_tensor = cropped_tensor[:, :, None, :, :]

        return cropped_tensor

    def _resize_mask(self, mask, latent, process_first_frame_only=True):
        latent_size = latent.size()
        batch_size, channels, num_frames, height, width = mask.shape

        if process_first_frame_only:
            target_size = list(latent_size[2:])
            target_size[0] = 1
            first_frame_resized = F.interpolate(
                mask[:, :, 0:1, :, :],
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

            target_size = list(latent_size[2:])
            target_size[0] = target_size[0] - 1
            if target_size[0] != 0:
                remaining_frames_resized = F.interpolate(
                    mask[:, :, 1:, :, :],
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
                mask, size=target_size, mode="trilinear", align_corners=False
            )
        return resized_mask

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        self.logger.info(f"Rendering step for model type: {self.model_type}")
        if self.model_type == "t2i":
            if os.environ.get("ENABLE_IMAGE_RENDER_STEP", "true") == "true":
                tensor_image = self.vae_decode(latents)[:, :, 0]
                image = self._tensor_to_frame(tensor_image)
                render_on_step_callback(image[0])
        else:
            super()._render_step(latents, render_on_step_callback)

    def encode_prompt(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        use_cfg_guidance: bool = True,
        num_videos: int = 1,
        max_sequence_length: int = 226,
        progress_callback: Callable | None = None,
        text_encoder_kwargs: Dict[str, Any] = {},
        offload: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        safe_emit_progress(progress_callback, 0.05, "Text encoder ready")

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **{"max_sequence_length": max_sequence_length, **text_encoder_kwargs},
        )

        safe_emit_progress(progress_callback, 0.10, "Encoded prompt")

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **{"max_sequence_length": max_sequence_length, **text_encoder_kwargs},
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

        return prompt_embeds, negative_prompt_embeds

    def _encode_ip_image(
        self,
        ip_image: Image.Image | str | np.ndarray | torch.Tensor,
        dtype: torch.dtype = None,
    ):
        ip_image = self._load_image(ip_image)
        ip_image = (
            torch.tensor(np.array(ip_image)).permute(2, 0, 1).float() / 255.0
        )  # [3, H, W]
        ip_image = ip_image.unsqueeze(1).unsqueeze(0).to(dtype=dtype)  # [B, 3, 1, H, W]
        ip_image = ip_image * 2 - 1

        encoded_image = self.vae_encode(ip_image, sample_mode="mode", dtype=dtype)
        return encoded_image

    def _estimate_loaded_module_tensor_bytes(
        self, module: Any, *, include_buffers: bool = True
    ) -> int:
        """Best-effort estimate of a loaded torch module's tensor footprint in bytes."""
        try:
            if module is None or not isinstance(module, torch.nn.Module):
                return 0
        except Exception:
            return 0

        total = 0
        try:
            for p in module.parameters(recurse=True):
                try:
                    total += int(p.numel()) * int(p.element_size())
                except Exception:
                    continue
        except Exception:
            pass

        if include_buffers:
            try:
                for b in module.buffers(recurse=True):
                    try:
                        total += int(b.numel()) * int(b.element_size())
                    except Exception:
                        continue
            except Exception:
                pass

        return int(total)

    def _estimate_component_bytes_by_name(self, component_name: str) -> int:
        """Estimate component weight size from config/weight files (works even if not loaded)."""
        try:
            comp_cfg = self.get_component_by_name(component_name)
            if not comp_cfg:
                return 0
            return int(self._estimate_component_model_size_bytes(comp_cfg) or 0)
        except Exception:
            return 0

    def _should_keep_moe_transformers_on_cpu(self) -> bool:
        """Return True if we can keep *both* MOE transformers resident on CPU when offloading."""
        if psutil is None:
            return False

        try:
            vm = psutil.virtual_memory()
            cpu_available_bytes = int(getattr(vm, "available", 0) or 0)
        except Exception:
            return False

        if cpu_available_bytes <= 0:
            return False

        low_obj = getattr(self, "low_noise_transformer", None)
        high_obj = getattr(self, "high_noise_transformer", None)

        low_bytes = self._estimate_loaded_module_tensor_bytes(low_obj)
        if low_bytes <= 0:
            low_bytes = self._estimate_component_bytes_by_name("low_noise_transformer")

        high_bytes = self._estimate_loaded_module_tensor_bytes(high_obj)
        if high_bytes <= 0:
            high_bytes = self._estimate_component_bytes_by_name(
                "high_noise_transformer"
            )

        # If we can't estimate both sizes at all, be conservative.
        if low_bytes <= 0 or high_bytes <= 0:
            return False

        try:
            headroom_gb = float(os.environ.get("WAN_MOE_CPU_KEEP_HEADROOM_GB", "4.0"))
        except Exception:
            headroom_gb = 4.0

        try:
            safety_mult = float(os.environ.get("WAN_MOE_CPU_KEEP_SAFETY_MULT", "1.15"))
        except Exception:
            safety_mult = 1.15

        try:
            max_frac = float(
                os.environ.get("WAN_MOE_CPU_KEEP_MAX_AVAILABLE_FRAC", "0.80")
            )
        except Exception:
            max_frac = 0.80

        needed_bytes = int((low_bytes + high_bytes) * safety_mult + headroom_gb * 1e9)
        return needed_bytes <= int(cpu_available_bytes * max_frac)

    def moe_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        latent_condition = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        boundary_timestep = kwargs.get("boundary_timestep", None)
        transformer_kwargs = kwargs.get("transformer_kwargs", {})
        unconditional_transformer_kwargs = (
            kwargs.get("unconditional_transformer_kwargs", {}) or {}
        )
        transformer_kwargs.pop("encoder_hidden_states_image", None)
        unconditional_transformer_kwargs.pop("encoder_hidden_states_image", None)
        mask_kwargs = kwargs.get("mask_kwargs", {})
        mask = mask_kwargs.get("mask", None)
        masked_video_latents = mask_kwargs.get("masked_video_latents", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)
        offload = kwargs.get("offload", True)
        extra_step_kwargs = kwargs.get("extra_step_kwargs", {})
        easy_cache_thresh = kwargs.get("easy_cache_thresh", 0.00)
        easy_cache_ret_steps = kwargs.get("easy_cache_ret_steps", 10)
        easy_cache_cutoff_steps = kwargs.get("easy_cache_cutoff_steps", None)
        total_steps = len(timesteps) if timesteps is not None else 0
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")
        keep_moe_transformers_on_cpu = self._should_keep_moe_transformers_on_cpu()

        if total_steps <= 8:
            render_on_step = False

        has_offloaded_low_noise_transformer = False
        has_loaded_low_noise_transformer = False
        has_offloaded_high_noise_transformer = False
        has_loaded_high_noise_transformer = False

        with self._progress_bar(len(timesteps), desc=f"Sampling MOE") as pbar:
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):

                if latent_condition is not None:
                    latent_model_input = torch.cat(
                        [latents, latent_condition], dim=1
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)

                timestep = t.expand(latents.shape[0])

                if (
                    boundary_timestep is not None
                    and t >= boundary_timestep
                    and not has_offloaded_low_noise_transformer
                ):
                    if getattr(self, "low_noise_transformer", None):
                        self.logger.info("Offloading low noise transformer")
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Offloading previous transformer",
                        )
                        # IMPORTANT: group-offloading hooks can keep CPU tensors alive.
                        self._offload(
                            "low_noise_transformer",
                            offload_type=(
                                "cpu" if keep_moe_transformers_on_cpu else "discard"
                            ),
                        )
                        has_offloaded_low_noise_transformer = True

                    if not getattr(self, "high_noise_transformer", None):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Loading new transformer",
                        )

                        self.load_component_by_name("high_noise_transformer")

                        self.to_device(self.high_noise_transformer)
                        self.high_noise_transformer.current_steps = i
                        self.high_noise_transformer.num_inference_steps = total_steps
                        if easy_cache_thresh > 0.0:
                            self.high_noise_transformer.enable_easy_cache(
                                total_steps,
                                easy_cache_thresh,
                                easy_cache_ret_steps,
                                should_reset_global_cache=True,
                            )
                            self.logger.info(
                                f"Enabled easy cache for high noise transformer with threshold {easy_cache_thresh}, ret steps {easy_cache_ret_steps}, cutoff steps {easy_cache_cutoff_steps}"
                            )

                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "New transformer ready",
                        )

                        has_loaded_high_noise_transformer = True

                    if not has_loaded_high_noise_transformer:
                        self.to_device(self.high_noise_transformer)
                        has_loaded_high_noise_transformer = True

                    transformer = self.high_noise_transformer

                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[0]
                else:
                    if (
                        getattr(self, "high_noise_transformer", None)
                        and not has_offloaded_high_noise_transformer
                    ):
                        self.logger.info("Offloading high noise transformer")
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Offloading previous transformer",
                        )
                        self._offload(
                            "high_noise_transformer",
                            offload_type=(
                                "cpu" if keep_moe_transformers_on_cpu else "discard"
                            ),
                        )
                        has_offloaded_high_noise_transformer = True

                    if not getattr(self, "low_noise_transformer", None):
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Loading alternate transformer",
                        )
                        self.load_component_by_name("low_noise_transformer")
                        self.to_device(self.low_noise_transformer)
                        if easy_cache_thresh > 0.0:
                            self.low_noise_transformer.enable_easy_cache(
                                total_steps,
                                easy_cache_thresh,
                                easy_cache_ret_steps,
                                should_reset_global_cache=False,
                            )
                        safe_emit_progress(
                            denoise_progress_callback,
                            float(i) / float(total_steps) if total_steps else 0.0,
                            "Alternate transformer ready",
                        )
                        has_loaded_low_noise_transformer = True

                    if not has_loaded_low_noise_transformer:
                        self.to_device(self.low_noise_transformer)
                        has_loaded_low_noise_transformer = True

                    transformer = self.low_noise_transformer
                    if isinstance(guidance_scale, list):
                        guidance_scale = guidance_scale[1]

                # Standard denoising
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]

                if use_cfg_guidance and kwargs.get(
                    "unconditional_transformer_kwargs", None
                ):
                    uncond_noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("unconditional_transformer_kwargs", {}),
                    )[0]

                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                latents = scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if (
                    self.vae_scale_factor_spatial >= 16
                    and mask is not None
                    and not mask[:, :, 0, :, :].any()
                ):
                    latents = (1 - mask) * masked_video_latents + mask * latents

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

                del transformer

            self.logger.info("Denoising completed.")

        if offload:
            if getattr(self, "low_noise_transformer", None):
                self._offload(
                    "low_noise_transformer",
                    offload_type="cpu" if keep_moe_transformers_on_cpu else "discard",
                )
            if getattr(self, "high_noise_transformer", None):
                self._offload(
                    "high_noise_transformer",
                    offload_type="cpu" if keep_moe_transformers_on_cpu else "discard",
                )

        return latents

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        latent_condition = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        expand_timesteps = kwargs.get("expand_timesteps", False)
        first_frame_mask = kwargs.get("first_frame_mask", None)
        ip_image = kwargs.get("ip_image", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)
        num_warmup_steps = kwargs.get("num_warmup_steps", 0)
        num_reference_images = kwargs.get("num_reference_images", 0)
        offload = kwargs.get("offload", True)
        total_steps = len(timesteps) if timesteps is not None else 0
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")

        if total_steps <= 8:
            render_on_step = False

        if ip_image is not None:
            ip_image_latent = self._encode_ip_image(ip_image, dtype=transformer_dtype)
        else:
            ip_image_latent = None

        if expand_timesteps and first_frame_mask is None:
            mask = torch.ones(latents.shape, dtype=torch.float32, device=self.device)
        else:
            mask = None

        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        model_type_str = getattr(self, "model_type", "WAN")

        with self._progress_bar(
            len(timesteps), desc=f"Sampling {model_type_str}"
        ) as pbar:
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):
                # check if transformer group offload is enabled
                if getattr(self.transformer, "_apex_group_offloading_enabled", False):
                    empty_cache()
                if expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    if latent_condition is not None and first_frame_mask is not None:
                        latent_model_input = (
                            1 - first_frame_mask
                        ) * latent_condition + first_frame_mask * latents
                        latent_model_input = latent_model_input.to(transformer_dtype)
                        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    else:
                        latent_model_input = latents.to(transformer_dtype)
                        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()

                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                    if latent_condition is not None:
                        latent_model_input = torch.cat(
                            [latents, latent_condition], dim=1
                        ).to(transformer_dtype)
                    else:
                        latent_model_input = latents.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    ip_image_hidden_states=ip_image_latent,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("transformer_kwargs", {}),
                )[0]

                ip_image_latent = None

                if use_cfg_guidance and kwargs.get(
                    "unconditional_transformer_kwargs", None
                ):
                    uncond_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **kwargs.get("unconditional_transformer_kwargs", {}),
                    )[0]
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(
                        latents[:, :, num_reference_images:], render_on_step_callback
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % getattr(self.scheduler, "order", 1) == 0
                ):
                    pbar.update(1)

                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps),
                    f"Denoising step {i + 1}/{total_steps}",
                )

            if expand_timesteps and first_frame_mask is not None:
                latents = (
                    1 - first_frame_mask
                ) * latent_condition + first_frame_mask * latents

            self.logger.info("Denoising completed.")

        return latents
