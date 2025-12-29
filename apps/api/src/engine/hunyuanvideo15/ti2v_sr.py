import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from einops import rearrange

SR_PIPELINE_CONFIGS = {
    "720p_sr_distilled": {
        "flow_shift": 2.0,
        "base_resolution": "480p",
        "guidance_scale": 1.0,
        "embedded_guidance_scale": None,
        "num_inference_steps": 6,
    },
    "1080p_sr_distilled": {
        "flow_shift": 2.0,
        "base_resolution": "720p",
        "guidance_scale": 1.0,
        "embedded_guidance_scale": None,
        "num_inference_steps": 8,
    },
}

from .ti2v import HunyuanVideo15TI2VEngine


SIZE_MAP = {"480p": 640, "720p": 960, "1080p": 1440}
SR_DEFAULTS = {
    "720p": {"base_resolution": "480p", **SR_PIPELINE_CONFIGS["720p_sr_distilled"]},
    "1080p": {"base_resolution": "720p", **SR_PIPELINE_CONFIGS["1080p_sr_distilled"]},
}


class HunyuanVideo15TI2VSRengine(HunyuanVideo15TI2VEngine):
    """
    HunyuanVideo 1.5 engine with optional super-resolution (720p / 1080p).
    """

    def _load_sr_transformer(self):
        if getattr(self, "sr_transformer", None) is not None:
            return
        for component in self.config.get("components", []):
            if component.get("type") != "transformer":
                continue
            name = component.get("name", "")
            if name in {"sr_transformer", "transformer_sr", "transformer_upsampler"}:
                comp = self.load_component(
                    component,
                    self.component_load_dtypes.get("transformer"),
                )
                setattr(self, name, comp)
                self.sr_transformer = comp
                break
        if getattr(self, "sr_transformer", None) is None:
            raise ValueError(
                "SR transformer component not found; add a transformer component named 'sr_transformer'."
            )

    def _load_upsampler(self, target_resolution: str):
        if getattr(self, "upsampler", None) is not None:
            return
        for component in self.config.get("components", []):
            if component.get("name") == "upsampler":
                comp = self.load_component(component, None)
                self.upsampler = comp
                return
        # Fallback: build from checkpoint path on disk if user embedded weights in config paths
        # but keep type safety with expected upsampler class.
        raise ValueError(
            "Upsampler component not found; add an 'upsampler' component for SR."
        )

    def _bucket_map(
        self, lr_bucket: Tuple[int, int], base_resolution: str, target_resolution: str
    ) -> Tuple[int, int]:
        lr_base = SIZE_MAP[base_resolution]
        hr_base = SIZE_MAP[target_resolution]
        lr_buckets = self._generate_crop_size_list(base_size=lr_base, patch_size=16)
        hr_buckets = self._generate_crop_size_list(base_size=hr_base, patch_size=16)
        lr_aspect = np.array([float(w) / float(h) for w, h in lr_buckets])
        hr_aspect = np.array([float(w) / float(h) for w, h in hr_buckets])
        lr_ratio = float(lr_bucket[0]) / float(lr_bucket[1])
        closest_hr_id = np.abs(hr_aspect - lr_ratio).argmin()
        return hr_buckets[closest_hr_id]

    def _prepare_lq_cond_latents(self, lq_latents: torch.Tensor) -> torch.Tensor:
        b, _, f, h, w = lq_latents.shape
        mask_ones = torch.ones(b, 1, f, h, w, device=lq_latents.device)
        return torch.concat([lq_latents, mask_ones], dim=1)

    def add_noise_to_lq(
        self, lq_latents: torch.Tensor, strength: float = 0.7
    ) -> torch.Tensor:
        noise = torch.randn_like(lq_latents)
        timestep = torch.tensor([1000.0], device=lq_latents.device) * strength
        t = timestep.view(-1, 1, 1, 1, 1)
        return (1 - t / 1000.0) * lq_latents + (t / 1000.0) * noise

    def run(
        self,
        *args,
        sr_resolution: Optional[str] = None,
        sr_num_inference_steps: Optional[int] = None,
        sr_guidance_scale: Optional[float] = None,
        sr_noise_strength: float = 0.7,
        sr_return_pre: bool = False,
        **kwargs: Any,
    ):
        """
        Run base generation; optionally upsample to `sr_resolution` ("720p" or "1080p").
        """
        sr_resolution = (
            sr_resolution.lower() if isinstance(sr_resolution, str) else None
        )
        if sr_resolution not in {None, "720p", "1080p"}:
            raise ValueError("sr_resolution must be None, '720p', or '1080p'")

        # Generate low-resolution latents first
        base_latents = super().run(*args, return_latents=True, offload=True, **kwargs)

        if sr_resolution is None:
            latents_to_decode = (
                base_latents if base_latents.ndim == 5 else base_latents.unsqueeze(2)
            )
            video_frames = self.vae_decode(latents_to_decode, offload=True)
            video_frames = self._tensor_to_frames(video_frames, output_type=output_type)
            return video_frames

        sr_cfg = SR_DEFAULTS[sr_resolution]
        base_resolution = sr_cfg["base_resolution"]
        guidance_scale = (
            sr_guidance_scale
            if sr_guidance_scale is not None
            else sr_cfg.get("guidance_scale", 1.0)
        )
        embedded_guidance_scale = sr_cfg.get("embedded_guidance_scale", None)
        num_steps = sr_num_inference_steps or sr_cfg.get(
            "num_inference_steps", 6 if sr_resolution == "720p" else 8
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        prompt = kwargs.get("prompt", args[0] if len(args) > 0 else None)
        video_length = kwargs.get("video_length", args[2] if len(args) > 2 else None)
        negative_prompt = kwargs.get("negative_prompt", None)
        reference_image = kwargs.get("reference_image", None)
        num_videos_per_prompt = kwargs.get("num_videos_per_prompt", 1)
        generator = kwargs.get("generator", None)
        output_type = kwargs.get("output_type", "pt")

        self._load_sr_transformer()
        self._load_upsampler(sr_resolution)
        sr_use_meanflow = bool(
            getattr(getattr(self.sr_transformer, "config", None), "use_meanflow", False)
        )

        # Ensure devices/dtypes
        sr_transformer_dtype = self.component_dtypes.get("transformer")
        self.to_device(self.sr_transformer)
        if hasattr(self, "upsampler"):
            self.to_device(self.upsampler)

        # Re-create scheduler for SR
        if getattr(self, "scheduler", None) is None:
            self.load_component_by_type("scheduler")

        self.to_device(self.scheduler)

        # Infer sizes
        lr_video_height = base_latents.shape[-2] * self.vae_scale_factor_spatial
        lr_video_width = base_latents.shape[-1] * self.vae_scale_factor_spatial
        width, height = self._bucket_map(
            (lr_video_width, lr_video_height),
            base_resolution=base_resolution,
            target_resolution=sr_resolution,
        )

        # Text encodings
        if getattr(self, "text_encoder", None) is None:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)
        self.text_len = getattr(self.text_encoder, "max_length", 1000)

        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt")
        num_videos_per_prompt = kwargs.get("num_videos_per_prompt", 1)
        device = self.device

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            guidance_scale > 1.0,
            negative_prompt,
            data_type="video",
        )
        if kwargs.get("offload", True):
            self._offload("text_encoder")

        extra_kwargs = self._prepare_byt5_embeddings(prompt, device)
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

        latent_target_length, latent_height, latent_width = self.get_latent_size(
            video_length, height, width
        )
        n_tokens = latent_target_length * latent_height * latent_width

        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_steps,
            **extra_set_timesteps_kwargs,
        )

        latents = self.prepare_latents(
            base_latents.shape[0],
            32,
            latent_height,
            latent_width,
            latent_target_length,
            self.target_dtype,
            device,
            generator,
        )

        # Condition preparation
        if reference_image is not None and isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert("RGB")
        task_type = "i2v" if reference_image is not None else "t2v"
        image_cond = self.get_image_condition_latents(
            task_type, reference_image, height, width
        )

        tgt_shape = latents.shape[-2:]
        bsz = base_latents.shape[0]
        lq_latents = rearrange(base_latents, "b c f h w -> (b f) c h w")
        lq_latents = F.interpolate(
            lq_latents, size=tgt_shape, mode="bilinear", align_corners=False
        )
        lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=bsz)
        lq_latents = self.upsampler(
            lq_latents.to(dtype=torch.float32, device=self.device)
        )
        lq_latents = lq_latents.to(dtype=latents.dtype)
        lq_latents = self.add_noise_to_lq(lq_latents, sr_noise_strength)

        multitask_mask = self.get_task_mask(task_type, latent_target_length)
        cond_latents = self._prepare_cond_latents(
            task_type, image_cond, latents, multitask_mask
        )
        lq_cond_latents = self._prepare_lq_cond_latents(lq_latents)
        condition = torch.concat([cond_latents, lq_cond_latents], dim=1)

        c = lq_latents.shape[1]
        zero_lq_condition = condition.clone()
        zero_lq_condition[:, c + 1 : 2 * c + 1] = torch.zeros_like(lq_latents)
        zero_lq_condition[:, 2 * c + 1] = 0

        vision_states = self._prepare_vision_states(
            None if reference_image is None else np.array(reference_image),
            sr_resolution,
            latents,
            device,
        )

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": kwargs.get("eta", 0.0)}
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self._progress_bar(
            total=num_inference_steps, desc=f"SR {sr_resolution}"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if t < 1000 * sr_noise_strength:
                    condition = zero_lq_condition

                latents_concat = torch.concat([latents, condition], dim=1)
                latent_model_input = (
                    torch.cat([latents_concat] * 2)
                    if guidance_scale > 1.0
                    else latents_concat
                )

                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                t_expand = t.repeat(latent_model_input.shape[0])
                if sr_use_meanflow:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=self.device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                else:
                    timesteps_r = None

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(self.target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=sr_transformer_dtype,
                    enabled=self.autocast_enabled and self.device.type == "cuda",
                ):
                    output = self.sr_transformer(
                        latent_model_input,
                        t_expand,
                        prompt_embeds,
                        None,
                        prompt_mask,
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=task_type,
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]

                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    if self.guidance_rescale > 0.0:
                        noise_pred = self._rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if pbar is not None:
                        pbar.update()

        # Offload heavy sr modules

        self._offload("sr_transformer")
        if hasattr(self, "upsampler"):
            self._offload("upsampler")

        latents_to_decode = latents if latents.ndim == 5 else latents.unsqueeze(2)

        video_frames = self.vae_decode(latents_to_decode, offload=True)
        if output_type == "np":
            video_frames = video_frames.numpy()
        elif output_type == "pil":
            video_frames = self._tensor_to_frames(video_frames, output_type=output_type)

        return video_frames
