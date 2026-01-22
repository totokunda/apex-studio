import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor


class HunyuanVideoShared(BaseEngine):
    """Base class for Hunyuan engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor_temporal = (
            getattr(self.vae, "temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(self.vae, "spatial_compression_ratio", None) or 8
            if getattr(self, "vae", None)
            else 8
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def _calculate_shift(
        self,
        image_seq_len,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    ):
        """Calculate shift parameter for timestep scheduling"""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str], None] = None,
        image: Union[
            Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor, None
        ] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
        image_embed_interleave: int = 2,
        hyavatar: bool = False,
        **kwargs,
    ):
        """Encode prompts using both LLaMA and CLIP text encoders"""
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        self.llama_text_encoder = self.helpers["hunyuanvideo.llama"]

        if isinstance(prompt, str):
            prompt = [prompt]

        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]

        if prompt_2 is None:
            prompt_2 = prompt

        prompt_embeds, prompt_attention_mask = self.llama_text_encoder(
            prompt,
            image=image,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
            image_embed_interleave=image_embed_interleave,
            hyavatar=hyavatar,
        )

        pooled_prompt_embeds = self.text_encoder.encode(
            prompt_2,
            max_sequence_length=77,
            pad_to_max_length=True,
            use_attention_mask=False,
            use_position_ids=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
            **kwargs,
        )

        return pooled_prompt_embeds, prompt_embeds, prompt_attention_mask

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg

    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        timesteps = kwargs.get("timesteps", None)
        latents = kwargs.get("latents", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_true_cfg_guidance = kwargs.get("use_true_cfg_guidance", False)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        true_guidance_scale = kwargs.get("true_guidance_scale", 1.0)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        image_condition_type = kwargs.get("image_condition_type", None)
        image_latents = kwargs.get("image_latents", None)
        mask = kwargs.get("mask", None)
        noise_pred_kwargs = kwargs.get("noise_pred_kwargs", {})
        unconditional_noise_pred_kwargs = kwargs.get(
            "unconditional_noise_pred_kwargs", {}
        )

        with self._progress_bar(
            total=num_inference_steps, desc=f"Denoising {self.denoise_type}"
        ) as pbar:
            for i, t in enumerate(timesteps):
                if image_condition_type == "latent_concat":
                    latent_model_input = torch.cat(
                        [latents, image_latents, mask], dim=1
                    ).to(transformer_dtype)
                elif image_condition_type == "token_replace":
                    latent_model_input = torch.cat(
                        [image_latents, latents[:, :, 1:]], dim=2
                    ).to(transformer_dtype)
                else:
                    latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Conditional forward pass
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        return_dict=False,
                        **noise_pred_kwargs,
                    )[0]

                # Unconditional forward pass for CFG
                if use_true_cfg_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            return_dict=False,
                            **unconditional_noise_pred_kwargs,
                        )[0]
                    noise_pred = neg_noise_pred + true_guidance_scale * (
                        noise_pred - neg_noise_pred
                    )

                # Scheduler step
                if image_condition_type == "latent_concat":
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                        0
                    ]

                elif image_condition_type == "token_replace":
                    latents_step = scheduler.step(
                        noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False
                    )[0]
                    latents = torch.cat([image_latents, latents_step], dim=2)
                else:
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                        0
                    ]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                pbar.update(1)

        self.logger.info("Denoising completed.")

        return latents
