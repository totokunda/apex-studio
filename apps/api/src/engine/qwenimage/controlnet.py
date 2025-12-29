import torch
from typing import Dict, Any, Callable, List, Union
from .shared import QwenImageShared
import numpy as np
from PIL import Image
from diffusers.models.controlnets.controlnet_qwenimage import (
    QwenImageControlNetModel,
    QwenImageMultiControlNetModel,
)
from src.utils.progress import make_mapped_progress


class QwenImageControlNetEngine(QwenImageShared):
    """QwenImage ControlNet Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: str | Image.Image | np.ndarray = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt,
            num_images_per_prompt=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
        )

        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                negative_prompt,
                num_images_per_prompt=num_images,
                text_encoder_kwargs=text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        if offload:
            self._offload("text_encoder")

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)

        if not hasattr(self, "controlnet"):
            self.load_component_by_name("controlnet")

        self.to_device(self.controlnet)

        if isinstance(self.controlnet, QwenImageControlNetModel):
            control_image = self.prepare_control_image(
                control_image,
                batch_size,
                height,
                width,
                transformer_dtype,
                use_cfg_guidance,
            )
            control_image_latents = self.vae_encode(control_image, offload=offload)
            control_image_latents = control_image_latents.permute(0, 2, 1, 3, 4)
            control_image = self._pack_latents(
                control_image_latents,
                batch_size=control_image_latents.shape[0],
                num_channels_latents=self.num_channels_latents,
                height=control_image_latents.shape[3],
                width=control_image_latents.shape[4],
            ).to(dtype=prompt_embeds.dtype, device=self.device)
        elif isinstance(self.controlnet, QwenImageMultiControlNetModel):
            control_images = []
            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    control_image_,
                    batch_size,
                    height,
                    width,
                    transformer_dtype,
                    use_cfg_guidance,
                )
                control_image_latents = self.vae_encode(control_image_, offload=offload)
                control_image_latents = control_image_latents.permute(0, 2, 1, 3, 4)
                control_image_ = self._pack_latents(
                    control_image_latents,
                    batch_size=control_image_latents.shape[0],
                    num_channels_latents=self.num_channels_latents,
                    height=control_image_latents.shape[3],
                    width=control_image_latents.shape[4],
                ).to(dtype=prompt_embeds.dtype, device=self.device)
                control_images.append(control_image_)
            control_image = control_images
        else:
            raise ValueError(f"Unsupported controlnet type.")

        latents = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            seed=seed,
            generator=generator,
        )

        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ] * num_images

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0]
                if isinstance(self.controlnet, QwenImageControlNetModel)
                else keeps
            )

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if negative_prompt_embeds_mask is not None
            else None
        )

        self.scheduler.set_begin_index(0)

        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [
                        c * s
                        for c, s in zip(
                            controlnet_conditioning_scale, controlnet_keep[i]
                        )
                    ]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                controlnet_block_samples = self.controlnet(
                    hidden_states=latents,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        controlnet_block_samples=controlnet_block_samples,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if use_cfg_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            controlnet_block_samples=controlnet_block_samples,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    comb_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(
                            min((i + 1) / total_steps, 1.0),
                            f"Denoising step {i + 1}/{total_steps}",
                        )
                    except Exception:
                        pass

        if offload:
            self._offload("transformer")
            self._offload("controlnet")

        if return_latents:
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        tensor_image = self.vae_decode(latents, offload=offload)[:, :, 0]
        image = self._tensor_to_frame(tensor_image)
        return image
