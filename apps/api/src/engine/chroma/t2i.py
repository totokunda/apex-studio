import torch
from typing import Dict, Any, Callable, List, Optional, Union
from PIL import Image
from src.engine.base_engine import BaseEngine
import numpy as np
from src.utils.progress import safe_emit_progress, make_mapped_progress
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor


class ChromaT2IEngine(BaseEngine):
    """Chroma Text-to-Image Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.num_channels_latents = (
            self.transformer.config.in_channels // 4 if self.transformer else 16
        )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        seed,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        return latents, latent_image_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def _prepare_attention_mask(
        self,
        batch_size,
        sequence_length,
        dtype,
        attention_mask=None,
    ):
        if attention_mask is None:
            return attention_mask

        # Extend the prompt attention mask to account for image tokens in the final sequence
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, sequence_length, device=attention_mask.device),
            ],
            dim=1,
        )
        attention_mask = attention_mask.to(dtype)

        return attention_mask

    def encode_image(self, image):
        image_encoder = self.helpers["image_encoder"]
        return image_encoder(image)

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images
    ):
        if not self.transformer:
            self.load_component_by_type("transformer")

        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if (
                len(ip_adapter_image)
                != self.transformer.encoder_hid_proj.num_ip_adapters
            ):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if (
                len(ip_adapter_image_embeds)
                != self.transformer.encoder_hid_proj.num_ip_adapters
            ):
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        use_cfg_guidance: bool = True,
        offload: bool = True,
        num_images: int = 1,
        text_encoder_kwargs: Optional[Dict[str, Any]] = {},
        progress_callback: Callable | None = None,
    ):
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            safe_emit_progress(progress_callback, 0.10, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.20, "Text encoder loaded")

        safe_emit_progress(progress_callback, 0.25, "Moving text encoder to device")
        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.30, "Text encoder on device")

        safe_emit_progress(progress_callback, 0.40, "Encoding prompt embeddings")
        prompt_embeds, prompt_embeds_mask = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_images,
            return_attention_mask=True,
            output_type="hidden_states",
            **text_encoder_kwargs,
        )
        safe_emit_progress(progress_callback, 0.60, "Prompt embeddings ready")

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=prompt_embeds.dtype
        )

        if negative_prompt is not None and use_cfg_guidance:
            safe_emit_progress(progress_callback, 0.70, "Encoding negative prompt embeddings")
            negative_prompt_embeds, negative_prompt_embeds_mask = (
                self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_images,
                    return_attention_mask=True,
                    output_type="hidden_states",
                    **text_encoder_kwargs,
                )
            )
            safe_emit_progress(progress_callback, 0.85, "Negative prompt embeddings ready")

            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(
                device=self.device, dtype=negative_prompt_embeds.dtype
            )

        else:
            negative_prompt_embeds = None
            negative_text_ids = None

        if offload:
            safe_emit_progress(progress_callback, 0.95, "Offloading text encoder")
            del self.text_encoder
        safe_emit_progress(progress_callback, 1.0, "Prompt encoding complete")
            
        return prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, text_ids, negative_text_ids
    
    
    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        """Override: unpack latents for image decoding and render a preview frame.

        Falls back to base implementation if preview dimensions are unavailable.

        """
        try:
            preview_height = getattr(self, "_preview_height", None)
            preview_width = getattr(self, "_preview_width", None)
            if preview_height is None or preview_width is None:
                return super()._render_step(latents, render_on_step_callback)

            unpacked = self._unpack_latents(
                latents, preview_height, preview_width, self.vae_scale_factor
            )
            tensor_image = self.vae_decode(
                unpacked, offload=getattr(self, "_preview_offload", True)
            )
            image = self._tensor_to_frame(tensor_image)
            render_on_step_callback(image[0])
        except Exception as e:
            try:
                super()._render_step(latents, render_on_step_callback)
            except Exception:
                pass

    def run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 3.0,
        true_cfg_scale: float = 4.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        progress_callback: Callable = None,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        sigmas: List[float] | None = None,
        render_on_step_interval: int = 3,
        attention_kwargs: Dict[str, Any] = {},
        ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        joint_attention_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        safe_emit_progress(progress_callback, 0.05, "Encoding prompt")
        (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            text_ids,
            negative_text_ids,
        ) = self.encode_prompt(
            prompt,
            negative_prompt,
            num_images=num_images,
            text_encoder_kwargs=text_encoder_kwargs,
            use_cfg_guidance=use_cfg_guidance,
            offload=offload,
            progress_callback=make_mapped_progress(progress_callback, 0.05, 0.20),
        )
        safe_emit_progress(progress_callback, 0.20, "Encoded prompt")

        batch_size = prompt_embeds.shape[0]

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.205, "Loading transformer")
            self.load_component_by_type("transformer")
            safe_emit_progress(progress_callback, 0.215, "Transformer loaded")

        safe_emit_progress(progress_callback, 0.218, "Moving transformer to device")
        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.22, "Transformer ready")

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(self.device)
        safe_emit_progress(progress_callback, 0.24, "Guidance embeddings prepared")

        latents, latent_image_ids = self._get_latents(
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
            seed=seed,
        )
        safe_emit_progress(progress_callback, 0.30, "Initialized latent noise")

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.31, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.33, "Scheduler loaded")

        safe_emit_progress(progress_callback, 0.34, "Moving scheduler to device")
        self.to_device(self.scheduler)
        safe_emit_progress(progress_callback, 0.35, "Scheduler ready")

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )

        image_seq_len = latents.shape[1]

        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        safe_emit_progress(progress_callback, 0.40, "Computed scheduler shift")

        attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=prompt_embeds_mask,
        )
        negative_attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=negative_prompt_embeds_mask,
        )
        safe_emit_progress(progress_callback, 0.45, "Prepared attention masks")

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            mu=mu,
        )
        safe_emit_progress(progress_callback, 0.48, "Timesteps computed")

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        if not hasattr(self, "transformer") or not self.transformer:
            safe_emit_progress(progress_callback, 0.485, "Loading transformer")
            self.load_component_by_type("transformer")

        safe_emit_progress(progress_callback, 0.49, "Ensuring transformer on device")
        self.to_device(self.transformer)

        # Set preview context for per-step rendering on the main engine

        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None
            and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            safe_emit_progress(progress_callback, 0.495, "Preparing IP adapter embeddings")
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                num_images,
            )
        if (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            safe_emit_progress(progress_callback, 0.498, "Preparing negative IP adapter embeddings")
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                self.device,
                num_images,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if image_embeds is not None:
                    joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if use_cfg_guidance:
                    if negative_image_embeds is not None:
                        joint_attention_kwargs["ip_adapter_image_embeds"] = (
                            negative_image_embeds
                        )
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        attention_mask=negative_attention_mask,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + guidance_scale * (
                        noise_pred - neg_noise_pred
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
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

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            try:
                self._offload("transformer")
            except Exception:
                pass
        safe_emit_progress(
            progress_callback,
            0.94,
            "Transformer offloaded" if offload else "Preparing decode",
        )

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        safe_emit_progress(progress_callback, 0.95, "Unpacking latents")
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        safe_emit_progress(progress_callback, 0.97, "Decoding latents")
        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
        return image
