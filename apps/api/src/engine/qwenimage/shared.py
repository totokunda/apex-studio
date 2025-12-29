import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any, Callable
from src.engine.base_engine import BaseEngine
from diffusers.image_processor import VaeImageProcessor
from loguru import logger
from src.utils.progress import safe_emit_progress

class QwenImageShared(BaseEngine):
    """Base class for QwenImage engine implementations containing common functionality"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
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
        self.prompt_template_encode_image = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_images = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_image_start_idx = 64
        self.prompt_template_encode_images_start_idx = 64
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

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def _get_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        generator=None,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        return latents

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

    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor | List[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 1024,
        num_images_per_prompt: int = 1,
        text_encoder_kwargs: Optional[Dict[str, Any]] = {},
        progress_callback: Callable | None = None,
    ):

        safe_emit_progress(progress_callback, 0.02, "Preparing prompt inputs")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        base_img_prompt = ""
        if image is None:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
        elif isinstance(image, list):
            img_prompt_template = (
                "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            )
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
            template = self.prompt_template_encode_images
            drop_idx = self.prompt_template_encode_images_start_idx
        else:
            template = self.prompt_template_encode_image
            drop_idx = self.prompt_template_encode_image_start_idx

        txt = [template.format(base_img_prompt + e) for e in prompt]

        

        prompt_hash = self.text_encoder.hash({
            "prompt": prompt,
            "image": image,
            "device": device,
            "dtype": dtype,
            "max_sequence_length": max_sequence_length,
            "num_images_per_prompt": num_images_per_prompt,
            "text_encoder_kwargs": text_encoder_kwargs,
        })

        cached = None
        if self.text_encoder.enable_cache:
            safe_emit_progress(progress_callback, 0.06, "Checking prompt cache")
            cached = self.text_encoder.load_cached(prompt_hash)

        if cached is not None:
            safe_emit_progress(progress_callback, 0.10, "Loaded cached prompt embeddings")
            prompt_embeds, prompt_embeds_mask = cached
        else:
            safe_emit_progress(progress_callback, 0.12, "Encoding prompt embeddings")
            if image is None:
                input_kwargs = {
                    "text": txt,
                    "max_sequence_length": self.tokenizer_max_length + drop_idx,
                    "use_attention_mask": True,
                    "pad_to_max_length": False,
                    "return_attention_mask": True,
                    "output_type": "raw",
                    "add_special_tokens": None,
                    "clean_text": False,
                    **text_encoder_kwargs,
                }
            else:
                safe_emit_progress(progress_callback, 0.14, "Preparing vision-language inputs")
                processor = self.helpers["image.processor"]
                model_inputs = processor(
                    text=txt,
                    images=image,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                input_kwargs = {
                    "input_ids": model_inputs.input_ids,
                    "attention_mask": model_inputs.attention_mask,
                    "pixel_values": model_inputs.pixel_values,
                    "image_grid_thw": model_inputs.image_grid_thw,
                    "output_hidden_states": True,
                }
            if image is None:
                safe_emit_progress(progress_callback, 0.22, "Running text encoder")
                encoder_outputs, attention_mask = self.text_encoder.encode(
                    **input_kwargs,
                )
            else:
                if not self.text_encoder.model_loaded:
                    safe_emit_progress(progress_callback, 0.18, "Loading text encoder model")
                    self.text_encoder.model = self.text_encoder.load_model(no_weights=False)
                    self.text_encoder.model = self.text_encoder.model.to(dtype)
                    self.text_encoder.model_loaded = True

                safe_emit_progress(progress_callback, 0.22, "Running text+vision encoder")

                encoder_outputs = self.text_encoder.model(**input_kwargs)
                attention_mask = model_inputs.attention_mask

            safe_emit_progress(progress_callback, 0.55, "Post-processing embeddings")
            hidden_states = encoder_outputs.hidden_states[-1]
            split_hidden_states = self._extract_masked_hidden(
                hidden_states, attention_mask
            )

            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [
                torch.ones(e.size(0), dtype=torch.long, device=e.device)
                for e in split_hidden_states
            ]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                    for u in split_hidden_states
                ]
            )
            prompt_embeds_mask = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                    for u in attn_mask_list
                ]
            )

            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
            batch_size = len(prompt)

            if image is not None:
                _, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )
                prompt_embeds_mask = prompt_embeds_mask.repeat(
                    1, num_images_per_prompt, 1
                )
                prompt_embeds_mask = prompt_embeds_mask.view(
                    batch_size * num_images_per_prompt, seq_len
                )
            else:
                prompt_embeds = prompt_embeds[:, :max_sequence_length]
                prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
                _, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(num_images_per_prompt, seq_len, -1)
                prompt_embeds_mask = prompt_embeds_mask.repeat(
                    1, num_images_per_prompt, 1
                )
                prompt_embeds_mask = prompt_embeds_mask.view(
                    num_images_per_prompt, seq_len
                )

            if self.text_encoder.enable_cache:
                safe_emit_progress(progress_callback, 0.92, "Caching prompt embeddings")
                self.text_encoder.cache(
                    prompt_hash,
                    prompt_embeds,
                    prompt_embeds_mask,
                )

        safe_emit_progress(progress_callback, 1.0, "Prompt encoding complete")
        return prompt_embeds, prompt_embeds_mask

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def prepare_control_image(
        self,
        control_image,
        batch_size,
        height,
        width,
        transformer_dtype,
        use_cfg_guidance,
    ):
        control_image = self._load_image(control_image)
        control_image = self.image_processor.preprocess(
            control_image, height=height, width=width
        )
        control_image = control_image.repeat_interleave(batch_size, dim=0)
        control_image = control_image.to(device=self.device, dtype=transformer_dtype)
        if use_cfg_guidance:
            control_image = torch.cat([control_image] * 2)

        if control_image.ndim == 4:
            control_image = control_image.unsqueeze(2)

        return control_image

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        """Override: unpack latents for image decoding and render a preview frame.

        Falls back to base implementation if preview dimensions are unavailable.

        """
        self.logger.info(f"Rendering step {latents.shape} OVERRIDDEN")
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
            )[:, :, 0]
            image = self._tensor_to_frame(tensor_image)
            render_on_step_callback(image[0])
        except Exception:
            try:
                super()._render_step(latents, render_on_step_callback)
            except Exception:
                pass

    def base_denoise(self, *args, **kwargs):
        latents = kwargs.get("latents")
        timesteps = kwargs.get("timesteps")
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance = kwargs.get("guidance")
        prompt_embeds = kwargs.get("prompt_embeds")
        image_latents = kwargs.get("image_latents", None)
        prompt_embeds_mask = kwargs.get("prompt_embeds_mask")
        img_shapes = kwargs.get("img_shapes")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = kwargs.get("negative_prompt_embeds_mask")
        txt_seq_lens = kwargs.get("txt_seq_lens")
        negative_txt_seq_lens = kwargs.get("negative_txt_seq_lens")
        attention_kwargs = kwargs.get("attention_kwargs")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        num_warmup_steps = kwargs.get("num_warmup_steps")
        true_cfg_scale = kwargs.get("true_cfg_scale")
        mask_image_latents = kwargs.get("mask_image_latents", None)
        mask = kwargs.get("mask", None)
        noise = kwargs.get("noise", None)
        denoise_progress_callback = kwargs.get("denoise_progress_callback", None)
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass
        
        try:
            self.scheduler.set_begin_index(0)
        except Exception:
            pass
        

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                else:
                    latent_model_input = latents

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    if image_latents is not None:
                        noise_pred = noise_pred[:, : latents.size(1)]

                if use_cfg_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        if image_latents is not None:
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
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

                if mask_image_latents is not None and mask is not None:
                    init_latents_proper = mask_image_latents
                    init_mask = mask
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.scale_noise(
                            init_latents_proper, torch.tensor([noise_timestep]), noise
                        )

                    latents = (
                        1 - init_mask
                    ) * init_latents_proper + init_mask * latents

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

        return latents
