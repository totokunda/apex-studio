from src.engine.base_engine import BaseEngine
from diffusers.image_processor import VaeImageProcessor
import torch
from typing import Union, List, Optional, Callable, Dict, Any
import os
from src.utils.progress import safe_emit_progress, make_mapped_progress
import numpy as np
from diffusers.utils.torch_utils import randn_tensor


class OvisT2IEngine(BaseEngine):
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        # Ovis-Image latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.system_prompt = "Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: "
        self.user_prompt_begin_id = 28
        self.tokenizer_max_length = 256 + self.user_prompt_begin_id
        self.default_sample_size = 128

    def _get_messages(
        self,
        prompt: Union[str, List[str]] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        messages = []
        for each_prompt in prompt:
            message = [
                {
                    "role": "user",
                    "content": self.system_prompt + each_prompt,
                }
            ]
            message = self.text_encoder.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            messages.append(message)
        return messages

    def _get_ovis_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        messages = self._get_messages(prompt)
        batch_size = len(messages)
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        prompt_embeds, attention_mask = self.text_encoder.encode(
            messages,
            pad_with_zero=False,
            max_sequence_length=self.tokenizer_max_length,
            num_videos_per_prompt=num_images_per_prompt,
            add_special_tokens=False,
            clean_text=False,
            output_type="hidden_states",
            use_attention_mask=True,
            return_attention_mask=True,
            reshape_prompt_embeds=False,
        )

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)

        prompt_embeds = prompt_embeds * attention_mask[..., None]
        prompt_embeds = prompt_embeds[:, self.user_prompt_begin_id :, :]

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""

        Args:
            prompt (`str`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self.device

        if prompt_embeds is None:
            prompt_embeds = self._get_ovis_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        dtype = self.component_dtypes["transformer"]
        text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        text_ids[..., 1] = (
            text_ids[..., 1] + torch.arange(prompt_embeds.shape[1])[None, :]
        )
        text_ids[..., 2] = (
            text_ids[..., 2] + torch.arange(prompt_embeds.shape[1])[None, :]
        )
        text_ids = text_ids.to(device=device, dtype=dtype)
        return prompt_embeds, text_ids

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

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
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

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def run(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = "",
        guidance_scale: float = 5.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_latents: bool = False,
        offload: bool = True,
        render_on_step: bool = False,
        render_on_step_callback: Optional[Callable] = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        if not height or not width:
            height = self.default_sample_size * self.vae_scale_factor
            width = self.default_sample_size * self.vae_scale_factor

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        do_classifier_free_guidance = guidance_scale > 1
        (
            prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if do_classifier_free_guidance:
            (
                negative_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        if offload:
            self._offload("text_encoder")

        if not self.transformer:
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)

        if not self.scheduler:
            self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None
        image_seq_len = latents.shape[1]
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
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                if do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
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

        self._current_timestep = None

        if return_latents:
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        images = self.vae_decode(latents, offload=offload)
        images = self._tensor_to_frame(images)

        return images

    def _render_step(self, latents, render_on_step_callback):
        if os.environ.get("ENABLE_IMAGE_RENDER_STEP", "true") == "false":
            return
        latents = self._unpack_latents(
            latents, self._preview_height, self._preview_width, self.vae_scale_factor
        )
        images = self.vae_decode(latents, offload=self._preview_offload)
        images = self._tensor_to_frame(images)
        render_on_step_callback(images[0])
