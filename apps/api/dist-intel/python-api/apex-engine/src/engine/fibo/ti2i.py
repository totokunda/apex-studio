import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.engine.base_engine import BaseEngine
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
import json
import textwrap
from src.types import InputImage


class FiboTI2IEngine(BaseEngine):

    def __init__(self, *args, **kwargs):
        super(FiboTI2IEngine, self).__init__(*args, **kwargs)
        self.vae_scale_factor = 16
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.default_sample_size = 64

    @staticmethod
    # Based on diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
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
    def _unpack_latents_no_patch(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    @staticmethod
    def _pack_latents_no_patch(
        latents, batch_size, num_channels_latents, height, width
    ):
        latents = latents.permute(0, 2, 3, 1)
        latents = latents.reshape(batch_size, height * width, num_channels_latents)
        return latents

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

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if not prompt:
            raise ValueError("`prompt` must be a non-empty string or list of strings.")

        batch_size = len(prompt)
        bot_token_id = 128000

        text_encoder_device = device if device is not None else torch.device("cpu")

        def process_inputs_func(text_inputs):
            if all(p == "" for p in prompt):
                text_inputs.input_ids = torch.full(
                    (batch_size, 1),
                    bot_token_id,
                    dtype=torch.long,
                    device=text_encoder_device,
                )
                text_inputs.attention_mask = torch.ones_like(text_inputs.input_ids)
                return text_inputs
            if any(p == "" for p in prompt):
                empty_rows = torch.tensor(
                    [p == "" for p in prompt],
                    dtype=torch.bool,
                    device=text_encoder_device,
                )
                text_inputs.input_ids[empty_rows] = bot_token_id
                text_inputs.attention_mask[empty_rows] = 1

            return text_inputs

        outputs, attention_mask = self.text_encoder.encode(
            text=prompt,
            max_sequence_length=max_sequence_length,
            device=text_encoder_device,
            use_attention_mask=True,
            pad_to_max_length=False,
            process_inputs_func=process_inputs_func,
            clean_text=False,
            output_type="raw",
            return_attention_mask=True,
        )

        hidden_states = outputs.hidden_states
        prompt_embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        hidden_states = tuple(
            layer.repeat_interleave(num_images_per_prompt, dim=0).to(device=device)
            for layer in hidden_states
        )

        attention_mask = attention_mask.repeat_interleave(
            num_images_per_prompt, dim=0
        ).to(device=device)

        return prompt_embeds, hidden_states, attention_mask

    @staticmethod
    def pad_embedding(prompt_embeds, max_tokens, attention_mask=None):
        # Pad embeddings to `max_tokens` while preserving the mask of real tokens.
        batch_size, seq_len, dim = prompt_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )
        else:
            attention_mask = attention_mask.to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )

        if max_tokens < seq_len:
            raise ValueError(
                "`max_tokens` must be greater or equal to the current sequence length."
            )

        if max_tokens > seq_len:
            pad_length = max_tokens - seq_len
            padding = torch.zeros(
                (batch_size, pad_length, dim),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)

            mask_padding = torch.zeros(
                (batch_size, pad_length),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        return prompt_embeds, attention_mask

    @staticmethod
    def _prepare_attention_mask(attention_mask):
        attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)

        # convert to 0 - keep, -inf ignore
        attention_matrix = torch.where(
            attention_matrix == 1, 0.0, -torch.inf
        )  # Apply -inf to ignored tokens for nulling softmax score
        return attention_matrix

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 3000,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            guidance_scale (`float`):
                Guidance scale for classifier free guidance.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        device = device or self.device
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        transformer_dtype = self.component_dtypes["transformer"]

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_attention_mask = None
        negative_prompt_attention_mask = None
        if prompt_embeds is None:
            prompt_embeds, prompt_layers, prompt_attention_mask = (
                self.get_prompt_embeds(
                    prompt=prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                )
            )
            prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
            prompt_layers = [
                tensor.to(dtype=transformer_dtype) for tensor in prompt_layers
            ]

        if guidance_scale > 1:
            if isinstance(negative_prompt, list) and negative_prompt[0] is None:
                negative_prompt = ""
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            (
                negative_prompt_embeds,
                negative_prompt_layers,
                negative_prompt_attention_mask,
            ) = self.get_prompt_embeds(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=transformer_dtype)
            negative_prompt_layers = [
                tensor.to(dtype=transformer_dtype) for tensor in negative_prompt_layers
            ]

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        # Pad to longest
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )

        if negative_prompt_embeds is not None:
            if negative_prompt_attention_mask is not None:
                negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                    device=negative_prompt_embeds.device,
                    dtype=negative_prompt_embeds.dtype,
                )
            max_tokens = max(negative_prompt_embeds.shape[1], prompt_embeds.shape[1])

            prompt_embeds, prompt_attention_mask = self.pad_embedding(
                prompt_embeds, max_tokens, attention_mask=prompt_attention_mask
            )
            prompt_layers = [
                self.pad_embedding(layer, max_tokens)[0] for layer in prompt_layers
            ]

            negative_prompt_embeds, negative_prompt_attention_mask = self.pad_embedding(
                negative_prompt_embeds,
                max_tokens,
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_layers = [
                self.pad_embedding(layer, max_tokens)[0]
                for layer in negative_prompt_layers
            ]
        else:
            max_tokens = prompt_embeds.shape[1]
            prompt_embeds, prompt_attention_mask = self.pad_embedding(
                prompt_embeds, max_tokens, attention_mask=prompt_attention_mask
            )
            negative_prompt_layers = None

        dtype = self.text_encoder.dtype
        text_ids = torch.zeros(prompt_embeds.shape[0], max_tokens, 3).to(
            device=device, dtype=dtype
        )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_layers,
            negative_prompt_layers,
        )

    def _get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents=None,
        do_patching=False,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        device = self.device

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height, width, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if do_patching:
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, height, width
            )
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
        else:
            latents = self._pack_latents_no_patch(
                latents, batch_size, num_channels_latents, height, width
            )
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height, width, device, dtype
            )

        return latents, latent_image_ids

    def get_default_negative_prompt(self, existing_json: dict) -> str:
        negative_prompt = ""
        style_medium = existing_json.get("style_medium", "").lower()
        if style_medium in ["photograph", "photography", "photo"]:
            negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
        return negative_prompt

    def run(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        structured_prompt: str | None = None,
        image: InputImage | None = None,
        generate_prompt_kwargs: Dict[str, Any] = {},
        guidance_scale: float = 5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        progress_callback: Callable = None,
        max_sequence_length: int = 3000,
        attention_kwargs: Dict[str, Any] = {},
        do_patching: bool = False,
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        offload: bool = True,
        render_on_step: bool = False,
        return_latents: bool = False,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")

        # Reserve progress ranges:
        # - [0.05, 0.40]: prompt / JSON generation
        # - [0.50, 0.90]: denoising loop
        prompt_progress_callback = make_mapped_progress(progress_callback, 0.05, 0.40)
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(prompt_progress_callback, 0.0, "Preparing prompt")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        lora_scale = (
            attention_kwargs.get("scale", None)
            if attention_kwargs is not None
            else None
        )

        try:
            json.loads(prompt)
            safe_emit_progress(
                prompt_progress_callback, 1.0, "Using provided structured JSON prompt"
            )
        except Exception:
            safe_emit_progress(
                prompt_progress_callback, 0.2, "Generating structured JSON prompt"
            )
            prompt = self._generate_json_prompt(
                prompt,
                structured_prompt,
                image,
                offload=offload,
                progress_callback=prompt_progress_callback,
                **generate_prompt_kwargs,
            )
            safe_emit_progress(
                prompt_progress_callback, 1.0, "Generated structured JSON prompt"
            )

        safe_emit_progress(progress_callback, 0.42, "Structured prompt ready")

        if not negative_prompt:
            negative_prompt = self.get_default_negative_prompt(json.loads(prompt))

        (
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_layers,
            negative_prompt_layers,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=self.device,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            lora_scale=lora_scale,
        )
        prompt_batch_size = prompt_embeds.shape[0]
        safe_emit_progress(progress_callback, 0.48, "Encoded prompt")

        if offload:
            self._offload("text_encoder")

        if not hasattr(self, "transformer") or not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        if guidance_scale > 1:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_layers = [
                torch.cat([negative_prompt_layers[i], prompt_layers[i]], dim=0)
                for i in range(len(prompt_layers))
            ]
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        total_num_layers_transformer = len(self.transformer.transformer_blocks) + len(
            self.transformer.single_transformer_blocks
        )
        if len(prompt_layers) >= total_num_layers_transformer:
            # remove first layers
            prompt_layers = prompt_layers[
                len(prompt_layers) - total_num_layers_transformer :
            ]
        else:
            # duplicate last layer
            prompt_layers = prompt_layers + [prompt_layers[-1]] * (
                total_num_layers_transformer - len(prompt_layers)
            )

        # 5. Prepare latent variables
        transformer_config = self.load_config_by_type("transformer")

        num_channels_latents = transformer_config.in_channels
        if do_patching:
            num_channels_latents = int(num_channels_latents / 4)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        latents, latent_image_ids = self._get_latents(
            prompt_batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
            do_patching,
        )

        latent_attention_mask = torch.ones(
            [latents.shape[0], latents.shape[1]],
            dtype=latents.dtype,
            device=latents.device,
        )
        if guidance_scale > 1:
            latent_attention_mask = latent_attention_mask.repeat(2, 1)

        attention_mask = torch.cat(
            [prompt_attention_mask, latent_attention_mask], dim=1
        )
        attention_mask = self._prepare_attention_mask(
            attention_mask
        )  # batch, seq => batch, seq, seq
        attention_mask = attention_mask.unsqueeze(dim=1).to(
            dtype=self.transformer.dtype
        )  # for head broadcasting

        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs["attention_mask"] = attention_mask

        # Adapt scheduler to dynamic shifting (resolution dependent)
        if not self.scheduler:
            self.load_component_by_type("scheduler")

        if do_patching:
            seq_len = (height // (self.vae_scale_factor * 2)) * (
                width // (self.vae_scale_factor * 2)
            )
        else:
            seq_len = (height // self.vae_scale_factor) * (
                width // self.vae_scale_factor
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        mu = self.calculate_shift(
            seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)
        safe_emit_progress(
            progress_callback, 0.50, "Timesteps computed; starting denoise"
        )

        # Support old different diffusers versions
        if len(latent_image_ids.shape) == 3:
            latent_image_ids = latent_image_ids[0]

        if len(text_ids.shape) == 3:
            text_ids = text_ids[0]

        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload
        self._do_patching = do_patching

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            total_steps = len(timesteps)
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if guidance_scale > 1 else latents
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(
                    device=latent_model_input.device, dtype=latent_model_input.dtype
                )

                # This is predicts "v" from flow-matching or eps from diffusion
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_encoder_layers=prompt_layers,
                    joint_attention_kwargs=attention_kwargs,
                    return_dict=False,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                )[0]

                # perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
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

                # Map denoising loop progress into [0.50, 0.90]
                if denoise_progress_callback is not None and total_steps > 0:
                    step_progress = float(i + 1) / float(total_steps)
                    safe_emit_progress(
                        denoise_progress_callback,
                        step_progress,
                        f"Denoising step {i + 1}/{total_steps}",
                    )

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            self._offload("transformer")
            safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            if do_patching:
                latents = self._unpack_latents(
                    latents, height, width, self.vae_scale_factor
                )
            else:
                latents = self._unpack_latents_no_patch(
                    latents, height, width, self.vae_scale_factor
                )

            latents = latents.unsqueeze(dim=2)
            images = self.vae_decode(latents, offload=offload)
            images = images.squeeze(dim=2)
            images = self._tensor_to_frame(images)
            safe_emit_progress(
                progress_callback, 1.0, "Completed text-to-image pipeline"
            )
            return images

    def _render_step(self, latents, render_on_step_callback):
        if self._do_patching:
            latents = self._unpack_latents(
                latents,
                self._preview_height,
                self._preview_width,
                self.vae_scale_factor,
            )
        else:
            latents = self._unpack_latents_no_patch(
                latents,
                self._preview_height,
                self._preview_width,
                self.vae_scale_factor,
            )
        latents = latents.unsqueeze(dim=2)
        images = self.vae_decode(latents, offload=self._preview_offload)
        images = images.squeeze(dim=2)
        images = self._tensor_to_frame(images)
        render_on_step_callback(images[0])

    def _build_messages(
        self,
        task: str,
        *,
        image: Optional[Image.Image] = None,
        refine_image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        structured_prompt: Optional[str] = None,
        editing_instructions: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = []

        if task == "inspire":
            user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": "<inspire>"})
        elif task == "generate":
            text_value = (prompt or "").strip()
            formatted = f"<generate>\n{text_value}"
            user_content.append({"type": "text", "text": formatted})
        else:  # refine
            if refine_image is None:
                base_prompt = (structured_prompt or "").strip()
                edits = (editing_instructions or "").strip()
                formatted = textwrap.dedent(f"""<refine>
    Input:
    {base_prompt}
    Editing instructions:
    {edits}""").strip()
                user_content.append({"type": "text", "text": formatted})
            else:
                user_content.append({"type": "image", "image": refine_image})
                edits = (editing_instructions or "").strip()
                formatted = textwrap.dedent(f"""<refine>
    Editing instructions:
    {edits}""").strip()
                user_content.append({"type": "text", "text": formatted})

        messages: List[Dict[str, Any]] = []
        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate_json_prompt(
        self,
        prompt: Union[str, List[str], None] = None,
        structured_prompt: str | None = None,
        image: InputImage | None = None,
        top_p: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop=["<|im_end|>", "<|end_of_text|>"],
        offload: bool = True,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ):
        if isinstance(prompt, list):
            prompt = prompt[0]

        self.logger.info(f"Loading image: {image} {prompt} {structured_prompt}")

        if image is not None:
            image = self._load_image(image)
        llm = self.helpers["prompt_gen"]
        self.to_device(llm)

        refine_image = None
        if not image and not structured_prompt:
            # only got prompt
            task = "generate"
            editing_instructions = None
        elif not image and structured_prompt and prompt:
            # got structured prompt and prompt
            task = "refine"
            editing_instructions = prompt
        elif image and not structured_prompt and prompt:
            # got image and prompt
            task = "refine"
            editing_instructions = prompt
            refine_image = image
        elif image and not structured_prompt and not prompt:
            # only got image
            task = "inspire"
            editing_instructions = None
        else:
            raise ValueError("Invalid input")

        messages = self._build_messages(
            task,
            image=image,
            prompt=prompt,
            refine_image=refine_image,
            structured_prompt=structured_prompt,
            editing_instructions=editing_instructions,
        )

        json_str = llm.generate(
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            progress_callback=progress_callback,
        )
        if offload:
            self._offload("prompt_gen")
        return json_str
