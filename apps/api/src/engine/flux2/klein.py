from src.engine.base_engine import BaseEngine
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from src.types import InputImage
from typing import Union, List, Dict, Any, Callable
import torch
from typing import Optional, Tuple
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from src.engine.flux2.shared import Flux2Shared
from src.utils.progress import safe_emit_progress, make_mapped_progress


class Flux2KleinEngine(Flux2Shared):
    """Flux2 Klein Text-to-Image Image-to-Image Engine Implementation"""

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

    def _get_qwen3_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: List[int] = (9, 18, 27),
    ):
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        texts = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.text_encoder.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        hidden_states = self.text_encoder.encode(
            texts,
            max_sequence_length=max_sequence_length,
            clean_text=False,
            use_attention_mask=True,
            pad_with_zero=False,
            add_special_tokens=None,
            output_type="hidden_states_all",
        )
        hidden_states = hidden_states.unbind(dim=0)

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
    ):
        device = device or self.device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def run(
        self,
        image: InputImage | List[InputImage] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        attention_kwargs: Dict[str, Any] = {},
        render_on_step: bool = False,
        render_on_step_callback: Optional[Callable] = None,
        render_on_step_interval: int = 3,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
        offload: bool = True,
        seed: int = None,
        progress_callback: Callable = None,
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting ti2i pipeline")

        if seed is not None:
            safe_emit_progress(progress_callback, 0.01, "Seeding generator")
            generator = torch.Generator(device=self.device).manual_seed(seed)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        safe_emit_progress(progress_callback, 0.02, "Initialized run state")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        safe_emit_progress(progress_callback, 0.03, "Preparing prompts")

        # 3. prepare text embeddings
        safe_emit_progress(progress_callback, 0.05, "Encoding prompts")
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        safe_emit_progress(progress_callback, 0.20, "Encoded prompts")

        if offload:
            safe_emit_progress(progress_callback, 0.22, "Offloading text encoder")
            self._offload("text_encoder")
        safe_emit_progress(progress_callback, 0.25, "Text encoder offloaded")

        # 4. process images
        safe_emit_progress(progress_callback, 0.26, "Loading and preprocessing images")
        if image is not None and not isinstance(image, list):
            image = [image]

        if image is not None:
            image = [self._load_image(img) for img in image]

        condition_images = None
        if image is not None:
            safe_emit_progress(progress_callback, 0.28, "Validating images")
            for img in image:
                self.image_processor.check_image_input(img)

            safe_emit_progress(
                progress_callback, 0.30, "Resizing and preprocessing images"
            )
            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(
                    img, height=image_height, width=image_width, resize_mode="crop"
                )
                condition_images.append(img)
                height = height or image_height
                width = width or image_width
            safe_emit_progress(progress_callback, 0.33, "Prepared condition images")
        else:
            safe_emit_progress(
                progress_callback, 0.33, "No input image provided; running pure t2i"
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        safe_emit_progress(progress_callback, 0.34, "Resolved target dimensions")

        # 5. prepare latent variables
        safe_emit_progress(progress_callback, 0.35, "Preparing latent variables")
        transformer_config = self.load_config_by_type("transformer")
        num_channels_latents = transformer_config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        safe_emit_progress(progress_callback, 0.38, "Initialized latent noise")

        # Store preview context for step-wise rendering
        self._preview_offload = offload
        self._preview_latent_ids = latent_ids

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            safe_emit_progress(progress_callback, 0.40, "Preparing image latents")
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.component_dtypes["vae"],
            )
            safe_emit_progress(progress_callback, 0.42, "Prepared image latents")

        if not self.scheduler:
            safe_emit_progress(progress_callback, 0.425, "Loading scheduler")
            self.load_component_by_type("scheduler")
            safe_emit_progress(progress_callback, 0.43, "Scheduler loaded")
        safe_emit_progress(progress_callback, 0.435, "Moving scheduler to device")
        self.to_device(self.scheduler)

        if not self.transformer:
            safe_emit_progress(progress_callback, 0.44, "Loading transformer")
            self.load_component_by_type("transformer")
            safe_emit_progress(progress_callback, 0.445, "Transformer loaded")
        safe_emit_progress(progress_callback, 0.448, "Moving transformer to device")
        self.to_device(self.transformer)

        safe_emit_progress(progress_callback, 0.45, "Scheduler prepared")

        # 6. Prepare timesteps
        safe_emit_progress(progress_callback, 0.46, "Preparing sigmas and timesteps")
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
        mu = self.compute_empirical_mu(
            image_seq_len=image_seq_len, num_steps=num_inference_steps
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

        safe_emit_progress(
            progress_callback, 0.50, "Timesteps computed; starting denoise"
        )

        # handle guidance
        safe_emit_progress(progress_callback, 0.49, "Preparing guidance embeddings")
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)

        safe_emit_progress(progress_callback, 0.50, "Starting denoising")
        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(
                        self.transformer.dtype
                    )
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,  # (B, image_seq_len, C)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=latent_image_ids,  # B, image_seq_len, 4
                    joint_attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred[:, : latents.size(1) :]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                    total_steps = len(timesteps)
                    step_progress = (
                        min(float(i + 1) / float(total_steps), 1.0)
                        if total_steps > 0
                        else 0.0
                    )
                    if denoise_progress_callback is not None and total_steps > 0:
                        safe_emit_progress(
                            denoise_progress_callback,
                            step_progress,
                            f"Denoising step {i + 1}/{total_steps}",
                        )

                    if (
                        render_on_step
                        and render_on_step_callback
                        and total_steps > 0
                        and (
                            (i + 1) % max(int(render_on_step_interval), 1) == 0
                            or i == 0
                        )
                        and i != len(timesteps) - 1
                    ):
                        safe_emit_progress(
                            denoise_progress_callback,
                            step_progress,
                            "Rendering preview",
                        )
                        self._render_step(latents, render_on_step_callback)

        self._current_timestep = None

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            safe_emit_progress(progress_callback, 0.94, "Offloading transformer")
            self._offload("transformer")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            image = latents
        else:
            safe_emit_progress(progress_callback, 0.95, "Decoding latents")
            latents = self._unpack_latents_with_ids(latents, latent_ids)
            if not self.vae:
                safe_emit_progress(progress_callback, 0.955, "Loading VAE")
                self.load_component_by_type("vae")
                safe_emit_progress(progress_callback, 0.96, "VAE loaded")
            safe_emit_progress(progress_callback, 0.965, "Moving VAE to device")
            self.to_device(self.vae)

            latents = self.vae.denormalize_latents(latents)
            latents = self._unpatchify_latents(latents)

            image = self.vae_decode(latents, offload=offload, denormalize_latents=False)
            image = self._tensor_to_frame(image)
            safe_emit_progress(progress_callback, 1.0, "Completed ti2i pipeline")

        return image

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        """Render a preview frame from packed Flux2 latents during denoising (best-effort)."""
        import os

        if os.environ.get("ENABLE_IMAGE_RENDER_STEP", "true") == "false":
            return

        latent_ids = getattr(self, "_preview_latent_ids", None)
        if latent_ids is None:
            return

        try:
            unpacked = self._unpack_latents_with_ids(latents, latent_ids)

            if not self.vae:
                self.load_component_by_type("vae")
            self.to_device(self.vae)

            unpacked = self.vae.denormalize_latents(unpacked)
            unpacked = self._unpatchify_latents(unpacked)

            tensor_image = self.vae_decode(
                unpacked,
                offload=getattr(self, "_preview_offload", True),
                denormalize_latents=False,
            )
            image = self._tensor_to_frame(tensor_image)
            render_on_step_callback(image[0])
        except Exception:
            # Never break sampling due to preview rendering.
            return
