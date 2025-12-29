from typing import Dict, Any, Callable, List, Union, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from .shared import FluxShared
from src.types import InputImage
from src.utils.progress import safe_emit_progress, make_mapped_progress
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class DreamOmni2Engine(FluxShared):
    """DreamOmni engine implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels_latents = (   
            self.transformer.config.out_channels // 4 if self.transformer else 16
        )

    def run(
        self,
        prompt: List[str] | str,
        image_list: List[InputImage],
        task: str = "generation",
        prompt_2: List[str] | str = None,
        negative_prompt: List[str] | str = None,
        negative_prompt_2: List[str] | str = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        resize_to_preferred_resolution: bool = True,
        num_images: int = 1,
        seed: int | None = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 3.5,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        text_encoder_2_kwargs: Dict[str, Any] = {},
        joint_attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        sigmas: List[float] | None = None,
        timesteps: List[int] | None = None,
        ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Image.Image | str | np.ndarray] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        _auto_resize: bool = True,
        max_area: int = 1024**2,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting DreamOmni2 pipeline")
        if task not in ["generation", "editing"]:
            raise ValueError(f"Invalid task: {task}")

        images = image_list or None

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        safe_emit_progress(progress_callback, 0.05, "Preparing prompt")

        prompt = self._prepare_prompt(prompt, images, task, offload, progress_callback)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        use_cfg_guidance = true_cfg_scale > 1.0 and negative_prompt is not None

        safe_emit_progress(progress_callback, 0.10, "Encoding prompts")
        (
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            negative_text_ids,
        ) = self.encode_prompt(
            prompt,
            negative_prompt,
            prompt_2,
            negative_prompt_2,
            use_cfg_guidance,
            offload,
            num_images,
            text_encoder_kwargs,
            text_encoder_2_kwargs,
        )

        if offload:
            self._offload("text_encoder_2")

        transformer_dtype = self.component_dtypes.get("transformer", None)

        safe_emit_progress(progress_callback, 0.125, "Loading and preprocessing image")
        for idx, image in enumerate(images):
            images[idx] = self._load_image(image)

        if images is not None and not (
            isinstance(images[0], torch.Tensor)
            and images[0].size(1) == self.num_channels_latents
        ):
            tp_images = []
            for img in images:
                image = img
                image_height, image_width = (
                    self.image_processor.get_default_height_width(img)
                )
                aspect_ratio = image_width / image_height
                if _auto_resize:
                    image, image_width, image_height = self._resize_input(image)
                image = self.image_processor.preprocess(
                    image, image_height, image_width
                )
                tp_images.append(image)
            images = tp_images

        batch_size = prompt_embeds.shape[0]

        safe_emit_progress(progress_callback, 0.15, "Preparing latents")
        latents, image_latents, latent_ids, image_ids = self._get_latents(
            images=images,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=transformer_dtype,
            device=self.device,
            generator=generator,
        )

        if image_ids is not None:
            latent_ids = torch.cat(
                [latent_ids, image_ids], dim=0
            )  # dim 0 is sequence dimension

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )

        image_seq_len = latents.shape[1]
        safe_emit_progress(progress_callback, 0.20, "Configuring scheduler")
        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        safe_emit_progress(progress_callback, 0.25, "Computing timesteps")
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            mu=mu,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        if not hasattr(self, "transformer") or not self.transformer:
            self.load_component_by_type("transformer")

        lora_paths = self.config.get("loras", [])
        safe_emit_progress(progress_callback, 0.27, "Applying LoRAs")
        if task == "generation":
            gen_lora_path = next(
                (lora for lora in lora_paths if lora.get("name") == "gen"), None
            )
            if gen_lora_path:
                self.apply_lora(gen_lora_path.get("source"))
        elif task == "editing":
            edit_lora_path = next(
                (lora for lora in lora_paths if lora.get("name") == "edit"), None
            )
            if edit_lora_path:
                self.apply_lora(edit_lora_path.get("source"))

        self.to_device(self.transformer)

        safe_emit_progress(progress_callback, 0.30, "Transformer ready")

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=self.device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

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
            safe_emit_progress(progress_callback, 0.34, "Preparing IP-Adapter embeds")
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
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                self.device,
                num_images,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)

        # Reserve a progress gap for denoising [0.50, 0.90]
        denoise_progress_callback = make_mapped_progress(progress_callback, 0.50, 0.90)
        safe_emit_progress(progress_callback, 0.50, "Starting denoise")

        # Set preview context for per-step rendering on the main engine (denoise runs there)
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            use_cfg_guidance=use_cfg_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            latent_ids=latent_ids,
            image_latents=image_latents,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_warmup_steps=num_warmup_steps,
            true_cfg_scale=true_cfg_scale,
            denoise_progress_callback=denoise_progress_callback,
            render_on_step_interval=render_on_step_interval,
        )
        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        image = self.vae_decode(latents, offload=offload)
        image = self._tensor_to_frame(image)
        safe_emit_progress(progress_callback, 1.0, "Completed DreamOmni2 pipeline")
        return image

    def _get_latents(
        self,
        images: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        offload: bool = True,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        h_offset = 0
        w_offset = 0
        image_latents = image_ids = None
        if images is not None:
            tp_image_latents = []
            tp_image_ids = []
            for i, image in enumerate(images):
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.num_channels_latents:
                    image_latents = self.vae_encode(
                        image, sample_generator=generator, offload=offload
                    )
                else:
                    image_latents = image
                if (
                    batch_size > image_latents.shape[0]
                    and batch_size % image_latents.shape[0] == 0
                ):
                    # expand init_latents for batch_size
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat(
                        [image_latents] * additional_image_per_prompt, dim=0
                    )
                elif (
                    batch_size > image_latents.shape[0]
                    and batch_size % image_latents.shape[0] != 0
                ):
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latent_height, image_latent_width = image_latents.shape[2:]
                image_latents = self._pack_latents(
                    image_latents,
                    batch_size,
                    num_channels_latents,
                    image_latent_height,
                    image_latent_width,
                )
                image_ids = self._prepare_latent_image_ids(
                    batch_size,
                    image_latent_height // 2,
                    image_latent_width // 2,
                    device,
                    dtype,
                )
                image_ids[..., 0] = i + 1
                image_ids[..., 2] += w_offset
                tp_image_latents.append(image_latents)
                tp_image_ids.append(image_ids)
                h_offset += image_latent_height // 2
                w_offset += image_latent_width // 2
            image_latents = torch.cat(tp_image_latents, dim=1)
            image_ids = torch.cat(tp_image_ids, dim=0)

        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, height, width
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids

    def _prepare_prompt(
        self,
        prompt: List[str] | str,
        images: List[InputImage],
        task: str,
        offload: bool = True,
        progress_callback: Callable = None,
    ):
        safe_emit_progress(progress_callback, 0.06, "Loading LLM")
        llm: Qwen2_5_VLForConditionalGeneration | None = self.helpers.get("llm")
        processor: Qwen2_5_VLProcessor | None = self.helpers.get("llm_processor")

        if not llm or not processor:
            return prompt

        tp = []
        prefix = " It is editing task." if task == "editing" else ""
        input_instruction = prompt + prefix
        for image in images:
            tp.append({"type": "image", "image": image})
        tp.append({"type": "text", "text": input_instruction + prefix})
        messages = [{"role": "user", "content": tp}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = self._process_images(images)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        safe_emit_progress(progress_callback, 0.07, "Generating prompt from LLM")
        generated_ids = llm.generate(**inputs, do_sample=False, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if offload:
            safe_emit_progress(progress_callback, 0.08, "Offloading LLM")
            self._offload("llm")

        return output_text[0]

    def _process_images(
        self, images: List[InputImage], patch_size: int = 14
    ) -> List[Image.Image]:
        output_images = []
        for image in images:
            image = self._load_image(image)
            image, image_width, image_height = self._resize_input(image)
            output_images.append(image)
        return output_images

    def _resize_input(self, image: Image.Image, multiple_of: int = 16) -> Image.Image:
        image_height, image_width = image.height, image.width
        aspect_ratio = image_width / image_height
        _, image_width, image_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of
        img = image.resize((image_width, image_height), Image.LANCZOS)
        return img, image_width, image_height
