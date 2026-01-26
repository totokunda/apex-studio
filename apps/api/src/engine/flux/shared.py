import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any, Callable
from PIL import Image
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from src.engine.base_engine import BaseEngine
from src.utils.progress import safe_emit_progress
from loguru import logger
import os


class FluxShared(TextualInversionLoaderMixin, BaseEngine):
    """Shared functionality for Flux engine implementations"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )

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

    def _get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        offload=True,
        timestep=None,
    ):

        if image is not None:
            image_latents = self.vae_encode(image, offload=offload)
            image_latent_height, image_latent_width = image_latents.shape[2:]
            if timestep is None:
                image_latents = self._pack_latents(
                    image_latents,
                    batch_size,
                    num_channels_latents,
                    image_latent_height,
                    image_latent_width,
                )
                latent_image_ids = self._prepare_latent_image_ids(
                    batch_size,
                    image_latent_height // 2,
                    image_latent_width // 2,
                    device,
                    dtype,
                )
                # image ids are the same as latent ids with the first dimension set to 1 instead of 0
                latent_image_ids[..., 0] = 1
            else:
                latent_image_ids = None
        else:
            image_latents = None
            latent_image_ids = None

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_ids

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if timestep is not None:
            assert (
                image_latents is not None
            ), "Image latents are required for timestep scaling"
            image_latents = torch.cat([image_latents] * batch_size, dim=0)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            latents = noise

        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        if image_latents is not None and timestep is None:
            return latents, image_latents, latent_ids, latent_image_ids

        return latents, latent_ids

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

    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        use_cfg_guidance: bool = True,
        offload: bool = True,
        num_images: int = 1,
        text_encoder_kwargs: Optional[Dict[str, Any]] = {},
        text_encoder_2_kwargs: Optional[Dict[str, Any]] = {},
        progress_callback: Callable | None = None,
    ):
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            safe_emit_progress(progress_callback, 0.10, "Loading text encoder")
            self.load_component_by_name("text_encoder")
            safe_emit_progress(progress_callback, 0.20, "Text encoder loaded")

        safe_emit_progress(progress_callback, 0.25, "Moving text encoder to device")
        self.to_device(self.text_encoder)
        safe_emit_progress(progress_callback, 0.30, "Text encoder on device")

        if isinstance(prompt, str):
            prompt = [prompt]
        prompt = self.maybe_convert_prompt(prompt, self.text_encoder.tokenizer)
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            negative_prompt = self.maybe_convert_prompt(
                negative_prompt, self.text_encoder.tokenizer
            )

        safe_emit_progress(progress_callback, 0.40, "Encoding pooled prompt embeddings")
        pooled_prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="pooler_output",
            **text_encoder_kwargs,
        )
        safe_emit_progress(progress_callback, 0.55, "Pooled prompt embeddings ready")

        if negative_prompt is not None and use_cfg_guidance:
            safe_emit_progress(
                progress_callback, 0.60, "Encoding pooled negative prompt embeddings"
            )
            negative_pooled_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="pooler_output",
                **text_encoder_kwargs,
            )
            safe_emit_progress(
                progress_callback, 0.70, "Pooled negative prompt embeddings ready"
            )
        else:
            negative_pooled_prompt_embeds = None

        if offload:
            safe_emit_progress(progress_callback, 0.72, "Offloading text encoder")
            del self.text_encoder

        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            safe_emit_progress(progress_callback, 0.75, "Loading text encoder 2")
            self.load_component_by_name("text_encoder_2")
            safe_emit_progress(progress_callback, 0.80, "Text encoder 2 loaded")

        safe_emit_progress(progress_callback, 0.82, "Moving text encoder 2 to device")
        self.to_device(self.text_encoder_2)
        safe_emit_progress(progress_callback, 0.85, "Text encoder 2 on device")

        if not prompt_2:
            prompt_2 = prompt

        if not negative_prompt_2:
            negative_prompt_2 = negative_prompt

        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]
        prompt_2 = self.maybe_convert_prompt(prompt_2, self.text_encoder_2.tokenizer)
        if negative_prompt_2 is not None:
            if isinstance(negative_prompt_2, str):
                negative_prompt_2 = [negative_prompt_2]
            negative_prompt_2 = self.maybe_convert_prompt(
                negative_prompt_2, self.text_encoder_2.tokenizer
            )

        safe_emit_progress(progress_callback, 0.90, "Encoding prompt embeddings")
        prompt_embeds = self.text_encoder_2.encode(
            prompt_2,
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="hidden_states",
            **text_encoder_2_kwargs,
        )
        safe_emit_progress(progress_callback, 0.95, "Prompt embeddings ready")

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=self.device, dtype=prompt_embeds.dtype
        )

        if negative_prompt_2 is not None and use_cfg_guidance:
            safe_emit_progress(
                progress_callback, 0.96, "Encoding negative prompt embeddings"
            )
            negative_prompt_embeds = self.text_encoder_2.encode(
                negative_prompt_2,
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="hidden_states",
                **text_encoder_2_kwargs,
            )
            safe_emit_progress(
                progress_callback, 0.99, "Negative prompt embeddings ready"
            )
            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(
                device=self.device, dtype=negative_prompt_embeds.dtype
            )
        else:
            negative_prompt_embeds = None
            negative_text_ids = None

        safe_emit_progress(progress_callback, 1.0, "Prompt encoding complete")
        return (
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            negative_text_ids,
        )

    def resize_to_preferred_resolution(self, image: Image.Image):
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

        original_width, original_height = image.size
        original_aspect = original_width / original_height

        best_resolution = None
        min_area_diff = float("inf")

        for width, height in PREFERRED_KONTEXT_RESOLUTIONS:
            target_aspect = width / height
            area_diff = abs((width * height) - (original_width * original_height))
            aspect_diff = abs(target_aspect - original_aspect)

            if area_diff < min_area_diff and aspect_diff < 0.2:
                min_area_diff = area_diff
                best_resolution = (width, height)

        if best_resolution is None:
            best_resolution = min(
                PREFERRED_KONTEXT_RESOLUTIONS,
                key=lambda res: abs(
                    (res[0] * res[1]) - (original_width * original_height)
                ),
            )

        return image.resize(best_resolution, Image.Resampling.LANCZOS)

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        offload=False,
    ):
        # 1. calculate the height and width of the latents
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # 2. encode the masked image
        if masked_image.shape[1] == num_channels_latents:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self.vae_encode(masked_image, offload=offload)

        # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        # 4. pack the masked_image_latents
        # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask = mask[
            :, 0, :, :
        ]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width

        # 6. pack the mask:
        # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
        mask = self._pack_latents(
            mask,
            batch_size,
            self.vae_scale_factor * self.vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        """Override: unpack latents for image decoding and render a preview frame.

        Falls back to base implementation if preview dimensions are unavailable.
        """
        if os.environ.get("ENABLE_IMAGE_RENDER_STEP", "true") == "false":
            return
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
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = kwargs.get("negative_pooled_prompt_embeds")
        true_cfg_scale = kwargs.get("true_cfg_scale")
        latent_ids = kwargs.get("latent_ids")
        text_ids = kwargs.get("text_ids")
        negative_text_ids = kwargs.get("negative_text_ids")
        image_embeds = kwargs.get("image_embeds", None)
        negative_image_embeds = kwargs.get("negative_image_embeds", None)
        num_warmup_steps = kwargs.get("num_warmup_steps")
        use_cfg_guidance = kwargs.get("use_cfg_guidance")
        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        render_on_step = kwargs.get("render_on_step")
        render_on_step_callback = kwargs.get("render_on_step_callback")
        image_latents = kwargs.get("image_latents")
        concat_latents = kwargs.get("concat_latents")
        denoise_progress_callback = kwargs.get("denoise_progress_callback")
        render_on_step_interval = kwargs.get("render_on_step_interval", 3)

        safe_emit_progress(denoise_progress_callback, 0.0, "Starting denoise")
        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if image_embeds is not None:
                    joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                elif concat_latents is not None:
                    latent_model_input = torch.cat([latents, concat_latents], dim=2)
                else:
                    latent_model_input = latents

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if image_latents is not None:
                        noise_pred = noise_pred[:, : latents.size(1)]

                if use_cfg_guidance:
                    if negative_image_embeds is not None:
                        joint_attention_kwargs["ip_adapter_image_embeds"] = (
                            negative_image_embeds
                        )

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        if image_latents is not None:
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                    noise_pred = neg_noise_pred + true_cfg_scale * (
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

                # external progress callback
                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(len(timesteps)) if len(timesteps) > 0 else 1.0,
                    f"Denoise {i + 1}/{len(timesteps)}",
                )

        safe_emit_progress(denoise_progress_callback, 1.0, "Denoise finished")
        return latents
