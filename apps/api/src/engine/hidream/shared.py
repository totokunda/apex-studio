import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import math
from src.engine.base_engine import BaseEngine  # noqa: F401
from diffusers.image_processor import VaeImageProcessor


class HidreamShared(BaseEngine):
    """Shared functionality for Hidream engine implementations"""

    def __init__(self, yaml_path: str, **kwargs):

        super().__init__(yaml_path, **kwargs)

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.default_sample_size = 128
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.num_channels_latents = (
            self.transformer.config.in_channels // 4 if self.transformer else 16
        )

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
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

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
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        negative_prompt,
        negative_prompt_2,
        negative_prompt_3,
        negative_prompt_4,
        text_encoder_kwargs,
        text_encoder_2_kwargs,
        text_encoder_3_kwargs,
        num_images,
        use_cfg_guidance,
        offload,
    ):
        if not hasattr(self, "text_encoder") or not self.text_encoder:
            self.load_component_by_name("text_encoder")

        self.to_device(self.text_encoder)

        pooled_prompt_embeds_1 = self.text_encoder.encode(
            f"<|startoftext|>{prompt}",
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="text_embeds",
            **text_encoder_kwargs,
        )

        if use_cfg_guidance and negative_prompt is None:
            negative_prompt = ""

        if negative_prompt is not None and use_cfg_guidance:
            negative_pooled_prompt_embeds_1 = self.text_encoder.encode(
                f"<|startoftext|>{negative_prompt}",
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="text_embeds",
                **text_encoder_kwargs,
            )
        else:
            negative_pooled_prompt_embeds_1 = None

        if offload:
            del self.text_encoder

        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            self.load_component_by_name("text_encoder_2")

        self.to_device(self.text_encoder_2)

        if not prompt_2:
            prompt_2 = prompt

        if not negative_prompt_2:
            negative_prompt_2 = negative_prompt

        pooled_prompt_embeds_2 = self.text_encoder_2.encode(
            f"<|startoftext|>{prompt_2}<|endoftext|>",
            device=self.device,
            num_videos_per_prompt=num_images,
            output_type="text_embeds",
            **text_encoder_2_kwargs,
        )

        if negative_prompt_2 is not None and use_cfg_guidance:
            negative_pooled_prompt_embeds_2 = self.text_encoder_2.encode(
                f"<|startoftext|>{negative_prompt_2}<|endoftext|>",
                device=self.device,
                num_videos_per_prompt=num_images,
                output_type="text_embeds",
                **text_encoder_2_kwargs,
            )
        else:
            negative_pooled_prompt_embeds_2 = None

        if offload:
            del self.text_encoder_2

        if not hasattr(self, "text_encoder_3") or not self.text_encoder_3:
            self.load_component_by_name("text_encoder_3")

        self.to_device(self.text_encoder_3)

        if not prompt_3:
            prompt_3 = prompt

        if not negative_prompt_3:
            negative_prompt_3 = negative_prompt

        prompt_embeds = self.text_encoder_3.encode(
            prompt_3,
            device=self.device,
            num_videos_per_prompt=num_images,
            **text_encoder_3_kwargs,
        )

        if negative_prompt_3 is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder_3.encode(
                negative_prompt_3,
                device=self.device,
                num_videos_per_prompt=num_images,
                **text_encoder_3_kwargs,
            )
        else:
            negative_prompt_embeds = None

        if offload:
            del self.text_encoder_3
            
        pooled_prompt_embeds = torch.cat(
            [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
        ).view(num_images, -1)

        if use_cfg_guidance:
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2],
                dim=-1,
            ).view(num_images, -1)

        if not prompt_4:
            prompt_4 = prompt

        if not negative_prompt_4:
            negative_prompt_4 = negative_prompt

        llama_encoder = self.helpers["llama"]
        self.to_device(llama_encoder)

        llama_prompt_embeds = llama_encoder(
            prompt_4,
            device=self.device,
            dtype=prompt_embeds.dtype,
            num_images_per_prompt=num_images,
        )

        if negative_prompt_4 is not None and use_cfg_guidance:
            llama_negative_prompt_embeds = llama_encoder(
                negative_prompt_4,
                device=self.device,
                dtype=prompt_embeds.dtype,
                num_images_per_prompt=num_images,
            )
        else:
            llama_negative_prompt_embeds = None

        if offload:
            del llama_encoder

        return (
            prompt_embeds,
            negative_prompt_embeds,
            llama_prompt_embeds,
            llama_negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def resize_image(
        self, pil_image: Image.Image, image_size: int = 1024
    ) -> Image.Image:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        m = 16
        width, height = pil_image.width, pil_image.height
        S_max = image_size * image_size
        scale = S_max / (width * height)
        scale = math.sqrt(scale)

        new_sizes = [
            (round(width * scale) // m * m, round(height * scale) // m * m),
            (round(width * scale) // m * m, math.floor(height * scale) // m * m),
            (math.floor(width * scale) // m * m, round(height * scale) // m * m),
            (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
        ]
        new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

        for new_size in new_sizes:
            if new_size[0] * new_size[1] <= S_max:
                break

        s1 = width / new_size[0]
        s2 = height / new_size[1]
        if s1 < s2:
            pil_image = pil_image.resize(
                [new_size[0], round(height / s1)], resample=Image.BICUBIC
            )
            top = (round(height / s1) - new_size[1]) // 2
            pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
        else:
            pil_image = pil_image.resize(
                [round(width / s2), new_size[1]], resample=Image.BICUBIC
            )
            left = (round(width / s2) - new_size[0]) // 2
            pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

        return pil_image

    def _render_step(self, latents, render_on_step_callback):
        """Decode latents and render a preview image during denoising."""
        try:
            preview_height = getattr(self, "_preview_height", None)
            preview_width = getattr(self, "_preview_width", None)
            if preview_height is None or preview_width is None:
                return super()._render_step(latents, render_on_step_callback)
            tensor_image = self.vae_decode(
                latents, offload=getattr(self, "_preview_offload", True)
            )
            image = self._tensor_to_frame(tensor_image)
            render_on_step_callback(image[0])
        except Exception:
            return super()._render_step(latents, render_on_step_callback)
