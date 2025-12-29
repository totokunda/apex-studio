import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, List, Optional, Dict, Any, Callable, Tuple, Literal
from PIL import Image
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from src.engine.base_engine import BaseEngine
from src.utils.progress import safe_emit_progress
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor


class Flux2Shared(BaseEngine):

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = Flux2ImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.diffusers_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        # fmt: off
        self.system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."
        # fmt: on

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_mistral_3_small_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
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

    def _encode_vae_image(
        self, image: torch.Tensor, generator: torch.Generator, offload: bool = True, offload_type: Literal["cpu", "discard"] = "discard"
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = self.vae_encode(
            image,
            sample_generator=generator,
            sample_mode="mode",
            offload=offload,
            normalize_latents=False,
            offload_type=offload_type,
        )
        image_latents = self._patchify_latents(image_latents)
        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        image_latents = self.vae.normalize_latents(image_latents)
        if offload:
            self._offload("vae")
        return image_latents

    @staticmethod
    def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    @staticmethod
    def format_text_input(prompts: List[str], system_message: str = None):
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    def _get_mistral_3_small_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        # fmt: off
        system_message: str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.",
        # fmt: on
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = self.component_dtypes["text_encoder"]
        device = self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = self.format_text_input(
            prompts=prompt, system_message=system_message
        )

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        tokenizer = self.text_encoder.tokenizer

        # Process all messages at once
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # check if inputs in cache

        if self.text_encoder.enable_cache:
            hash = self.text_encoder.hash(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_sequence_length": max_sequence_length,
                    "system_message": system_message,
                    "hidden_states_layers": hidden_states_layers,
                }
            )
            cached = self.text_encoder.load_cached(hash)
            if cached is not None:
                return cached[0].to(device=device, dtype=dtype)
        else:
            hash = None

        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model(no_weights=False)

        self.to_device(self.text_encoder, device=device)

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        self.text_encoder.model.language_model.embed_tokens.to(device)
        # Forward pass through the model
        output = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack(
            [output.hidden_states[k] for k in hidden_states_layers], dim=1
        )
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )

        if self.text_encoder.enable_cache and hash is not None:
            self.text_encoder.cache(hash, prompt_embeds)

        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(
        latents: torch.Tensor,  # (B, C, H, W)
    ):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _prepare_image_ids(
        image_latents: List[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ):
        r"""
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        This function creates a unique coordinate for every pixel/patch across all input latent with different
        dimensions.

        Args:
            image_latents (List[torch.Tensor]):
                A list of image latent feature tensors, typically of shape (C, H, W).
            scale (int, optional):
                A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
                latent is: 'scale + scale * i'. Defaults to 10.

        Returns:
            torch.Tensor:
                The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
                input latents.

        Coordinate Components (Dimension 4):
            - T (Time): The unique index indicating which latent image the coordinate belongs to.
            - H (Height): The row index within that latent image.
            - W (Width): The column index within that latent image.
            - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
        """

        if not isinstance(image_latents, list):
            raise ValueError(
                f"Expected `image_latents` to be a list, got {type(image_latents)}."
            )

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(
                t, torch.arange(height), torch.arange(width), torch.arange(1)
            )
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(
            batch_size, num_channels_latents * 4, height // 2, width // 2
        )
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(
            batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
        )
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(
            batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
        )
        return latents

    @staticmethod
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(
            0, 2, 1
        )

        return latents

    @staticmethod
    def _unpack_latents_with_ids(
        x: torch.Tensor, x_ids: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def prepare_image_latents(
        self,
        images: List[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
        offload: bool = True,
    ):
        image_latents = []
        for idx, image in enumerate(images):
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self._encode_vae_image(
                image=image, generator=generator, offload=offload, offload_type="cpu" if idx != len(images) - 1 else "discard"
            )
            image_latents.append(imagge_latent)  # (1, 128, 32, 32)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, 32, 32)
            packed = self._pack_latents(latent)  # (1, 1024, 128)
            packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids
