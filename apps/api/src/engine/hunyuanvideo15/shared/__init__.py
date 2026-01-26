import numpy as np
from PIL import Image
import torch
from src.engine.base_engine import BaseEngine
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.hunyuan_video1_5.image_processor import (
    HunyuanVideo15ImageProcessor,
)
from PIL import Image
import re
from diffusers.utils.torch_utils import randn_tensor


def get_gpu_memory(device=None):
    if not torch.cuda.is_available():
        return 0
    device = device if device is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    if hasattr(torch.cuda, "get_per_process_memory_fraction"):
        memory_fraction = torch.cuda.get_per_process_memory_fraction()
    else:
        memory_fraction = 1.0
    return props.total_memory * memory_fraction


class HunyuanVideo15Shared(BaseEngine):
    """HunyuanVideo15 Shared Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        )
        self.video_processor = HunyuanVideo15ImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            do_resize=False,
            do_convert_rgb=True,
        )
        self.target_size = (
            self.transformer.config.target_size
            if getattr(self, "transformer", None)
            else 640
        )
        self.vision_states_dim = (
            self.transformer.config.image_embed_dim
            if getattr(self, "transformer", None)
            else 1152
        )
        self.num_channels_latents = (
            self.vae.config.latent_channels if getattr(self, "vae", None) else 32
        )
        # fmt: off
        self.system_message = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
        # fmt: on
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729

    def _extract_glyph_texts(self, prompt: str) -> str:
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string

        Returns:
            List of extracted glyph texts
        """
        pattern = r"\"(.*?)\"|“(.*?)”"
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result

        if result:
            formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
        else:
            formatted_result = None

        return formatted_result

    def _format_text_input(
        self, prompt: List[str], system_message: str
    ) -> List[Dict[str, Any]]:
        """
        Apply text to template.

        Args:
            prompt (List[str]): Input text.
            system_message (str): System message.

        Returns:
            List[Dict[str, Any]]: List of chat conversation.
        """

        template = [
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": p if p else " "},
            ]
            for p in prompt
        ]

        return template

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _get_mllm_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        tokenizer_max_length: int = 1000,
        num_hidden_layers_to_skip: int = 2,
        # fmt: off
        system_message: str = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.",
        # fmt: on
        crop_start: int = 108,
        offload: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt = self._format_text_input(prompt, system_message)
        text_encoder_dtype = self.component_dtypes.get("text_encoder")

        if not self.text_encoder:
            self.load_component_by_name("text_encoder")

        tokenizer = self.text_encoder.tokenizer

        # check if inputs in cache
        hashable_inputs = {
            "prompt": prompt,
            "tokenizer_max_length": tokenizer_max_length,
            "crop_start": crop_start,
            "system_message": system_message,
            "num_hidden_layers_to_skip": num_hidden_layers_to_skip,
        }

        hash = self.text_encoder.hash(hashable_inputs)
        if self.text_encoder.enable_cache:
            cached = self.text_encoder.load_cached(hash)
            if cached is not None:
                return cached[0].to(device=device, dtype=text_encoder_dtype), cached[
                    1
                ].to(device=device)

        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model()

        text_inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=tokenizer_max_length + crop_start,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder.model(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        prompt_embeds = prompt_embeds.to(device=device, dtype=text_encoder_dtype)
        prompt_attention_mask = prompt_attention_mask.to(
            device=device, dtype=torch.int64
        )

        if self.text_encoder.enable_cache:
            self.text_encoder.cache(hash, prompt_embeds, prompt_attention_mask)

        self._offload("text_encoder")

        return prompt_embeds, prompt_attention_mask

    def _get_byt5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        tokenizer_max_length: int = 256,
        offload: bool = True,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        glyph_texts = [self._extract_glyph_texts(p) for p in prompt]

        prompt_embeds_list = []
        prompt_embeds_mask_list = []
        text_encoder_config = self.load_config_by_name("text_encoder_2")
        text_encoder_dtype = self.component_dtypes.get("text_encoder")

        for glyph_text in glyph_texts:
            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, tokenizer_max_length, text_encoder_config["d_model"]),
                    device=device,
                    dtype=text_encoder_dtype,
                )
                glyph_text_embeds_mask = torch.zeros(
                    (1, tokenizer_max_length), device=device, dtype=torch.int64
                )
            else:

                if not self.text_encoder_2:
                    self.load_component_by_name("text_encoder_2")

                glyph_text_embeds, glyph_text_embeds_mask = self.text_encoder_2.encode(
                    glyph_text,
                    max_sequence_length=tokenizer_max_length,
                    pad_to_max_length=True,
                    use_attention_mask=True,
                    return_attention_mask=True,
                    output_type="hidden_states",
                    clean_text=False,
                    pad_with_zero=False,
                )

                glyph_text_embeds = glyph_text_embeds.to(
                    device=device, dtype=text_encoder_dtype
                )
                glyph_text_embeds_mask = glyph_text_embeds_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)

        self._offload("text_encoder_2")

        return prompt_embeds, prompt_embeds_mask

    def _get_image_latents(
        self,
        image: Image.Image,
        height: int,
        width: int,
        device: torch.device,
        offload: bool = True,
    ) -> torch.Tensor:

        image_tensor = self.video_processor.preprocess(
            image, height=height, width=width
        )
        image_tensor = image_tensor.unsqueeze(2)
        image_latents = self.vae_encode(image_tensor, offload=False, sample_mode="mode")
        return image_latents

    def _get_image_embeds(
        self,
        image: Image.Image,
        device: torch.device,
        offload: bool = True,
    ) -> torch.Tensor:

        image_encoder = self.helpers["image_encoder"]
        feature_extractor = self.helpers["feature_extractor"]
        image_encoder_dtype = next(image_encoder.parameters()).dtype
        image = feature_extractor.preprocess(
            images=image, do_resize=True, return_tensors="pt", do_convert_rgb=True
        )
        image = image.to(device=device, dtype=image_encoder_dtype)
        image_enc_hidden_states = image_encoder(**image).last_hidden_state

        if offload:
            del image_encoder
            del feature_extractor
        self._offload("image_encoder")
        self._offload("feature_extractor")

        return image_enc_hidden_states

    def encode_image(
        self,
        image: Image.Image,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        offload: bool = True,
    ) -> torch.Tensor:

        image_embeds = self._get_image_embeds(
            image=image,
            device=device,
            offload=offload,
        )
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(device=device, dtype=dtype)
        return image_embeds

    # Copied from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5.HunyuanVideo15Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        offload: bool = True,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            batch_size (`int`):
                batch size of prompts, defaults to 1
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input
                argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
        """
        device = device or self.device
        dtype = dtype or self.component_dtypes.get("text_encoder")

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_max_length,
                system_message=self.system_message,
                crop_start=self.prompt_template_encode_start_idx,
                offload=offload,
            )

        if prompt_embeds_2 is None:
            prompt_embeds_2, prompt_embeds_mask_2 = self._get_byt5_prompt_embeds(
                prompt=prompt,
                device=device,
                tokenizer_max_length=self.tokenizer_2_max_length,
                offload=offload,
            )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(
            batch_size * num_videos_per_prompt, seq_len
        )

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(
            batch_size * num_videos_per_prompt, seq_len_2, -1
        )
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(
            batch_size * num_videos_per_prompt, seq_len_2
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2
