from PIL import Image
from typing import Union, List
import numpy as np
import torch
from transformers import AutoProcessor
from src.utils.defaults import DEFAULT_COMPONENTS_PATH, DEFAULT_CONFIG_SAVE_PATH
from src.utils.module import find_class_recursive
import importlib
from typing import Dict, Any
from transformers.image_processing_utils import ImageProcessingMixin
from src.text_encoder.tokenizer import fetch_and_save_tokenizer_from_config
from src.helpers.helpers import helpers
from src.mixins.cache_mixin import CacheMixin
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
import torch.nn as nn


def _expand_input_ids_with_image_tokens(
    text_input_ids,
    prompt_attention_mask,
    max_sequence_length,
    image_token_index,
    image_emb_len,
    image_emb_start,
    image_emb_end,
    pad_token_id,
):
    special_image_token_mask = text_input_ids == image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    batch_indices, non_image_indices = torch.where(text_input_ids != image_token_index)

    max_expanded_length = max_sequence_length + (
        num_special_image_tokens.max() * (image_emb_len - 1)
    )
    new_token_positions = (
        torch.cumsum((special_image_token_mask * (image_emb_len - 1) + 1), -1) - 1
    )
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    expanded_input_ids = torch.full(
        (text_input_ids.shape[0], max_expanded_length),
        pad_token_id,
        dtype=text_input_ids.dtype,
        device=text_input_ids.device,
    )
    expanded_input_ids[batch_indices, text_to_overwrite] = text_input_ids[
        batch_indices, non_image_indices
    ]
    expanded_input_ids[batch_indices, image_emb_start:image_emb_end] = image_token_index

    expanded_attention_mask = torch.zeros(
        (text_input_ids.shape[0], max_expanded_length),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    attn_batch_indices, attention_indices = torch.where(
        expanded_input_ids != pad_token_id
    )
    expanded_attention_mask[attn_batch_indices, attention_indices] = 1.0
    expanded_attention_mask = expanded_attention_mask.to(prompt_attention_mask.dtype)
    position_ids = (expanded_attention_mask.cumsum(-1) - 1).masked_fill_(
        (expanded_attention_mask == 0), 1
    )

    return {
        "input_ids": expanded_input_ids,
        "attention_mask": expanded_attention_mask,
        "position_ids": position_ids,
    }


@helpers("hunyuanvideo.llama")
class HunyuanLlama(nn.Module, LoaderMixin, OffloadMixin, CacheMixin):
    def __init__(
        self,
        model_path: str,
        image_processor_path: str | None = None,
        config_path: str | None = None,
        config: Dict[str, Any] | None = None,
        save_path: str = DEFAULT_COMPONENTS_PATH,
        config_save_path: str = DEFAULT_CONFIG_SAVE_PATH,
        enable_cache: bool = True,
        cache_file: str = None,
        max_cache_size: int = 100,
        tokenizer_name: str | None = None,
        tokenizer_class: str | None = None,
        tokenizer_kwargs: Dict[str, Any] | None = None,
        base_model: str = "LlamaModel",
        **kwargs,
    ):
        super().__init__()
        self.enable_cache = enable_cache
        self.cache_file = cache_file
        self.max_cache_size = max_cache_size

        self.model_path = self._download(model_path, save_path)
        # Default prompt template for HunyuanVideo
        self.default_prompt_template_text = {
            "template": (
                "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
                "1. The main content and theme of the video."
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
                "4. background environment, light, style and atmosphere."
                "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
            ),
            "crop_start": 95,
        }

        self.default_prompt_template_image = {
            "template": (
                "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
                "1. The main content and theme of the video."
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
                "4. background environment, light, style and atmosphere."
                "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "crop_start": 103,
            "image_emb_start": 5,
            "image_emb_end": 581,
            "image_emb_len": 576,
            "double_return_token_id": 271,
        }

        self.model = self._load_model(
            {
                "type": "hunyuanvideo.llama",
                "base": base_model,
                "model_path": self.model_path,
                "config_path": config_path,
                "config": config,
            },
            module_name="transformers",
        )

        if hasattr(self.model, "vision_tower"):
            self.model.vision_tower.to(self.model.vision_tower.config.torch_dtype)

        self.config_save_path = config_save_path

        self.processor_class = "CLIPImageProcessor"

        if image_processor_path is not None:
            self.image_processor = self.load_processor(image_processor_path)
        else:
            self.image_processor = None

        config_path = self._download(config_path, self.config_save_path)

        self.tokenizer = fetch_and_save_tokenizer_from_config(
            self.model_path,
            config_path=config_path,
            config=config,
            tokenizer_name=tokenizer_name,
            tokenizer_class=tokenizer_class,
            **(tokenizer_kwargs if tokenizer_kwargs is not None else {}),
        )

    def load_processor(self, processor_path: Dict[str, Any] | str) -> AutoProcessor:
        try:
            processor_class = find_class_recursive(
                importlib.import_module("transformers"), self.processor_class
            )
            if self._is_huggingface_repo(processor_path):
                if len(processor_path.split("/")) > 2:
                    subfolder = "/".join(processor_path.split("/")[2:])
                    processor_path = "/".join(processor_path.split("/")[:2])
                    return processor_class.from_pretrained(
                        processor_path,
                        subfolder=subfolder,
                        save_dir=self.config_save_path,
                    )
                else:
                    return processor_class.from_pretrained(
                        processor_path, save_dir=self.config_save_path
                    )
            else:
                return processor_class.from_pretrained(processor_path)
        except Exception as e:
            processor_config = self.fetch_config(processor_path)
            processor_class = find_class_recursive(
                importlib.import_module("transformers"),
                processor_config[self.find_key_with_type(processor_config)],
            )
            if not issubclass(processor_class, ImageProcessingMixin):
                processor_class = find_class_recursive(
                    importlib.import_module("transformers"), self.processor_class
                )
            return processor_class(**processor_config)

    def __str__(self):
        return (
            f"LlamaPreprocessor(model={self.model}, preprocessor={self.preprocessor})"
        )

    def __repr__(self):
        return self.__str__()

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        max_sequence_length: int = 256,
        image_embed_interleave: int = 2,
        num_hidden_layers_to_skip: int = 2,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        num_videos_per_prompt: int = 1,
        hyavatar: bool = False,
        **kwargs,
    ):

        input_kwargs = {
            "prompt": prompt,
            "image": image,
            "max_sequence_length": max_sequence_length,
            "image_embed_interleave": image_embed_interleave,
            "num_hidden_layers_to_skip": num_hidden_layers_to_skip,
            "device": device,
            "dtype": dtype,
            "num_videos_per_prompt": num_videos_per_prompt,
            "hyavatar": hyavatar,
        }

        prompt_hash = self.hash(input_kwargs)

        if self.enable_cache:
            cached = self.load_cached(prompt_hash)
            if cached is not None:
                return cached

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_template = (
            self.default_prompt_template_text
            if image is None
            else self.default_prompt_template_image
        )
        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)

        image_emb_len = prompt_template.get("image_emb_len", 576)
        image_emb_start = prompt_template.get("image_emb_start", 5)
        image_emb_end = prompt_template.get("image_emb_end", 581)
        double_return_token_id = prompt_template.get("double_return_token_id", 271)

        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|start_header_id|>, <|end_header_id|>, assistant, <|eot_id|>, and placeholder {}
            crop_start -= 5

        max_sequence_length += crop_start

        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        text_attention_mask = text_inputs.attention_mask.to(device=device)

        if self.image_processor is not None:
            loaded_image = self._load_image(image)
            image_embeds = self.image_processor(
                loaded_image, return_tensors="pt"
            ).pixel_values.to(device)
        else:
            image_embeds = None

        if image_embeds is not None:
            image_token_index = self.model.config.image_token_index
            pad_token_id = self.model.config.pad_token_id
            expanded_inputs = _expand_input_ids_with_image_tokens(
                text_input_ids,
                text_attention_mask,
                max_sequence_length,
                image_token_index,
                image_emb_len,
                image_emb_start,
                image_emb_end,
                pad_token_id,
            )
            expanded_inputs["pixel_values"] = image_embeds.to(
                self.model.vision_tower.config.torch_dtype
            )
            expanded_inputs = {
                k: v.to(self.model.device) for k, v in expanded_inputs.items()
            }
        else:
            expanded_inputs = {
                "input_ids": text_input_ids.to(self.model.device),
                "attention_mask": text_attention_mask.to(self.model.device),
            }

        prompt_attention_mask = expanded_inputs["attention_mask"]

        prompt_embeds = self.model(
            **expanded_inputs,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if (
            crop_start is not None
            and crop_start > 0
            and image_embeds is not None
            and not hyavatar
        ):
            text_crop_start = crop_start - 1 + image_emb_len
            batch_indices, last_double_return_token_indices = torch.where(
                text_input_ids == double_return_token_id
            )

            if last_double_return_token_indices.shape[0] == 3:
                # in case the prompt is too long
                last_double_return_token_indices = torch.cat(
                    (
                        last_double_return_token_indices,
                        torch.tensor([text_input_ids.shape[-1]]),
                    )
                )
                batch_indices = torch.cat((batch_indices, torch.tensor([0])))

            last_double_return_token_indices = last_double_return_token_indices.reshape(
                text_input_ids.shape[0], -1
            )[:, -1]
            batch_indices = batch_indices.reshape(text_input_ids.shape[0], -1)[:, -1]
            assistant_crop_start = (
                last_double_return_token_indices - 1 + image_emb_len - 4
            )

            assistant_crop_end = last_double_return_token_indices - 1 + image_emb_len
            attention_mask_assistant_crop_start = last_double_return_token_indices - 4
            attention_mask_assistant_crop_end = last_double_return_token_indices

            prompt_embed_list = []
            prompt_attention_mask_list = []
            image_embed_list = []
            image_attention_mask_list = []

            for i in range(text_input_ids.shape[0]):
                prompt_embed_list.append(
                    torch.cat(
                        [
                            prompt_embeds[
                                i, text_crop_start : assistant_crop_start[i].item()
                            ],
                            prompt_embeds[i, assistant_crop_end[i].item() :],
                        ]
                    ).to(device)
                )

                prompt_attention_mask_list.append(
                    torch.cat(
                        [
                            prompt_attention_mask[
                                i,
                                crop_start : attention_mask_assistant_crop_start[
                                    i
                                ].item(),
                            ],
                            prompt_attention_mask[
                                i, attention_mask_assistant_crop_end[i].item() :
                            ],
                        ]
                    ).to(device)
                )

                image_embed_list.append(
                    prompt_embeds[i, image_emb_start:image_emb_end].to(device)
                )
                image_attention_mask_list.append(
                    torch.ones(image_embed_list[-1].shape[0])
                    .to(prompt_attention_mask.dtype)
                    .to(device)
                )

            prompt_embed_list = torch.stack(prompt_embed_list)
            prompt_attention_mask_list = torch.stack(prompt_attention_mask_list)
            image_embed_list = torch.stack(image_embed_list)
            image_attention_mask_list = torch.stack(image_attention_mask_list)

            if 0 < image_embed_interleave < 6:
                image_embed_list = image_embed_list[:, ::image_embed_interleave, :]
                image_attention_mask_list = image_attention_mask_list[
                    :, ::image_embed_interleave
                ]

            assert (
                prompt_embed_list.shape[0] == prompt_attention_mask_list.shape[0]
                and image_embed_list.shape[0] == image_attention_mask_list.shape[0]
            )

            prompt_embeds = torch.cat([image_embed_list, prompt_embed_list], dim=1)

            prompt_attention_mask = torch.cat(
                [image_attention_mask_list, prompt_attention_mask_list], dim=1
            )
        else:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs * num_videos_per_prompt, seq_len, -1)

            bs, seq_len = prompt_attention_mask.shape
            prompt_attention_mask = prompt_attention_mask.repeat(
                1, num_videos_per_prompt
            )
            prompt_attention_mask = prompt_attention_mask.view(
                bs * num_videos_per_prompt, seq_len
            )

        if self.enable_cache:
            self.cache(prompt_hash, prompt_embeds, prompt_attention_mask)

        return prompt_embeds, prompt_attention_mask
