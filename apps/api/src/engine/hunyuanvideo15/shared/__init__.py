import numpy as np
from PIL import Image
import torch
from src.engine.base_engine import BaseEngine
import json
from src.text_encoder.text_encoder import TextEncoder
from loguru import logger
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


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

    # ----------------------------
    # Prompt templating (moved from helpers/hunyuanvideo15/text.py)
    # ----------------------------
    PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Describe the image by detailing the following aspects: "
            "1. The main content and theme of the image. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. The background environment, light, style and atmosphere.",
        },
        {"role": "user", "content": "{}"},
    ]

    PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video.",
        },
        {"role": "user", "content": "{}"},
    ]

    PROMPT_TEMPLATES: Dict[str, Dict[str, Any]] = {
        "li-dit-encode-image-json": {"template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON},
        "li-dit-encode-video-json": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON},
    }

    @staticmethod
    def _apply_text_to_template(
        text: str, template: Union[str, List[Dict[str, Any]]], prevent_empty_text: bool = True
    ):
        if isinstance(template, str):
            return template.format(text)
        if isinstance(template, list):
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    item["content"] = item["content"].format(
                        text if text else (" " if prevent_empty_text else "")
                    )
            return template_copy
        raise TypeError(f"Unsupported template type: {type(template)}")

    # ----------------------------
    # Tokenization + crop_start (ported from helpers/hunyuanvideo15/text.py)
    # ----------------------------

    @dataclass
    class HunyuanTextEncoderOutputs:
        hidden_state: torch.Tensor
        attention_mask: Optional[torch.Tensor] = None
        hidden_states_list: Optional[Tuple[torch.Tensor, ...]] = None

    def _get_prompt_template_key(self, text_encoder: TextEncoder, data_type: str) -> Optional[str]:
        cfg = getattr(text_encoder, "config", {}) or {}
        if data_type == "image":
            return cfg.get("prompt_template")
        if data_type == "video":
            return cfg.get("prompt_template_video")
        raise ValueError(f"Unsupported data type: {data_type}")

    def _get_crop_start(self, text_encoder: TextEncoder, data_type: str) -> int:
        if not hasattr(self, "_hunyuan_crop_start"):
            self._hunyuan_crop_start = {}
        te_key = id(text_encoder)
        per_te = self._hunyuan_crop_start.setdefault(te_key, {"image": -1, "video": -1})
        return int(per_te.get(data_type, -1))

    def _set_crop_start(self, text_encoder: TextEncoder, data_type: str, crop_start: int) -> None:
        if not hasattr(self, "_hunyuan_crop_start"):
            self._hunyuan_crop_start = {}
        te_key = id(text_encoder)
        per_te = self._hunyuan_crop_start.setdefault(te_key, {"image": -1, "video": -1})
        per_te[data_type] = int(crop_start)

    def calculate_crop_start(self, text_encoder: TextEncoder, tokenized_input) -> int:
        """
        Automatically calculate crop_start by locating the user marker tokens.
        Ported from `src/helpers/hunyuanvideo15/text.py`.
        """
        input_ids = tokenized_input["input_ids"][0].tolist()
        marker = "<|im_start|>user\n"
        tok = getattr(text_encoder, "tokenizer", None)
        if tok is None:
            return 0

        marker_tokens = tok(marker, add_special_tokens=False)["input_ids"]
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i : i + len(marker_tokens)] == marker_tokens:
                return i + len(marker_tokens)
        return 0

    def text2tokens(
        self,
        text_encoder: TextEncoder,
        text: Union[str, List[str]],
        data_type: str = "image",
        max_length: int = 300,
    ):
        """
        Tokenize input text exactly like `src/helpers/hunyuanvideo15/text.py::TextEncoder.text2tokens`,
        including 2-pass crop_start auto-detection and chat-template tokenization.
        """
        tok = getattr(text_encoder, "tokenizer", None)
        if tok is None:
            raise ValueError("Text encoder tokenizer not available.")

        # Original helper forces right-padding
        try:
            tok.padding_side = "right"
        except Exception:
            pass

        tpl_key = self._get_prompt_template_key(text_encoder, data_type=data_type)
        tpl_cfg = self.PROMPT_TEMPLATES.get(tpl_key) if tpl_key else None

        tokenize_input_type = "str"
        crop_start = 0

        if tpl_cfg is not None and tpl_cfg.get("template") is not None:
            prompt_template = tpl_cfg["template"]
            crop_start = self._get_crop_start(text_encoder, data_type=data_type)

            if isinstance(text, (list, tuple)):
                text = [self._apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self._apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

            # First pass: tokenize with temp max_length to compute crop_start
            if crop_start == -1:
                temp_kwargs = dict(
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                    return_tensors="pt",
                )
                if tokenize_input_type == "str":
                    temp_tokenized = tok(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **temp_kwargs,
                    )
                elif tokenize_input_type == "list":
                    temp_tokenized = tok.apply_chat_template(
                        text,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **temp_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

                crop_start = self.calculate_crop_start(text_encoder, temp_tokenized)
                self._set_crop_start(text_encoder, data_type=data_type, crop_start=crop_start)

        # Second pass: tokenize with final length (+crop_start)
        kwargs = dict(
            truncation=True,
            max_length=max_length + (crop_start if crop_start > 0 else 0),
            padding="max_length",
            return_tensors="pt",
        )

        if tokenize_input_type == "str":
            tokenized_output = tok(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            tokenized_output = tok.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

        return tokenized_output

    @staticmethod
    def _use_default(value, default):
        return value if value is not None else default

    def _resolve_final_layer_norm(self, model: Any):
        # Match original helper intent: apply the model's final norm if available.
        if model is None:
            return None
        if hasattr(model, "text_model") and hasattr(model.text_model, "final_layer_norm"):
            return model.text_model.final_layer_norm
        if hasattr(model, "final_layer_norm"):
            return model.final_layer_norm
        if hasattr(model, "norm"):
            return model.norm
        return None

    @torch.no_grad()
    def encode_tokenized(
        self,
        text_encoder: TextEncoder,
        batch_encoding,
        *,
        use_attention_mask: bool = True,
        output_hidden_states: bool = False,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        data_type: str = "image",
        device: Optional[torch.device] = None,
    ) -> "HunyuanTextEncoderOutputs":
        """
        Encode tokenized inputs like `src/helpers/hunyuanvideo15/text.py::TextEncoder.encode`,
        including crop_start slicing and optional skip-layer selection.
        """
        device = self.device if device is None else device

        # Ensure model is loaded
        if not getattr(text_encoder, "model_loaded", False):
            text_encoder.model = text_encoder.load_model(no_weights=False)
            text_encoder.model_loaded = True

        model = getattr(text_encoder, "model", None)
        if model is None:
            raise ValueError("Text encoder model is not loaded.")

        # Match helper behavior: use the language_model if present
        lm = getattr(model, "language_model", None) or model

        hidden_state_skip_layer = self._use_default(
            hidden_state_skip_layer, (getattr(text_encoder, "config", {}) or {}).get("hidden_state_skip_layer", None)
        )
        apply_final_norm = self._use_default(
            apply_final_norm, bool((getattr(text_encoder, "config", {}) or {}).get("apply_final_norm", False))
        )

        attention_mask = batch_encoding["attention_mask"].to(device) if use_attention_mask else None

        # Helper-style caching hash (includes token tensors)
        hash_kwargs = dict(
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            hidden_state_skip_layer=hidden_state_skip_layer,
            apply_final_norm=apply_final_norm,
            data_type=data_type,
            input_ids=batch_encoding["input_ids"],
            attention_mask=batch_encoding.get("attention_mask", None),
        )
        prompt_hash = text_encoder.hash(hash_kwargs)

        if getattr(text_encoder, "enable_cache", False):
            cached = text_encoder.load_cached(prompt_hash)
            if cached is not None:
                if output_hidden_states:
                    last_hidden_state, attn, hidden_states = cached
                    return self.HunyuanTextEncoderOutputs(
                        hidden_state=last_hidden_state,
                        attention_mask=attn,
                        hidden_states_list=tuple(hidden_states),
                    )
                last_hidden_state, attn = cached
                return self.HunyuanTextEncoderOutputs(hidden_state=last_hidden_state, attention_mask=attn)

        outputs = lm(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=attention_mask,
            output_hidden_states=(output_hidden_states or hidden_state_skip_layer is not None),
        )

        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            if hidden_state_skip_layer > 0 and apply_final_norm:
                final_norm = self._resolve_final_layer_norm(lm)
                if callable(final_norm):
                    last_hidden_state = final_norm(last_hidden_state)
        else:
            last_hidden_state = outputs["last_hidden_state"]

        # Crop instruction tokens
        crop_start = self._get_crop_start(text_encoder, data_type=data_type)
        if crop_start > 0:
            last_hidden_state = last_hidden_state[:, crop_start:]
            if use_attention_mask and attention_mask is not None:
                attention_mask = attention_mask[:, crop_start:]

        if getattr(text_encoder, "enable_cache", False):
            if output_hidden_states:
                text_encoder.cache(prompt_hash, last_hidden_state, attention_mask, torch.stack(outputs.hidden_states))
            else:
                text_encoder.cache(prompt_hash, last_hidden_state, attention_mask)

        if output_hidden_states:
            return self.HunyuanTextEncoderOutputs(
                hidden_state=last_hidden_state,
                attention_mask=attention_mask,
                hidden_states_list=outputs.hidden_states,
            )

        return self.HunyuanTextEncoderOutputs(hidden_state=last_hidden_state, attention_mask=attention_mask)

    def _format_prompt_for_text_encoder(
        self,
        text_encoder: TextEncoder,
        prompt: str,
        data_type: str = "image",
    ) -> str:
        # Choose configured template names if present; else fall back to defaults.
        tpl_name = None
        if data_type == "image":
            tpl_name = self.config.get("prompt_template", "li-dit-encode-image-json")
        elif data_type == "video":
            tpl_name = self.config.get("prompt_template_video", "li-dit-encode-video-json")
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        tpl_cfg = self.PROMPT_TEMPLATES.get(tpl_name)
        if not tpl_cfg:
            # Unknown template: just return raw prompt
            return prompt

        template = tpl_cfg.get("template")
        applied = self._apply_text_to_template(prompt, template)

        # If tokenizer supports chat templating, turn the conversation into a single string.
        tok = getattr(text_encoder, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template") and isinstance(applied, list):
            try:
                return tok.apply_chat_template(
                    applied,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                # Fall back to naive join if chat templating fails.
                pass

        # Fallback: stringify conversation-ish list into a plain prompt.
        if isinstance(applied, list):
            parts = []
            for m in applied:
                if isinstance(m, dict):
                    role = m.get("role", "")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}".strip())
            return "\n".join([p for p in parts if p])

        return str(applied)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: str = "image",
        max_sequence_length: int = 256,
        text_encoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Shared prompt encoding matching the original hunyuanvideo15 helper behavior,
        but without using the helper class/module.
        """
        text_encoder_kwargs = text_encoder_kwargs or {}

        if text_encoder is None:
            if getattr(self, "text_encoder", None) is None:
                self.load_component_by_type("text_encoder")
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt_list = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            prompt_list = prompt
        else:
            batch_size = prompt_embeds.shape[0]
            prompt_list = None

        # Pick dtype like original `ti2v.encode_prompt`
        prompt_embeds_dtype = None
        if text_encoder is not None and getattr(text_encoder, "model", None) is not None:
            prompt_embeds_dtype = getattr(text_encoder.model, "dtype", None) or self.component_dtypes.get(
                "text_encoder", None
            )
        elif getattr(self, "transformer", None) is not None:
            prompt_embeds_dtype = getattr(self.transformer, "dtype", None) or self.component_dtypes.get(
                "transformer", None
            )

        if prompt_embeds is None and prompt_list is not None:
            text_inputs = self.text2tokens(
                text_encoder, prompt_list, data_type=data_type, max_length=max_sequence_length
            )
            if clip_skip is None:
                prompt_outputs = self.encode_tokenized(
                    text_encoder,
                    text_inputs,
                    use_attention_mask=True,
                    output_hidden_states=False,
                    data_type=data_type,
                    device=device,
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = self.encode_tokenized(
                    text_encoder,
                    text_inputs,
                    use_attention_mask=True,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                final_norm = self._resolve_final_layer_norm(
                    getattr(getattr(text_encoder, "model", None), "language_model", None)
                    or getattr(text_encoder, "model", None)
                )
                if callable(final_norm):
                    prompt_embeds = final_norm(prompt_embeds)

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt` has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.text2tokens(
                text_encoder, uncond_tokens, data_type=data_type, max_length=max_sequence_length
            )
            negative_prompt_outputs = self.encode_tokenized(
                text_encoder,
                uncond_input,
                use_attention_mask=True,
                output_hidden_states=(clip_skip is not None),
                data_type=data_type,
                device=device,
            )
            if clip_skip is None:
                negative_prompt_embeds = negative_prompt_outputs.hidden_state
            else:
                negative_prompt_embeds = negative_prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                final_norm = self._resolve_final_layer_norm(
                    getattr(getattr(text_encoder, "model", None), "language_model", None)
                    or getattr(text_encoder, "model", None)
                )
                if callable(final_norm):
                    negative_prompt_embeds = final_norm(negative_prompt_embeds)

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        # Match original `ti2v.encode_prompt` behavior: repeat negative embeds for each generation.
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    @staticmethod
    def is_sparse_attn_supported():
        return "nvidia h" in torch.cuda.get_device_properties(0).name.lower()

    def _rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg

    def _resize_and_center_crop(self, image, target_width, target_height):
        if target_height == image.shape[0] and target_width == image.shape[1]:
            return image

        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(
            target_width / original_width, target_height / original_height
        )
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    def _get_closest_ratio(
        self, height: float, width: float, ratios: list, buckets: list
    ):
        """
        Get the closest ratio in the buckets.

        Args:
            height (float): video height
            width (float): video width
            ratios (list): video aspect ratio
            buckets (list): buckets generated by `generate_crop_size_list`

        Returns:
            the closest size in the buckets and the corresponding ratio
        """
        aspect_ratio = float(height) / float(width)

        ratios_array = np.array(ratios)
        closest_ratio_id = np.abs(ratios_array - aspect_ratio).argmin()
        closest_size = buckets[closest_ratio_id]
        closest_ratio = ratios_array[closest_ratio_id]

        return closest_size, closest_ratio

    def _generate_crop_size_list(sef, base_size=256, patch_size=16, max_ratio=4.0):
        num_patches = round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.0
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list

    def _merge_tensor_by_mask(self, tensor_1, tensor_2, mask, dim):
        assert tensor_1.shape == tensor_2.shape
        # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
        masked_indices = torch.nonzero(mask).squeeze(1)
        tmp = tensor_1.clone()
        if dim == 0:
            tmp[masked_indices] = tensor_2[masked_indices]
        elif dim == 1:
            tmp[:, masked_indices] = tensor_2[:, masked_indices]
        elif dim == 2:
            tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
        return tmp

    def _add_special_token(
        self,
        text_encoder: TextEncoder,
        add_color,
        add_font,
        color_ann_path,
        font_ann_path,
        multilingual=False,
    ):
        """
        Add special tokens for color and font to tokenizer and text encoder.

        Args:
            tokenizer: Huggingface tokenizer.
            text_encoder: Huggingface T5 encoder.
            add_color (bool): Whether to add color tokens.
            add_font (bool): Whether to add font tokens.
            color_ann_path (str): Path to color annotation JSON.
            font_ann_path (str): Path to font annotation JSON.
            multilingual (bool): Whether to use multilingual font tokens.
        """
        with open(font_ann_path, "r") as f:
            idx_font_dict = json.load(f)
        with open(color_ann_path, "r") as f:
            idx_color_dict = json.load(f)

        if multilingual:
            font_token = [
                f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>"
                for font_code in idx_font_dict
            ]
        else:
            font_token = [f"<font-{i}>" for i in range(len(idx_font_dict))]
        color_token = [f"<color-{i}>" for i in range(len(idx_color_dict))]
        additional_special_tokens = []
        if add_color:
            additional_special_tokens += color_token
        if add_font:
            additional_special_tokens += font_token

        tokenizer = text_encoder.tokenizer

        tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
        if not text_encoder.model_loaded:
            text_encoder.model = text_encoder.load_model()
        # Set mean_resizing=False to avoid PyTorch LAPACK dependency

        if hasattr(text_encoder.model, "resize_token_embeddings"):
            text_encoder.model.resize_token_embeddings(
                len(tokenizer), mean_resizing=False
            )
        else:
            logger.warning(
                "Text encoder model does not support resizing token embeddings."
            )

    @staticmethod
    def get_vae_inference_config(memory_limitation=None):
        if memory_limitation is None:
            memory_limitation = get_gpu_memory()
        GB = 1024 * 1024 * 1024
        if memory_limitation < 23 * GB:
            sample_size = 160
            tile_overlap_factor = 0.2
            dtype = torch.float16
        else:
            sample_size = 256
            tile_overlap_factor = 0.25
            dtype = torch.float32
        return {
            "sample_size": sample_size,
            "tile_overlap_factor": tile_overlap_factor,
            "dtype": dtype,
        }
