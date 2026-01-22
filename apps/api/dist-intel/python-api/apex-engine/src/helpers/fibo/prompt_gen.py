from src.helpers.helpers import helpers
import torch
import json
import math
from typing import Dict, Any, Iterable, List, Optional, Callable
from PIL import Image
import ujson
from boltons.iterutils import remap
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

try:
    # HF helper for detecting FlashAttention-2 availability (if present in this transformers version).
    from transformers.utils import is_flash_attn_2_available  # type: ignore
except Exception:  # pragma: no cover - best-effort optional import

    def is_flash_attn_2_available() -> bool:
        return False


from src.helpers.base import BaseHelper
from src.utils.defaults import get_components_path, DEFAULT_CACHE_PATH, get_torch_device
from src.mixins.cache_mixin import CacheMixin
from src.utils.progress import safe_emit_progress, make_mapped_progress
import os


def clean_json(caption):
    caption["pickascore"] = 1.0
    caption["aesthetic_score"] = 10.0
    caption = prepare_clean_caption(caption)
    return caption


def parse_aesthetic_score(record: dict) -> str:
    ae = record["aesthetic_score"]
    if ae < 5.5:
        return "very low"
    elif ae < 6:
        return "low"
    elif ae < 7:
        return "medium"
    elif ae < 7.6:
        return "high"
    else:
        return "very high"


def parse_pickascore(record: dict) -> str:
    ps = record["pickascore"]
    if ps < 0.78:
        return "very low"
    elif ps < 0.82:
        return "low"
    elif ps < 0.87:
        return "medium"
    elif ps < 0.91:
        return "high"
    else:
        return "very high"


def prepare_clean_caption(record: dict) -> str:
    def keep(p, k, v):
        is_none = v is None
        is_empty_string = isinstance(v, str) and v == ""
        is_empty_dict = isinstance(v, dict) and not v
        is_empty_list = isinstance(v, list) and not v
        is_nan = isinstance(v, float) and math.isnan(v)
        if is_none or is_empty_string or is_empty_list or is_empty_dict or is_nan:
            return False
        return True

    try:
        scores = {}
        if "pickascore" in record:
            scores["preference_score"] = parse_pickascore(record)
        if "aesthetic_score" in record:
            scores["aesthetic_score"] = parse_aesthetic_score(record)

        # Create structured caption dict of original values
        fields = [
            "short_description",
            "objects",
            "background_setting",
            "lighting",
            "aesthetics",
            "photographic_characteristics",
            "style_medium",
            "text_render",
            "context",
            "artistic_style",
        ]

        original_caption_dict = {f: record[f] for f in fields if f in record}

        # filter empty values recursivly (i.e. None, "", {}, [], float("nan"))
        clean_caption_dict = remap(original_caption_dict, visit=keep)

        # Set aesthetics scores
        if "aesthetics" not in clean_caption_dict:
            if len(scores) > 0:
                clean_caption_dict["aesthetics"] = scores
        else:
            clean_caption_dict["aesthetics"].update(scores)

        # Dumps clean structured caption as minimal json string (i.e. no newlines\whitespaces seps)
        clean_caption_str = ujson.dumps(
            clean_caption_dict, escape_forward_slashes=False
        )
        return clean_caption_str
    except Exception as ex:
        print("Error: ", ex)
        raise ex


def _collect_images(messages: Iterable[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            image_value = item.get("image")
            if isinstance(image_value, Image.Image):
                images.append(image_value)
            else:
                raise ValueError("Expected PIL.Image for image content in messages.")
    return images


def _strip_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not stop_sequences:
        return text.strip()
    cleaned = text
    for stop in stop_sequences:
        if not stop:
            continue
        index = cleaned.find(stop)
        if index >= 0:
            cleaned = cleaned[:index]
    return cleaned.strip()


class PromptGenHelper(BaseHelper, CacheMixin):
    """Inference wrapper using Hugging Face transformers."""

    def __init__(
        self,
        model_path: str,
        save_path: str = get_components_path(),
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        enable_cache: bool = True,
        cache_file: str = None,
        max_cache_size: int = 100,
        **kwargs,
    ) -> None:
        default_processor_kwargs: Dict[str, Any] = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1024 * 28 * 28,
        }
        processor_kwargs = {**default_processor_kwargs, **(processor_kwargs or {})}
        model_kwargs = model_kwargs or {}
        self.enable_cache = enable_cache
        self.cache_file = cache_file
        self.max_cache_size = max_cache_size

        # Ensure this helper has an explicit torch device (GPU when available)
        # so that it runs on GPU even when instantiated outside a BaseEngine.
        self.device = get_torch_device()

        # Prefer FlashAttention-2 when available for faster attention; fall back silently otherwise.
        if "attn_implementation" not in model_kwargs:
            try:
                use_flash_attn = is_flash_attn_2_available()
            except Exception:
                use_flash_attn = False
            if use_flash_attn and getattr(self.device, "type", None) == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"

        if self.enable_cache and self.cache_file is None:
            self.cache_file = os.path.join(
                DEFAULT_CACHE_PATH,
                f"prompt_gen_{model_path.replace('/', '_')}.safetensors",
            )

        super(PromptGenHelper, self).__init__()
        model_path = self._download(model_path, save_path)
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()
        self.to_device(self.model, device=self.device)

        tokenizer_obj = self.processor.tokenizer
        if tokenizer_obj.pad_token_id is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
        self._pad_token_id = tokenizer_obj.pad_token_id
        eos_token_id = tokenizer_obj.eos_token_id
        if isinstance(eos_token_id, list) and eos_token_id:
            self._eos_token_id = eos_token_id
        elif eos_token_id is not None:
            self._eos_token_id = [eos_token_id]
        else:
            raise ValueError("Tokenizer must define an EOS token for generation.")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:

        return self._generate_with_transformers(
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            progress_callback=progress_callback,
        )

    def _generate_with_transformers(
        self,
        messages: List[Dict[str, Any]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        # Map this helper's internal progress to the caller's overall progress range if provided.
        local_progress = (
            make_mapped_progress(progress_callback, 0.0, 1.0)
            if progress_callback
            else None
        )

        safe_emit_progress(
            local_progress, 0.0, "Starting prompt generation with Qwen3-VL"
        )

        prompt_hash = self.hash(
            {
                k: v
                for k, v in locals().items()
                if k not in {"progress_callback", "local_progress"}
            }
        )
        cached = self.load_cached(prompt_hash)
        if cached is not None:
            cached_prompt = cached[0]
            decoded = self.processor.tokenizer.batch_decode(
                cached_prompt, skip_special_tokens=True
            )
            if decoded:
                text = decoded[0]
                stripped_text = _strip_stop_sequences(text, stop)
                json_prompt = json.loads(stripped_text)
                safe_emit_progress(
                    local_progress, 1.0, "Loaded structured prompt from cache"
                )
                return prepare_clean_caption(json_prompt)

        tokenizer = self.processor.tokenizer
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_inputs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        images = _collect_images(messages)
        if images:
            processor_inputs["images"] = images

        safe_emit_progress(local_progress, 0.2, "Prepared inputs for prompt generation")

        inputs = self.processor(**processor_inputs)
        device = self.model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
            "use_cache": True,
        }

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError(
                "Processor did not return input_ids; cannot compute new tokens."
            )

        # Use non-streaming generation to avoid per-token printing overhead,
        # which can dominate latency even when the model itself is fast.
        # try to compile the model
        safe_emit_progress(local_progress, 0.4, "Running prompt generation model")
        try:
            compiled_model = torch.compile(self.model)
        except Exception as e:
            print(f"Error compiling model: {e}")
            compiled_model = self.model
        with torch.inference_mode():
            generated_ids = compiled_model.generate(**inputs, **generation_kwargs)

        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        if self.enable_cache:
            self.cache(prompt_hash, new_token_ids)

        decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        if not decoded:
            safe_emit_progress(
                local_progress, 1.0, "Prompt generation produced no output"
            )
            return ""
        text = decoded[0]
        stripped_text = _strip_stop_sequences(text, stop)
        json_prompt = json.loads(stripped_text)
        result = prepare_clean_caption(json_prompt)
        safe_emit_progress(local_progress, 1.0, "Completed prompt generation")
        return result
