from typing import Dict, Any, List, Literal, Callable
import torch
from src.text_encoder.tokenizer import fetch_and_save_tokenizer_from_config
from src.mixins.loader_mixin import LoaderMixin
import ftfy
import re
import html
from src.utils.defaults import DEFAULT_CACHE_PATH, DEFAULT_COMPONENTS_PATH
import os
from src.mixins.to_mixin import ToMixin
from src.mixins.cache_mixin import CacheMixin
from src.utils.module import find_class_recursive
import transformers
import inspect

def nan_hook(module, inputs, outputs):
    if isinstance(outputs, torch.Tensor):
        if not torch.isfinite(outputs).all():
            raise RuntimeError(f"NaN/Inf in {module}")

class TextEncoder(torch.nn.Module, LoaderMixin, CacheMixin, ToMixin):
    def __init__(
        self,
        config: Dict[str, Any],
        no_weights: bool = True,
        enable_cache: bool = True,
        cache_file: str = None,
        max_cache_size: int = 1000,
        device: torch.device | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.base = config.get("base")
        # check if base has transformers in it if so remove it
        if "transformer" in self.base:
            self.base = self.base.replace("transformer.", "")
        self.model_path = config.get("model_path")
        self.config_path = config.get("config_path", None)
        self.tokenizer_path = config.get("tokenizer_path", None)
        self.model_config = config.get("config", {})
        self.config = config
        self.enable_cache = config.get("enable_cache", enable_cache)
        self.load_dtype = config.get("load_dtype", None)
        self.dtype = config.get("dtype", None)
        self.module_name = config.get("module_name", "transformers")
        self.cache_file = cache_file
        self.device = device
        self.max_cache_size = max_cache_size

        if self.enable_cache and self.cache_file is None:
            self.cache_file = os.path.join(
                DEFAULT_CACHE_PATH,
                f"text_encoder_{self.model_path.replace('/', '_')}.safetensors",
            )
        if self.tokenizer_path is not None:
            self.tokenizer_path = self._download(
                self.tokenizer_path, DEFAULT_COMPONENTS_PATH
            )
            tokenizer_class = find_class_recursive(
                transformers, config.get("tokenizer_class", "AutoTokenizer")
            )
            self.tokenizer = tokenizer_class.from_pretrained(self.tokenizer_path)
        else:
            self.tokenizer = fetch_and_save_tokenizer_from_config(
                self.model_path,
                self.config_path,
                self.model_config,
                tokenizer_class=self.config.get("tokenizer_class", None),
                tokenizer_name=self.config.get("tokenizer_name", None),
                **self.config.get("tokenizer_kwargs", {}),
            )

        self.model = None
        self.model_loaded = False

    def load_model(
        self, no_weights: bool = False, override_kwargs: Dict[str, Any] = None, to_device: bool = True
    ):
        input_kwargs = dict(
            component={
                "config": self.model_config,
                "config_path": self.config_path,
                "model_path": self.model_path,
                "base": self.base,
                "type": "text_encoder",
                "gguf_kwargs": self.config.get("gguf_kwargs", {}),
                "extra_model_paths": self.config.get("extra_model_paths", []),
            },
            load_dtype=self.dtype,
            module_name=self.module_name,
            no_weights=no_weights,
            key_map=self.config.get("key_map", {}),
            extra_kwargs=self.config.get("extra_kwargs", {}),
            load_device=self.device,
        )
        if override_kwargs is not None:
            input_kwargs.update(override_kwargs)
        model = self._load_model(**input_kwargs)

        if not no_weights and to_device:
            self.to_device(model, device=self.device)

        return model

    def basic_clean(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(self, text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def prompt_clean(self, text, lower_case: bool = False):
        text = self.whitespace_clean(self.basic_clean(text))
        if lower_case:
            text = text.lower()
        return text

    def get_prompt_hash(
        self,
        text: str | List[str],
        max_sequence_length: int = 512,
        pad_to_max_length: bool = True,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype | str | None = None,
        device: torch.device | None = None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = False,
        use_attention_mask: bool = False,
        use_position_ids: bool = False,
        use_token_type_ids: bool = False,
        arrange_attention_mask: bool = False,
        pad_with_zero: bool = True,
        clean_text: bool = True,
        process_inputs_func: Callable = None,
        output_type: Literal[
            "hidden_states", "pooler_output", "text_embeds", "raw"
        ] = "hidden_states",
        lower_case: bool = False,
    ) -> str:
        kwargs = {
            "text": text,
            "max_sequence_length": max_sequence_length,
            "pad_to_max_length": pad_to_max_length,
            "num_videos_per_prompt": num_videos_per_prompt,
            "dtype": dtype,
            "device": device,
            "add_special_tokens": add_special_tokens,
            "return_attention_mask": return_attention_mask,
            "use_attention_mask": use_attention_mask,
            "arrange_attention_mask": arrange_attention_mask,
            "use_position_ids": use_position_ids,
            "use_token_type_ids": use_token_type_ids,
            "pad_with_zero": pad_with_zero,
            "clean_text": clean_text,
            "output_type": output_type,
            "lower_case": lower_case,
            "model_path": self.model_path,
            "process_inputs_func": (
                inspect.signature(process_inputs_func).parameters
                if process_inputs_func is not None
                else None
            ),
        }

        return self.hash(kwargs)

    @torch.no_grad()
    def encode(
        self,
        text: str | List[str],
        max_sequence_length: int = 512,
        pad_to_max_length: bool = True,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype | str | None = None,
        device: torch.device | None = None,
        add_special_tokens: bool | None = True,
        return_attention_mask: bool = False,
        use_attention_mask: bool = False,
        use_position_ids: bool = False,
        use_token_type_ids: bool = False,
        arrange_attention_mask: bool = False,
        pad_with_zero: bool = True,
        clean_text: bool = True,
        process_inputs_func: Callable = None,
        reshape_prompt_embeds: bool = True,
        output_type: Literal[
            "hidden_states", "hidden_states_all", "pooler_output", "text_embeds", "raw"
        ] = "hidden_states",
        lower_case: bool = False,
    ):
        if isinstance(text, str):
            text = [text]
        if clean_text:
            text = [self.prompt_clean(t, lower_case=lower_case) for t in text]

        if dtype is not None:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype.lstrip("torch."))

        batch_size = len(text)

        kwargs = {
            "text": text,
            "max_sequence_length": max_sequence_length,
            "pad_to_max_length": pad_to_max_length,
            "num_videos_per_prompt": num_videos_per_prompt,
            "dtype": dtype,
            "device": device,
            "add_special_tokens": add_special_tokens,
            "return_attention_mask": return_attention_mask,
            "use_attention_mask": use_attention_mask,
            "arrange_attention_mask": arrange_attention_mask,
            "use_position_ids": use_position_ids,
            "use_token_type_ids": use_token_type_ids,
            "pad_with_zero": pad_with_zero,
            "clean_text": clean_text,
            "output_type": output_type,
            "lower_case": lower_case,
            "model_path": self.model_path,
            "process_inputs_func": (
                inspect.signature(process_inputs_func).parameters
                if process_inputs_func is not None
                else None
            ),
        }

        prompt_hash = self.hash(kwargs)

        if self.enable_cache:
            cached = self.load_cached(prompt_hash)
            if cached is not None:
                cached_embeds, cached_mask = cached

                # Move to requested dtype/device without altering mask dtype
                if dtype is not None:
                    cached_embeds = cached_embeds.to(dtype=dtype)
                if device is not None:
                    cached_embeds = cached_embeds.to(device=device)
                    if cached_mask is not None:
                        cached_mask = cached_mask.to(device=device)

                if return_attention_mask:
                    return cached_embeds, cached_mask
                else:
                    return cached_embeds

        if not self.model_loaded:
            self.model = self.load_model(no_weights=False)
            self.model_loaded = True
        

        dict_kwargs = dict(
            padding="max_length" if pad_to_max_length else "longest",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        if add_special_tokens is None:
            dict_kwargs.pop("add_special_tokens")

        text_inputs = self.tokenizer(
            text,
            **dict_kwargs,
        )
        if process_inputs_func is not None:
            text_inputs = process_inputs_func(text_inputs)

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        seq_lens = mask.gt(0).sum(dim=1).long()
        inputs = {"input_ids": text_input_ids.to(device=self.model.device)}

        if use_position_ids:
            position_ids = torch.arange(text_input_ids.shape[1]).expand(
                batch_size, text_input_ids.shape[1]
            )
            position_ids = position_ids.to(dtype=torch.long, device=self.model.device)
            inputs["position_ids"] = position_ids

        if use_token_type_ids:
            inputs["token_type_ids"] = torch.zeros_like(text_input_ids).to(
                device=self.model.device
            )

        if use_attention_mask:
            inputs["attention_mask"] = mask.to(device=self.model.device)

        if arrange_attention_mask:
            seq_lengths = mask.sum(dim=1)
            mask_indices = (
                torch.arange(mask.size(1)).unsqueeze(0).expand(batch_size, -1)
            )
            mask = (mask_indices <= seq_lengths.unsqueeze(1)).long()
            inputs["attention_mask"] = mask.to(device=self.model.device)
            
        self.model.apply(lambda m: m.register_forward_hook(nan_hook))
      
        result = self.model(
            **inputs,
            output_hidden_states=(
                output_type == "hidden_states"
                or output_type == "raw"
                or output_type == "hidden_states_all"
            ),
        )
        
        if output_type == "hidden_states_all" and hasattr(result, "hidden_states"):
            prompt_embeds = result.hidden_states
            prompt_embeds = torch.stack(prompt_embeds, dim=0)
        elif output_type == "hidden_states" and hasattr(result, "last_hidden_state"):
            prompt_embeds = result.last_hidden_state
        elif output_type == "hidden_states" and hasattr(result, "hidden_states"):
            prompt_embeds = result.hidden_states[-1]
        elif output_type == "pooler_output" and hasattr(result, "pooler_output"):
            prompt_embeds = result.pooler_output
        elif output_type == "text_embeds" and hasattr(result, "text_embeds"):
            prompt_embeds = result.text_embeds
        elif output_type == "raw":
            if return_attention_mask:
                return result, mask
            else:
                return result
        else:
            raise ValueError(f"Invalid output type: {output_type}")

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        if output_type == "pooler_output":
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
        elif pad_with_zero and output_type == "hidden_states":
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack(
                [
                    (
                        torch.cat(
                            [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
                        )
                        if pad_to_max_length
                        else u
                    )
                    for u in prompt_embeds
                ],
                dim=0,
            )
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            if reshape_prompt_embeds:
                _, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                mask = mask.repeat(1, num_videos_per_prompt)
                mask = mask.view(batch_size * num_videos_per_prompt, seq_len)

                prompt_embeds = prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        elif output_type == "text_embeds":
            if reshape_prompt_embeds:
                prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1)
                mask = mask.repeat(num_videos_per_prompt, 1)
        elif output_type == "hidden_states":
            if reshape_prompt_embeds:
                _, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)

                prompt_embeds = prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

                mask = mask.repeat(1, num_videos_per_prompt)
                mask = mask.view(batch_size * num_videos_per_prompt, seq_len)

        if self.enable_cache:
            self.cache(
                prompt_hash,
                prompt_embeds,
                mask,
            )

        if return_attention_mask:
            return prompt_embeds, mask
        else:
            return prompt_embeds
