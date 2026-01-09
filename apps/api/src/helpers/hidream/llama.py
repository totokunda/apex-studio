from src.helpers.helpers import helpers
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
from src.mixins.cache_mixin import CacheMixin
from src.mixins.to_mixin import ToMixin
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
from src.text_encoder.tokenizer import fetch_and_save_tokenizer_from_config
import os
import torch
import torch.nn as nn
from typing import List

@helpers("hidream.llama")
class HidreamLlama(CacheMixin, LoaderMixin, OffloadMixin, ToMixin, nn.Module):
    def __init__(
        self,
        model_path: str = "unsloth/Llama-3.1-8B-Instruct",
        model_class: str = "LlamaForCausalLM",
        tokenizer_name: str = "unsloth/Llama-3.1-8B-Instruct",
        tokenizer_class: str = "PreTrainedTokenizerFast",
        save_path: str = DEFAULT_COMPONENTS_PATH,
        enable_cache: bool = True,
        cache_file: str = None,
        max_cache_size: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.model_path = self._download(model_path, save_path)

        config_path = os.path.join(self.model_path, "config.json")
        config = self._load_config_file(config_path)

        self.model = None
        self.model_loaded = False
        self.model_class = model_class
        self.enable_cache = enable_cache
        self.cache_file = cache_file
        self.max_cache_size = max_cache_size

        self.tokenizer = fetch_and_save_tokenizer_from_config(
            self.model_path,
            config=config,
            tokenizer_name=tokenizer_name,
            tokenizer_class=tokenizer_class,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, device: torch.device = None, dtype: torch.dtype = None):
        model = self._load_model(
            {
                "type": "hidream.llama",
                "base": self.model_class,
                "model_path": self.model_path,
            },
            module_name="transformers",
            load_dtype=dtype,
        )
        self.to_device(model, device=device)
        
        

        return model

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | List[str],
        max_sequence_length: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = None,
        num_images_per_prompt: int = 1,
    ):

        kwargs = {
            "prompt": prompt,
            "max_sequence_length": max_sequence_length,
            "device": device,
            "dtype": dtype,
            "num_images_per_prompt": num_images_per_prompt,
        }

        prompt_hash = self.hash_prompt(kwargs)

        if self.enable_cache:
            cached = self.load_cached(prompt_hash)
            if cached is not None:
                prompt_embeds = cached[0]
                prompt_embeds = prompt_embeds.to(device).to(dtype)
                return prompt_embeds

        if not self.model_loaded:
            self.model = self.load_model(device=device, dtype=dtype)
            self.model_loaded = True

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        outputs = self.model(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
        )

        prompt_embeds = outputs.hidden_states[1:]

        prompt_embeds = torch.stack(prompt_embeds, dim=0)

        _, bs_embed, seq_len, dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(-1, num_images_per_prompt, seq_len, dim)

        if self.enable_cache:
            self.cache(prompt_hash, prompt_embeds, attention_mask)

        return prompt_embeds
