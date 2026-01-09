import os
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, Dict
import json
import transformers
from src.utils.module import find_class_recursive

def fetch_and_save_tokenizer_from_config(
    model_path: str,
    config_path: str | None = None,
    config: Dict[str, Any] | None = None,
    tokenizer_class: str | None = None,
    tokenizer_name: str | None = None,
    **tokenizer_kwargs: Any,
) -> AutoTokenizer:
    """
    1) Finds the HF repo whose config.json matches your local config file name.
    2) Downloads its tokenizer.
    3) Saves the tokenizer files alongside your model weights.

    Args:
        model_path (str):
            Local path to your .safetensors (or .bin) file.
        config_path (str):
            Local path to the JSON config (e.g. text_encoder_config.json).
        revision (str):
            Which Git revision/branch/tag to search on the Hub.
        use_auth_token (bool or str):
            OAuth token for private repos, or False for public.
        **tokenizer_kwargs:
            Forwarded into AutoTokenizer.from_pretrained (e.g. use_fast=True).

    Returns:
        transformers.AutoTokenizer: the loaded tokenizer instance.
    """

    if config_path:
        loaded_config = json.load(open(config_path, "r"))
    else:
        loaded_config = {}

    if config:
        loaded_config.update(config)

    _name_or_path = tokenizer_name or loaded_config.get("_name_or_path", None)

    if _name_or_path is not None:
        tokenizer_kwargs["from_tiktoken"] = False
        if tokenizer_class is not None:
            tokenizer_class = find_class_recursive(transformers, tokenizer_class)
        else:
            tokenizer_class = AutoTokenizer
        try:
            tokenizer = tokenizer_class.from_pretrained(_name_or_path, **tokenizer_kwargs)
        except Exception as e:
            tokenizer_class = AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(_name_or_path, **tokenizer_kwargs)
        save_dir = Path(model_path).parent
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        return tokenizer
    else:
        raise ValueError("No name_or_path found in config")
