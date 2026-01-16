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
    hf_token: str | None = None,
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
        hf_token (str | None):
            Hugging Face token used for gated/private repos. If omitted, will also
            check (in order): `tokenizer_kwargs["hf_token"]`, `config["hf_token"]`,
            env `HUGGING_FACE_HUB_TOKEN`, then env `HF_TOKEN`.
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

        # Include Hugging Face token if available for gated/private repos.
        # transformers>=4.38 prefers `token=...`, older versions use `use_auth_token=...`.
        token = (
            hf_token
            or tokenizer_kwargs.pop("hf_token", None)
            or loaded_config.get("hf_token", None)
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        if isinstance(token, str):
            token = token.strip()
        if not token:
            token = None

        if tokenizer_class is not None:
            tokenizer_class = find_class_recursive(transformers, tokenizer_class)
        else:
            tokenizer_class = AutoTokenizer

        def _load_tokenizer_with_fallbacks(_tokenizer_cls, name_or_path: str):
            # If caller already supplied auth kwargs, respect them.
            if tokenizer_kwargs.get("token") is not None or tokenizer_kwargs.get(
                "use_auth_token"
            ) is not None:
                return _tokenizer_cls.from_pretrained(name_or_path, **tokenizer_kwargs)

            if token is None:
                return _tokenizer_cls.from_pretrained(name_or_path, **tokenizer_kwargs)

            # Try modern arg first; fall back for older transformers.
            try:
                return _tokenizer_cls.from_pretrained(
                    name_or_path, token=token, **tokenizer_kwargs
                )
            except TypeError:
                return _tokenizer_cls.from_pretrained(
                    name_or_path, use_auth_token=token, **tokenizer_kwargs
                )

        try:
            tokenizer = _load_tokenizer_with_fallbacks(tokenizer_class, _name_or_path)
        except Exception as e:
            tokenizer_class = AutoTokenizer
            tokenizer = _load_tokenizer_with_fallbacks(tokenizer_class, _name_or_path)
        save_dir = Path(model_path).parent
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        return tokenizer
    else:
        raise ValueError("No name_or_path found in config")
