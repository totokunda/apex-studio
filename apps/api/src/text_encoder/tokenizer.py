import os
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, Dict
import json
import transformers
from src.utils.module import find_class_recursive
import traceback


_TRANSFORMERS_CHAT_TEMPLATE_SHIM_INSTALLED = False


def _install_transformers_chat_template_shim() -> None:
    """
    transformers>=4.57 calls `transformers.utils.hub.list_repo_templates()` during
    tokenizer loading to discover optional chat templates under
    `additional_chat_templates/`.

    Some hub servers/repos legitimately don't have this folder, and certain
    version combinations raise `RemoteEntryNotFoundError` (404) instead of
    treating it as "no templates". We shim this to return an empty list.
    """

    global _TRANSFORMERS_CHAT_TEMPLATE_SHIM_INSTALLED
    if _TRANSFORMERS_CHAT_TEMPLATE_SHIM_INSTALLED:
        return

    try:
        from transformers.utils import hub as _tf_hub
        from transformers import tokenization_utils_base as _tf_tok_base
        from transformers import utils as _tf_utils
    except Exception:
        return

    original_list_repo_templates = getattr(_tf_hub, "list_repo_templates", None)
    if original_list_repo_templates is None:
        return

    # Avoid double-wrapping if this module is reloaded.
    if getattr(original_list_repo_templates, "__apex_chat_template_shim__", False):
        _TRANSFORMERS_CHAT_TEMPLATE_SHIM_INSTALLED = True
        return

    def _wrapped_list_repo_templates(*args: Any, **kwargs: Any) -> list[str]:
        try:
            return original_list_repo_templates(*args, **kwargs)
        except Exception as e:
            try:
                from huggingface_hub.errors import RemoteEntryNotFoundError
            except Exception:
                RemoteEntryNotFoundError = None

            # Missing `additional_chat_templates/` should not be fatal.
            if RemoteEntryNotFoundError is not None and isinstance(
                e, RemoteEntryNotFoundError
            ):
                return []
            raise

    setattr(_wrapped_list_repo_templates, "__apex_chat_template_shim__", True)
    # Patch both the canonical implementation and any by-value imports.
    _tf_hub.list_repo_templates = _wrapped_list_repo_templates
    if getattr(_tf_utils, "list_repo_templates", None) is not None:
        _tf_utils.list_repo_templates = _wrapped_list_repo_templates
    if getattr(_tf_tok_base, "list_repo_templates", None) is not None:
        _tf_tok_base.list_repo_templates = _wrapped_list_repo_templates
    _TRANSFORMERS_CHAT_TEMPLATE_SHIM_INSTALLED = True


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

    _install_transformers_chat_template_shim()

    if config_path:
        loaded_config = json.load(open(config_path, "r"))
    else:
        loaded_config = {}

    if config:
        loaded_config.update(config)

    _name_or_path = tokenizer_name or loaded_config.get("_name_or_path", None)

    if _name_or_path is not None:
        save_dir = Path(model_path).parent / _name_or_path.replace("/", "_") / "tokenizer"
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

        def _maybe_load_local_tokenizer_first(_tokenizer_cls, local_dir: Path):
            """
            Prefer the tokenizer we previously saved alongside model weights.
            This avoids repeated Hub HEAD/GET calls on every run.
            """

            # Heuristic: only attempt local load if the directory contains tokenizer artifacts.
            # (Some tokenizers may not have all of these, but usually have at least one.)
            marker_files = (
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
                "tokenizer.model",
                "preprocessor_config.json",
                "processor_config.json",
            )
            if not local_dir.exists() or not local_dir.is_dir():
                return None
            if not any((local_dir / f).exists() for f in marker_files):
                return None

            # If the caller explicitly asked for local-only behavior, respect it.
            local_kwargs = dict(tokenizer_kwargs)
            local_kwargs.setdefault("local_files_only", True)
            local_kwargs.pop("subfolder", None)
            try:
                return _tokenizer_cls.from_pretrained(str(local_dir), **local_kwargs)
            except Exception as e:
                # If local load fails, fall back to Hub below.
                return None

        def _load_tokenizer_with_fallbacks(_tokenizer_cls, name_or_path: str):
            # If caller already supplied auth kwargs, respect them.
            if (
                tokenizer_kwargs.get("token") is not None
                or tokenizer_kwargs.get("use_auth_token") is not None
            ):
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
            local_tokenizer = _maybe_load_local_tokenizer_first(tokenizer_class, save_dir)
            if local_tokenizer is not None:
                return local_tokenizer
            tokenizer = _load_tokenizer_with_fallbacks(tokenizer_class, _name_or_path)
        except Exception as e:
            traceback.print_exc()
            tokenizer_class = AutoTokenizer
            local_tokenizer = _maybe_load_local_tokenizer_first(tokenizer_class, save_dir)
            if local_tokenizer is not None:
                return local_tokenizer
            tokenizer = _load_tokenizer_with_fallbacks(tokenizer_class, _name_or_path)
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        return tokenizer
    else:
        raise ValueError("No name_or_path found in config")
