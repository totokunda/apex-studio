import os
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, Dict
import json
import transformers
from src.utils.module import find_class_recursive
import traceback
import hashlib
import pickle
import re


_TRANSFORMERS_CHAT_TEMPLATE_SHIM_INSTALLED = False


def _model_dir_from_path(model_path: str, *, config_path: str | None = None) -> Path:
    """
    Return the directory that "owns" model assets.

    `model_path` can be either:
    - a file path (e.g. `/.../text_encoder.safetensors`) -> returns its parent dir
    - a directory path (e.g. `/.../Llama-3.1-8B-Instruct`) -> returns itself
    """
    p = Path(model_path)
    try:
        # Prefer a real local path if it exists.
        if p.exists():
            return p if p.is_dir() else p.parent
    except Exception:
        pass

    # If the model weights path isn't local yet (common: tokenizer loads before weights download),
    # fall back to the (already-downloaded) config path directory if available.
    if config_path:
        cp = Path(config_path)
        try:
            if cp.exists():
                return cp if cp.is_dir() else cp.parent
        except Exception:
            pass

    # Last resort: previous behavior.
    return p.parent


def _normalize_name_or_path(value: Any) -> Any:
    """Normalize a tokenizer name/path value.

    - Strips whitespace for strings.
    - Treats empty/whitespace-only strings as missing (None).
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def _sanitize_for_dirname(value: str | None, *, max_len: int = 64) -> str:
    """Make a filesystem-friendly directory name component."""
    if not value:
        return "unknown"
    s = str(value)
    # Replace path separators and collapse consecutive underscores.
    s = s.replace("\\", "_").replace("/", "_")
    s = re.sub(r'[<>:"|?*\x00-\x1f]', "_", s)
    s = re.sub(r"_+", "_", s).strip("._ ")
    if not s:
        return "unknown"
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    return s if s else "unknown"


def _stable_object_hash(obj: Any) -> str:
    """Return a deterministic hash for nested dict/list/set structures."""

    def canonicalize(x: Any) -> Any:
        if isinstance(x, dict):
            return tuple((k, canonicalize(v)) for k, v in sorted(x.items(), key=lambda kv: kv[0]))
        if isinstance(x, (list, tuple)):
            return tuple(canonicalize(v) for v in x)
        if isinstance(x, set):
            return tuple(sorted(canonicalize(v) for v in x))
        # Avoid embedding secrets or huge objects; fall back to repr().
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return repr(x)

    data = pickle.dumps(canonicalize(obj), protocol=5)
    return hashlib.sha256(data).hexdigest()


def _tokenizer_class_id(tokenizer_cls: Any) -> str:
    try:
        mod = getattr(tokenizer_cls, "__module__", "") or ""
        name = getattr(tokenizer_cls, "__name__", "") or ""
        if mod and name:
            return f"{mod}.{name}"
        return name or mod or repr(tokenizer_cls)
    except Exception:
        return repr(tokenizer_cls)


def _compute_tokenizer_save_dirs(
    *,
    model_path: str,
    name_or_path: Any,
    tokenizer_cls: Any,
    tokenizer_kwargs: Dict[str, Any],
    loaded_config: Dict[str, Any],
    config_path: str | None,
) -> tuple[Path, Path]:
    """
    Returns (unique_save_dir, legacy_save_dir).

    - unique_save_dir is stable and collision-resistant even when name/path is blank.
    - legacy_save_dir matches the previous behavior for backwards-compatible reads.
    """
    normalized = _normalize_name_or_path(name_or_path)
    model_dir = _model_dir_from_path(model_path, config_path=config_path)

    # Previous behavior:
    #   Path(model_path).parent / _name_or_path.replace("/", "_") / "tokenizer"
    # When _name_or_path == "" it collapsed to "<parent>/tokenizer" (collision-prone).
    if normalized is None:
        legacy_save_dir = model_dir / "tokenizer"
    else:
        legacy_save_dir = (
            model_dir / str(normalized).replace("/", "_") / "tokenizer"
        )

    # New behavior: include a short hash of the tokenizer "variant" *and* the model identity
    # so different models never collapse into the same tokenizer directory (avoids races).
    variant_keys = ("revision", "subfolder", "use_fast", "trust_remote_code", "tokenizer_type")
    variant = {k: tokenizer_kwargs.get(k) for k in variant_keys if k in tokenizer_kwargs}

    # Avoid hashing secrets if present in config/kwargs.
    safe_config = dict(loaded_config or {})
    safe_config.pop("hf_token", None)
    safe_config.pop("token", None)
    safe_config.pop("use_auth_token", None)

    payload: Dict[str, Any] = {
        "tokenizer_class": _tokenizer_class_id(tokenizer_cls),
        "name_or_path": str(normalized) if normalized is not None else "",
        "variant": variant,
        # Ensure per-model uniqueness even when multiple models share a tokenizer repo name.
        "model_path": str(model_path),
        "config_path": str(config_path) if config_path else None,
    }

    # If the name/path is missing, we also mix in (sanitized) config content to reduce ambiguity.
    if normalized is None:
        payload["config"] = safe_config

    suffix = _stable_object_hash(payload)[:12]
    base = (
        _sanitize_for_dirname(str(normalized), max_len=64)
        if normalized is not None
        else _sanitize_for_dirname(f"local_{model_dir.name}", max_len=64)
    )
    unique_save_dir = model_dir / f"{base}__{suffix}" / "tokenizer"
    return unique_save_dir, legacy_save_dir


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
    _name_or_path = _normalize_name_or_path(_name_or_path)
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

    unique_save_dir, legacy_save_dir = _compute_tokenizer_save_dirs(
        model_path=model_path,
        name_or_path=_name_or_path,
        tokenizer_cls=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        loaded_config=loaded_config,
        config_path=config_path,
    )

    def _looks_like_tokenizer_dir(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        # Minimal heuristics: any one of these files indicates a real saved tokenizer.
        markers = (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        )
        return any((p / m).exists() for m in markers)

    # When loading from a locally-saved tokenizer directory, we must not keep hub-oriented
    # kwargs like `subfolder=...` (it would point inside the local dir and fail, causing
    # a fallback to hub that triggers HEAD/GET requests).
    local_tokenizer_kwargs = dict(tokenizer_kwargs)
    local_tokenizer_kwargs.pop("subfolder", None)
    # Be explicit: when we think we have local files, do not touch the network.
    local_tokenizer_kwargs.setdefault("local_files_only", True)

    def _load_tokenizer_with_fallbacks(_tokenizer_cls:AutoTokenizer, name_or_path: str):
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

    def _load_local_tokenizer(_tokenizer_cls: AutoTokenizer, local_path: str):
        # Local loads should never require auth kwargs. If present, strip them to avoid
        # compatibility issues across transformers versions.
        kwargs = dict(local_tokenizer_kwargs)
        kwargs.pop("token", None)
        kwargs.pop("use_auth_token", None)
        return _tokenizer_cls.from_pretrained(local_path, **kwargs)

    # Determine the source identifier for loading.
    # If name/path is missing, prefer local tokenizer next to weights over ambiguous ""/cwd.
    model_dir = _model_dir_from_path(model_path, config_path=config_path)

    # Prefer an on-disk tokenizer saved next to the model (no hub calls).
    for candidate in (unique_save_dir, legacy_save_dir):
        if _looks_like_tokenizer_dir(candidate):
            try:
                return _load_local_tokenizer(tokenizer_class, str(candidate))
            except Exception:
                traceback.print_exc()
                # Fall through to hub/local-dir fallback below.

    load_name_or_path = str(_name_or_path) if _name_or_path is not None else str(model_dir)
    try:
        tokenizer = _load_tokenizer_with_fallbacks(tokenizer_class, load_name_or_path)
    except Exception:
        tokenizer_class = AutoTokenizer
        tokenizer = _load_tokenizer_with_fallbacks(tokenizer_class, load_name_or_path)

    # Persist for future runs/generations so we can load locally without hitting the hub.
    try:
        unique_save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(unique_save_dir))
        # Backwards-compatible copy to the legacy location (best-effort).
        if legacy_save_dir != unique_save_dir and not _looks_like_tokenizer_dir(legacy_save_dir):
            legacy_save_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(legacy_save_dir))
    except Exception:
        pass
    return tokenizer
