from typing import Dict, Any, Optional
from accelerate import init_empty_weights
import importlib
import re
from collections import Counter
import torch
from src.quantize.ggml_ops import ggml_cat, ggml_chunk
from diffusers import ModelMixin
from transformers import PreTrainedModel, PretrainedConfig
import inspect


def get_model_class(
    model_base: str, model_config: dict | None = None, model_type: str = "transformer"
):
    """
    Import the model module for the given base and return the class that is a subclass
    of `ModelMixin` from `diffusers`. This no longer relies on `_class_name` inside
    `model_config`; `model_config` is kept only for backward compatibility and may be
    omitted.
    """
    module_type = importlib.import_module(f"src.{model_type}.{model_base}.model")

    # Scan attributes on the module and return the first concrete subclass of ModelMixin.
    candidate_class = None
    for attr_name in dir(module_type):
        attr = getattr(module_type, attr_name)
        if isinstance(attr, type) and issubclass(attr, ModelMixin):
            # Skip the base class itself if it ever appears on the module.
            if attr is ModelMixin:
                continue
            candidate_class = attr
            break

    if candidate_class is None:
        raise RuntimeError(
            f"No subclass of ModelMixin found in module "
            f"'src.{model_type}.{model_base}.model'"
        )

    return candidate_class


def get_empty_model(model_class, config: dict, **extra_kwargs):
    with init_empty_weights():
        # Check the constructor signature to determine what it expects
        sig = inspect.signature(model_class.__init__)
        params = list(sig.parameters.values())
        # Skip 'self' parameter
        if params and params[0].name == "self":
            params = params[1:]
        # Check if the first parameter expects a PretrainedConfig object
        expects_pretrained_config = False
        if params:
            first_param = params[0]
            if (
                first_param.annotation == PretrainedConfig
                or (
                    hasattr(first_param.annotation, "__name__")
                    and "Config" in first_param.annotation.__name__
                )
                or first_param.name in ["config"]
                and issubclass(model_class, PreTrainedModel)
            ):
                expects_pretrained_config = True
        if expects_pretrained_config:
            # Use the model's specific config class if available, otherwise fall back to PretrainedConfig
            config_class = getattr(model_class, "config_class", PretrainedConfig)
            conf = config_class(**config)
            if hasattr(model_class, "_from_config"):
                model = model_class._from_config(conf, **extra_kwargs)
            else:
                model = model_class.from_config(conf, **extra_kwargs)
        else:
            if hasattr(model_class, "_from_config"):
                model = model_class._from_config(config, **extra_kwargs)
            else:
                model = model_class.from_config(config, **extra_kwargs)
    return model


def swap_scale_shift(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """[shift, scale]  ->  [scale, shift] along `dim`."""
    shift, scale = ggml_chunk(t, 2, dim=dim)
    return ggml_cat([scale, shift], dim=dim)


def swap_proj_gate(t: torch.Tensor) -> torch.Tensor:
    """[proj, gate]  ->  [gate, proj] for Gated-GeLU/SiLU MLPs."""
    proj, gate = ggml_chunk(t, 2, dim=0)
    return ggml_cat([gate, proj], dim=0)


def update_state_dict_(sd: Dict[str, Any], old_key: str, new_key: str):
    """Pop `old_key` (if still present) and write it back under `new_key`."""
    if old_key in sd:
        sd[new_key] = sd.pop(old_key)


def strip_common_prefix(
    src_state: Dict[str, Any],
    ref_state: Optional[Dict[str, Any]] = None,
    model_keys: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Strip a (safe) common prefix from keys *in place* when doing so improves matching.

    Parameters
    ----------
    src_state : dict
        The state_dict that might contain an unwanted prefix.
    ref_state : dict, optional
        A “clean” state_dict whose keys represent the desired names.
        If given, the function tries to find the shortest prefix `p` such that
        `k[len(p):] in ref_state` for *many* keys in `src_state`.
        If omitted, the function simply removes a unanimous first token
        (everything before the first '.') if all keys share it.

    Examples
    --------
    >>> clean_sd = {"blocks.0.attn1.to_q.weight": torch.randn(1)}
    >>> dirty_sd = {"model.diffusion_model.blocks.0.attn1.to_q.weight": torch.randn(1)}
    >>> strip_common_prefix(dirty_sd, clean_sd).keys()
    dict_keys(['blocks.0.attn1.to_q.weight'])
    """
    # --- Helpers -----------------------------------------------------------
    # We only want to strip *wrapper* prefixes like `model.` / `diffusion_model.`
    # (and short combinations like `model.diffusion_model.`), never deeper structural
    # tokens like `blocks.` / `output_blocks.`.
    _SAFE_PREFIX_TOKENS = {
        # common wrappers
        "model",
        "module",
        "diffusion_model",
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "first_stage_model",
        "cond_stage_model",
        "base_model",
        "network",
        "net",
        "backbone",
        "encoder",
        "decoder",
    }

    def _normalize_lora_key_for_ref_match(k: str) -> str:
        """
        Normalize LoRA-style keys so they can be compared against base-model keys.
        Only applies when the key looks like a LoRA key; otherwise returns `k` unchanged.
        """
        tokens = k.split(".")
        if not tokens:
            return k

        has_lora_marker = any(
            t == "lora" or t.startswith("lora_") or t.endswith("_lora") for t in tokens
        )
        if not has_lora_marker:
            return k

        drop_tokens = {
            "lora",
            "loras",
            "lora_up",
            "lora_down",
            "lora_A",
            "lora_B",
            "lora_embedding_A",
            "lora_embedding_B",
            "alpha",
            "rank",
            "rank_num",
            "ranknum",
        }

        cleaned: list[str] = []
        for t in tokens:
            if t in drop_tokens:
                continue
            # Some formats use "lora.up"/"lora.down" or just "A"/"B" tokens
            if t in {"up", "down", "A", "B"}:
                continue
            cleaned.append(t)

        return ".".join(cleaned)

    def _apply_prefix(p: str, k: str) -> str:
        return k[len(p) :] if p and k.startswith(p) else k

    # --- Determine reference keys -----------------------------------------
    ref_keys: Optional[set[str]] = None
    if ref_state is not None:
        ref_keys = set(ref_state.keys())
    elif model_keys:
        ref_keys = set(model_keys)

    keys = list(src_state.keys())
    if not keys:
        return src_state

    # --- Candidate prefixes (1-2 leading safe tokens) ----------------------
    candidates: set[str] = set()
    for k in keys:
        parts = k.split(".")
        if len(parts) < 2:
            continue
        if parts[0] in _SAFE_PREFIX_TOKENS:
            candidates.add(parts[0] + ".")
            if len(parts) >= 3 and parts[1] in _SAFE_PREFIX_TOKENS:
                candidates.add(parts[0] + "." + parts[1] + ".")

    # --- If we don't have ref keys, only do a very safe unanimous strip ----
    if ref_keys is None:
        first_tokens = {k.split(".", 1)[0] for k in keys if "." in k}
        if len(first_tokens) == 1:
            token = next(iter(first_tokens))
            if token in _SAFE_PREFIX_TOKENS:
                prefix = token + "."
                items = list(src_state.items())
                src_state.clear()
                for k, v in items:
                    src_state[_apply_prefix(prefix, k)] = v
                return src_state
        return src_state

    # --- Otherwise, pick the safe prefix that improves overlap -------------
    def score(prefix: str) -> int:
        c = 0
        for k in keys:
            k2 = _apply_prefix(prefix, k)
            if _normalize_lora_key_for_ref_match(k2) in ref_keys:
                c += 1
        return c

    baseline = score("")
    best_prefix = ""
    best_score = baseline

    for p in candidates:
        s = score(p)
        if s > best_score or (
            s == best_score and best_prefix and len(p) < len(best_prefix)
        ):
            best_prefix = p
            best_score = s

    if best_prefix and best_score > baseline:
        items = list(src_state.items())
        src_state.clear()
        for k, v in items:
            src_state[_apply_prefix(best_prefix, k)] = v
        return src_state

    return src_state
