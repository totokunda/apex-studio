
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
) -> Dict[str, Any]:
    """
    Return a *new* state-dict whose keys no longer have an extra prefix.

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
    if ref_state is None:
        # Heuristic: do *all* keys share the same first token?
        first_tokens = {k.split(".", 1)[0] for k in src_state.keys()}

        if len(first_tokens) == 1:  # unanimous
            prefix = next(iter(first_tokens)) + "."
        else:
            return src_state  # nothing to strip
    else:
        ref_keys = set(ref_state.keys())
        prefix_counter: Counter[str] = Counter()

        # Normalize LoRA-style suffixes so we can compare against base model keys.
        # Examples:
        #   "attn1.to_q.lora_down.weight" -> "attn1.to_q.weight"
        #   "attn1.to_q.lora_A.weight"   -> "attn1.to_q.weight"
        #   "... .lora.up.weight"        -> "... .weight"
        def _normalize_lora_suffix(suffix: str) -> str:
            tokens = suffix.split(".")
            if not tokens:
                return suffix

            has_lora_marker = any(
                t == "lora" or t.startswith("lora_") or t.endswith("_lora")
                for t in tokens
            )

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

            cleaned = []
            for t in tokens:
                if t in drop_tokens:
                    continue
                # Some formats use "lora.up"/"lora.down" or just "A"/"B" tokens
                if has_lora_marker and t in {"up", "down", "A", "B"}:
                    continue
                cleaned.append(t)

            return ".".join(cleaned)

        # Look for candidate prefixes
        for k in src_state.keys():
            # Skip keys that already match
            if k in ref_keys:
                continue
            # Try every prefix ending at a dot
            for m in re.finditer(r"\.", k):
                p = k[: m.start() + 1]  # keep the trailing dot
                suffix = k[len(p) :]
                normalized_suffix = _normalize_lora_suffix(suffix)
                if normalized_suffix in ref_keys:
                    prefix_counter[p] += 1
                    # shortest prefix that works is good enough
                    break

        if not prefix_counter:
            return src_state  # nothing matched → keep as-is

        # Use the prefix that matched the *most* keys
        prefix, _ = prefix_counter.most_common(1)[0]

    # Actually build a new state-dict with the prefix removed
    stripped_state = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in src_state.items()
    }
    return stripped_state
