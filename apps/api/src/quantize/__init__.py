from __future__ import annotations

from typing import TYPE_CHECKING

from .scaled_layer import (
    patch_fpscaled_model,
    patch_fpscaled_model_from_state_dict,
    restore_fpscaled_parameters,
)

if TYPE_CHECKING:  # pragma: no cover
    # Imported lazily at runtime to avoid pulling heavy deps on import.
    from .ggml_layer import patch_model, patch_model_from_state_dict
    from .load import load_gguf
    from .quantize import TextEncoderQuantizer, TransformerQuantizer

__all__ = [
    "patch_model",
    "patch_model_from_state_dict",
    "patch_fpscaled_model",
    "patch_fpscaled_model_from_state_dict",
    "restore_fpscaled_parameters",
    "load_gguf",
    "TextEncoderQuantizer",
    "TransformerQuantizer",
]


def __getattr__(name: str):
    """
    Lazy attribute loading.

    `src.quantize` is used in lightweight inference paths (patching / loading)
    and in heavier quantization/authoring paths (quantizer classes). Importing
    the latter can pull optional heavyweight dependencies (e.g. transformers).
    """
    if name in {"patch_model", "patch_model_from_state_dict"}:
        from .ggml_layer import patch_model, patch_model_from_state_dict

        return patch_model if name == "patch_model" else patch_model_from_state_dict

    if name == "load_gguf":
        from .load import load_gguf

        return load_gguf

    if name in {"TextEncoderQuantizer", "TransformerQuantizer"}:
        from .quantize import TextEncoderQuantizer, TransformerQuantizer

        return TextEncoderQuantizer if name == "TextEncoderQuantizer" else TransformerQuantizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
