"""
Keep `src.converters` imports lightweight.

Some workflows (e.g. standalone scripts that only need `transformer_converters`) should
not fail just because optional deps of `convert.py` aren't installed in the current env.
"""

try:
    from .convert import (
        convert_transformer,
        get_text_encoder_converter,
        get_transformer_converter,
        get_vae_converter,
        get_transformer_keys,
        convert_vae,
        get_vae_keys,
    )

    __all__ = [
        "convert_transformer",
        "get_text_encoder_converter",
        "get_transformer_keys",
        "convert_vae",
        "get_vae_keys",
    ]
except ModuleNotFoundError:
    # Allow importing submodules like `src.converters.transformer_converters` without
    # requiring all optional dependencies of `src.converters.convert`.
    __all__ = []
