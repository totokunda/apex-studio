"""
Anima transformer registry entry.

The transformer auto-registration system expects models to live under:
  src/transformer/<family>/<variant>/model.py

The core implementation for Anima lives in `src/transformer/anima/model.py`, so
this thin wrapper defines a local subclass to make the auto-registration logic
pick it up under the key `anima.base`.
"""

from src.transformer.anima.model import AnimaTransformer3DModel as _AnimaTransformer3DModel


class AnimaTransformer3DModel(_AnimaTransformer3DModel):
    pass


__all__ = ["AnimaTransformer3DModel"]

