import importlib
import inspect
import logging
from pathlib import Path
from diffusers.models.modeling_utils import ModelMixin
from .base import TRANSFORMERS_REGISTRY

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


def _auto_register_transformers():
    """
    Automatically register transformer models by scanning the transformer package.

    A transformer is assumed to live under ``<root>/<family>/<variant>/model.py`` and expose
    at least one class that inherits from ``ModelMixin``. It is registered under the key
    ``"{family}.{variant}"`` (lowercased).

    Existing registrations (for example via the TRANSFORMERS_REGISTRY decorator) are preserved.
    """
    global TRANSFORMERS_REGISTRY
    root = Path(__file__).resolve().parent

    for family_dir in root.iterdir():
        if not family_dir.is_dir():
            continue
        if family_dir.name.startswith("_") or family_dir.name == "__pycache__":
            continue

        for variant_dir in family_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            model_file = variant_dir / "model.py"
            if not model_file.is_file():
                continue

            key = f"{family_dir.name}.{variant_dir.name}".lower()
            # If something already registered this key (e.g. via decorator), leave it alone.
            try:
                TRANSFORMERS_REGISTRY.get(key)
                continue
            except KeyError:
                pass

            module_name = f"{__name__}.{family_dir.name}.{variant_dir.name}.model"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                msg = str(e)
                if "libcudart.so.13" in msg:
                    msg = (
                        f"{msg} (Nunchaku wheel requires CUDA runtime 13. "
                        "Install CUDA runtime 13 / make it discoverable via LD_LIBRARY_PATH, "
                        "or uninstall/avoid Nunchaku variants.)"
                    )
                logger.error(f"Error importing module {module_name}: {msg}")
                # If import fails for any reason, skip auto-registration for this module.
                continue

            # Find the first class defined in this module that subclasses ModelMixin.
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                try:
                    if issubclass(cls, ModelMixin) and cls is not ModelMixin:
                        try:
                            TRANSFORMERS_REGISTRY.register(key, cls)
                        except Exception as e:
                            continue
                        break
                except TypeError:
                    # Builtins and certain extension types can raise here; just ignore them.
                    continue


# Ensure all transformers are registered on import.
_auto_register_transformers()

__all__ = ["TRANSFORMERS_REGISTRY"]
