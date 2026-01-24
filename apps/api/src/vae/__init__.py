import importlib
import inspect
from pathlib import Path
from typing import Dict, Type
from diffusers.models.modeling_utils import ModelMixin
from src.register import ClassRegister
from loguru import logger

VAE_REGISTRY = ClassRegister()


def _auto_register_vaes():
    """
    Automatically register VAE models by scanning the ``src.vae`` package.

    A VAE is assumed to live under ``<root>/<name>/model.py`` and expose at least one class
    that inherits from ``ModelMixin``. It is registered under the key ``"<name>"`` (lowercased).

    Existing entries in ``VAE_REGISTRY`` are preserved so current VAEs keep their explicit mapping.
    """

    root = Path(__file__).resolve().parent

    for vae_dir in root.iterdir():
        if not vae_dir.is_dir():
            continue
        if vae_dir.name.startswith("_") or vae_dir.name == "__pycache__":
            continue

        key = vae_dir.name.lower()
        if key in VAE_REGISTRY:
            # Already mapped explicitly above; leave as-is.
            continue

        model_file = vae_dir / "model.py"
        if not model_file.is_file():
            continue

        module_name = f"{__name__}.{vae_dir.name}.model"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            print(
                f"\n\nError importing module {module_name} with exception {Exception}\n\n"
            )
            import traceback

            traceback.print_exc()
            # If import fails for any reason, skip auto-registration for this module.
            continue

        # Find the first class defined in this module that subclasses ModelMixin.
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            try:
                if issubclass(cls, ModelMixin) and cls is not ModelMixin:
                    VAE_REGISTRY.register(key, cls)
                    break
            except TypeError:
                # Builtins and certain extension types can raise here; just ignore them.
                continue


_auto_register_vaes()


def get_vae(vae_name: str):
    key = vae_name.lower()

    if key in VAE_REGISTRY:
        return VAE_REGISTRY.get(key)
    raise ValueError(f"VAE {vae_name} not found")
