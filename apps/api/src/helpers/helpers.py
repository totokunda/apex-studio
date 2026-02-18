from pathlib import Path
import importlib
import inspect
import traceback
from loguru import logger

from src.register import ClassRegister
from src.helpers.base import BaseHelper

helpers = ClassRegister()


def _auto_register_helpers() -> None:
    """
    Automatically register helper classes by scanning the helpers package.

    A helper is assumed to live under ``src/helpers/<family>/<variant>.py`` and expose
    at least one class that inherits from :class:`BaseHelper`. It is registered under
    the key ``"{family}.{variant}"`` (lowercased).

    Top-level helpers (e.g. ``src/helpers/clip.py``) are also supported and are
    registered under the key ``"{module}"`` (lowercased).

    Existing registrations are preserved: if a key already exists in ``helpers``,
    it is left unchanged.
    """
    root = Path(__file__).resolve().parent

    # 1) Nested helpers: src/helpers/<family>/<variant>.py → "family.variant"
    for family_dir in root.iterdir():
        if not family_dir.is_dir():
            continue
        if family_dir.name.startswith("_") or family_dir.name == "__pycache__":
            continue

        for module_path in family_dir.glob("*.py"):
            stem = module_path.stem
            if stem in {"__init__"}:
                continue

            key = f"{family_dir.name}.{stem}".lower()

            # Preserve any explicit registrations
            try:
                helpers.get(key)
                continue
            except KeyError:
                pass

            module_name = f"src.helpers.{family_dir.name}.{stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                logger.error(f"Error importing helper module {module_name}: {e}")
                traceback.print_exc()
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module
                if cls.__module__ != module.__name__:
                    continue

                try:
                    if issubclass(cls, BaseHelper) and cls is not BaseHelper:
                        try:
                            helpers.register(key, cls)
                        except Exception as reg_err:
                            logger.warning(
                                f"Failed to register helper '{key}' from {module_name}: {reg_err}"
                            )
                        break
                except TypeError:
                    # Non-user or extension types can raise here; just skip them.
                    continue

    # 2) Top-level helpers: src/helpers/<module>.py → "<module>"
    for module_path in root.glob("*.py"):
        stem = module_path.stem
        if stem in {"__init__", "base", "helpers"}:
            continue

        key = stem.lower()

        # Preserve any explicit registrations
        try:
            helpers.get(key)
            continue
        except KeyError:
            pass

        module_name = f"src.helpers.{stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"Error importing top-level helper module {module_name}: {e}")
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue

            try:
                if issubclass(cls, BaseHelper) and cls is not BaseHelper:
                    try:
                        helpers.register(key, cls)
                    except Exception as reg_err:
                        logger.warning(
                            f"Failed to register top-level helper '{key}' from {module_name}: {reg_err}"
                        )
                    break
            except TypeError:
                continue


# Ensure all helpers are registered on import.
_auto_register_helpers()
