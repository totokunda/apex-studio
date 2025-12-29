import importlib
import pkgutil
from types import ModuleType
from typing import Any

from loguru import logger


def find_class_recursive(module: ModuleType, class_name: str) -> Any:
    """
    Recursively searches for a class in a module and its submodules using pkgutil.
    """
    # Check the top-level module first
    if hasattr(module, class_name):
        attr = getattr(module, class_name)
        if isinstance(attr, type):
            return attr

    # If it's a package, search submodules
    if not hasattr(module, "__path__"):
        return None

    for _, submodule_name, _ in pkgutil.walk_packages(
        module.__path__, prefix=module.__name__ + "."
    ):
        try:
            submodule = importlib.import_module(submodule_name)
            if hasattr(submodule, class_name):
                attr = getattr(submodule, class_name)
                if isinstance(attr, type):
                    logger.debug(f"Found class {class_name} in {submodule_name}")
                    return attr
        except (ImportError, AttributeError, SyntaxError) as e:
            logger.debug(f"Could not import or check submodule {submodule_name}: {e}")
            continue

    return None
