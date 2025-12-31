from typing import Dict, Type, Any, Optional, List, Literal, Tuple
from pathlib import Path
import importlib
import inspect
from src.utils.yaml import load_yaml
import torch
from src.engine.base_engine import BaseEngine
from src.manifest.resolver import resolve_manifest_reference
from loguru import logger
import traceback


class EngineRegistry:
    """Central registry for all engine implementations.

    There are two sources of truth:

    - **Legacy, family-level engines** registered explicitly in
      :meth:`_register_engines`. These typically expose a ``model_type`` enum
      and internally route to per‑mode implementations.
    - **New, auto-discovered engines**, where each concrete engine subclass
      :class:`BaseEngine` lives in its own module
      ``src.engine.<engine>/<model_type>.py`` (optionally inheriting from a
      shared base defined in ``shared.py``). For these, the ``model_type`` is
      inferred from the filename and no enum is required.
    """

    def __init__(self):
        # Auto-discovered engines:
        #   engine_type (folder name) → model_type (filename) → engine class
        # All keys are stored lowercase for consistency.
        self._discovered: Dict[str, Dict[str, Type[BaseEngine]]] = {}

        self._auto_discover_engines()

    def _auto_discover_engines(self) -> None:
        """Discover concrete engine implementations from the filesystem.

        Discovery rules:
        - Look under ``src/engine/<engine_type>/`` directories.
        - Import every ``*.py`` module except ``__init__.py`` and ``shared.py``.
        - Within each module, register any classes that subclass
          :class:`BaseEngine` (excluding :class:`BaseEngine` itself).
        - The ``engine_type`` is the folder name, and the ``model_type`` is
          inferred from the filename (e.g. ``t2i.py`` → ``model_type="t2i"``).
        """

        root: Path = Path(__file__).resolve().parent
        for pkg_path in root.iterdir():
            if pkg_path.name.startswith("_"):
                continue

            engine_type = pkg_path.name.lower()
            for module_path in pkg_path.glob("*.py"):
                stem = module_path.stem
                # Skip package and shared helpers; these do not represent
                # concrete model variants.
                if stem in {"__init__", "shared"}:
                    continue

                module_name = f"src.engine.{engine_type}.{stem}"
                try:
                    module = importlib.import_module(module_name)
                except Exception as e:
                    # Best-effort discovery; failures here should not block startup
                    logger.error(f"Failed to import module {module_name}: {Exception}")
                    logger.debug("Traceback:\n" + traceback.format_exc())
                    continue
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        inspect.isclass(attr)
                        and issubclass(attr, BaseEngine)
                        and attr is not BaseEngine  # ignore classes with shared in name
                        and not (
                            attr_name.lower().startswith("shared")
                            or attr_name.lower().endswith("shared")
                        )
                    ):
                        model_type = stem.lower()
                        engine_map = self._discovered.setdefault(engine_type, {})
                        # First one wins for a given (engine_type, model_type)
                        engine_map.setdefault(model_type, attr)

    def get_engine_class(
        self, engine_type: str, model_type: Optional[str] = None
    ) -> Optional[Type[BaseEngine]]:
        """Get an engine class.

        Resolution:
        - If ``model_type`` is provided and a matching auto‑discovered class
          exists (``src.engine.<engine_type>/<model_type>.py``), return it.
        - Otherwise, return ``None``.
        """

        engine_key = engine_type.lower()
        model_key = model_type.lower() if isinstance(model_type, str) else None
        if model_key is not None:
            family = self._discovered.get(engine_key, {})
            if model_key in family:
                return family[model_key]

        return None

    def list_engines(self) -> List[str]:
        """List all available engines.

        Returns combined identifiers of the form ``"<engine>/<model_type>"``
        (e.g. ``"chroma/t2i"``) for all auto‑discovered engines.
        """

        names = set()
        for engine_type, model_map in self._discovered.items():
            for model_type in model_map.keys():
                names.add(f"{engine_type}/{model_type}")
        return sorted(names)

    def create_engine(
        self,
        yaml_path: str,
        engine_type: str | None = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Create an engine instance.

        Behaviour:
        - If a concrete auto‑discovered engine is available for the given
          ``engine_type`` and ``model_type`` (based on filename), that class
          is instantiated directly.
        - Otherwise, a ``ValueError`` is raised.
        """

        resolved = resolve_manifest_reference(yaml_path) or yaml_path
        engine_kwargs = {}
        if engine_type is None or model_type is None:
            # read from yaml_path
            data = load_yaml(resolved)
            spec = data.get("spec", {})
            engine_type = engine_type or spec.get("engine")
            model_type = model_type or spec.get("model_type")
            engine_kwargs = spec.get("engine_kwargs", {})
            if engine_type is None or model_type is None:
                raise ValueError(
                    f"Engine type and model type must be provided in the yaml file: {resolved}"
                )

        # Prefer auto‑discovered concrete engines when model_type is given.
        impl_class = (
            self.get_engine_class(engine_type, model_type) if model_type else None
        )

        if impl_class is not None and impl_class is not BaseEngine:
            return impl_class(
                yaml_path=resolved, model_type=model_type, **{**engine_kwargs, **kwargs}
            )

        # No autodiscovered implementation found
        raise ValueError(
            "Unknown engine implementation for "
            f"engine_type='{engine_type}' model_type='{model_type}'. "
            "Ensure there is a "
            f"src/engine/{engine_type}/{model_type}.py defining a BaseEngine subclass."
        )


class UniversalEngine:
    """Universal engine interface that can run any registered engine"""

    def __init__(
        self,
        engine_type: str | None = None,
        yaml_path: str | None = None,
        model_type: Optional[str] = None,
        **kwargs,
    ):
        self.registry = EngineRegistry()
        self.engine: BaseEngine = self.registry.create_engine(
            engine_type=engine_type,
            yaml_path=yaml_path,
            model_type=model_type,
            **kwargs,
        )
        self.engine_type = engine_type
        self.model_type = model_type

        if not self.engine_type:
            self.engine_type = self.engine.config.get("engine", None)
        if not self.model_type:
            self.model_type = self.engine.model_type

    @torch.inference_mode()
    def run(self, *args, **kwargs):
        """Run the engine with given parameters"""
        default_kwargs = self.engine._get_default_kwargs("run")
        merged_kwargs = {**default_kwargs, **kwargs}
        return self.engine.run(*args, **merged_kwargs)

    def offload_engine(self):
        self.engine.offload_engine()

    def __getattr__(self, name):
        """Delegate any missing attributes to the underlying engine"""
        return getattr(self.engine, name)

    def __str__(self):
        return f"UniversalEngine(engine_type={self.engine_type}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


# Global registry instance
_global_registry = EngineRegistry()


def get_engine_registry() -> EngineRegistry:
    """Get the global engine registry"""
    return _global_registry


def create_engine(
    yaml_path: str,
    engine_type: str | None = None,
    model_type: str | None = None,
    **kwargs,
) -> BaseEngine:
    """Convenience function to create a concrete engine instance.

    This uses the global :class:`EngineRegistry` and supports both legacy
    family-level engines and the new autodiscovered engines, where
    ``engine_type`` is the directory name under ``src/engine`` and
    ``model_type`` is the stem of the Python file.
    """

    return _global_registry.create_engine(
        engine_type=engine_type,
        yaml_path=yaml_path,
        model_type=model_type,
        **kwargs,
    )


def list_available_engines() -> List[str]:
    """List all available engine types"""
    return _global_registry.list_engines()
