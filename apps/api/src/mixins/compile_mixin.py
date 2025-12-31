import torch
from loguru import logger as _logger
from typing import Any, Dict


class CompileMixin:
    """
    Mixin that optionally compiles components with torch.compile based on
    per‑component configuration from the manifest.

    Expected component config shape (example):
        compile:
          type: torch.compile
          enabled: true
          mode: reduce-overhead
          dynamic: true
          fullgraph: false
          backend: aot_eager        # optional
          options: {...}            # optional torch.compile options dict
    """

    def _compile_get_logger(self):
        # Prefer an existing engine/logger attribute when present.
        return getattr(self, "logger", _logger)

    def _compile_should_enable(self, component: Dict[str, Any]) -> bool:
        cfg = component.get("compile")
        if not isinstance(cfg, dict):
            return False

        enabled = cfg.get("enabled", False)
        if not enabled:
            return False

        compile_type = cfg.get("type", "torch.compile")
        # For now we only support torch.compile based flows.
        return isinstance(compile_type, str) and compile_type.startswith(
            "torch.compile"
        )

    def _compile_from_config(self, module: Any, component: Dict[str, Any]):
        """
        Apply torch.compile to `module` according to the component's `compile` dict.

        This is intentionally conservative:
        - Only runs when engine_type == 'torch'
        - Only for torch.nn.Module instances
        - Silently falls back to the original module if compilation fails
        """
        log = self._compile_get_logger()

        # Only PyTorch engines are supported for now.
        engine_type = getattr(self, "engine_type", "torch")
        if engine_type != "torch":
            return module

        if not hasattr(torch, "compile"):
            log.warning(
                "torch.compile is not available on this PyTorch install; skipping compilation."
            )
            return module

        if module is None or not isinstance(module, torch.nn.Module):
            return module

        if not self._compile_should_enable(component):
            return module

        cfg = component.get("compile") or {}

        backend = cfg.get("backend")
        mode = cfg.get("mode")
        fullgraph = cfg.get("fullgraph")
        dynamic = cfg.get("dynamic")
        options = cfg.get("options")

        compile_kwargs = {}
        if backend is not None:
            compile_kwargs["backend"] = backend
        if mode is not None:
            compile_kwargs["mode"] = mode
        if fullgraph is not None:
            compile_kwargs["fullgraph"] = bool(fullgraph)
        if dynamic is not None:
            compile_kwargs["dynamic"] = bool(dynamic)
        if options is not None:
            compile_kwargs["options"] = options

        name = component.get("name") or component.get("type") or type(module).__name__

        try:
            log.info(
                f"Compiling component '{name}' with torch.compile (kwargs={compile_kwargs})"
            )
            compiled = torch.compile(module, **compile_kwargs)
            return compiled
        except Exception as e:
            # Do not fail engine construction if compilation fails; just warn and continue.
            log.warning(f"Failed to compile component '{name}' with torch.compile: {e}")
            return module

    def _maybe_compile_module(self, module: Any, component: Dict[str, Any]):
        """
        Public hook used by loaders to opt‑in to compilation when supported.

        Returns either the compiled module or the original one if compilation is
        disabled/unsupported or fails.
        """
        try:
            return self._compile_from_config(module, component)
        except Exception:
            # Very defensive: never let compilation break model loading.
            return module
