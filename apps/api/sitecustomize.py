"""
Optional environment-gated shims that run at Python startup.

Python auto-imports `sitecustomize` (if present on `sys.path`) after `site`.
We use this to spoof system RAM for heuristics that rely on psutil.
"""

from __future__ import annotations

import os
import sys
import importlib.abc
import importlib.machinery
from typing import Any


def _patch_torch_cuda_queries(torch_mod: Any) -> None:
    """
    Prevent third-party libs (notably xFormers) from crashing at import-time on
    CPU-only environments.

    Some versions of xFormers probe CUDA capability during import via
    `torch.cuda.get_device_capability("cuda")`, which can raise:
      AssertionError: Invalid device id
    when CUDA is compiled in but no GPU is available / visible.

    We keep the patch lightweight and safe:
    - We DO NOT import torch here (sitecustomize runs for every Python process).
    - We only change behavior for "no CUDA / invalid device" failure cases.
    - On such failures we return (0, 0) so optional CUDA/Triton paths disable.
    """

    # Encourage xFormers to avoid Triton probing even if installed.
    # (These are no-ops for versions that don't read them, but harmless.)
    os.environ.setdefault("XFORMERS_DISABLE_TRITON", "1")
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

    try:
        cuda = torch_mod.cuda
    except Exception:
        return

    if not hasattr(cuda, "get_device_capability"):
        return

    try:
        _real_get_device_capability = cuda.get_device_capability
    except Exception:
        return

    # Avoid double-patching if multiple reload workers import torch.
    if getattr(_real_get_device_capability, "__apex_safe__", False):
        return

    def _safe_get_device_capability(device: Any = None):  # type: ignore[no-untyped-def]
        try:
            return _real_get_device_capability(device)
        except AssertionError as e:
            if "Invalid device id" in str(e):
                return (0, 0)
            raise
        except Exception as e:
            # Treat common CUDA-init failures as "no GPU".
            msg = str(e).lower()
            if (
                "invalid device" in msg
                or "no cuda" in msg
                or "not compiled with cuda" in msg
                or "cuda driver" in msg
                or "driver" in msg
            ):
                return (0, 0)
            raise

    _safe_get_device_capability.__apex_safe__ = True  # type: ignore[attr-defined]
    cuda.get_device_capability = _safe_get_device_capability  # type: ignore[assignment]


class _TorchPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader):
        self._wrapped = wrapped

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]
        self._wrapped.exec_module(module)  # type: ignore[misc]
        _patch_torch_cuda_queries(module)


class _TorchPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None):  # type: ignore[no-untyped-def]
        if fullname != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _TorchPatchLoader(spec.loader)  # type: ignore[assignment]
        return spec


def _install_torch_patch_hook() -> None:
    # Patch immediately if torch is already imported.
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        _patch_torch_cuda_queries(torch_mod)
        return
    # Otherwise install a one-shot import hook.
    for f in sys.meta_path:
        if isinstance(f, _TorchPatchFinder):
            return
    sys.meta_path.insert(0, _TorchPatchFinder())


_install_torch_patch_hook()