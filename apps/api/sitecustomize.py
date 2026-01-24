"""
Optional environment-gated shims that run at Python startup.

Python auto-imports `sitecustomize` (if present on `sys.path`) after `site`.
We use this to spoof system RAM for heuristics that rely on psutil.
"""

from __future__ import annotations

import logging
import os
import sys
import importlib.abc
import importlib.machinery
from typing import Any

# Mitigate CUDA allocator fragmentation across long-lived processes.
# Keep user overrides intact.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def _patch_nunchaku_cuda_mempool_utils(nunchaku_c_mod: Any) -> None:
    """
    Nunchaku's `_C.utils.disable_memory_auto_release()` uses CUDA mempool APIs:
      - cudaDeviceGetDefaultMemPool
      - cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold, UINT64_MAX)

    On some Windows driver/runtime combos (notably newer GPUs), this can raise:
      RuntimeError: CUDA error: operation not supported (at .../utils.h:20)

    That call is an optimization (avoid auto-trimming). If it fails, we can safely
    skip it and continue loading the model.
    """

    try:
        cutils = getattr(nunchaku_c_mod, "utils", None)
    except Exception:
        return
    if cutils is None:
        return

    def _wrap_best_effort(fn_name: str) -> None:
        try:
            real = getattr(cutils, fn_name)
        except Exception:
            return

        if getattr(real, "__apex_best_effort__", False):
            return

        def _best_effort(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            try:
                return real(*args, **kwargs)
            except RuntimeError as e:
                msg = str(e).lower()
                # Treat "not supported" mempool calls as non-fatal.
                if "operation not supported" in msg or "cuda error" in msg:
                    logging.getLogger("apex.sitecustomize").warning(
                        "Nunchaku CUDA mempool tweak '%s' failed (%s); continuing without it.",
                        fn_name,
                        str(e).strip(),
                    )
                    return None
                raise

        _best_effort.__apex_best_effort__ = True  # type: ignore[attr-defined]
        setattr(cutils, fn_name, _best_effort)

    # Both functions rely on CUDA mempool APIs.
    _wrap_best_effort("disable_memory_auto_release")
    _wrap_best_effort("trim_memory")


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


class _NunchakuCPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader):
        self._wrapped = wrapped

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]
        self._wrapped.exec_module(module)  # type: ignore[misc]
        _patch_nunchaku_cuda_mempool_utils(module)


class _NunchakuCPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None):  # type: ignore[no-untyped-def]
        if fullname != "nunchaku._C":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _NunchakuCPatchLoader(spec.loader)  # type: ignore[assignment]
        return spec


def _install_nunchaku_c_patch_hook() -> None:
    # Patch immediately if already imported (rare).
    mod = sys.modules.get("nunchaku._C")
    if mod is not None:
        _patch_nunchaku_cuda_mempool_utils(mod)
        return
    for f in sys.meta_path:
        if isinstance(f, _NunchakuCPatchFinder):
            return
    sys.meta_path.insert(0, _NunchakuCPatchFinder())


_install_torch_patch_hook()
_install_nunchaku_c_patch_hook()
