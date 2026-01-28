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
import threading
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


def _looks_like_onnx_gpu_provider_failure(exc: BaseException) -> bool:
    """
    Best-effort filter so we only fallback to CPU for GPU/provider-related failures,
    not for genuine model/input errors (which would also fail on CPU).
    """

    msg = str(exc).lower()
    needles = (
        # Common EP names / hints
        "cudaexecutionprovider",
        "directmlexecutionprovider",
        "tensorrtexecutionprovider",
        "rocmexecutionprovider",
        "openvinoexecutionprovider",
        "coremlexecutionprovider",
        # CUDA / cuDNN / driver errors
        "cuda",
        "cudnn",
        "cublas",
        "cudart",
        "nvcuda",
        "driver",
        "dll load failed",
        "could not load",
        "failed to load",
        "failed to create",
        "failed to initialize",
        "initialization failed",
        "no cuda",
        "no gpu",
        "gpu",
        "device ordinal",
        "invalid device",
        "device not found",
        "out of memory",
        "insufficient memory",
        "memory allocation",
    )
    return any(n in msg for n in needles)


def _patch_onnxruntime_inference_session(ort_mod: Any) -> None:
    """
    Monkey-patch `onnxruntime.InferenceSession` so GPU EP failures (at init or run)
    automatically retry once on CPU. This avoids touching every call site.
    """

    if os.getenv("APEX_ORT_FALLBACK_TO_CPU", "1").strip() in ("0", "false", "False"):
        return

    try:
        RealInferenceSession = ort_mod.InferenceSession
    except Exception:
        return

    # Avoid double-patching.
    if getattr(RealInferenceSession, "__apex_patched__", False):
        return

    log = logging.getLogger("apex.sitecustomize.onnxruntime")
    _lock = threading.Lock()

    def _cpu_only_provider_list() -> list[str]:
        return ["CPUExecutionProvider"]

    def _requested_non_cpu_providers(providers: Any) -> bool:
        try:
            if providers is None:
                return True  # default behavior could pick GPU EPs
            return any(p != "CPUExecutionProvider" for p in providers)
        except Exception:
            return True

    class InferenceSession:  # noqa: N801 - keep public API name
        """
        Wrapper around ONNX Runtime's compiled `InferenceSession` with CPU fallback.
        """

        __apex_patched__ = True

        def __init__(  # type: ignore[no-untyped-def]
            self,
            path_or_bytes: Any,
            sess_options: Any = None,
            providers: Any = None,
            provider_options: Any = None,
            **kwargs: Any,
        ) -> None:
            self._apex_model_source = path_or_bytes
            self._apex_sess_options = sess_options
            self._apex_providers = providers
            self._apex_provider_options = provider_options
            self._apex_kwargs = kwargs

            self._apex_using_cpu = False
            self._apex_fallback_attempted = False
            self._apex_session = self._apex_create_session(allow_fallback=True)

        def _apex_create_session(self, allow_fallback: bool):  # type: ignore[no-untyped-def]
            try:
                return RealInferenceSession(
                    self._apex_model_source,
                    sess_options=self._apex_sess_options,
                    providers=self._apex_providers,
                    provider_options=self._apex_provider_options,
                    **self._apex_kwargs,
                )
            except Exception as e:
                if (
                    allow_fallback
                    and not self._apex_using_cpu
                    and not self._apex_fallback_attempted
                    and _requested_non_cpu_providers(self._apex_providers)
                    and _looks_like_onnx_gpu_provider_failure(e)
                ):
                    self._apex_fallback_attempted = True
                    self._apex_using_cpu = True
                    log.warning(
                        "onnxruntime InferenceSession init failed with GPU/EP error (%s); falling back to CPUExecutionProvider.",
                        str(e).strip(),
                    )
                    return RealInferenceSession(
                        self._apex_model_source,
                        sess_options=self._apex_sess_options,
                        providers=_cpu_only_provider_list(),
                        **self._apex_kwargs,
                    )
                raise

        def run(self, output_names, input_feed, run_options=None):  # type: ignore[no-untyped-def]
            try:
                if run_options is None:
                    return self._apex_session.run(output_names, input_feed)
                return self._apex_session.run(
                    output_names, input_feed, run_options=run_options
                )
            except Exception as e:
                # Retry exactly once on CPU if we were trying a non-CPU provider.
                if (
                    not self._apex_using_cpu
                    and not self._apex_fallback_attempted
                    and _looks_like_onnx_gpu_provider_failure(e)
                ):
                    with _lock:
                        if not self._apex_using_cpu and not self._apex_fallback_attempted:
                            self._apex_fallback_attempted = True
                            self._apex_using_cpu = True
                            log.warning(
                                "onnxruntime session.run failed with GPU/EP error (%s); recreating session on CPU and retrying once.",
                                str(e).strip(),
                            )
                            self._apex_session = RealInferenceSession(
                                self._apex_model_source,
                                sess_options=self._apex_sess_options,
                                providers=_cpu_only_provider_list(),
                                **self._apex_kwargs,
                            )

                    # Retry once (outside lock).
                    if run_options is None:
                        return self._apex_session.run(output_names, input_feed)
                    return self._apex_session.run(
                        output_names, input_feed, run_options=run_options
                    )
                raise

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            # Delegate all other attributes/methods to the real session.
            return getattr(self._apex_session, name)

    # Patch the primary public entrypoint.
    ort_mod.InferenceSession = InferenceSession  # type: ignore[assignment]


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


class _OnnxRuntimePatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader):
        self._wrapped = wrapped

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]
        self._wrapped.exec_module(module)  # type: ignore[misc]
        _patch_onnxruntime_inference_session(module)


class _OnnxRuntimePatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None):  # type: ignore[no-untyped-def]
        if fullname != "onnxruntime":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _OnnxRuntimePatchLoader(spec.loader)  # type: ignore[assignment]
        return spec


def _install_onnxruntime_patch_hook() -> None:
    # Patch immediately if already imported.
    ort_mod = sys.modules.get("onnxruntime")
    if ort_mod is not None:
        _patch_onnxruntime_inference_session(ort_mod)
        return
    # Otherwise install a one-shot import hook.
    for f in sys.meta_path:
        if isinstance(f, _OnnxRuntimePatchFinder):
            return
    sys.meta_path.insert(0, _OnnxRuntimePatchFinder())


_install_torch_patch_hook()
_install_nunchaku_c_patch_hook()
_install_onnxruntime_patch_hook()