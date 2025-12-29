import re
import torch
from typing import List
from src.utils.cache import empty_cache
from diffusers import ModelMixin
from mlx import nn as mx_nn
import mlx.core as mx
from src.utils.mlx import convert_dtype_to_mlx
from src.mlx.mixins.from_model_mixin import _flatten_leaf_arrays, _set_by_path
from src.quantize.dequant import is_quantized
from src.quantize.ggml_tensor import GGMLTensor

try:
    # Private helpers/constants from Diffusers used to detect whether group offloading
    # is enabled, and whether a particular module is controlled by a group-offload hook.
    from diffusers.hooks.group_offloading import (
        _is_group_offload_enabled as _hf_is_group_offload_enabled,
        _GROUP_OFFLOADING as _HF_GROUP_OFFLOADING,
    )
except Exception:  # pragma: no cover - defensive fallback for older Diffusers versions
    _hf_is_group_offload_enabled = None
    _HF_GROUP_OFFLOADING = None


def _module_has_group_offload_hook(module: torch.nn.Module) -> bool:
    """
    Return True if this *specific* module has a Diffusers group-offloading hook
    registered on it (as opposed to any of its children).
    """
    if _HF_GROUP_OFFLOADING is None:
        return False

    registry = getattr(module, "_diffusers_hook", None)
    if registry is None:
        return False

    get_hook = getattr(registry, "get_hook", None)
    if get_hook is None:
        return False

    try:
        return get_hook(_HF_GROUP_OFFLOADING) is not None
    except Exception:
        return False


def _move_module_to_device_excluding_group_offload(
    module: torch.nn.Module, device: torch.device
) -> None:
    """
    Recursively move a module tree to `device`, but *skip* any subtrees that are
    controlled by Diffusers group offloading.

    This lets us handle cases like `HunyuanImage3ForCausalMM`, where the transformer
    backbone is group-offloaded but surrounding layers (e.g. patch_embed / final_layer)
    should still be moved to the target device.
    """
    # If this module itself is group-offloaded, skip the entire subtree.
    if _module_has_group_offload_hook(module):
        return

    # Move parameters and buffers owned by this module.
    for param in module.parameters(recurse=False):
        if param is not None:
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)

    for buf in module.buffers(recurse=False):
        if buf is not None:
            buf.data = buf.data.to(device)

    # Recurse into children; group-offloaded children will early-return above.
    for child in module.children():
        _move_module_to_device_excluding_group_offload(child, device)


class ToMixin:
    """
    Mixin providing utilities to move Diffusers ModelMixin components
    to specified devices and data types, respecting model-specific settings.
    """

    def _parse_dtype(self, dtype: str | torch.dtype) -> torch.dtype:
        """
        Convert a string or torch.dtype into a torch.dtype.
        """
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            **{alias: torch.float16 for alias in ("float16", "fp16", "f16")},
            **{alias: torch.bfloat16 for alias in ("bfloat16", "bf16")},
            **{alias: torch.float32 for alias in ("float32", "fp32", "f32")},
            **{alias: torch.float64 for alias in ("float64", "fp64", "f64")},
            **{alias: torch.int8 for alias in ("int8", "i8")},
            **{alias: torch.uint8 for alias in ("uint8", "u8")},
        }
        key = dtype.lower() if isinstance(dtype, str) else dtype
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unsupported dtype: {dtype}")

    def check_quantized(
        self, module: ModelMixin | GGMLTensor | torch.nn.Parameter
    ) -> bool:
        """
        Check if the module is quantized.
        """
        if isinstance(module, GGMLTensor):
            return is_quantized(module)
        elif isinstance(module, torch.nn.Parameter):
            return is_quantized(module.data)
        elif isinstance(module, ModelMixin):
            for _, param in module.named_parameters():
                if is_quantized(param.data):
                    return True
            return False
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")

    def _is_scaled_module(self, module: ModelMixin) -> bool:
        """
        Heuristically detect whether a module uses scaled / quantized weights.

        We treat a module as "scaled" when:
          * it is explicitly detected as quantized via `check_quantized`, or
          * any *parameter* has a dtype that is not one of {fp16, bf16, fp32},
            including integer dtypes (e.g. int8) and newer float types (e.g. fp8).

        In those cases we skip blanket casting to avoid breaking quantized models.
        """
        # First, honour any explicit quantization markers we know about.
        try:
            if self.check_quantized(module):
                return True
        except Exception:
            # If quantization inspection fails for any reason, fall back to dtype checks.
            pass

        allowed_fp = {torch.float16, torch.bfloat16, torch.float32}

        # Only look at trainable parameters (weights); buffers often hold indices,
        # masks, etc. and should not by themselves mark a module as "scaled".
        for param in module.parameters():
            if param is None:
                continue
            # Non-floating parameters (e.g. int8) imply quantization/scaling.
            if not param.is_floating_point():
                return True
            
            if hasattr(param, "logical_dtype") and param.logical_dtype is not None:
                return True
            # Floating dtypes outside the standard trio (fp16/bf16/fp32) are treated
            # as scaled (e.g. fp8 variants, float64, etc.).

            if param.dtype not in allowed_fp:
                return True

        return False

    def to_dtype(
        self,
        module: ModelMixin,
        dtype: str | torch.dtype,
        # ---- new knobs -----------------------------------------------
        layerwise: bool = False,
        storage_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> ModelMixin:
        """
        Cast *either* uniformly (like `from_pretrained(torch_dtype=…)`)
        *or* via Diffusers' on-the-fly layer-wise casting.

        Parameters
        ----------
        dtype:
            Target dtype for a uniform cast **or** the *default* storage
            dtype when `layerwise=True`.
        layerwise:
            • False  → blanket cast; ignore `_skip_layerwise_casting_patterns`.
            • True   → call `enable_layerwise_casting`, honouring
                        `_skip_layerwise_casting_patterns`.
        storage_dtype / compute_dtype:
            Only used when `layerwise=True`.
        """
        target_dtype = self._parse_dtype(dtype)

        # ------------------------------------------------------------------
        # 1. Layer-wise casting – delegate entirely to Diffusers
        # ------------------------------------------------------------------
        if layerwise:
            skip_patterns = tuple(
                getattr(module, "_skip_layerwise_casting_patterns", []) or []
            )
            module.enable_layerwise_casting(
                storage_dtype=storage_dtype or target_dtype,
                compute_dtype=compute_dtype or torch.float32,
                skip_modules_pattern=skip_patterns,
            )
            return module

        # ------------------------------------------------------------------
        # 2. Casting behaviour based on whether the module is "scaled"
        # ------------------------------------------------------------------
        if self._is_scaled_module(module):
            # For scaled / quantized modules, only downcast true FP32 weights to the
            # requested dtype and leave all other dtypes (e.g. int8, fp8) untouched.
            for param in module.parameters():
                if param is None:
                    continue
                if (
                    param.is_floating_point()
                    and param.dtype is torch.float32
                    and param.dtype != target_dtype
                ):
                    param.data = param.data.to(dtype=target_dtype)
            return module

        # For regular (unscaled) modules, rely on the standard blanket `.to(...)`.
        module.to(dtype=target_dtype)

        return module

    def to_mlx_dtype(
        self,
        module: mx_nn.Module,
        dtype: str | torch.dtype | mx.Dtype,
    ) -> mx_nn.Module:
        """
        Cast all floating-point mlx arrays inside an `mlx.nn.Module` to the
        requested MLX dtype, preserving structure and honoring model-specific
        FP32 keep-lists when available.

        Parameters
        ----------
        module:
            The MLX module whose arrays should be cast.
        dtype:
            Target dtype. May be a string (e.g. "float16", "bfloat16"),
            a torch.dtype, or an MLX dtype; will be converted to an MLX dtype.

        Returns
        -------
        module: mx_nn.Module
            The same module instance, with floating-point arrays cast in-place.
        """
        target_dtype: mx.Dtype = convert_dtype_to_mlx(dtype)

        keep_fp32_patterns = tuple(getattr(module, "_keep_in_fp32_modules", []) or [])

        def _matches(patterns: tuple[str, ...], name: str) -> bool:
            return any(re.search(p, name) for p in patterns)

        def _is_float_dtype(d: mx.Dtype) -> bool:
            return d in (mx.float16, mx.bfloat16, mx.float32, mx.float64)

        leaves = _flatten_leaf_arrays(module)

        for name, arr in leaves.items():
            # Only cast floating types; skip integer/bool arrays
            if not _is_float_dtype(arr.dtype):
                continue

            wanted_dtype = (
                mx.float32 if _matches(keep_fp32_patterns, name) else target_dtype
            )

            if arr.dtype != wanted_dtype:
                _set_by_path(module, name, arr.astype(wanted_dtype))

        return module

    def to_device(
        self,
        *components: torch.nn.Module | mx_nn.Module,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Move specified modules (or defaults) to a device, then clear CUDA cache.

        If no components are provided, tries attributes:
        vae, text_encoder, transformer, scheduler, and self.helpers.values().
        """
        # Determine target device
        if device is None:
            device = getattr(self, "device", None) or torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)

        # Default components if none specified
        if not components:
            defaults = []
            for attr in ("vae", "text_encoder", "transformer", "scheduler"):
                comp = getattr(self, attr, None)
                if comp is not None:
                    defaults.append(comp)
            extras = getattr(self, "helpers", {}).values()
            components = (*defaults, *extras)

        # Move each to device
        for comp in components:
            if not hasattr(comp, "to"):
                continue

            # For torch modules that contain any group-offloaded submodules, we need
            # finer-grained control: we move only the non-offloaded parts while
            # leaving group-offloaded subtrees alone so their hooks can manage them.
            if (
                isinstance(comp, torch.nn.Module)
                and _hf_is_group_offload_enabled is not None
            ):
                try:
                    if _hf_is_group_offload_enabled(comp):
                        _move_module_to_device_excluding_group_offload(comp, device)
                        continue
                except Exception:
                    # If detection fails for any reason, fall back to the default behavior.
                    pass

            comp.to(device)

        # Free up any unused CUDA memory
        try:
            empty_cache()
        except Exception:
            pass
