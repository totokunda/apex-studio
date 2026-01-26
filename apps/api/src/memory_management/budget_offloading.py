from __future__ import annotations

"""
Budgeted VRAM offloading (Apex Studio).

This module is a clean-room *behavioral port* of the "budgets" offloading
mechanism from `mmgp/offload.py`, adapted to Apex Studio's "apply to one module"
API.

Important:
- Internal names are cleaned up for maintainability, but offload behavior is
  intentionally aligned with mmgp's logic (block detection, tied-weight handling,
  async prefetch stream usage, cache-empty heuristics).
- Public API remains stable for the rest of the Apex codebase:
  - `apply_budget_offloading`
  - `_maybe_remove_and_reapply_budget_offloading`
"""

import builtins
import functools
import gc
import os
import time
import enum
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from loguru import logger
import torch

from .group_offloading import _is_group_offload_enabled

ONE_MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Basic helpers (ported behavior)
# ---------------------------------------------------------------------------


def _is_budget_offload_enabled(module: torch.nn.Module) -> bool:
    return bool(getattr(module, "_apex_budget_offloading_enabled", False))


def _extract_num_suffix(name: str) -> tuple[str, int]:
    """
    Port of mmgp `_extract_num_from_str`.
    Returns (prefix, number) where `number` is the trailing integer suffix, or -1.
    """
    s = str(name or "")
    size = len(s)
    for i in range(size):
        if not s[-i - 1 :].isnumeric():
            if i == 0:
                return s, -1
            return s[: -i], int(s[-i:])
    return ("", -1) if size == 0 else ("", int(s))


def _matches_ignore(name: str, ignore_modules: Optional[Iterable[str]]) -> bool:
    if not ignore_modules:
        return False
    for ignore in ignore_modules:
        if not ignore:
            continue
        if name == ignore or name.startswith(f"{ignore}."):
            return True
    return False


def _device_total_bytes(device: torch.device) -> int:
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else torch.cuda.current_device()
            return int(torch.cuda.get_device_properties(idx).total_memory)
    except Exception:
        pass
    return 0


def _get_perc_reserved_mem_max(value: float = 0.0) -> float:
    """
    mmgp behavior:
    - reads env `perc_reserved_mem_max` if value <= 0
    - default ~0.5 on non-windows, ~0.4 on windows
    """
    if value <= 0:
        try:
            value = float(os.getenv("perc_reserved_mem_max", "0") or "0")
        except Exception:
            value = 0.0
    if value <= 0:
        value = 0.40 if os.name == "nt" else 0.5
    return float(value)


# ---------------------------------------------------------------------------
# Quantized / tied-weight helpers (ported behavior)
# ---------------------------------------------------------------------------


def _unwrap_quantized_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # mmgp: QTensor sometimes wraps the underlying tensor in `_data._data`
    if hasattr(tensor, "_data") and torch.is_tensor(getattr(tensor, "_data")):
        return getattr(tensor, "_data")
    return tensor


def _get_quantized_subtensors(param: Any) -> Optional[list[tuple[str, torch.Tensor]]]:
    """
    Port of mmgp `_get_quantized_subtensors`.
    Works for quanto QTensor-like objects that expose `get_quantized_subtensors()`.
    """
    getter = getattr(param, "get_quantized_subtensors", None)
    if getter is None:
        return None
    try:
        sub_tensors = getter()
    except Exception:
        return None
    if not sub_tensors:
        return None
    if isinstance(sub_tensors, dict):
        sub_tensors = list(sub_tensors.items())
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in sub_tensors:
        if tensor is None:
            continue
        if torch.is_tensor(tensor):
            out.append((str(name), tensor))
    return out or None


def _set_quantized_subtensors(param: Any, sub_tensors: Union[dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]) -> bool:
    setter = getattr(param, "set_quantized_subtensors", None)
    if setter is None:
        return False
    try:
        setter(sub_tensors)
        return True
    except Exception:
        return False


def _subtensors_nbytes(sub_tensors: list[tuple[str, torch.Tensor]]) -> int:
    return int(sum(torch.numel(t) * t.element_size() for _, t in sub_tensors))


def _subtensors_itemsize(sub_tensors: list[tuple[str, torch.Tensor]], fallback: int) -> int:
    sizes = [int(t.element_size()) for _, t in sub_tensors]
    return max(sizes) if sizes else int(fallback)


def _tensor_ref(param: Any) -> int:
    """
    Port of mmgp `_get_tensor_ref`.
    Uses the data_ptr of the first quantized subtensor if present.
    """
    sub_tensors = _get_quantized_subtensors(param)
    if sub_tensors:
        for _, t in sub_tensors:
            try:
                ref = int(t.data_ptr())
            finally:
                del sub_tensors
            return ref
        del sub_tensors
    try:
        return int(param.data_ptr())  # type: ignore[attr-defined]
    except Exception:
        # Fallback: Python identity (still stable within process)
        return id(param)


def _wrap_as_module_tensor(t: torch.Tensor, *, is_buffer: bool) -> torch.Tensor:
    """
    Ensure `t` is registered correctly on a module when assigned via setattr().

    Important:
    - Some models use custom `nn.Parameter` subclasses (e.g. FPScaledParameter) that
      cannot be wrapped again with `torch.nn.Parameter(...)` due to PyTorch's
      subclass+detach semantics. If it's already a Parameter, keep it.
    - For inference/offloading we enforce `requires_grad=False` to match the
      previous behaviour of wrapping with `nn.Parameter(..., requires_grad=False)`.
    """
    if is_buffer:
        # `torch.nn.Buffer` registers a tensor as a module buffer on assignment.
        return torch.nn.Buffer(t)

    if isinstance(t, torch.nn.Parameter):
        try:
            # Align with previous behaviour: always non-trainable in offload mode.
            if t.requires_grad:
                t.requires_grad_(False)
        except Exception:
            pass
        return t

    return torch.nn.Parameter(t, requires_grad=False)


# ---------------------------------------------------------------------------
# Reserved/pinned CPU memory (ported behavior; optional)
# ---------------------------------------------------------------------------

BIG_TENSOR_MAX_SIZE = 2**28  # 256MB
BIG_TENSOR_MIN_SIZE = 2**26  # 64MB
RESERVED_RAM_MIN_AVAILABLE = BIG_TENSOR_MAX_SIZE

_total_pinned_bytes: int = 0
_max_pinnable_bytes: int = 0


def _physical_memory_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def _max_reservable_memory_bytes(perc_reserved_mem_max: float) -> int:
    pm = _physical_memory_bytes()
    if pm <= 0:
        return 0
    return int(float(perc_reserved_mem_max) * pm)


def _move_to_pinned_tensor(source_tensor: torch.Tensor, big_tensor: torch.Tensor, offset: int, length: int) -> torch.Tensor:
    """
    Port of mmgp `_move_to_pinned_tensor`.
    Reinterprets the bytes of `source_tensor` onto a pinned uint8 slab slice.
    """
    dtype = source_tensor.dtype
    shape = source_tensor.shape
    if len(shape) > 0:
        t = source_tensor.view(torch.uint8)
        t = torch.reshape(t, (length,))
    else:
        # Preserve raw bytes for 0-dim tensors (scalar buffers like embed_scale).
        t = source_tensor.view(1).view(torch.uint8)
        t = torch.reshape(t, (length,))
    big_tensor[offset : offset + length] = t
    t = big_tensor[offset : offset + length]
    t = t.view(dtype)
    t = torch.reshape(t, shape)
    assert t.is_pinned()
    return t


def _force_load_buffer(buf: torch.Tensor) -> None:
    # mmgp forces a clone swap to avoid memory leak in some cases
    q = torch.nn.Buffer(buf.clone())
    torch.utils.swap_tensors(buf, q)
    del q


def _flush_caches() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    try:
        torch._C._host_emptyCache()  # type: ignore[attr-defined]
    except Exception:
        pass


def _pin_module_to_reserved_memory(
    model: torch.nn.Module,
    *,
    partial_pinning: bool,
    pinned_peft_lora: bool,
    perc_reserved_mem_max: float,
    verbose_level: int,
) -> None:
    """
    Port of mmgp `_pin_to_memory`, adapted to a live `torch.nn.Module`.
    This is optional and only used when `pin_cpu_memory=True` in Apex config.
    """
    global _total_pinned_bytes, _max_pinnable_bytes

    if _max_pinnable_bytes > 0 and _total_pinned_bytes >= _max_pinnable_bytes:
        return

    towers_names: list[str] = []
    if partial_pinning:
        towers_names, _ = _detect_main_towers(model)

    perc_reserved_mem_max = _get_perc_reserved_mem_max(perc_reserved_mem_max)
    max_reservable = _max_reservable_memory_bytes(perc_reserved_mem_max)

    # Gather params/buffers to pin (name -> (tensor-like, is_buffer))
    params_dict: dict[str, tuple[Any, bool]] = {}
    for k, sub_module in model.named_modules():
        include = True
        if partial_pinning:
            include = any(k.startswith(pre) for pre in towers_names) if towers_names else False
        if include and (not pinned_peft_lora) and ".lora_" in k:
            include = False
        if include:
            params_dict.update({f"{k}.{n}": (p, False) for n, p in sub_module.named_parameters(recurse=False)})
            params_dict.update({f"{k}.{n}": (b, True) for n, b in sub_module.named_buffers(recurse=False)})

    if not params_dict:
        return

    # Plan slabs & detect tied weights
    current_big_tensor_size = 0
    big_tensor_no = 0
    big_tensors_sizes: list[int] = []
    tensor_map_indexes: list[tuple[int, int, int]] = []

    ref_cache: dict[int, tuple[str, int]] = {}
    tied_weights: dict[str, str] = {}

    for n, (p, _) in params_dict.items():
        if p is None:
            continue
        ref = _tensor_ref(p)
        match = ref_cache.get(ref)
        if match is not None:
            match_name, _match_size = match
            tied_weights[n] = match_name
            continue

        sub_tensors = _get_quantized_subtensors(p)
        if sub_tensors:
            if builtins.all(t.is_pinned() for _, t in sub_tensors):
                params_dict[n] = (None, False)
                del sub_tensors
                continue
            length = _subtensors_nbytes(sub_tensors)
            itemsize = _subtensors_itemsize(sub_tensors, getattr(p, "dtype", torch.uint8).itemsize)
            del sub_tensors
        else:
            if hasattr(p, "data") and torch.is_tensor(p.data) and p.data.is_pinned():
                params_dict[n] = (None, False)
                continue
            length = int(torch.numel(p.data) * p.data.element_size()) if hasattr(p, "data") else 0
            itemsize = int(getattr(p.data, "dtype", torch.uint8).itemsize) if hasattr(p, "data") else 1

        ref_cache[ref] = (n, length)
        if current_big_tensor_size + length > BIG_TENSOR_MAX_SIZE and current_big_tensor_size != 0:
            big_tensors_sizes.append(current_big_tensor_size)
            current_big_tensor_size = 0
            big_tensor_no += 1

        if current_big_tensor_size % itemsize:
            current_big_tensor_size += itemsize - (current_big_tensor_size % itemsize)
        tensor_map_indexes.append((big_tensor_no, current_big_tensor_size, length))
        current_big_tensor_size += length

    big_tensors_sizes.append(current_big_tensor_size)

    # Verify we can allocate at least a small pinned tensor 
    gc.collect()
    dummy_pinned_tensor = None
    try:
        dummy_pinned_tensor = torch.empty(RESERVED_RAM_MIN_AVAILABLE, dtype=torch.uint8, pin_memory=True, device="cpu")
    except Exception:
        return

    big_tensors: list[torch.Tensor] = []
    total = 0
    failed_planned_allocation = False

    last_allocated_big_tensor = -1
    tensor_no = 0
    for n, (p, is_buffer) in params_dict.items():
        if p is None:
            continue
        q_name = tied_weights.get(n)
        if q_name is not None:
            q, _ = params_dict[q_name]
            if q is None:
                continue
            sub_tensors = _get_quantized_subtensors(q)
            if sub_tensors:
                sub_map = {name: tensor for name, tensor in sub_tensors}
                _set_quantized_subtensors(p, sub_map)
                del sub_map, sub_tensors
            else:
                p.data = q.data
                assert p.data.is_pinned()
            continue

        slab_no, offset, _length = tensor_map_indexes[tensor_no]
        if last_allocated_big_tensor < slab_no:
            last_allocated_big_tensor += 1
            size = max(int(big_tensors_sizes[last_allocated_big_tensor]), BIG_TENSOR_MIN_SIZE)
            try:
                if max_reservable > 0 and ((_total_pinned_bytes + total + size) >= max_reservable):
                    dummy_pinned_tensor = None
                    failed_planned_allocation = True
                    _max_pinnable_bytes = _total_pinned_bytes + total
                    break
                current_slab = torch.empty(size, dtype=torch.uint8, pin_memory=True, device="cpu")
                big_tensors.append(current_slab)
            except Exception:
                dummy_pinned_tensor = None
                failed_planned_allocation = True
                _max_pinnable_bytes = _total_pinned_bytes + total
                _flush_caches()
                break
            total += size

        current_slab = big_tensors[slab_no]
        if is_buffer:
            try:
                _force_load_buffer(p)
            except Exception:
                pass

        sub_tensors = _get_quantized_subtensors(p)
        if sub_tensors:
            sub_offset = offset
            new_subs: dict[str, torch.Tensor] = {}
            for sub_name, tensor in sub_tensors:
                length = int(torch.numel(tensor) * tensor.element_size())
                new_subs[sub_name] = _move_to_pinned_tensor(tensor, current_slab, sub_offset, length)
                sub_offset += length
            _set_quantized_subtensors(p, new_subs)
            del new_subs, sub_tensors
        else:
            length = int(torch.numel(p.data) * p.data.element_size())
            p.data = _move_to_pinned_tensor(p.data, current_slab, offset, length)

        tensor_no += 1
    
    del dummy_pinned_tensor

    model._pinned_bytes = total  # type: ignore[attr-defined]
    _total_pinned_bytes += total
    model._already_pinned = True  # type: ignore[attr-defined]

    if verbose_level >= 2 and (partial_pinning or failed_planned_allocation):
        # mmgp prints; we keep very verbose-only to avoid noisy server logs
        pass


def _detect_main_towers(model: torch.nn.Module, min_floors: int = 5) -> tuple[list[str], list[torch.nn.Module]]:
    """
    Port of mmgp `_detect_main_towers`.
    Detects repeating numeric "tower" structures (ModuleList-ish stacks) used to
    plan preloading.
    """
    cur_blocks_prefix: Optional[str] = None
    towers_modules: list[torch.nn.Module] = []
    towers_names: list[str] = []
    floors_modules: list[torch.nn.Module] = []
    tower_name: Optional[str] = None
    cur_blocks_seq = -1

    for submodule_name, submodule in model.named_modules():
        if submodule_name == "":
            continue

        if cur_blocks_prefix is not None:
            if submodule_name.startswith(cur_blocks_prefix):
                depth_prefix = cur_blocks_prefix.split(".")
                depth_name = submodule_name.split(".")
                level = depth_name[len(depth_prefix) - 1]
                _pre, num = _extract_num_suffix(level)
                if num != cur_blocks_seq:
                    floors_modules.append(submodule)
                cur_blocks_seq = num
            else:
                if len(floors_modules) >= min_floors and tower_name is not None:
                    towers_modules += floors_modules
                    towers_names.append(tower_name)
                tower_name = None
                floors_modules = []
                cur_blocks_prefix, cur_blocks_seq = None, -1

        if cur_blocks_prefix is None:
            pre, num = _extract_num_suffix(submodule_name)
            if isinstance(submodule, torch.nn.ModuleList):
                cur_blocks_prefix, cur_blocks_seq = pre + ".", -1
                tower_name = submodule_name + "."
            elif num >= 0:
                cur_blocks_prefix, cur_blocks_seq = pre, num
                tower_name = submodule_name[:-1]
                floors_modules.append(submodule)

    if len(floors_modules) >= min_floors and tower_name is not None:
        towers_modules += floors_modules
        towers_names.append(tower_name)
    return towers_names, towers_modules


# ---------------------------------------------------------------------------
# Core offloader (mmgp-like behavior for a single module)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BlockParam:
    parent_module: torch.nn.Module
    attr_name: str
    cpu_obj: Any  # original Parameter/Buffer object stored on CPU
    is_buffer: bool
    tied_to: Optional[tuple[torch.nn.Module, str]] = None  # (module, attr_name)
    is_lora: bool = False


class _HfHook:
    """
    mmgp installs a fake accelerate hook so `_execution_device` probes think "cuda".
    We keep the minimal surface area.
    """

    def __init__(self):
        self.execution_device = "cuda"

    def init_hook(self, module: torch.nn.Module):
        return module

    def detach_hook(self, module: torch.nn.Module):
        return module


class BudgetOffloader:
    """
    Apex wrapper around a mmgp-style budgeted offloader for a single module.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        onload_device: torch.device,
        *,
        offload_device: torch.device,
        block_modules: Optional[List[str]] = None,
        ignore_modules: Optional[List[str]] = None,
        budget_mb: Optional[Union[int, str]] = None,
        async_transfers: bool = True,
        prefetch: bool = True,  # kept for API; mmgp's plan always prefetches next when async
        pin_cpu_memory: bool = False,
        vram_safety_coefficient: float = 0.8,
        offload_after_forward: bool = False,
        # Additional mmgp-style knobs (optional; not wired by current Apex config)
        partial_pinning: bool = False,
        pinned_peft_lora: bool = False,
        perc_reserved_mem_max: float = 0.0,
        verbose_level: int = -1,
        compile: bool = False,
        compile_mode: str = "default",
        model_id: str = "transformer",
    ) -> None:
        self.module = module
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.block_modules = block_modules or []
        self.ignore_modules = ignore_modules or []
        self.async_transfers = bool(async_transfers)
        self.prefetch = bool(prefetch)
        self.pin_cpu_memory = bool(pin_cpu_memory)
        self.vram_safety_coefficient = float(vram_safety_coefficient)
        self.offload_after_forward = bool(offload_after_forward)
        self.partial_pinning = bool(partial_pinning)
        self.pinned_peft_lora = bool(pinned_peft_lora)
        self.perc_reserved_mem_max = float(perc_reserved_mem_max)
        self.verbose_level = int(verbose_level)
        self.compile = bool(compile)
        self.compile_mode = str(compile_mode or "default")

        if not (0.0 < self.vram_safety_coefficient < 1.0):
            raise ValueError("vram_safety_coefficient must be a float between 0 and 1 (exclusive)")

        self._budget_mb = budget_mb
        # mmgp uses model ids (e.g. "transformer", "vae"); keep for parity.
        self._model_id = str(model_id or "transformer")

        # mmgp-like internal state
        self.blocks_of_modules: dict[str, list[_BlockParam]] = {}
        self.blocks_of_modules_sizes: dict[str, int] = {}
        self.loaded_blocks: dict[str, Optional[str]] = {self._model_id: None}
        self.prev_blocks_names: dict[str, Optional[str]] = {}
        self.next_blocks_names: dict[str, Optional[str]] = {}
        self.preloaded_blocks_per_model: dict[str, dict[str, int]] = {self._model_id: {}}

        self._parameters_ref: dict[int, tuple[torch.nn.Module, str]] = {}
        # Fast lookup for restoring a module's params/buffers to device at call time.
        # Keyed by (id(parent_module), attr_name) -> _BlockParam.
        self._block_param_index: dict[tuple[int, str], _BlockParam] = {}
        self._type_wrappers: dict[type, Any] = {}
        self._loras_on_gpu: bool = False

        self.device_mem_capacity = _device_total_bytes(torch.device("cuda"))
        self._last_reserved_mem_check: float = 0.0
        self.any_compiled_module = bool(self.compile)

        # Device-move guards to prevent accidental `.to()/cuda()/cpu()` calls
        # from eagerly placing budget-managed weights on the accelerator.
        # Stored as id(module) -> dict of original callables.
        self._device_move_guard_originals: Dict[int, Dict[str, Any]] = {}

        # Runtime safety: forward-pre hooks that ensure a module's weights are on
        # the correct device *right before it runs*. This covers:
        # - modules whose `forward` got replaced after we wrapped it
        # - dynamically-created submodules after offloading was applied
        # - edge cases where tied/shared weights get moved by other mechanisms
        self._ensure_on_call_device_pre_hook_handles: list[Any] = []
        self._ensure_on_call_device_hooked_module_ids: set[int] = set()

        # Streams (mmgp always creates both)
        self.default_stream = torch.cuda.default_stream(torch.device("cuda")) if torch.cuda.is_available() else None
        self._transfer_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Parse budget exactly like mmgp (`get_parsed_budget`)
        self._budget_bytes = self._parse_budget(budget_mb)


        # Ensure CPU baseline
        try:
            self.module.to(self.offload_device).eval()
        except Exception:
            pass
        for p in self.module.parameters():
            try:
                p.requires_grad_(False)
            except Exception:
                pass

        # Optional reserved pinned-memory pinning (mapped from pin_cpu_memory)
        if self.pin_cpu_memory:
            try:
                if not hasattr(self.module, "_already_pinned"):
                    _pin_module_to_reserved_memory(
                        self.module,
                        partial_pinning=self.partial_pinning,
                        pinned_peft_lora=self.pinned_peft_lora,
                        perc_reserved_mem_max=self.perc_reserved_mem_max,
                        verbose_level=self.verbose_level,
                    )
            except Exception:
                # Best-effort only; offloading still works without pinning
                pass
        
        

        # Build blocks, install hooks, tune preloading plan
        self._build_and_hook()
        self._tune_preloading()

        # Optional: wrap root forward to offload after each call
        if self.offload_after_forward:
            self._wrap_root_forward_for_offload_after()

    def _install_device_move_guards(self) -> None:
        """
        Soft-protect against accidental device moves via `.to()/cuda()/cpu()`.

        - Device moves are ignored (no-op) and logged as a warning.
        - Dtype-only `.to(dtype=...)` is allowed to pass through (user requested).
        """
        if getattr(self.module, "_apex_budget_device_move_guards", False):
            return

        def _make_guarded_to(orig_to):
            def _guarded_to(mod, *args, **kwargs):
                requested_device = None
                requested_dtype = kwargs.get("dtype", None)

                # Handle kwargs device.
                if "device" in kwargs and kwargs.get("device", None) is not None:
                    try:
                        requested_device = torch.device(kwargs["device"])
                    except Exception:
                        requested_device = kwargs["device"]

                # Handle positional variants (best-effort, mirrors common PyTorch patterns).
                if args:
                    a0 = args[0]
                    try:
                        if torch.is_tensor(a0):
                            requested_device = a0.device
                            if requested_dtype is None:
                                requested_dtype = a0.dtype
                        elif isinstance(a0, torch.dtype):
                            if requested_dtype is None:
                                requested_dtype = a0
                        else:
                            # Might be a device-like (str/int/torch.device)
                            try:
                                requested_device = torch.device(a0)
                            except Exception:
                                pass
                            # Common: to(device, dtype)
                            if len(args) >= 2 and requested_dtype is None:
                                a1 = args[1]
                                if isinstance(a1, torch.dtype):
                                    requested_dtype = a1
                    except Exception:
                        # If parsing fails, fall back to original call.
                        return orig_to(*args, **kwargs)

                # If a device move was requested, ignore it but keep dtype if present.
                if requested_device is not None:
                    try:
                        logger.warning(
                            "Budget offloading is enabled; ignoring `.to(device=...)` on a budget-managed module "
                            f"({type(mod)}). Use engine `to_device(...)` (updates the budget manager) instead."
                        )
                    except Exception:
                        pass

                    if requested_dtype is not None:
                        try:
                            new_kwargs = dict(kwargs)
                            new_kwargs.pop("device", None)
                            # Avoid passing a tensor/device positional arg; use explicit dtype-only.
                            return orig_to(dtype=requested_dtype, **new_kwargs)
                        except Exception:
                            # If dtype-only call fails, degrade to no-op.
                            return mod
                    return mod

                # No device move detected: pass through (dtype-only allowed).
                return orig_to(*args, **kwargs)

            return _guarded_to

        def _make_guarded_device_call(name: str):
            def _guarded(mod, *args, **kwargs):
                try:
                    logger.warning(
                        f"Budget offloading is enabled; ignoring `.{name}()` on a budget-managed module "
                        f"({type(mod)}). Use engine `to_device(...)` instead."
                    )
                except Exception:
                    pass
                return mod

            return _guarded

        for _name, sub in self.module.named_modules():
            if getattr(sub, "_apex_budget_device_move_guarded", False):
                continue

            try:
                # Record whether the instance already had overrides; if not, we
                # restore by deleting our instance attributes to fall back to the
                # class-level implementation.
                originals: Dict[str, Any] = {
                    "had_to": "to" in getattr(sub, "__dict__", {}),
                    "had_cuda": "cuda" in getattr(sub, "__dict__", {}),
                    "had_cpu": "cpu" in getattr(sub, "__dict__", {}),
                    "to": getattr(sub, "__dict__", {}).get("to", None),
                    "cuda": getattr(sub, "__dict__", {}).get("cuda", None),
                    "cpu": getattr(sub, "__dict__", {}).get("cpu", None),
                    # Also keep the resolved callables for wrapping.
                    "_resolved_to": getattr(sub, "to", None),
                    "_resolved_cuda": getattr(sub, "cuda", None),
                    "_resolved_cpu": getattr(sub, "cpu", None),
                }
                self._device_move_guard_originals[id(sub)] = originals

                if callable(originals.get("_resolved_to")):
                    sub.to = types.MethodType(  # type: ignore[assignment]
                        _make_guarded_to(originals["_resolved_to"]), sub
                    )
                if callable(originals.get("_resolved_cuda")):
                    sub.cuda = types.MethodType(_make_guarded_device_call("cuda"), sub)  # type: ignore[assignment]
                if callable(originals.get("_resolved_cpu")):
                    sub.cpu = types.MethodType(_make_guarded_device_call("cpu"), sub)  # type: ignore[assignment]

                setattr(sub, "_apex_budget_device_move_guarded", True)
            except Exception:
                # Best-effort only.
                continue

        setattr(self.module, "_apex_budget_device_move_guards", True)

    def _remove_device_move_guards(self) -> None:
        if not getattr(self.module, "_apex_budget_device_move_guards", False):
            return
        for _name, sub in self.module.named_modules():
            if not getattr(sub, "_apex_budget_device_move_guarded", False):
                continue
            orig = self._device_move_guard_originals.get(id(sub))
            try:
                if orig:
                    # Restore by removing our instance attributes if there were
                    # no instance-level overrides before.
                    if not orig.get("had_to", False):
                        if "to" in getattr(sub, "__dict__", {}):
                            delattr(sub, "to")
                    else:
                        setattr(sub, "to", orig.get("to"))

                    if not orig.get("had_cuda", False):
                        if "cuda" in getattr(sub, "__dict__", {}):
                            delattr(sub, "cuda")
                    else:
                        setattr(sub, "cuda", orig.get("cuda"))

                    if not orig.get("had_cpu", False):
                        if "cpu" in getattr(sub, "__dict__", {}):
                            delattr(sub, "cpu")
                    else:
                        setattr(sub, "cpu", orig.get("cpu"))
            except Exception:
                pass
            try:
                delattr(sub, "_apex_budget_device_move_guarded")
            except Exception:
                pass
        try:
            delattr(self.module, "_apex_budget_device_move_guards")
        except Exception:
            pass
        self._device_move_guard_originals.clear()

    def _install_ensure_on_call_device_pre_hooks(self) -> None:
        """
        Install forward-pre hooks on all (current) submodules that:
        - load required blocks (if budget metadata exists)
        - guarantee params/buffers are on the call device (typically CUDA)

        This is deliberately redundant with the forward wrappers: it is a
        correctness backstop that continues to work even if user code replaces
        `forward` methods or dynamically adds submodules after hook installation.
        """
        if not torch.cuda.is_available():
            return
        if getattr(self.module, "_apex_budget_ensure_on_call_device_hooks", False):
            return

        def _register_one(m: torch.nn.Module) -> None:
            mid = id(m)
            if mid in self._ensure_on_call_device_hooked_module_ids:
                return

            def _ensure_new_children(mod: torch.nn.Module) -> None:
                # Dynamically register for newly-attached children. We keep this
                # shallow (direct children only); recursion is handled naturally
                # when those parents execute.
                try:
                    for _child_name, child in mod.named_children():
                        if id(child) not in self._ensure_on_call_device_hooked_module_ids:
                            _register_one(child)
                except Exception:
                    return

            def _pre_hook(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):  # noqa: ARG001
                try:
                    if torch.compiler.is_compiling():
                        return
                except Exception:
                    pass

                _ensure_new_children(mod)

                # Ensure correct block residency if budget metadata exists.
                try:
                    self._pre_check(mod)
                except Exception:
                    pass

                # Always enforce device correctness (this is the core guarantee).
                try:
                    self._ensure_params_on_call_device(mod, args, kwargs)
                except Exception:
                    pass

            def _pre_hook_no_kwargs(mod: torch.nn.Module, args: tuple[Any, ...]):  # noqa: ARG001
                return _pre_hook(mod, args, {})

            try:
                handle = m.register_forward_pre_hook(_pre_hook, with_kwargs=True)  # type: ignore[call-arg]
            except Exception:
                try:
                    handle = m.register_forward_pre_hook(_pre_hook_no_kwargs)
                except Exception:
                    return

            self._ensure_on_call_device_pre_hook_handles.append(handle)
            self._ensure_on_call_device_hooked_module_ids.add(mid)

        for _name, sub in self.module.named_modules():
            _register_one(sub)

        setattr(self.module, "_apex_budget_ensure_on_call_device_hooks", True)

    def _remove_ensure_on_call_device_pre_hooks(self) -> None:
        if not getattr(self.module, "_apex_budget_ensure_on_call_device_hooks", False):
            return
        for h in list(self._ensure_on_call_device_pre_hook_handles):
            try:
                h.remove()
            except Exception:
                pass
        self._ensure_on_call_device_pre_hook_handles.clear()
        self._ensure_on_call_device_hooked_module_ids.clear()
        try:
            delattr(self.module, "_apex_budget_ensure_on_call_device_hooks")
        except Exception:
            pass

    def _build_block_param_index(self) -> None:
        if self._block_param_index:
            return
        # Note: a given (module, attr) can appear multiple times via tied weights.
        # We only need one entry to restore CPU->GPU correctly.
        for params in self.blocks_of_modules.values():
            for bp in params:
                self._block_param_index.setdefault((id(bp.parent_module), bp.attr_name), bp)

    def _infer_call_device(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[torch.device]:
        """
        Best-effort: find the first Tensor device in args/kwargs.
        Keep this shallow for performance (common case: first arg is `x`).
        """

        def _probe(obj: Any) -> Optional[torch.device]:
            if torch.is_tensor(obj):
                return obj.device
            if isinstance(obj, (list, tuple)) and obj:
                for item in obj[:4]:
                    d = _probe(item)
                    if d is not None:
                        return d
            if isinstance(obj, dict) and obj:
                for item in list(obj.values())[:4]:
                    d = _probe(item)
                    if d is not None:
                        return d
            return None

        for a in args:
            d = _probe(a)
            if d is not None:
                return d
        for v in kwargs.values():
            d = _probe(v)
            if d is not None:
                return d
        return None

    @torch.compiler.disable()
    def _ensure_params_on_call_device(self, target_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """
        Runtime safety: ensure the target module's params/buffers live on the same
        device as the call inputs (typically CUDA) right before executing.

        This protects against accidental CPU-offloaded weights being invoked with
        CUDA inputs (device-mismatch crashes).
        """
        if not torch.cuda.is_available():
            return
        call_device = self._infer_call_device(args, kwargs)
        if call_device is None or call_device.type != "cuda":
            return

        # Fast path: if everything is already on the call device, do nothing.
        try:
            for t in target_module.parameters(recurse=False):
                if torch.is_tensor(t) and t.device != call_device:
                    raise StopIteration
            for t in target_module.buffers(recurse=False):
                if torch.is_tensor(t) and t.device != call_device:
                    raise StopIteration
            return
        except StopIteration:
            pass
        except Exception:
            # If inspection fails, fall back to attempting to move via index below.
            pass

        self._build_block_param_index()

        stream = torch.cuda.current_stream(device=call_device)
        with torch.cuda.stream(stream):
            # Parameters
            for name, cur in target_module.named_parameters(recurse=False):
                if cur is None or (torch.is_tensor(cur) and cur.device == call_device):
                    continue
                bp = self._block_param_index.get((id(target_module), name))
                src = bp.cpu_obj if bp is not None else cur
                q = src.to(call_device, non_blocking=True)
                q = _wrap_as_module_tensor(q, is_buffer=False)
                setattr(target_module, name, q)
                if bp is not None and bp.tied_to is not None:
                    setattr(bp.tied_to[0], bp.tied_to[1], q)

            # Buffers
            for name, cur in target_module.named_buffers(recurse=False):
                if cur is None or (torch.is_tensor(cur) and cur.device == call_device):
                    continue
                bp = self._block_param_index.get((id(target_module), name))
                src = bp.cpu_obj if bp is not None else cur
                q = src.to(call_device, non_blocking=True)
                q = _wrap_as_module_tensor(q, is_buffer=True)
                setattr(target_module, name, q)

    # -------- budget parsing / heuristics --------

    def _parse_budget(self, b: Optional[Union[int, str]]) -> int:
        if b is None:
            return 0
        if isinstance(b, str) and b.endswith("%"):
            # mmgp: "0.8%" means 0.8 * VRAM capacity (coefficient, not percent)
            try:
                coef = float(b[:-1])
            except Exception:
                return 0
            return int(coef * float(self.device_mem_capacity))
        try:
            return int(float(b) * ONE_MB)  # MB
        except Exception:
            return 0

    def _ready_to_check_mem(self) -> bool:
        if self.any_compiled_module:
            return False
        if not torch.cuda.is_available():
            return False
        cur = time.time()
        if (cur - self._last_reserved_mem_check) < 0.200:
            return False
        self._last_reserved_mem_check = cur
        return True

    def _empty_cache_if_needed(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            mem_reserved = torch.cuda.memory_reserved()
            if mem_reserved < 0.9 * float(self.device_mem_capacity):
                return
            mem_allocated = torch.cuda.memory_allocated()
            if mem_allocated <= 0.70 * mem_reserved:
                torch.cuda.empty_cache()
        except Exception:
            return

    # -------- block construction --------

    def _add_module_to_blocks(
        self,
        *,
        blocks_name: Optional[str],
        submodule: torch.nn.Module,
        prev_block_name: Optional[str],
        submodule_path: str,
    ) -> int:
        """
        Port of mmgp `add_module_to_blocks` for the single model_id.
        """
        if blocks_name is not None and ".lora_" in blocks_name:
            blocks_name = None

        model_id = self._model_id
        entry_name = model_id if blocks_name is None else f"{model_id}/{blocks_name}"

        blocks_params = self.blocks_of_modules.get(entry_name)
        blocks_params_size = self.blocks_of_modules_sizes.get(entry_name, 0)
        if blocks_params is None:
            blocks_params = []
            self.blocks_of_modules[entry_name] = blocks_params
            if blocks_name is not None:
                prev_entry = None if prev_block_name is None else f"{model_id}/{prev_block_name}"
                self.prev_blocks_names[entry_name] = prev_entry
                if prev_entry is not None:
                    self.next_blocks_names[prev_entry] = entry_name

        for n, p in submodule.named_parameters(recurse=False):
            ref = _tensor_ref(p)
            tied_param = self._parameters_ref.get(ref)
            is_lora = "lora" in submodule_path.lower()
            blocks_params.append(
                _BlockParam(submodule, n, p, False, tied_param, is_lora=is_lora)
            )
            sub_tensors = _get_quantized_subtensors(p)
            if sub_tensors:
                param_size = _subtensors_nbytes(sub_tensors)
                del sub_tensors
            else:
                param_size = int(torch.numel(p.data) * p.data.element_size())
            if tied_param is None:
                blocks_params_size += int(param_size)
                self._parameters_ref[ref] = (submodule, n)

        for n, b in submodule.named_buffers(recurse=False):
            is_lora = "lora" in submodule_path.lower()
            blocks_params.append(_BlockParam(submodule, n, b, True, None, is_lora=is_lora))
            try:
                blocks_params_size += int(b.data.nbytes)
            except Exception:
                blocks_params_size += int(torch.numel(b.data) * b.data.element_size())

        self.blocks_of_modules_sizes[entry_name] = int(blocks_params_size)
        return int(blocks_params_size)

    def _build_and_hook(self) -> None:
        """
        Port of mmgp scanning logic from `all()` that:
        - builds block membership
        - installs forward wrappers that ensure the current block is loaded
        """
        model_id = self._model_id

        current_budget = int(self._budget_bytes or 0)
        self.loaded_blocks[model_id] = None

        cur_blocks_prefix: Optional[str] = None
        prev_blocks_name: Optional[str] = None
        cur_blocks_name: Optional[str] = None
        cur_blocks_seq: int = -1
        is_mod_seq: bool = False

        # Root entry always exists to hold the "base" block size
        self.blocks_of_modules.setdefault(model_id, [])
        self.blocks_of_modules_sizes.setdefault(model_id, 0)

        for submodule_name, submodule in self.module.named_modules():
            # IMPORTANT:
            # Even if a module is "ignored" for block detection/hooking, its parameters
            # still need to be part of the load plan. Otherwise the initial CPU baseline
            # (`self.module.to(offload_device)`) would move it to CPU and it would never
            # be brought back to GPU, causing device-mismatch crashes at runtime.
            if _matches_ignore(submodule_name, self.ignore_modules):
                self._add_module_to_blocks(
                    blocks_name=None,
                    submodule=submodule,
                    prev_block_name=None,
                    submodule_path=submodule_name,
                )
                continue

            # Fake accelerate hook so some pipelines think device is "cuda"
            if not hasattr(submodule, "_hf_hook"):
                try:
                    setattr(submodule, "_hf_hook", _HfHook())
                except Exception:
                    pass

            # mmgp: detect blocks only when budget > 0 and for non-root names
            if current_budget > 0 and len(submodule_name) > 0:
                if cur_blocks_prefix is not None:
                    if submodule_name.startswith(cur_blocks_prefix):
                        depth_prefix = cur_blocks_prefix.split(".")
                        depth_name = submodule_name.split(".")
                        level = depth_name[len(depth_prefix) - 1]
                        _pre, num = _extract_num_suffix(level)
                        if num != cur_blocks_seq and not (is_mod_seq and cur_blocks_seq >= 0):
                            prev_blocks_name = cur_blocks_name
                            cur_blocks_name = cur_blocks_prefix + str(num)
                        cur_blocks_seq = num
                    else:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_name, cur_blocks_seq, is_mod_seq = (
                            None,
                            None,
                            None,
                            -1,
                            False,
                        )

                if cur_blocks_prefix is None:
                    pre, num = _extract_num_suffix(submodule_name)
                    if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = (
                            pre + ".",
                            None,
                            -1,
                            isinstance(submodule, torch.nn.Sequential),
                        )
                    elif num >= 0:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre, None, num, False
                        cur_blocks_name = submodule_name

            top_submodule = len(submodule_name.split(".")) == 1
            offload_hooks = getattr(submodule, "_offload_hooks", []) if hasattr(submodule, "_offload_hooks") else []
            assert top_submodule or len(offload_hooks) == 0, "custom offload hooks can only be set at the top of the module"
            method_names = ["forward"] + list(offload_hooks)

            for method_name in method_names:
                if not hasattr(submodule, method_name):
                    continue
                previous_method = getattr(submodule, method_name)
                if not callable(previous_method):
                    continue

                # In mmgp, top-level modules without a blocks_name get a special wrapper
                if top_submodule and cur_blocks_name is None:
                    self._hook_change_module(submodule, previous_method, method_name)
                else:
                    # mmgp uses different wrapper paths for compilation; we keep both
                    if self.compile and torch.cuda.is_available():
                        self._hook_check_load_into_gpu(submodule, cur_blocks_name, previous_method)
                    else:
                        self._hook_check_load_into_gpu_default(submodule, cur_blocks_name, previous_method)

            # Always add parameters/buffers to the block plan
            self._add_module_to_blocks(
                blocks_name=cur_blocks_name,
                submodule=submodule,
                prev_block_name=prev_blocks_name,
                submodule_path=submodule_name,
            )

        # Ensure next pointer exists for the last entry if any
        # (mmgp uses next_blocks_names map sparsely; OK to omit)

    # -------- hooking (ported behavior) --------

    @torch._dynamo.disable
    def _pre_check(self, module: torch.nn.Module) -> None:
        model_id = getattr(module, "_apex_mm_model_id", None)
        blocks_name = getattr(module, "_apex_mm_blocks_name", None)
        if not model_id:
            model_id = self._model_id
        self.ensure_model_loaded(model_id)
        # If LoRA adapters are active, ensure LoRA weights are on GPU before forward.
        # This prevents LoRA weights from occupying VRAM when adapters are inactive.
        if self._has_active_adapters() and not self._loras_on_gpu:
            try:
                self._load_lora_params_to_gpu()
                self._loras_on_gpu = True
            except Exception:
                # Best-effort: if this fails, the subsequent forward may fail.
                self._loras_on_gpu = False
        if blocks_name is None:
            if self._ready_to_check_mem():
                self._empty_cache_if_needed()
            return

        if (
            blocks_name != self.loaded_blocks.get(model_id)
            and blocks_name not in self.preloaded_blocks_per_model.get(model_id, {})
        ):
            self.gpu_load_blocks(model_id, blocks_name)

    def _get_wrapper_for_type(self, mod_cls: type) -> Any:
        fn = self._type_wrappers.get(mod_cls)
        if fn is not None:
            return fn

        fname = f"_apex_mm_wrap_{mod_cls.__module__.replace('.', '_')}_{mod_cls.__name__}"
        src = f"""
def {fname}(module, *args, **kwargs):
    _ = __TYPE_CONST
    mgr = module._apex_mm_manager
    mgr._pre_check(module)
    mgr._ensure_params_on_call_device(module, args, kwargs)
    return module._apex_mm_forward(*args, **kwargs)
"""
        ns = {"__TYPE_CONST": mod_cls}
        exec(src, ns)
        fn = ns[fname]
        self._type_wrappers[mod_cls] = fn
        return fn

    def _hook_check_load_into_gpu(self, target_module: torch.nn.Module, blocks_name: Optional[str], previous_method: Any) -> None:
        # mmgp: store instance data on the module (not captured by wrapper)
        target_module._apex_mm_manager = self  # type: ignore[attr-defined]
        target_module._apex_mm_model_id = self._model_id  # type: ignore[attr-defined]
        target_module._apex_mm_blocks_name = blocks_name  # type: ignore[attr-defined]
        target_module._apex_mm_forward = previous_method  # type: ignore[attr-defined]

        wrapper_fn = self._get_wrapper_for_type(type(target_module))
        target_module.forward = functools.update_wrapper(  # type: ignore[assignment]
            functools.partial(wrapper_fn, target_module), previous_method
        )

    def _hook_check_load_into_gpu_default(
        self, target_module: torch.nn.Module, blocks_name: Optional[str], previous_method: Any
    ) -> None:
        # mmgp default wrapper uses closures/partials per instance
        def _check_load() -> None:
            self.ensure_model_loaded(self._model_id)
            if blocks_name is None:
                if self._ready_to_check_mem():
                    self._empty_cache_if_needed()
            elif (
                blocks_name != self.loaded_blocks.get(self._model_id)
                and blocks_name not in self.preloaded_blocks_per_model.get(self._model_id, {})
            ):
                self.gpu_load_blocks(self._model_id, blocks_name)

        def wrapped(_module, *args, **kwargs):
            _check_load()
            try:
                self._ensure_params_on_call_device(_module, args, kwargs)
            except Exception:
                # Best-effort: never block the original forward due to guard logic.
                pass
            return previous_method(*args, **kwargs)

        if not hasattr(target_module, "_apex_mm_id"):
            setattr(target_module, "_apex_mm_id", self._model_id)
            setattr(target_module, "_apex_mm_manager", self)
            setattr(target_module, "_apex_mm_forward", previous_method)
            setattr(target_module, "_apex_mm_blocks_name", blocks_name)
            target_module.forward = functools.update_wrapper(  # type: ignore[assignment]
                functools.partial(wrapped, target_module), previous_method
            )

    def _hook_change_module(self, target_module: torch.nn.Module, previous_method: Any, previous_method_name: str) -> None:
        # mmgp: move args to GPU if created on CPU; we skip argument dtype juggling here (Apex handles separately)
        def wrapped(_module, *args, **kwargs):
            self.ensure_model_loaded(self._model_id)
            try:
                self._ensure_params_on_call_device(_module, args, kwargs)
            except Exception:
                pass
            return previous_method(*args, **kwargs)

        key = f"_apex_mm_{previous_method_name}"
        if hasattr(target_module, key):
            return
        setattr(target_module, key, previous_method)
        # Keep parity with other wrappers for device enforcement.
        if not hasattr(target_module, "_apex_mm_blocks_name"):
            setattr(target_module, "_apex_mm_blocks_name", None)
        setattr(
            target_module,
            previous_method_name,
            functools.update_wrapper(functools.partial(wrapped, target_module), previous_method),
        )

    def _wrap_root_forward_for_offload_after(self) -> None:
        if getattr(self.module, "_apex_budget_root_wrapped", False):
            return
        original_forward = self.module.forward
        self.module._apex_budget_root_original_forward = original_forward  # type: ignore[attr-defined]
        self.module._apex_budget_root_wrapped = True  # type: ignore[attr-defined]
        self.module._apex_budget_manager = self  # type: ignore[attr-defined]

        @torch._dynamo.disable
        def _wrapped(*args, **kwargs):
            try:
                return original_forward(*args, **kwargs)
            finally:
                try:
                    self.unload_all()
                except Exception:
                    pass

        self.module.forward = _wrapped  # type: ignore[assignment]

    # -------- load/unload (ported behavior) --------

    def ensure_model_loaded(self, model_id: str) -> None:
        # Single model: load once and keep "active"
        if getattr(self, "_active", False):
            return
        self._active = True
        self.gpu_load(model_id)

    def _has_active_adapters(self) -> bool:
        """
        Best-effort PEFT adapter activity detection.
        - `PeftAdapterMixin` typically exposes `active_adapters` (list[str])
        - Some variants expose `active_adapter` (str)
        """
        try:
            aa = getattr(self.module, "active_adapters", None)
            # Diffusers' `PeftAdapterMixin.active_adapters` is a method.
            if callable(aa):
                aa = aa()
            if isinstance(aa, (list, tuple)) and len(aa) > 0:
                return True
        except Exception:
            pass
        try:
            a = getattr(self.module, "active_adapter", None)
            if isinstance(a, str) and a.strip():
                return True
        except Exception:
            pass
        return False

    def _load_lora_params_to_gpu(self) -> None:
        """
        Move all LoRA-tagged params/buffers to GPU.
        We rely on `gpu_unload_blocks(..., None)` / `unload_all()` to restore them.
        """
        if not torch.cuda.is_available():
            return
        # Use current stream; keep this synchronous with the upcoming forward.
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for entry_name, params in self.blocks_of_modules.items():
                for param in params:
                    if not param.is_lora:
                        continue
                    p = param.cpu_obj
                    # If already on GPU, skip
                    try:
                        existing = getattr(param.parent_module, param.attr_name, None)
                        if torch.is_tensor(existing) and getattr(existing, "is_cuda", False):
                            continue
                    except Exception:
                        pass
                    q = p.to("cuda", non_blocking=True)
                    q = _wrap_as_module_tensor(q, is_buffer=param.is_buffer)
                    setattr(param.parent_module, param.attr_name, q)
        torch.cuda.synchronize()

    @torch.compiler.disable()
    def gpu_load_blocks(self, model_id: str, blocks_name: Optional[str], preload: bool = False) -> None:
        if not torch.cuda.is_available():
            return
        entry_name = model_id if blocks_name is None else f"{model_id}/{blocks_name}"

        def cpu_to_gpu(stream_to_use: torch.cuda.Stream, block_params: list[_BlockParam]) -> None:
            with torch.cuda.stream(stream_to_use):
                for param in block_params:
                    parent_module, n, p, is_buffer, tied_param = (
                        param.parent_module,
                        param.attr_name,
                        param.cpu_obj,
                        param.is_buffer,
                        param.tied_to,
                    )

                    # Skip moving LoRA weights unless adapters are active.
                    if param.is_lora and not self._has_active_adapters():
                        continue

                    if tied_param is not None:
                        tied_p = getattr(tied_param[0], tied_param[1], None)
                        if torch.is_tensor(tied_p) and getattr(tied_p, "is_cuda", False):
                            setattr(parent_module, n, tied_p)
                            continue

                    q = p.to("cuda", non_blocking=True)
                    q = _wrap_as_module_tensor(q, is_buffer=is_buffer)
                    setattr(parent_module, n, q)
                    if tied_param is not None:
                        setattr(tied_param[0], tied_param[1], q)
                    del p, q

        loaded_block = self.loaded_blocks.get(model_id)
        if (not preload) and loaded_block is not None:
            self.gpu_unload_blocks(model_id, loaded_block)
            if self._ready_to_check_mem():
                self._empty_cache_if_needed()

        if self.async_transfers and blocks_name is not None and self._transfer_stream is not None:
            prev = self.prev_blocks_names.get(entry_name)
            first = prev is None or prev != loaded_block
            next_entry = self.next_blocks_names.get(entry_name)
            if first:
                cpu_to_gpu(torch.cuda.current_stream(), self.blocks_of_modules.get(entry_name, []))
            torch.cuda.synchronize()
            if next_entry is not None:
                cpu_to_gpu(self._transfer_stream, self.blocks_of_modules.get(next_entry, []))
        else:
            stream = self.default_stream or torch.cuda.current_stream()
            cpu_to_gpu(stream, self.blocks_of_modules.get(entry_name, []))
            torch.cuda.synchronize()

        if not preload:
            self.loaded_blocks[model_id] = blocks_name

    @torch.compiler.disable()
    def gpu_unload_blocks(self, model_id: str, blocks_name: Optional[str]) -> None:
        if not torch.cuda.is_available():
            return

        if blocks_name is not None and blocks_name == self.loaded_blocks.get(model_id):
            self.loaded_blocks[model_id] = None

        entry_name = model_id if blocks_name is None else f"{model_id}/{blocks_name}"
        for param in self.blocks_of_modules.get(entry_name, []):
            parent_module, n, p, is_buffer = (
                param.parent_module,
                param.attr_name,
                param.cpu_obj,
                param.is_buffer,
            )
            q = _wrap_as_module_tensor(p, is_buffer=is_buffer)
            setattr(parent_module, n, q)
            del p, q
        if blocks_name is None:
            # Base unload implies LoRAs are no longer resident on GPU.
            self._loras_on_gpu = False

    def gpu_load(self, model_id: str) -> None:
        # Base block + preloaded blocks
        self.gpu_load_blocks(model_id, None, preload=True)
        for block_name in self.preloaded_blocks_per_model.get(model_id, {}):
            self.gpu_load_blocks(model_id, block_name, preload=True)

    def unload_all(self) -> None:
        model_id = self._model_id
        try:
            self.gpu_unload_blocks(model_id, None)
            for block_name in self.preloaded_blocks_per_model.get(model_id, {}):
                self.gpu_unload_blocks(model_id, block_name)
            loaded_block = self.loaded_blocks.get(model_id)
            if loaded_block is not None:
                self.gpu_unload_blocks(model_id, loaded_block)
                entry_name = f"{model_id}/{loaded_block}"
                next_entry = self.next_blocks_names.get(entry_name)
                if next_entry is not None:
                    pos = next_entry.rfind("/")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self.gpu_unload_blocks(model_id, next_entry[pos + 1 :])
                self.loaded_blocks[model_id] = None
        finally:
            self._active = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self._last_reserved_mem_check = time.time()

    # Backward compatible names expected elsewhere in Apex
    def offload_all(self) -> None:
        self.unload_all()

    # -------- preloading plan (ported behavior) --------

    def _tune_preloading(self) -> None:
        """
        Port of mmgp `tune_preloading` for one model.
        """
        model_id = self._model_id
        current_budget = int(self._budget_bytes or 0)
        towers_names, _towers_modules = _detect_main_towers(self.module)
        preloaded_blocks: dict[str, int] = {}
        self.preloaded_blocks_per_model[model_id] = preloaded_blocks

        if current_budget == 0 or not towers_names or not self.async_transfers:
            return

        base_size = int(self.blocks_of_modules_sizes.get(model_id, 0))
        current_budget -= base_size
        current_budget = max(0, current_budget)

        towers: list[tuple[list[tuple[str, int, int]], int, int]] = []
        total_size = 0
        for tower_name in towers_names:
            max_floor_size = 0
            tower_size = 0
            floors: list[tuple[str, int, int]] = []
            prefix = model_id + "/" + tower_name
            for name, size in self.blocks_of_modules_sizes.items():
                if name.startswith(prefix):
                    tower_size += int(size)
                    floor_no = int(name[len(prefix) :])
                    floors.append((name, floor_no, int(size)))
                    max_floor_size = max(max_floor_size, int(size))
            towers.append((floors, max_floor_size, tower_size))
            total_size += tower_size
            current_budget -= 2 * max_floor_size
            current_budget = max(0, current_budget)

        for floors, max_floor_size, tower_size in towers:
            if total_size <= 0:
                continue
            tower_budget = (tower_size / total_size) * current_budget
            preload_blocks_count = int(tower_budget / max_floor_size) if max_floor_size > 0 else 0
            nb_blocks = len(floors)
            if preload_blocks_count == 0:
                space_between = 0.0
                cursor = float(len(floors))
            else:
                space_between = float(nb_blocks - preload_blocks_count) / float(preload_blocks_count)
                cursor = space_between

            first_non_preloaded: Optional[str] = None
            prev_non_preloaded: Optional[str] = None

            for name, i, size in floors:
                if i < cursor:
                    if prev_non_preloaded is None:
                        first_non_preloaded = name
                    else:
                        self.next_blocks_names[prev_non_preloaded] = name
                        self.prev_blocks_names[name] = prev_non_preloaded
                    prev_non_preloaded = name
                else:
                    self.next_blocks_names[name] = None
                    self.prev_blocks_names[name] = None
                    preloaded_blocks[name[len(model_id) + 1 :]] = int(size)
                    cursor += 1.0 + space_between

            if prev_non_preloaded is not None and len(towers) == 1 and first_non_preloaded is not None:
                self.next_blocks_names[prev_non_preloaded] = first_non_preloaded
                self.prev_blocks_names[first_non_preloaded] = prev_non_preloaded
            elif prev_non_preloaded is not None:
                self.next_blocks_names[prev_non_preloaded] = None

        self.preloaded_blocks_per_model[model_id] = preloaded_blocks
     
    # -------- hook lifecycle --------

    def install_hooks(self) -> None:
        # Hooks are installed during init in `_build_and_hook`.
        # This method is kept for API compatibility.
        try:
            self._install_device_move_guards()
        except Exception:
            pass
        try:
            self._install_ensure_on_call_device_pre_hooks()
        except Exception:
            pass
        return

    def remove_hooks(self) -> None:
        """
        Best-effort restore of wrapped forwards.
        We only undo Apex-specific attributes; modules that were wrapped multiple
        times will keep the outermost wrapper.
        """
        try:
            self._remove_device_move_guards()
        except Exception:
            pass
        try:
            self._remove_ensure_on_call_device_pre_hooks()
        except Exception:
            pass
        # Root wrapper
        if getattr(self.module, "_apex_budget_root_wrapped", False):
            try:
                self.module.forward = self.module._apex_budget_root_original_forward  # type: ignore[attr-defined]
            except Exception:
                pass
            for attr in ("_apex_budget_root_wrapped", "_apex_budget_root_original_forward", "_apex_budget_manager"):
                if hasattr(self.module, attr):
                    try:
                        delattr(self.module, attr)
                    except Exception:
                        pass

        # Best-effort restore for submodules
        for _name, sub in self.module.named_modules():
            # restore mmgp-style compiled wrapper
            if hasattr(sub, "_apex_mm_forward") and hasattr(sub, "forward"):
                try:
                    sub.forward = sub._apex_mm_forward  # type: ignore[attr-defined]
                except Exception:
                    pass
            for attr in (
                "_apex_mm_manager",
                "_apex_mm_model_id",
                "_apex_mm_blocks_name",
                "_apex_mm_forward",
                "_apex_mm_id",
            ):
                if hasattr(sub, attr):
                    try:
                        delattr(sub, attr)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Public API (stable for Apex)
# ---------------------------------------------------------------------------


def apply_budget_offloading(
    module: torch.nn.Module,
    *,
    onload_device: Union[str, torch.device],
    offload_device: Union[str, torch.device] = torch.device("cpu"),
    block_modules: Optional[List[str]] = None,
    ignore_modules: Optional[List[str]] = None,
    budget_mb: Optional[Union[int, str]] = None,
    async_transfers: bool = True,
    prefetch: bool = True,
    pin_cpu_memory: bool = False,
    vram_safety_coefficient: float = 0.8,
    offload_after_forward: bool = False,
    # Optional extra knobs (mmgp-aligned)
    partial_pinning: bool = False,
    pinned_peft_lora: bool = False,
    perc_reserved_mem_max: float = 0.0,
    verbose_level: int = -1,
    compile: bool = False,
    compile_mode: str = "default",
    model_id: str = "transformer",
) -> BudgetOffloader:
    if module is None:
        raise ValueError("module is required for budget offloading")
    if _is_group_offload_enabled(module) or getattr(module, "_apex_group_offloading_enabled", False):
        logger.warning("Group offloading cannot be combined with budget offloading")
        return
    if _is_budget_offload_enabled(module):
        return getattr(module, "_apex_budget_offloading_manager")
    
    

    device = onload_device if isinstance(onload_device, torch.device) else torch.device(onload_device)
    offload = offload_device if isinstance(offload_device, torch.device) else torch.device(offload_device)

    manager = BudgetOffloader(
        module,
        device,
        offload_device=offload,
        block_modules=block_modules,
        ignore_modules=ignore_modules,
        budget_mb=budget_mb,
        async_transfers=async_transfers,
        prefetch=prefetch,
        pin_cpu_memory=pin_cpu_memory,
        vram_safety_coefficient=vram_safety_coefficient,
        offload_after_forward=offload_after_forward,
        partial_pinning=partial_pinning,
        pinned_peft_lora=pinned_peft_lora,
        perc_reserved_mem_max=perc_reserved_mem_max,
        verbose_level=verbose_level,
        compile=compile,
        compile_mode=compile_mode,
        model_id=model_id,
    )

    manager.offload_all()
    manager.install_hooks()
    setattr(module, "_apex_budget_offloading_enabled", True)
    setattr(module, "_apex_budget_offloading_manager", manager)

    return manager


def _maybe_remove_and_reapply_budget_offloading(
    module: torch.nn.Module,
    *,
    onload_device: Optional[torch.device] = None,
) -> None:
    if not _is_budget_offload_enabled(module):
        return
    # If group offloading is enabled, do not attempt to reapply budget offloading.
    # This preserves the "one or the other" guarantee across in-place mutations.
    if _is_group_offload_enabled(module) or bool(
        getattr(module, "_apex_group_offloading_enabled", False)
    ):
        return
    manager: BudgetOffloader = getattr(module, "_apex_budget_offloading_manager")

    # Capture prior settings (API-stable subset + extra knobs)
    block_modules = manager.block_modules
    ignore_modules = manager.ignore_modules
    budget_mb = manager._budget_mb
    async_transfers = manager.async_transfers
    prefetch = manager.prefetch
    pin_cpu_memory = manager.pin_cpu_memory
    vram_safety_coefficient = manager.vram_safety_coefficient
    offload_after_forward = manager.offload_after_forward
    device = onload_device or manager.onload_device

    partial_pinning = getattr(manager, "partial_pinning", False)
    pinned_peft_lora = getattr(manager, "pinned_peft_lora", True)
    perc_reserved_mem_max = getattr(manager, "perc_reserved_mem_max", 0.0)
    verbose_level = getattr(manager, "verbose_level", -1)
    compile = getattr(manager, "compile", False)
    compile_mode = getattr(manager, "compile_mode", "default")
    model_id = getattr(manager, "_model_id", "transformer")

    # Tear down
    try:
        manager.remove_hooks()
    except Exception:
        pass
    try:
        manager.offload_all()
    except Exception:
        pass

    setattr(module, "_apex_budget_offloading_enabled", False)
    setattr(module, "_apex_budget_offloading_manager", None)

    apply_budget_offloading(
        module,
        onload_device=device,
        offload_device=manager.offload_device,
        block_modules=block_modules,
        ignore_modules=ignore_modules,
        budget_mb=budget_mb,
        async_transfers=async_transfers,
        prefetch=prefetch,
        pin_cpu_memory=pin_cpu_memory,
        vram_safety_coefficient=vram_safety_coefficient,
        offload_after_forward=offload_after_forward,
        partial_pinning=partial_pinning,
        pinned_peft_lora=pinned_peft_lora,
        perc_reserved_mem_max=perc_reserved_mem_max,
        verbose_level=verbose_level,
        compile=compile,
        compile_mode=compile_mode,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# Profiles (mmgp parity)
# ---------------------------------------------------------------------------


class BudgetOffloadProfile(int, enum.Enum):
    """
    Mirror of mmgp's `profile_type`.
    """

    HighRAM_HighVRAM = 1
    HighRAM_LowVRAM = 2
    LowRAM_HighVRAM = 3
    LowRAM_LowVRAM = 4
    VerylowRAM_LowVRAM = 5

    @staticmethod
    def tostr(v: "BudgetOffloadProfile | int") -> str:
        try:
            vv = int(v)
        except Exception:
            vv = 5
        if vv == 1:
            return "HighRAM_HighVRAM"
        if vv == 2:
            return "HighRAM_LowVRAM"
        if vv == 3:
            return "LowRAM_HighVRAM"
        if vv == 4:
            return "LowRAM_LowVRAM"
        return "VerylowRAM_LowVRAM"


def apply_budget_offloading_profile(
    module: torch.nn.Module,
    profile: BudgetOffloadProfile = BudgetOffloadProfile.VerylowRAM_LowVRAM,
    *,
    onload_device: Union[str, torch.device],
    offload_device: Union[str, torch.device] = torch.device("cpu"),
    model_id: str = "transformer",
    # Allow overrides of any apply_budget_offloading kwarg
    **override_kwargs: Any,
) -> BudgetOffloader:
    """
    Apex wrapper equivalent to `mmgp.offload.profile(...)`, but for a single module.

    Mapping notes:
    - mmgp uses multi-model budgets; here we map them to `budget_mb` for the single module.
    - When mmgp would set `budgets=None` (no budgeted block offload), we use `budget_mb=0`,
      which results in "load full module to GPU" behavior in this implementation.
    - `pinnedMemory` maps to Apex `pin_cpu_memory` (reserved pinned RAM).
    """
    p = BudgetOffloadProfile(int(profile))

    # mmgp profile defaults:
    # - budgets["*"]=3000 for LowVRAM profiles, budgets["transformer"]=1200/400
    # - pinnedMemory True for HighRAM_* profiles, and pinnedMemory="transformer" for LowRAM_*.
    budget_mb: Union[int, str, None] = 0
    pin_cpu_memory = False

    if p == BudgetOffloadProfile.HighRAM_HighVRAM:
        pin_cpu_memory = True
        budget_mb = 0
    elif p == BudgetOffloadProfile.HighRAM_LowVRAM:
        pin_cpu_memory = True
        budget_mb = 3000
    elif p == BudgetOffloadProfile.LowRAM_HighVRAM:
        pin_cpu_memory = True if str(model_id) == "transformer" else False
        budget_mb = 0
    elif p == BudgetOffloadProfile.LowRAM_LowVRAM:
        pin_cpu_memory = True if str(model_id) == "transformer" else False
        budget_mb = 3000
    elif p == BudgetOffloadProfile.VerylowRAM_LowVRAM:
        pin_cpu_memory = False
        budget_mb = 400 if str(model_id) == "transformer" else 3000
    else:
        pin_cpu_memory = False
        budget_mb = 3000

    # Apply overrides
    if "budget_mb" in override_kwargs:
        budget_mb = override_kwargs.pop("budget_mb")
    if "pin_cpu_memory" in override_kwargs:
        pin_cpu_memory = bool(override_kwargs.pop("pin_cpu_memory"))

    return apply_budget_offloading(
        module,
        onload_device=onload_device,
        offload_device=offload_device,
        budget_mb=budget_mb,
        pin_cpu_memory=pin_cpu_memory,
        model_id=model_id,
        **override_kwargs,
    )
