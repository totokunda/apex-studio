"""
Memory management utilities inspired by ComfyUI's model management.

This module now exposes a `ComponentMemoryManager` that tracks engine components,
monitors VRAM/RAM pressure, and proactively offloads/loads modules to keep
inference runs stable. It installs lightweight forward hooks on registered
modules so we can:
  • Pre‑flight GPU allocations before forward passes.
  • Offload cold components when VRAM falls below configured headroom.
  • Keep the engine's view of component locations (CPU/GPU) in sync after
    manual offloads.

The manager is attached per‑engine via `install_memory_hooks(engine)` and
wraps `engine.to_device` / `_offload` to coordinate RAM/VRAM usage across runs.
"""

from __future__ import annotations


import os
import glob
import threading
import time
import types
import weakref
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    import torch
except OSError as e:
    # Common on Windows when system commit (RAM + pagefile) is exhausted while
    # loading CUDA DLLs (e.g. cufft64_11.dll). This is frequently surfaced as:
    #   OSError: [WinError 1455] The paging file is too small ...
    if getattr(e, "winerror", None) == 1455:
        raise RuntimeError(
            "PyTorch failed to load CUDA libraries due to low Windows virtual memory "
            "(WinError 1455: paging file too small). "
            "Fix: increase your system pagefile size (System-managed is fine, or set a "
            "larger custom size on an SSD), then restart the app. "
            "Mitigations: avoid starting many Ray workers/actors at once; keep only one "
            "GPU EngineRunner per GPU; close other GPU-heavy apps."
        ) from e
    raise

try:
    import psutil
except Exception:  # pragma: no cover - psutil may be missing in limited envs
    psutil = None

from .config import MemoryConfig
from .group_offloading import apply_group_offloading
from .budget_offloading import (
    apply_budget_offloading,
    apply_budget_offloading_profile,
    BudgetOffloadProfile,
)
from loguru import logger


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ManagedComponent:
    label: str
    tags: Set[str] = field(default_factory=set)
    module_ref: weakref.ReferenceType | None = None
    engine_ref: weakref.ReferenceType | None = None
    hooks: Tuple[Any, ...] = field(default_factory=tuple)
    estimated_bytes: int = 0
    last_used: float = field(default_factory=lambda: time.time())
    in_forward: bool = False
    pinned: bool = False
    # "Lease" pins acquired by engine.load_component(). These prevent the manager
    # from accidentally evicting/discarding the component mid-run. They are
    # released only when an explicit engine._offload(...) is called for that
    # label (or when force_offload_engine_components is invoked).
    load_pin_count: int = 0
    device: Optional[torch.device] = None

    # -----------------------------
    # Debug-only: per-component VRAM profiling (CUDA)
    #
    # When enabled, we measure the *peak* allocator usage during each top-level
    # component forward() and attribute it to that component. This includes
    # all nested ops/submodules executed inside that forward.
    # -----------------------------
    profile_start_allocated_bytes: int | None = None
    profile_start_reserved_bytes: int | None = None
    profile_device_index: int | None = None
    forward_calls: int = 0
    forward_peak_alloc_delta_bytes_max: int = 0
    forward_peak_reserved_delta_bytes_max: int = 0
    forward_peak_alloc_delta_bytes_total: int = 0
    forward_peak_reserved_delta_bytes_total: int = 0
    forward_last_peak_alloc_delta_bytes: int = 0
    forward_last_peak_reserved_delta_bytes: int = 0
    # Best-effort RAM tracking (process RSS delta) during forward.
    forward_rss_delta_bytes_last: int = 0
    forward_rss_delta_bytes_max: int = 0
    forward_rss_delta_bytes_total: int = 0

    def module(self) -> Optional[torch.nn.Module]:
        return None if self.module_ref is None else self.module_ref()

    def engine(self) -> Any:
        return None if self.engine_ref is None else self.engine_ref()


# -----------------------------
# Core manager
# -----------------------------
class ComponentMemoryManager:
    """
    Lightweight VRAM/RAM manager that mirrors ComfyUI's model management ideas.

    Responsibilities:
    - Track engine components and their approximate weight size.
    - Guard GPU moves (`to_device`) to ensure headroom before allocating.
    - Attach forward hooks that evict cold components when pressure is detected.
    - Update bookkeeping when components are offloaded/discarded.
    """

    def __init__(self) -> None:
        self._components: Dict[str, ManagedComponent] = {}
        self._module_to_label: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._evicting_flag = (
            threading.local()
        )  # Track if we are currently forcing eviction
        self._run_counter = 0

    # --------- pin/lease helpers ----------
    def _is_effectively_pinned(self, comp: ManagedComponent) -> bool:
        # Static pins (e.g. scheduler) and dynamic pins (load_component leases).
        try:
            return bool(comp.pinned) or int(getattr(comp, "load_pin_count", 0)) > 0
        except Exception:
            return bool(getattr(comp, "pinned", False))

    def pin_component(self, label: str, *, reason: str = "load_component") -> None:
        """
        Pin a registered component so it cannot be evicted by pressure handling.
        Intended to be called after a successful engine.load_component(...).
        """
        if not label:
            return
        with self._lock:
            comp = self._components.get(label)
            if comp is None:
                return
            comp.load_pin_count = int(getattr(comp, "load_pin_count", 0)) + 1
            comp.last_used = time.time()
            self._log_debug(
                f"[mem] pin '{label}' (count={comp.load_pin_count}, reason={reason})"
            )

    def release_component_pin(
        self, label: str, *, reason: str = "_offload"
    ) -> None:
        """
        Release one pin lease for a component. If pins reach 0, it becomes
        eligible for eviction again.
        """
        if not label:
            return
        with self._lock:
            comp = self._components.get(label)
            if comp is None:
                return
            current = int(getattr(comp, "load_pin_count", 0))
            if current <= 0:
                return
            comp.load_pin_count = current - 1
            comp.last_used = time.time()
            self._log_debug(
                f"[mem] unpin '{label}' (count={comp.load_pin_count}, reason={reason})"
            )

    def clear_component_pins(self, label: str, *, reason: str = "force_offload") -> None:
        """
        Clear all dynamic pins for a component (used by forced cleanup paths).
        """
        if not label:
            return
        with self._lock:
            comp = self._components.get(label)
            if comp is None:
                return
            if int(getattr(comp, "load_pin_count", 0)) <= 0:
                return
            comp.load_pin_count = 0
            comp.last_used = time.time()
            self._log_debug(f"[mem] clear pins '{label}' (reason={reason})")

    def ensure_component_pinned(
        self, label: str, *, min_pins: int = 1, reason: str = "install"
    ) -> None:
        """
        Ensure a component has at least `min_pins` dynamic leases.
        Useful when the manager is installed after components were already loaded.
        """
        if not label or min_pins <= 0:
            return
        with self._lock:
            comp = self._components.get(label)
            if comp is None:
                return
            current = int(getattr(comp, "load_pin_count", 0))
            if current >= min_pins:
                return
            comp.load_pin_count = int(min_pins)
            comp.last_used = time.time()
            self._log_debug(
                f"[mem] ensure pin '{label}' (count={comp.load_pin_count}, reason={reason})"
            )

    # --------- environment helpers ----------
    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            raw = os.environ.get(name, "")
            if raw == "":
                return default
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            raw = os.environ.get(name, "")
            if raw == "":
                return default
            return int(float(raw))
        except Exception:
            return default

    def _debug_enabled(self) -> bool:
        return self._env_bool("APEX_MEM_DEBUG", False)

    def _log_debug(self, msg: str) -> None:
        if self._debug_enabled():
            logger.info(msg)
        else:
            logger.debug(msg)

    # --------- profiling helpers ----------
    def _profile_component_vram_enabled(self, engine: Any | None) -> bool:
        """
        Enable forward VRAM profiling when either:
        - Env var `APEX_PROFILE_COMPONENT_VRAM` is truthy, OR
        - Engine has `debug_component_vram=True` (or `profile_component_vram=True`).
        """
        try:
            if self._env_bool("APEX_PROFILE_COMPONENT_VRAM", False):
                return True
        except Exception:
            pass
        if engine is None:
            return False
        try:
            return bool(getattr(engine, "debug_component_vram", False)) or bool(
                getattr(engine, "profile_component_vram", False)
            )
        except Exception:
            return False

    def _profile_component_vram_isolate_enabled(self, engine: Any | None) -> bool:
        """
        When true, forcibly offload all *other* tracked components for the engine
        before each component forward. This helps attribute allocator peaks to the
        active component (instead of shared residency).
        """
        try:
            # Default to true when profiling is enabled (user asked for isolation).
            return self._env_bool("APEX_PROFILE_COMPONENT_VRAM_ISOLATE", True)
        except Exception:
            return True

    def _profile_component_vram_isolate_offload_type(self) -> str:
        """
        Offload strategy used for isolation.
        - "cpu": keep weights in RAM, free VRAM (default; preserves run correctness)
        - "discard": free both VRAM and (eventually) RAM, but may break the run if
          components are needed again later.
        """
        # Default to discard to isolate both VRAM and RAM attribution.
        raw = os.environ.get(
            "APEX_PROFILE_COMPONENT_VRAM_ISOLATE_OFFLOAD_TYPE", "discard"
        )
        val = str(raw or "").strip().lower()
        if val not in {"cpu", "discard"}:
            return "discard"
        return val

    def _get_process_rss_bytes(self) -> int | None:
        try:
            if psutil is None:
                return None
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return None

    def _force_offload_other_components_for_isolation(
        self,
        *,
        engine: Any,
        keep_label: str,
        target_device: torch.device,
    ) -> None:
        """
        Best-effort: move all other engine components off `target_device`.
        Uses ignore_pins=True so the debug path can override "lease" pins.
        """
        if target_device.type == "cpu":
            return
        offload_type = self._profile_component_vram_isolate_offload_type()

        try:
            with self._lock:
                comps = list(self._components.values())
        except Exception:
            comps = list(self._components.values())

        for comp in comps:
            try:
                if comp is None:
                    continue
                if comp.label == keep_label:
                    continue
                if comp.in_forward:
                    continue
                if comp.engine() is not engine:
                    continue
                if comp.device is None or comp.device.type != target_device.type:
                    continue
                # If the component isn't a module anymore (discarded), nothing to do.
                if offload_type != "discard" and comp.module() is None:
                    continue
                self._offload_component(comp, offload_type=offload_type, ignore_pins=True)
            except Exception:
                continue

    def _wrap_component_method_for_profiling(
        self, module: torch.nn.Module, label: str, method_name: str
    ) -> None:
        """
        Wrap a common "entrypoint" method (e.g. `encode`, `decode`) so profiling
        works even when pipelines call methods that do not trigger `forward()`.
        """
        try:
            if module is None or not label or not method_name:
                return
            flag = f"_apex_mem_profile_method_wrapped_{method_name}"
            if bool(getattr(module, flag, False)):
                return
            if not hasattr(module, method_name):
                return
            original = getattr(module, method_name)
            if not callable(original):
                return

            # Stash original for debugging/escape hatch.
            try:
                setattr(module, f"_apex_mem_profile_method_original_{method_name}", original)
            except Exception:
                pass

            def _wrapped(_self_mod, *args: Any, **kwargs: Any):
                comp = self._components.get(label)
                engine = None if comp is None else comp.engine()
                if comp is None or not self._profile_component_vram_enabled(engine):
                    # Fast path: profiling disabled.
                    try:
                        return original(*args, **kwargs)
                    except TypeError:
                        return original(_self_mod, *args, **kwargs)

                # Mark active so debug isolation doesn't accidentally offload us.
                comp.in_forward = True
                comp.last_used = time.time()
                try:
                    module_device = self._module_device(module)
                    target_device = module_device
                    try:
                        eng_dev = getattr(engine, "device", None)
                        if isinstance(eng_dev, str):
                            eng_dev = torch.device(eng_dev)
                        if isinstance(eng_dev, torch.device):
                            target_device = eng_dev
                    except Exception:
                        target_device = module_device
                    comp.device = module_device

                    if self._profile_component_vram_isolate_enabled(engine):
                        self._force_offload_other_components_for_isolation(
                            engine=engine, keep_label=label, target_device=target_device
                        )

                    rss_before = self._get_process_rss_bytes()
                    self._start_forward_vram_profile(comp, module, device=target_device)
                    try:
                        try:
                            return original(*args, **kwargs)
                        except TypeError:
                            return original(_self_mod, *args, **kwargs)
                    finally:
                        self._end_forward_vram_profile(comp, module)
                        rss_after = self._get_process_rss_bytes()
                        if rss_before is not None and rss_after is not None:
                            delta = max(0, int(rss_after) - int(rss_before))
                            comp.forward_rss_delta_bytes_last = int(delta)
                            comp.forward_rss_delta_bytes_total = int(
                                getattr(comp, "forward_rss_delta_bytes_total", 0)
                            ) + int(delta)
                            comp.forward_rss_delta_bytes_max = max(
                                int(getattr(comp, "forward_rss_delta_bytes_max", 0)),
                                int(delta),
                            )
                finally:
                    comp.in_forward = False
                    try:
                        comp.device = self._module_device(module)
                    except Exception:
                        pass
                    comp.last_used = time.time()

            setattr(module, method_name, types.MethodType(_wrapped, module))
            setattr(module, flag, True)
        except Exception:
            return

    def _install_entrypoint_method_wrappers(
        self, module: torch.nn.Module, label: str
    ) -> None:
        # Text encoders often use `encode(...)` (not forward).
        self._wrap_component_method_for_profiling(module, label, "encode")
        # VAEs commonly use `decode(...)` / `encode(...)` helpers (not forward).
        self._wrap_component_method_for_profiling(module, label, "decode")

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        try:
            n = float(num_bytes)
        except Exception:
            return str(num_bytes)
        for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
            if abs(n) < 1024.0:
                return f"{n:.2f}{unit}"
            n /= 1024.0
        return f"{n:.2f}PiB"

    def _start_forward_vram_profile(
        self,
        comp: ManagedComponent,
        module: torch.nn.Module,
        *,
        device: torch.device | None = None,
    ) -> None:
        try:
            dev = device or self._module_device(module)
            if dev.type != "cuda" or not torch.cuda.is_available():
                return
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            with torch.cuda.device(idx):
                # Sync to attribute any in-flight allocations to the previous op.
                torch.cuda.synchronize()
                comp.profile_start_allocated_bytes = int(torch.cuda.memory_allocated())
                comp.profile_start_reserved_bytes = int(torch.cuda.memory_reserved())
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    # Older torch builds may not support this on some backends.
                    pass
                comp.profile_device_index = int(idx)
        except Exception:
            # Best-effort only; profiling must never break inference.
            return

    def _end_forward_vram_profile(
        self, comp: ManagedComponent, module: torch.nn.Module
    ) -> None:
        idx = comp.profile_device_index
        start_alloc = comp.profile_start_allocated_bytes
        start_res = comp.profile_start_reserved_bytes
        if idx is None or start_alloc is None or start_res is None:
            return
        try:
            with torch.cuda.device(int(idx)):
                torch.cuda.synchronize()
                peak_alloc = int(torch.cuda.max_memory_allocated())
                peak_res = int(torch.cuda.max_memory_reserved())
                delta_alloc = max(0, peak_alloc - int(start_alloc))
                delta_res = max(0, peak_res - int(start_res))

                comp.forward_calls = int(getattr(comp, "forward_calls", 0)) + 1
                comp.forward_last_peak_alloc_delta_bytes = int(delta_alloc)
                comp.forward_last_peak_reserved_delta_bytes = int(delta_res)
                comp.forward_peak_alloc_delta_bytes_total = int(
                    getattr(comp, "forward_peak_alloc_delta_bytes_total", 0)
                ) + int(delta_alloc)
                comp.forward_peak_reserved_delta_bytes_total = int(
                    getattr(comp, "forward_peak_reserved_delta_bytes_total", 0)
                ) + int(delta_res)
                comp.forward_peak_alloc_delta_bytes_max = max(
                    int(getattr(comp, "forward_peak_alloc_delta_bytes_max", 0)),
                    int(delta_alloc),
                )
                comp.forward_peak_reserved_delta_bytes_max = max(
                    int(getattr(comp, "forward_peak_reserved_delta_bytes_max", 0)),
                    int(delta_res),
                )
        except Exception:
            return
        finally:
            # Always clear the per-forward scratch space.
            comp.profile_device_index = None
            comp.profile_start_allocated_bytes = None
            comp.profile_start_reserved_bytes = None

    def begin_run(self, engine: Any) -> None:
        """
        Called at the start of `engine.run(...)` (when wrapped).
        Clears previous per-forward stats for this engine's components.
        """
        try:
            self._run_counter += 1
        except Exception:
            pass
        try:
            setattr(engine, "_apex_component_vram_profile", None)
        except Exception:
            pass
        # Clear any previous stats for components owned by this engine.
        try:
            with self._lock:
                comps = list(self._components.values())
        except Exception:
            comps = list(self._components.values())
        for comp in comps:
            try:
                if comp.engine() is not engine:
                    continue
                comp.forward_calls = 0
                comp.forward_peak_alloc_delta_bytes_max = 0
                comp.forward_peak_reserved_delta_bytes_max = 0
                comp.forward_peak_alloc_delta_bytes_total = 0
                comp.forward_peak_reserved_delta_bytes_total = 0
                comp.forward_last_peak_alloc_delta_bytes = 0
                comp.forward_last_peak_reserved_delta_bytes = 0
                comp.forward_rss_delta_bytes_last = 0
                comp.forward_rss_delta_bytes_max = 0
                comp.forward_rss_delta_bytes_total = 0
                comp.profile_device_index = None
                comp.profile_start_allocated_bytes = None
                comp.profile_start_reserved_bytes = None
            except Exception:
                continue

    def end_run(self, engine: Any) -> None:
        """
        Called at the end of `engine.run(...)` (when wrapped).
        If profiling is enabled, summarize per-component peak VRAM deltas.
        """
        if not self._profile_component_vram_enabled(engine):
            return
        try:
            with self._lock:
                comps = list(self._components.values())
        except Exception:
            comps = list(self._components.values())

        profile: Dict[str, Dict[str, Any]] = {}
        for comp in comps:
            try:
                if comp.engine() is not engine:
                    continue
                if int(getattr(comp, "forward_calls", 0)) <= 0:
                    continue
                profile[str(comp.label)] = {
                    "calls": int(comp.forward_calls),
                    "weight_bytes": int(getattr(comp, "estimated_bytes", 0)),
                    "device": str(getattr(comp, "device", None)),
                    "peak_alloc_delta_bytes_max": int(
                        getattr(comp, "forward_peak_alloc_delta_bytes_max", 0)
                    ),
                    "peak_reserved_delta_bytes_max": int(
                        getattr(comp, "forward_peak_reserved_delta_bytes_max", 0)
                    ),
                    "peak_alloc_delta_bytes_total": int(
                        getattr(comp, "forward_peak_alloc_delta_bytes_total", 0)
                    ),
                    "peak_reserved_delta_bytes_total": int(
                        getattr(comp, "forward_peak_reserved_delta_bytes_total", 0)
                    ),
                    "rss_delta_bytes_max": int(
                        getattr(comp, "forward_rss_delta_bytes_max", 0)
                    ),
                    "rss_delta_bytes_total": int(
                        getattr(comp, "forward_rss_delta_bytes_total", 0)
                    ),
                }
            except Exception:
                continue

        try:
            setattr(engine, "_apex_component_vram_profile", profile)
        except Exception:
            pass

        # Log a concise summary (top components by peak alloc delta).
        try:
            items = list(profile.items())
            items.sort(key=lambda kv: int(kv[1].get("peak_alloc_delta_bytes_max", 0)), reverse=True)
            top_n = self._env_int("APEX_PROFILE_COMPONENT_VRAM_TOPN", 10)
            lines: List[str] = []
            for label, stats in items[: max(0, int(top_n))]:
                lines.append(
                    f"- {label}: "
                    f"calls={stats.get('calls')} "
                    f"weights={self._format_bytes(int(stats.get('weight_bytes', 0)))} "
                    f"peak_alloc+={self._format_bytes(int(stats.get('peak_alloc_delta_bytes_max', 0)))} "
                    f"peak_res+={self._format_bytes(int(stats.get('peak_reserved_delta_bytes_max', 0)))} "
                    f"rss+={self._format_bytes(int(stats.get('rss_delta_bytes_max', 0)))} "
                    f"device={stats.get('device')}"
                )
            if lines:
                logger.info(
                    "[mem] component forward VRAM profile (peak deltas per forward, top "
                    f"{min(len(lines), int(top_n))}):\n" + "\n".join(lines)
                )
        except Exception:
            pass

    # --------- sizing helpers ----------
    def _module_device(self, module: torch.nn.Module) -> torch.device:
        for param in module.parameters(recurse=True):
            return param.device
        for buf in module.buffers(recurse=True):
            return buf.device
        return torch.device("cpu")

    def _module_size_bytes(self, module: torch.nn.Module) -> int:
        total = 0
        try:
            for p in module.parameters(recurse=True):
                if p is not None:
                    total += p.numel() * p.element_size()
            for b in module.buffers(recurse=True):
                if b is not None:
                    total += b.numel() * b.element_size()
        except Exception:
            pass
        return int(total)

    def _estimate_allocation_bytes(self, modules: Sequence[torch.nn.Module]) -> int:
        """
        Estimate memory required to move the provided modules onto an accelerator.
        Mirrors ComfyUI's "weight + safety margin" approach.
        """
        base = sum(self._module_size_bytes(m) for m in modules)
        mult = self._env_float("APEX_LOAD_MODEL_VRAM_MULT", 1.20)
        extra = self._env_int("APEX_LOAD_MODEL_VRAM_EXTRA_BYTES", 512 * 1024**2)
        return int(base * mult + extra)

    # --------- memory stats ----------
    def _device_free_total(self, device: torch.device) -> Tuple[int, int]:
        """
        Return (free_bytes, total_bytes) for the given device.
        """
        try:
            if device.type == "cuda" and torch.cuda.is_available():
                idx = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                free, total = torch.cuda.mem_get_info(idx)
                return int(free), int(total)
            if (
                device.type == "xpu"
                and hasattr(torch, "xpu")
                and torch.xpu.is_available()
            ):
                idx = (
                    device.index
                    if device.index is not None
                    else torch.xpu.current_device()
                )
                stats = torch.xpu.memory_stats(idx)
                free = int(stats.get("available_bytes.all.current", 0))
                total = int(torch.xpu.get_device_properties(idx).total_memory)
                return free, total
            if (
                device.type == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                if psutil:
                    vm = psutil.virtual_memory()
                    return int(vm.available), int(vm.total)
                return 0, 0
        except Exception:
            pass

        if psutil:
            try:
                vm = psutil.virtual_memory()
                return int(vm.available), int(vm.total)
            except Exception:
                pass
        return 0, 0

    def _flush_device_caches(self, device: torch.device) -> None:
        """
        Best-effort cache flush without moving weights.

        Used when we skip a voluntary `_offload()` to keep models warm while still
        returning transient allocator cache back to the system.
        """
        try:
            if device.type == "cuda" and torch.cuda.is_available():
                idx = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                with torch.cuda.device(idx):
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            if (
                device.type == "mps"
                and getattr(torch, "mps", None) is not None
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    def _target_free_bytes(self, device: torch.device, reserve: int) -> int:
        free, total = self._device_free_total(device)
        if total == 0:
            return reserve
        if device.type in {"cuda", "xpu", "mps", "npu", "mlu"}:
            frac = self._env_float("APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION", 0.12)
        else:
            frac = self._env_float("APEX_WEIGHT_TARGET_FREE_RAM_FRACTION", 0.10)
        target = int(total * frac)
        if device.type in {"cuda", "xpu"}:
            target = max(
                target,
                self._env_int("APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES", 2 * 1024**3),
            )
        return reserve + target

    # --------- registration ----------
    def register_component(
        self,
        module: torch.nn.Module,
        label: str,
        tags: Optional[Iterable[str]] = None,
        *,
        engine: Any | None = None,
        pinned: bool = False,
        install_hooks: bool = True,
    ) -> ManagedComponent | None:
        if module is None:
            return None
        try:
            with self._lock:
                if id(module) in self._module_to_label:
                    existing_label = self._module_to_label[id(module)]
                    comp = self._components.get(existing_label)
                    if comp:
                        return comp

                comp = ManagedComponent(
                    label=label,
                    tags=set(tags or []),
                    module_ref=weakref.ref(module),
                    engine_ref=weakref.ref(engine) if engine is not None else None,
                    estimated_bytes=self._module_size_bytes(module),
                    pinned=pinned,
                    device=self._module_device(module),
                )
                if install_hooks:
                    comp.hooks = self._attach_forward_hooks(module, label)
                # Also wrap common non-forward entrypoints so profiling works for
                # components invoked via methods like `.encode()` / `.decode()`.
                try:
                    self._install_entrypoint_method_wrappers(module, label)
                except Exception:
                    pass
                self._components[label] = comp
                self._module_to_label[id(module)] = label
                self._log_debug(
                    f"[mem] registered component '{label}' "
                    f"(size={comp.estimated_bytes/1e6:.2f}MB, device={comp.device})"
                )
                return comp
        except Exception as exc:
            logger.warning(f"Failed to register component {label}: {exc}")
        return None

    def _attach_forward_hooks(
        self, module: torch.nn.Module, label: str
    ) -> Tuple[Any, ...]:
        handles: List[Any] = []
        try:

            def _pre(_mod, _inputs):
                # Allow returning modified inputs from the hook.
                return self._on_forward_start(label, _mod, _inputs)

            def _post(_mod, _inputs, _outputs):
                self._on_forward_end(label, _mod)

            handles.append(module.register_forward_pre_hook(_pre))
            handles.append(module.register_forward_hook(_post))
        except Exception as exc:
            logger.debug(f"Could not attach forward hooks for {label}: {exc}")
        return tuple(handles)

    # --------- lifecycle ----------
    def _infer_first_tensor_device(self, obj: Any) -> Optional[torch.device]:
        """
        Best-effort: find the first tensor device within a nested input structure.
        """
        try:
            if torch.is_tensor(obj):
                return obj.device
        except Exception:
            pass
        if isinstance(obj, (list, tuple)):
            for it in obj:
                dev = self._infer_first_tensor_device(it)
                if dev is not None:
                    return dev
        if isinstance(obj, dict):
            for it in obj.values():
                dev = self._infer_first_tensor_device(it)
                if dev is not None:
                    return dev
        return None

    def _move_tensors_to_device(self, obj: Any, device: torch.device) -> Any:
        """
        Recursively move tensors inside `obj` to `device`.
        Non-tensors are preserved.
        """
        try:
            if torch.is_tensor(obj):
                return obj.to(device)
        except Exception:
            pass
        if isinstance(obj, tuple):
            return tuple(self._move_tensors_to_device(x, device) for x in obj)
        if isinstance(obj, list):
            return [self._move_tensors_to_device(x, device) for x in obj]
        if isinstance(obj, dict):
            return {k: self._move_tensors_to_device(v, device) for k, v in obj.items()}
        return obj

    def _on_forward_start(
        self, label: str, module: torch.nn.Module, inputs: Any
    ) -> Any:
        comp = self._components.get(label)
        if comp is None:
            return None
        comp.in_forward = True
        comp.last_used = time.time()
        module_device = self._module_device(module)

        # -----------------------------
        # Device alignment guard
        # -----------------------------
        # If an input tensor is on a different device than the module weights,
        # fix it *before* the forward to avoid runtime errors like:
        #   "weight is on cpu but expected on mps".
        input_device = self._infer_first_tensor_device(inputs)
        if (
            input_device is not None
            and isinstance(input_device, torch.device)
            and input_device.type != module_device.type
        ):
            # Prefer moving the module to the input device (keeps the fast path fast).
            reserve = max(comp.estimated_bytes, 0)
            reserve = int(reserve * self._env_float("APEX_LOAD_MODEL_VRAM_MULT", 1.20))
            reserve += self._env_int(
                "APEX_LOAD_MODEL_VRAM_EXTRA_BYTES", 512 * 1024**2
            )
            try:
                # Ensure we have headroom on the *target* device before moving.
                self._ensure_room(input_device, reserve, exclude_label=label)
                engine = comp.engine()
                if engine is not None and hasattr(engine, "to_device"):
                    engine.to_device(module, device=input_device)
                else:
                    module.to(input_device)
                module_device = self._module_device(module)
            except Exception as exc:
                # If moving weights fails (e.g. OOM), fall back to moving inputs.
                self._log_debug(
                    f"[mem] device align failed for '{label}' "
                    f"({module_device} -> {input_device}): {exc}; falling back to input move"
                )
                try:
                    moved = self._move_tensors_to_device(inputs, module_device)
                    inputs = moved
                except Exception:
                    pass

        comp.device = module_device

        # Debug-only isolation: offload all other components *before* ensuring room
        # so we can attribute allocator peaks to this component.
        try:
            engine = comp.engine()
            if (
                self._profile_component_vram_enabled(engine)
                and self._profile_component_vram_isolate_enabled(engine)
            ):
                self._force_offload_other_components_for_isolation(
                    engine=engine, keep_label=label, target_device=module_device
                )
        except Exception:
            pass

        reserve = max(comp.estimated_bytes, 0)
        reserve = int(reserve * self._env_float("APEX_LOAD_MODEL_VRAM_MULT", 1.20))
        reserve += self._env_int("APEX_LOAD_MODEL_VRAM_EXTRA_BYTES", 512 * 1024**2)

        self._ensure_room(module_device, reserve, exclude_label=label)

        # Debug-only VRAM profiling: start peak tracking right before forward.
        try:
            engine = comp.engine()
            if self._profile_component_vram_enabled(engine):
                rss_before = self._get_process_rss_bytes()
                if rss_before is not None:
                    setattr(comp, "_apex_forward_rss_before", int(rss_before))
                self._start_forward_vram_profile(comp, module, device=module_device)
        except Exception:
            pass

        # If we modified inputs, return them to the hook caller.
        return inputs

    def _on_forward_end(self, label: str, module: torch.nn.Module) -> None:
        comp = self._components.get(label)
        if comp is None:
            return
        # Debug-only VRAM profiling: attribute peak usage to this component.
        try:
            engine = comp.engine()
            if self._profile_component_vram_enabled(engine):
                self._end_forward_vram_profile(comp, module)
                rss_after = self._get_process_rss_bytes()
                rss_before = getattr(comp, "_apex_forward_rss_before", None)
                if rss_after is not None and rss_before is not None:
                    # "RAM used" (best-effort): positive RSS increase during forward.
                    delta = max(0, int(rss_after) - int(rss_before))
                    comp.forward_rss_delta_bytes_last = int(delta)
                    comp.forward_rss_delta_bytes_total = int(
                        getattr(comp, "forward_rss_delta_bytes_total", 0)
                    ) + int(delta)
                    comp.forward_rss_delta_bytes_max = max(
                        int(getattr(comp, "forward_rss_delta_bytes_max", 0)), int(delta)
                    )
                try:
                    if hasattr(comp, "_apex_forward_rss_before"):
                        delattr(comp, "_apex_forward_rss_before")
                except Exception:
                    pass
        except Exception:
            pass
        comp.in_forward = False
        comp.device = self._module_device(module)
        comp.last_used = time.time()

    def mark_offloaded(
        self,
        module: Union[str, torch.nn.Module, None],
        *,
        offload_type: str = "cpu",
    ) -> None:
        label = self._resolve_label(module)
        if label is None:
            return
        comp = self._components.get(label)
        if comp is None:
            return
        comp.device = torch.device("cpu") if offload_type == "cpu" else None
        comp.last_used = time.time()
        self._log_debug(f"[mem] marked '{label}' offloaded ({offload_type})")

    def mark_moved(
        self, modules: Sequence[torch.nn.Module], device: torch.device
    ) -> None:
        now = time.time()
        for mod in modules:
            label = self._module_to_label.get(id(mod))
            if not label:
                continue
            comp = self._components.get(label)
            if comp is None:
                continue
            comp.device = device
            comp.last_used = now
            comp.estimated_bytes = max(
                comp.estimated_bytes, self._module_size_bytes(mod)
            )

    def mark_idle(self, module: Union[str, torch.nn.Module, None]) -> None:
        """
        Mark a component as no longer actively in use, without moving it.

        This enables "lazy offloading": pipelines may call `_offload()` after use,
        but we can keep weights resident until VRAM pressure forces eviction.
        """
        label = self._resolve_label(module)
        if label is None:
            return
        comp = self._components.get(label)
        if comp is None:
            return
        mod_obj: Optional[torch.nn.Module] = None
        if isinstance(module, torch.nn.Module):
            mod_obj = module
        else:
            mod_obj = comp.module()
        if mod_obj is not None:
            try:
                comp.device = self._module_device(mod_obj)
                comp.estimated_bytes = max(
                    comp.estimated_bytes, self._module_size_bytes(mod_obj)
                )
            except Exception:
                pass
        comp.in_forward = False
        comp.last_used = time.time()

    def force_offload_engine_components(
        self,
        engine: Any,
        *,
        active_labels: Optional[Set[str]] = None,
        offload_type: str = "discard",
    ) -> Dict[str, str]:
        """
        Best-effort: forcibly offload/discard tracked components belonging to `engine`.

        This bypasses "lazy offload" behavior by routing through the manager's
        forced eviction pathway (`_offload_component`), so cleanup calls can truly
        free VRAM/RAM when requested.
        """
        active = {str(x) for x in (active_labels or set()) if x is not None}
        results: Dict[str, str] = {}
        try:
            with self._lock:
                comps = list(self._components.values())
        except Exception:
            comps = list(self._components.values())

        for comp in comps:
            try:
                if comp is None:
                    continue
                if comp.in_forward:
                    continue
                if comp.label in active:
                    continue
                eng = comp.engine()
                if eng is None or eng is not engine:
                    continue
                self._offload_component(
                    comp, offload_type=offload_type, ignore_pins=True
                )
                results[str(comp.label)] = str(offload_type)
            except Exception as exc:
                results[str(getattr(comp, "label", "unknown"))] = f"error:{exc}"
        return results

    # --------- pressure handling ----------
    def _offload_min_free_bytes(self, device: torch.device) -> int:
        """
        Minimum free memory target used when deciding whether to honor *voluntary*
        offload requests (i.e. pipeline cleanup calls).
        """
        _, total = self._device_free_total(device)
        if total <= 0:
            return 0
        if device.type in {"cuda", "xpu", "mps", "npu", "mlu"}:
            frac = self._env_float("APEX_OFFLOAD_MIN_FREE_VRAM_FRACTION", 0.10)
        else:
            frac = self._env_float("APEX_OFFLOAD_MIN_FREE_RAM_FRACTION", 0.10)
        frac = max(0.0, min(float(frac), 0.95))
        return int(total * frac)

    def _ensure_min_free(
        self,
        device: torch.device,
        min_free_bytes: int,
        *,
        exclude_label: str | None = None,
    ) -> None:
        """
        Evict cold components until `free(device) >= min_free_bytes`.

        Unlike `_ensure_room`, this is used for post-run cleanup policies where the
        user may want to keep models warm on large VRAM cards.
        """
        if min_free_bytes <= 0:
            return
        free, total = self._device_free_total(device)
        if total == 0 or free >= min_free_bytes:
            return

        candidates = self._eviction_candidates(device, exclude_label=exclude_label)
        for label in candidates:
            comp = self._components.get(label)
            if comp is None or comp.in_forward:
                continue

            # Tiered offloading: accelerator -> CPU -> Discard (for CPU pressure only).
            if device.type != "cpu":
                try:
                    # Be conservative: ensure we have CPU room before offloading more weights.
                    self._ensure_room(
                        torch.device("cpu"),
                        comp.estimated_bytes,
                        exclude_label=exclude_label,
                    )
                    self._offload_component(comp, offload_type="cpu")
                except Exception:
                    self._offload_component(comp, offload_type="discard")
            else:
                self._offload_component(comp, offload_type="discard")

            free, _ = self._device_free_total(device)
            if free >= min_free_bytes:
                break

    def _ensure_room(
        self,
        device: torch.device,
        reserve_bytes: int,
        *,
        exclude_label: str | None = None,
    ) -> None:
        free, total = self._device_free_total(device)
        target_free = self._target_free_bytes(device, reserve_bytes)
        if total == 0:
            return

        if free >= target_free:
            return

        self._log_debug(
            f"[mem] pressure detected on {device}: free={free/1e9:.2f}GB "
            f"target={target_free/1e9:.2f}GB"
        )
        candidates = self._eviction_candidates(device, exclude_label=exclude_label)
        for label in candidates:
            comp = self._components.get(label)
            if comp is None or comp.in_forward:
                continue

            # Tiered offloading: GPU -> CPU -> Discard
            if device.type != "cpu":
                try:
                    self._ensure_room(
                        torch.device("cpu"),
                        comp.estimated_bytes,
                        exclude_label=exclude_label,
                    )
                    self._offload_component(comp, offload_type="cpu")
                except Exception:
                    self._offload_component(comp, offload_type="discard")
            else:
                self._offload_component(comp, offload_type="discard")

            free, _ = self._device_free_total(device)
            if free >= target_free:
                break

    def _eviction_candidates_allow_pinned(
        self,
        device: torch.device,
        *,
        exclude_label: str | None = None,
        engine: Any | None = None,
    ) -> List[str]:
        """
        Like `_eviction_candidates`, but includes pinned/leased components.

        This is used for **load-time** pressure handling: it's better to
        temporarily offload previously-loaded weights than to crash with OOM
        while loading a new component.
        """
        comps: List[Tuple[float, str]] = []
        for label, comp in self._components.items():
            if exclude_label is not None and label == exclude_label:
                continue
            if comp is None or comp.in_forward:
                continue
            if engine is not None and comp.engine() is not engine:
                continue
            if comp.device is None or comp.device.type != device.type:
                continue
            comps.append((comp.last_used, label))
        comps.sort(key=lambda x: x[0])
        return [label for _, label in comps]

    def _ensure_room_for_load(
        self,
        device: torch.device,
        reserve_bytes: int,
        *,
        exclude_label: str | None = None,
        engine: Any | None = None,
    ) -> None:
        """
        Ensure sufficient free memory **before loading weights** onto `device`.

        Key difference vs `_ensure_room`: load-time preflight is allowed to evict
        *leased/pinned* components (ignore_pins=True), because otherwise it is
        easy to hit OOM while simply sequencing component loads.
        """
        free, total = self._device_free_total(device)
        target_free = self._target_free_bytes(device, reserve_bytes)
        if total == 0 or free >= target_free:
            return

        self._log_debug(
            f"[mem] load pressure on {device}: free={free/1e9:.2f}GB "
            f"target={target_free/1e9:.2f}GB"
        )

        # First: prefer evicting components belonging to the same engine.
        for scope_engine in (engine, None):
            candidates = self._eviction_candidates_allow_pinned(
                device, exclude_label=exclude_label, engine=scope_engine
            )
            for label in candidates:
                comp = self._components.get(label)
                if comp is None or comp.in_forward:
                    continue

                if device.type != "cpu":
                    try:
                        # Ensure we have CPU room before offloading more weights.
                        self._ensure_room(
                            torch.device("cpu"),
                            comp.estimated_bytes,
                            exclude_label=exclude_label,
                        )
                        self._offload_component(comp, offload_type="cpu", ignore_pins=True)
                    except Exception:
                        self._offload_component(comp, offload_type="discard", ignore_pins=True)
                else:
                    self._offload_component(comp, offload_type="discard", ignore_pins=True)

                # Flush allocator caches to maximize contiguous free segments for upcoming loads.
                try:
                    self._flush_device_caches(device)
                except Exception:
                    pass

                free, _ = self._device_free_total(device)
                if free >= target_free:
                    return

    def preflight_component_load(
        self,
        *,
        engine: Any,
        component: Any,
        reserve_bytes: int | None = None,
    ) -> None:
        """
        Public entrypoint used by the engine wrapper around `load_component()`.

        This runs *before* the actual weight load begins, so we can proactively
        offload other resident components and prevent CUDA OOM during load.
        """
        try:
            dev = getattr(engine, "device", None) or torch.device("cpu")
            if isinstance(dev, str):
                dev = torch.device(dev)
            if not isinstance(dev, torch.device):
                dev = torch.device("cpu")
        except Exception:
            dev = torch.device("cpu")

        if dev.type == "cpu":
            return

        # IMPORTANT:
        # Do NOT exclude by component label during load preflight.
        #
        # Labels like "transformer"/"vae" are stable across engines, so excluding them
        # would incorrectly protect an *old* model owned by a different engine and
        # reintroduce the exact OOM we are trying to prevent.
        exclude_label = None

        # Estimate incoming weights (best-effort) so we reserve enough headroom
        # for the *load itself* (not just the post-load forward).
        try:
            if reserve_bytes is not None:
                reserve = max(0, int(reserve_bytes))
            else:
                extra = self._env_int(
                    "APEX_LOAD_MODEL_VRAM_EXTRA_BYTES", 512 * 1024**2
                )
                mult = self._env_float("APEX_LOAD_MODEL_VRAM_MULT", 1.20)
                weight_bytes = 0
                if isinstance(component, dict):
                    model_path = component.get("model_path")
                    if model_path and os.path.exists(str(model_path)):
                        mp = str(model_path)
                        if os.path.isdir(mp):
                            exts = component.get("extensions") or [
                                "safetensors",
                                "bin",
                                "pt",
                                "ckpt",
                                "pth",
                            ]
                            try:
                                exts = list(exts)
                            except Exception:
                                exts = ["safetensors", "bin", "pt", "ckpt", "pth"]
                            # Sum weight file sizes as a rough upper bound.
                            for ext in exts:
                                for fp in glob.glob(os.path.join(mp, f"*.{ext}")):
                                    try:
                                        weight_bytes += int(os.path.getsize(fp))
                                    except Exception:
                                        continue
                        else:
                            try:
                                weight_bytes = int(os.path.getsize(mp))
                            except Exception:
                                weight_bytes = 0
                # If we can't estimate, fall back to a conservative small reserve.
                if weight_bytes <= 0:
                    reserve = int(extra)
                else:
                    reserve = int(weight_bytes * float(mult) + int(extra))
        except Exception:
            reserve = 0

        # If we already have enough free memory, do nothing (do NOT offload eagerly).
        try:
            free, total = self._device_free_total(dev)
            target_free = self._target_free_bytes(dev, int(reserve))
            if total == 0 or free >= target_free:
                return
        except Exception:
            # If we cannot compute pressure, fall through to best-effort eviction.
            pass

        # ---------
        # Force-offload engine attributes (not just tracked components)
        # ---------
        #
        # Some engine components are wrappers that don't expose Parameters directly
        # (or load weights lazily). In those cases, relying solely on the tracked
        # registry can miss large VRAM residents. For load-time safety, we force
        # offload known engine attributes first.
        try:
            if hasattr(engine, "_offload"):
                # Signal to the engine `_offload` wrapper that this is a FORCED eviction
                # (so it must not keep modules warm on GPU).
                self._evicting_flag.is_evicting = True
                try:
                    # Prefer keeping weights in RAM (CPU) rather than discarding.
                    for name in ("text_encoder", "vae", "transformer"):
                        try:
                            obj = getattr(engine, name, None)
                        except Exception:
                            obj = None
                        if obj is None:
                            continue
                        try:
                            engine._offload(name, offload_type="cpu")  # type: ignore[attr-defined]
                        except Exception:
                            # Fallback: try passing the object reference.
                            try:
                                engine._offload(obj, offload_type="cpu")  # type: ignore[attr-defined]
                            except Exception:
                                pass

                    helpers_map = getattr(engine, "_helpers", None) or getattr(
                        engine, "helpers", None
                    )
                    if isinstance(helpers_map, dict):
                        for key, helper in helpers_map.items():
                            if helper is None:
                                continue
                            try:
                                engine._offload(helper, offload_type="cpu")  # type: ignore[attr-defined]
                            except Exception:
                                try:
                                    engine._offload(str(key), offload_type="discard")  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                finally:
                    self._evicting_flag.is_evicting = False
                try:
                    self._flush_device_caches(dev)
                except Exception:
                    pass
        except Exception:
            # Best-effort only; never block component load due to preflight.
            pass

        try:
            self._ensure_room_for_load(
                dev, reserve, exclude_label=exclude_label, engine=engine
            )
        except Exception:
            # Never break loading due to memory manager errors; worst case we fall
            # back to the original OOM (but do not introduce new failures).
            return

    def _eviction_candidates(
        self, device: torch.device, *, exclude_label: str | None = None
    ) -> List[str]:
        """
        Return component labels sorted by LRU (oldest first) for the given device.
        """
        comps: List[Tuple[float, str]] = []
        for label, comp in self._components.items():
            if exclude_label is not None and label == exclude_label:
                continue
            if comp.in_forward or self._is_effectively_pinned(comp):
                continue
            if comp.device is not None and comp.device.type != device.type:
                continue
            comps.append((comp.last_used, label))

        comps.sort(key=lambda x: x[0])
        return [label for _, label in comps]

    def _offload_component(
        self,
        comp: ManagedComponent,
        offload_type: str = "cpu",
        *,
        ignore_pins: bool = False,
    ) -> None:
        if (not ignore_pins) and self._is_effectively_pinned(comp):
            # Never evict components that were explicitly loaded and still "leased".
            # This is the core guarantee: load_component() cannot be accidentally
            # discarded/offloaded by pressure management.
            self._log_debug(f"[mem] skip offload pinned '{comp.label}' ({offload_type})")
            return

        module = comp.module()
        if module is None and offload_type != "discard":
            return

        logger.info(f"Offloading component: {comp.label} to {offload_type}")

        # Set thread-local flag to signal that this offload is FORCED by the manager
        self._evicting_flag.is_evicting = True
        try:
            if ignore_pins:
                # Forced cleanup explicitly overrides load_component leases.
                try:
                    self.clear_component_pins(comp.label, reason="force_offload")
                except Exception:
                    pass
            engine = comp.engine()
            try:
                if engine is not None and hasattr(engine, "_offload"):
                    # Must pass label (str) for discard so OffloadMixin can nullify the attribute.
                    # For CPU, passing module object is fine (and slightly faster/safer if attr changed).
                    target = (
                        comp.label
                        if offload_type == "discard"
                        else (module if module is not None else comp.label)
                    )
                    engine._offload(target, offload_type=offload_type)  # type: ignore[attr-defined]
                else:
                    if module is not None:
                        module.to("cpu")

                if offload_type == "discard":
                    comp.device = None
                else:
                    comp.device = torch.device("cpu")

                comp.last_used = time.time()
                self._log_debug(f"[mem] offloaded '{comp.label}' ({offload_type})")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Offload failed for {comp.label}: {exc}")
        finally:
            self._evicting_flag.is_evicting = False

    # --------- engine wiring ----------
    def _resolve_label(
        self, module: Union[str, torch.nn.Module, None]
    ) -> Optional[str]:
        if module is None:
            return None
        if isinstance(module, str):
            return module
        return self._module_to_label.get(id(module))

    def _normalize_modules(
        self,
        engine: Any,
        components: (
            Tuple[Union[str, torch.nn.Module], ...] | List[Union[str, torch.nn.Module]]
        ),
    ) -> List[torch.nn.Module]:
        mods: List[torch.nn.Module] = []
        for comp in components:
            mod = None
            if isinstance(comp, str):
                mod = getattr(engine, comp, None)
                if mod is None:
                    helpers = getattr(engine, "helpers", {})
                    mod = helpers.get(comp) if isinstance(helpers, dict) else None
            else:
                mod = comp
            if mod is not None and hasattr(mod, "to"):
                mods.append(mod)
        return mods

    def _wrap_to_device(self, engine: Any) -> None:
        if getattr(engine, "_apex_mem_to_device_wrapped", False):
            return
        if not hasattr(engine, "to_device"):
            return
        original = engine.to_device

        def _wrapped_to_device(self, *components, device=None):
            target_device = device
            if target_device is None:
                target_device = getattr(self, "device", None) or torch.device("cpu")
            if isinstance(target_device, str):
                target_device = torch.device(target_device)

            if components:
                modules = self._component_memory_manager._normalize_modules(
                    self, components
                )
            else:
                defaults: List[torch.nn.Module] = []
                for maybe_mod in (
                    getattr(self, "vae", None),
                    getattr(self, "text_encoder", None),
                    getattr(self, "transformer", None),
                    getattr(self, "scheduler", None),
                ):
                    if maybe_mod is not None:
                        defaults.append(maybe_mod)
                helpers_map = getattr(self, "_helpers", None) or getattr(
                    self, "helpers", None
                )
                if isinstance(helpers_map, dict):
                    defaults.extend([h for h in helpers_map.values() if h is not None])
                modules = self._component_memory_manager._normalize_modules(
                    self, tuple(defaults)
                )

            reserve = (
                self._component_memory_manager._estimate_allocation_bytes(modules)
                if modules
                else 0
            )
            self._component_memory_manager._ensure_room(target_device, reserve)

            result = original(*components, device=device)
            self._component_memory_manager.mark_moved(modules, target_device)
            return result

        engine.to_device = types.MethodType(_wrapped_to_device, engine)
        engine._apex_mem_to_device_wrapped = True

    def _wrap_offload(self, engine: Any) -> None:
        if getattr(engine, "_apex_mem_offload_wrapped", False):
            return
        if not hasattr(engine, "_offload"):
            return
        original = engine._offload

        def _wrapped_offload(self, module, *, offload_type: str = "discard"):
            # Intercept "discard" requests: try to keep in RAM (CPU) if space permits.
            # Lazy Offloading: If offload_type is "discard" or "cpu", we might prefer
            # to just mark it as IDLE and leave it on GPU until pressure forces it out.

            manager = getattr(self, "_component_memory_manager", None)

            final_offload_type = offload_type

            # 1. Resolve Module Object
            mod_obj = module
            if manager:
                try:
                    if isinstance(module, str):
                        # Try to resolve from registered components first
                        comp = manager._components.get(module)
                        if comp:
                            mod_obj = comp.module()

                        # Fallback to engine lookup
                        if mod_obj is None:
                            mod_obj = getattr(self, module, None)
                        if mod_obj is None:
                            # Check both _helpers and helpers
                            helpers = (
                                getattr(self, "_helpers", None)
                                or getattr(self, "helpers", None)
                                or {}
                            )
                            mod_obj = helpers.get(module)
                except Exception:
                    pass

            # Explicit offload indicates the caller is done using this component.
            # Release any lease pins so future pressure handling can evict/discard it.
            #
            # IMPORTANT: when `_offload(...)` is called with a string (common for helpers),
            # we must resolve to the *registered* label (via module id) to release the
            # correct lease.
            if manager:
                try:
                    label_to_release = None
                    if isinstance(module, str):
                        if mod_obj is not None and hasattr(manager, "_module_to_label"):
                            label_to_release = manager._module_to_label.get(id(mod_obj))
                        label_to_release = label_to_release or module
                    else:
                        if hasattr(manager, "_module_to_label"):
                            label_to_release = manager._module_to_label.get(id(module))
                        label_to_release = label_to_release or manager._resolve_label(module)

                    if label_to_release:
                        manager.release_component_pin(
                            str(label_to_release), reason="_offload"
                        )
                except Exception:
                    pass

            # 2. Downgrade Discard -> CPU if RAM allows
            if final_offload_type == "discard" and manager and mod_obj is not None:
                try:
                    size = manager._module_size_bytes(mod_obj)
                    cpu_dev = torch.device("cpu")
                    free, _ = manager._device_free_total(cpu_dev)
                    target_free = manager._target_free_bytes(cpu_dev, 0)

                    if free > (target_free + size + 2 * 1024**3):
                        final_offload_type = "cpu"
                except Exception:
                    pass

            # 3. Check for Forced Eviction vs Voluntary Offload
            is_forced = False
            if manager:
                if hasattr(manager, "_evicting_flag") and getattr(
                    manager._evicting_flag, "is_evicting", False
                ):
                    is_forced = True

            # 4. Lazy Offloading Logic
            # For VOLUNTARY cleanup calls, keep the module warm on accelerator if:
            # - The user-configured "keep free VRAM" target is already satisfied, OR
            # - We can satisfy it by evicting *other* cold components first.
            if (
                not is_forced
                and manager
                and mod_obj is not None
                and final_offload_type in {"cpu", "discard"}
            ):
                try:
                    if not manager._env_bool("APEX_DISABLE_LAZY_OFFLOAD", False):
                        current_dev = manager._module_device(mod_obj)
                        if current_dev.type != "cpu":
                            min_free = manager._offload_min_free_bytes(current_dev)
                            free_before, total_before = manager._device_free_total(
                                current_dev
                            )

                            exclude_label = None
                            try:
                                exclude_label = manager._resolve_label(module)
                            except Exception:
                                exclude_label = None

                            if total_before > 0 and free_before < min_free:
                                manager._ensure_min_free(
                                    current_dev, min_free, exclude_label=exclude_label
                                )

                            free_after, _ = manager._device_free_total(current_dev)
                            if min_free <= 0 or free_after >= min_free:
                                manager.mark_idle(module)
                                manager._flush_device_caches(current_dev)
                                return  # keep warm; pressure hooks will evict if needed
                except Exception:
                    pass

            # 5. Execute actual offload (if forced, or if moving to discard, or if already on CPU)
            result = original(module, offload_type=final_offload_type)
            try:
                if manager:
                    manager.mark_offloaded(module, offload_type=final_offload_type)
            except Exception:
                pass
            return result

        engine._offload = types.MethodType(_wrapped_offload, engine)
        engine._apex_mem_offload_wrapped = True

    def _wrap_load_component(self, engine: Any) -> None:
        """
        Ensure engine.load_component() creates a non-evictable lease for the loaded
        component. This prevents mid-run surprises like `engine.transformer` being
        set to None by pressure eviction before explicit offload.
        """
        if getattr(engine, "_apex_mem_load_component_wrapped", False):
            return
        if not hasattr(engine, "load_component"):
            return
        original = engine.load_component

        def _wrapped_load_component(self, component, *args, **kwargs):
            # Load-time preflight: ensure other resident components are offloaded
            # before we start materializing a new component's weights.
            manager = getattr(self, "_component_memory_manager", None)
            if manager is not None and hasattr(manager, "preflight_component_load"):
                try:
                    manager.preflight_component_load(engine=self, component=component)
                except Exception:
                    pass

            module = original(component, *args, **kwargs)
            # If a component loader returned None, that's a hard error: callers
            # expect a usable module, and we must not allow silent "None".
            if module is None:
                raise RuntimeError(
                    f"load_component returned None for component={component}"
                )

            # Mirror BaseEngine.load_component labeling rules.
            label = None
            tags: Set[str] | None = None
            try:
                if isinstance(component, dict):
                    ctype = component.get("type")
                    # Use stable labels for core engine-owned components so
                    # `_offload("vae")` / `_offload("transformer")` releases the
                    # correct lease. For non-core components, fall back to name.
                    core_labels = {"transformer", "vae", "text_encoder", "scheduler"}
                    if ctype in core_labels:
                        label = ctype
                    elif ctype == "helper":
                        label = component.get("name") or component.get("base") or ctype
                    else:
                        label = component.get("name") or ctype
                    if ctype:
                        tags = {str(ctype)}
            except Exception:
                label = None
                tags = None

            # Best-effort: ensure it's registered, then pin it.
            manager = getattr(self, "_component_memory_manager", None)
            if manager and label:
                try:
                    if hasattr(self, "_register_tracked_module"):
                        self._register_tracked_module(module, label, tags or {label})
                except Exception:
                    pass
                try:
                    # Ensure at least one lease exists (do not increment endlessly).
                    manager.ensure_component_pinned(label, min_pins=1, reason="load_component")
                except Exception:
                    pass

            return module

        engine.load_component = types.MethodType(_wrapped_load_component, engine)
        engine._apex_mem_load_component_wrapped = True

    def _wrap_run(self, engine: Any) -> None:
        """
        Wrap engine.run(...) so we can begin/end a "run session" for profiling.
        """
        if getattr(engine, "_apex_mem_run_wrapped", False):
            return
        if not hasattr(engine, "run"):
            return
        original = engine.run

        def _wrapped_run(self, *args: Any, **kwargs: Any):
            manager = getattr(self, "_component_memory_manager", None)
            if manager is not None and hasattr(manager, "begin_run"):
                try:
                    manager.begin_run(self)
                except Exception:
                    pass
            try:
                return original(*args, **kwargs)
            finally:
                if manager is not None and hasattr(manager, "end_run"):
                    try:
                        manager.end_run(self)
                    except Exception:
                        pass

        engine.run = types.MethodType(_wrapped_run, engine)
        engine._apex_mem_run_wrapped = True

    def _register_existing_components(self, engine: Any) -> None:
        for name in ("transformer", "vae", "text_encoder", "scheduler"):
            mod = getattr(engine, name, None)
            if mod is not None:
                self.register_component(
                    mod, name, {name}, engine=engine, pinned=name == "scheduler"
                )
                # If the engine already has these core modules loaded by the time
                # the manager is installed, treat them as leased: they must not
                # be accidentally discarded by pressure eviction mid-run.
                if name in {"transformer", "vae", "text_encoder"}:
                    self.ensure_component_pinned(name, min_pins=1, reason="install")
        helpers = getattr(engine, "_helpers", None) or getattr(engine, "helpers", None)
        if isinstance(helpers, dict):
            for key, helper in helpers.items():
                self.register_component(
                    helper, key, {"helper"}, engine=engine, pinned=False
                )
                # Treat pre-existing helpers as leased: do not allow pressure
                # eviction until an explicit engine `_offload(...)` happens.
                try:
                    self.ensure_component_pinned(str(key), min_pins=1, reason="install")
                except Exception:
                    pass

    def install_for_engine(self, engine: Any) -> None:
        """
        Attach manager to an engine instance: wrap to_device/offload, register
        any preloaded modules, and expose helper methods used inside BaseEngine.
        """
        setattr(engine, "_component_memory_manager", self)
        self._register_existing_components(engine)
        self._wrap_to_device(engine)
        self._wrap_offload(engine)
        self._wrap_load_component(engine)
        self._wrap_run(engine)

        # Expose helpers for BaseEngine.load_component path.
        def _register_tracked_module(self_obj, module, label, tags=None):
            return self.register_component(
                module, label, tags or set(), engine=self_obj
            )

        def _install_preforward_hook(self_obj, module, label):
            comp = self.register_component(
                module, label, {label}, engine=self_obj, install_hooks=False
            )
            if comp is not None and not comp.hooks:
                comp.hooks = self._attach_forward_hooks(module, label)
            return comp

        if not hasattr(engine, "_register_tracked_module"):
            engine._register_tracked_module = types.MethodType(
                _register_tracked_module, engine
            )
        if not hasattr(engine, "_install_preforward_hook"):
            engine._install_preforward_hook = types.MethodType(
                _install_preforward_hook, engine
            )


# -----------------------------
# Public helpers
# -----------------------------
_GLOBAL_MANAGER: ComponentMemoryManager | None = None
_GLOBAL_NOOP_MANAGER: "NoopComponentMemoryManager" | None = None


class NoopComponentMemoryManager:
    """
    No-op shim used when auto memory management is disabled.

    This preserves call sites that expect a manager object, while ensuring we do not
    wrap engine methods or install hooks.
    """

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def install_for_engine(self, engine: Any) -> None:
        # Keep a predictable attribute for downstream call sites.
        try:
            setattr(engine, "_component_memory_manager", self)
        except Exception:
            pass

    def register_component(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _attach_forward_hooks(self, *args: Any, **kwargs: Any) -> tuple:
        return tuple()

    def ensure_component_pinned(self, *args: Any, **kwargs: Any) -> None:
        return None

    def release_component_pin(self, *args: Any, **kwargs: Any) -> None:
        return None

    def mark_idle(self, *args: Any, **kwargs: Any) -> None:
        return None

    def mark_offloaded(self, *args: Any, **kwargs: Any) -> None:
        return None

    def force_offload_engine_components(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __getattr__(self, _name: str) -> Any:
        # Best-effort: swallow unexpected calls without exploding.
        def _noop(*_a: Any, **_kw: Any) -> None:
            return None

        return _noop


def _auto_memory_management_disabled() -> bool:
    try:
        return ComponentMemoryManager._env_bool(
            "APEX_DISABLE_AUTO_MEMORY_MANAGEMENT", False
        ) or ComponentMemoryManager._env_bool(
            "DISABLE_AUTO_MEMORY_MANAGEMENT", False
        )
    except Exception:
        raw = os.environ.get("APEX_DISABLE_AUTO_MEMORY_MANAGEMENT")
        return str(raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def get_memory_manager() -> ComponentMemoryManager | NoopComponentMemoryManager:
    global _GLOBAL_MANAGER, _GLOBAL_NOOP_MANAGER
    if _auto_memory_management_disabled():
        if _GLOBAL_NOOP_MANAGER is None:
            _GLOBAL_NOOP_MANAGER = NoopComponentMemoryManager()
        return _GLOBAL_NOOP_MANAGER  # type: ignore[return-value]
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ComponentMemoryManager()
    return _GLOBAL_MANAGER


def install_memory_hooks(engine: Any) -> ComponentMemoryManager | NoopComponentMemoryManager:
    """
    Attach the global memory manager to the given engine instance.

    This is idempotent: calling multiple times will reuse the same manager.
    """
    # If disabled, do not wrap engine methods; return a no-op manager instead.
    if _auto_memory_management_disabled():
        manager = get_memory_manager()
        try:
            manager.install_for_engine(engine)
        except Exception:
            pass
        return manager

    manager = get_memory_manager()
    manager.install_for_engine(engine)
    return manager


__all__ = [
    "MemoryConfig",
    "apply_group_offloading",
    "apply_budget_offloading",
    "apply_budget_offloading_profile",
    "BudgetOffloadProfile",
    "ComponentMemoryManager",
    "get_memory_manager",
    "install_memory_hooks",
]

__version__ = "2.1.0"
