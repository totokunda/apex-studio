from __future__ import annotations

import hashlib
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Set, Tuple

import torch

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil should be available, but stay defensive
    psutil = None  # type: ignore

from src.utils.defaults import get_offload_path

MemoryTier = Literal["gpu", "cpu", "disk"]


@dataclass
class TensorMeta:
    kind: Literal["param", "buffer"]
    name: str
    numel: int
    element_size: int
    dtype: torch.dtype
    device_type: str

    @property
    def size_bytes(self) -> int:
        return int(self.numel * self.element_size)


@dataclass
class TrackedModule:
    module_id: str
    owner: Optional[str]
    module: torch.nn.Module
    tensors: Dict[str, TensorMeta] = field(default_factory=dict)
    location: MemoryTier = "cpu"
    disk_path: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    last_used: float = field(default_factory=time.monotonic)

    @property
    def total_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors.values())


class GlobalWeightManager:
    """
    Global, process-wide manager that tracks where model weights live (GPU / CPU / disk)
    and moves them across tiers to keep weights warm while avoiding OOMs.

    The manager keeps a lightweight reference to every tracked tensor so we can
    offload or reload any module on demand, independent of the engine instance
    that created it.
    """

    def __init__(
        self,
        *,
        disk_root: Optional[str] = None,
        force_disk_only: Optional[bool] = None,
        warm_cache_enabled: Optional[bool] = None,
        gpu_stats_provider=None,
        ram_stats_provider=None,
    ) -> None:
        self._lock = threading.RLock()
        self._modules: Dict[str, TrackedModule] = {}
        self._module_index: "weakref.WeakKeyDictionary[torch.nn.Module, str]" = (
            weakref.WeakKeyDictionary()
        )
        # Track "currently running" modules in a thread-local stack so that
        # background loads/evictions never kick out the active component.
        self._active_local = threading.local()
        self.disk_root = Path(disk_root or get_offload_path())
        self.disk_root.mkdir(parents=True, exist_ok=True)

        self.warm_cache_enabled = (
            warm_cache_enabled
            if warm_cache_enabled is not None
            else not self._env_bool("APEX_DISABLE_WARM_WEIGHTS", False)
        )
        self.force_disk_only = (
            force_disk_only
            if force_disk_only is not None
            else self._env_bool("APEX_FORCE_DISK_ONLY", False)
            or not self.warm_cache_enabled
        )

        self.target_free_vram_fraction = self._env_float(
            "APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION", 0.12
        )
        self.target_free_ram_fraction = self._env_float(
            "APEX_WEIGHT_TARGET_FREE_RAM_FRACTION", 0.10
        )
        
        self._gpu_stats_provider = gpu_stats_provider or self._default_gpu_stats
        self._ram_stats_provider = ram_stats_provider or self._default_ram_stats

    # --------------------------------------------------------------------- #
    # Active module tracking (thread-local)
    # --------------------------------------------------------------------- #
    def _active_stack(self) -> list[str]:
        stack = getattr(self._active_local, "stack", None)
        if stack is None:
            stack = []
            setattr(self._active_local, "stack", stack)
        return stack

    def active_ids(self) -> Set[str]:
        try:
            return set(self._active_stack())
        except Exception:
            return set()

    def push_active(self, module_id: str) -> None:
        if not module_id:
            return
        try:
            self._active_stack().append(str(module_id))
        except Exception:
            return

    def pop_active(self, module_id: str | None = None) -> None:
        try:
            stack = self._active_stack()
        except Exception:
            return
        if not stack:
            return
        if module_id is None:
            try:
                stack.pop()
            except Exception:
                return
            return

        target = str(module_id)
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] == target:
                try:
                    stack.pop(i)
                except Exception:
                    pass
                break

    # --------------------------------------------------------------------- #
    # Environment helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    # --------------------------------------------------------------------- #
    # Stats providers
    # --------------------------------------------------------------------- #
    def _default_gpu_stats(self) -> Optional[Tuple[int, int, float]]:
        try:
            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            if total <= 0:
                return None
            return int(free), int(total), float(free) / float(total)
        except Exception:
            return None

    def _default_ram_stats(self) -> Optional[Tuple[int, int, float]]:
        if psutil is None:
            return None
        try:
            vm = psutil.virtual_memory()
            if vm.total <= 0:
                return None
            return int(vm.available), int(vm.total), float(vm.available) / float(
                vm.total
            )
        except Exception:
            return None

    # --------------------------------------------------------------------- #
    # Registration / bookkeeping
    # --------------------------------------------------------------------- #
    def register_module(
        self,
        module: torch.nn.Module,
        module_id: str,
        *,
        owner: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> str:
        """Register (or refresh) a module so it participates in global memory control."""
        if not isinstance(module, torch.nn.Module):
            return module_id

        with self._lock:
            tensors = self._collect_tensor_meta(module)
            location = self._detect_location(module)
            record = TrackedModule(
                module_id=module_id,
                owner=owner,
                module=module,
                tensors=tensors,
                location=location,
                tags=set(tags or ()),
            )
            record.last_used = time.monotonic()
            module.__dict__["_apex_mem_id"] = module_id
            self._modules[module_id] = record
            self._module_index[module] = module_id

            # Force-disk-only mode is treated as "no warm caching": we *do not*
            # try to serialize+shrink weights, because the intended policy is
            # to discard modules at the engine layer and reload on demand.
            #
            # Still, if a module was registered while on GPU, move it back to CPU
            # so we don't pin VRAM indefinitely.
            if self.force_disk_only and record.location == "gpu":
                try:
                    module.to("cpu")
                    record.location = "cpu"
                    record.tensors = self._collect_tensor_meta(module)
                except Exception:
                    pass
            elif record.location == "gpu":
                # If we registered a GPU-resident module and we're already tight on VRAM,
                # proactively evict older modules to keep headroom.
                try:
                    self.evict_for_vram(
                        reason="post_register",
                        active={module_id},
                        target_free_fraction=self.target_free_vram_fraction,
                    )
                except Exception:
                    pass
            return module_id

    def refresh(self, module_or_id) -> None:
        """Recompute tensor metadata after in-place changes (e.g. lazy model load)."""
        with self._lock:
            record = self._resolve_record(module_or_id)
            if record is None:
                return
            record.tensors = self._collect_tensor_meta(record.module)
            record.location = self._detect_location(record.module)
            record.last_used = time.monotonic()

    # --------------------------------------------------------------------- #
    # Core operations
    # --------------------------------------------------------------------- #
    @staticmethod
    def _is_group_offloaded_module(module: torch.nn.Module) -> bool:
        """
        Return True when `module` is controlled by Diffusers group offloading.

        Group-offloaded modules manage their own device placement via hooks, so
        forcing `.to(device)` here is redundant and can fight the hook logic.
        """
        try:
            from src.mixins.to_mixin import _module_has_group_offload_hook

            return bool(_module_has_group_offload_hook(module))
        except Exception:
            return False

    def ensure_on_device(
        self,
        module_or_id,
        *,
        device: torch.device | str | None = None,
        reason: str = "",
    ) -> bool:
        """
        Ensure a module is fully materialized on `device`, reloading from disk if needed.
        """
        record = self._resolve_record(module_or_id)
        if record is None:
            return False
        module = record.module
        if module is None:
            return False

        if device is None:
            target = None
        else:
            target = torch.device(device)

        with self._lock, torch.no_grad():
            is_group_offloaded = self._is_group_offloaded_module(module)

            if record.location == "disk":
                if record.disk_path:
                    state = torch.load(record.disk_path, map_location="cpu")
                    params = dict(module.named_parameters(recurse=True))
                    buffers = dict(module.named_buffers(recurse=True))
                    for name, tensor in state.items():
                        # Group-offloaded modules should be restored to CPU; the hook
                        # will move submodules/tensors on-demand.
                        dest_device = (
                            torch.device("cpu")
                            if is_group_offloaded
                            else (target or torch.device("cpu"))
                        )
                        restored = tensor.to(dest_device)
                        if name in params:
                            params[name].data = restored
                        elif name in buffers:
                            try:
                                buffers[name].data = restored
                            except Exception:
                                module._buffers[name] = restored  # type: ignore[attr-defined]
                    record.location = "cpu"
                    record.tensors = self._collect_tensor_meta(module)
                else:
                    # No serialized state to pull from; best effort: keep as-is and bail.
                    # Callers should reload from the original checkpoint when they see a False return.
                    return False

            # If this module is group-offloaded, it controls its own placement and
            # ensuring here is unnecessary (and can be harmful).
            if target is not None and not is_group_offloaded:
                module.to(target)
                record.location = target.type

            record.last_used = time.monotonic()
            record.tensors = self._collect_tensor_meta(module)
            return True

    def offload_module(
        self,
        module_or_id,
        *,
        target: MemoryTier = "cpu",
        drop_cpu: Optional[bool] = None,
        reason: str = "",
    ) -> Optional[MemoryTier]:
        """
        Move a module to CPU or disk while keeping a handle so it can be rehydrated.
        """
        record = self._resolve_record(module_or_id)
        if record is None:
            return None

        drop_cpu = self.force_disk_only or bool(drop_cpu)
        target_tier: MemoryTier = "disk" if target == "disk" else target

        with self._lock, torch.no_grad():
            module = record.module
            if module is None:
                return None

            if target_tier == "cpu":
                if record.location != "cpu":
                    module.to("cpu")
                record.location = "cpu"
            elif target_tier == "disk":
                # In force-disk-only mode, "disk" effectively means "caller should discard and reload".
                # The weight manager cannot safely null all external references, so here we avoid
                # producing meta/empty tensors (which would cause runtime 0-numel failures) and
                # simply move to CPU. Engines using OffloadMixin will drop references themselves.
                if self.force_disk_only:
                    try:
                        if record.location != "cpu":
                            module.to("cpu")
                    except Exception:
                        pass
                    record.location = "cpu"
                    record.tensors = self._collect_tensor_meta(module)
                    record.last_used = time.monotonic()
                    return record.location

                # If the module is already meta-backed, there is nothing to move.
                # Avoid attempting to materialize / save meta tensors.
                has_meta = any(
                    (getattr(p, "is_meta", False) or str(p.device) == "meta")
                    for p in module.parameters(recurse=True)
                ) or any(
                    (getattr(b, "is_meta", False) or str(b.device) == "meta")
                    for b in module.buffers(recurse=True)
                )

                if record.location != "cpu" and not has_meta:
                    module.to("cpu")

                # If we're truly evicting to disk (dropping CPU weights), we must persist
                # a copy first. Otherwise modules end up with 0-numel/meta tensors and
                # cannot be rehydrated via `ensure_on_device(...)`.
                if drop_cpu and not has_meta:
                    try:
                        disk_path = Path(
                            record.disk_path
                            or str(self._module_disk_path(record.module_id))
                        )
                        tmp_path = disk_path.with_suffix(disk_path.suffix + ".tmp")

                        # Save a CPU state dict. At this point the module should already
                        # be on CPU (see the `.to("cpu")` above), but be defensive.
                        state = {
                            k: (v.detach().cpu() if torch.is_tensor(v) else v)
                            for k, v in module.state_dict().items()
                        }
                        torch.save(state, tmp_path)
                        tmp_path.replace(disk_path)
                        record.disk_path = str(disk_path)
                    except Exception:
                        # Best-effort: if we fail to serialize, do NOT shrink the module;
                        # keep weights in CPU memory so the model remains runnable.
                        record.location = "cpu"
                        record.tensors = self._collect_tensor_meta(module)
                        record.last_used = time.monotonic()
                        return record.location

                    self._shrink_module_tensors(module)
                    record.location = "disk"
                else:
                    # Keep a CPU copy if we cannot safely drop to meta or weights are already meta.
                    record.location = "cpu" if not has_meta else "disk"

            record.tensors = self._collect_tensor_meta(module)
            record.last_used = time.monotonic()

        # Clearing CUDA cache after moving weights out prevents allocator spikes.
        if target_tier in {"cpu", "disk"} and self._gpu_stats_provider() is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return record.location

    def evict_for_vram(
        self,
        *,
        reason: str = "",
        active: Optional[Set[str]] = None,
        target_free_fraction: Optional[float] = None,
        request_bytes: Optional[int] = None,
    ) -> Dict[str, MemoryTier]:
        """
        Offload least-recently-used GPU modules until VRAM headroom is restored.
        """
        stats = self._gpu_stats_provider()
        if stats is None:
            return {}
        free_bytes, total_bytes, free_frac = stats
        target = (
            target_free_fraction
            if target_free_fraction is not None
            else self.target_free_vram_fraction
        )

        need_bytes = 0
        if total_bytes and target is not None:
            target_bytes = int(total_bytes * target)
            if free_bytes < target_bytes:
                need_bytes = target_bytes - free_bytes
        if request_bytes is not None and request_bytes > free_bytes:
            need_bytes = max(need_bytes, int(request_bytes - free_bytes))

        if need_bytes <= 0 and (free_frac is None or free_frac >= target):
            return {}

        active_set = set(active or set())
        try:
            active_set |= self.active_ids()
        except Exception:
            pass
        candidates = [
            rec
            for rec in self._modules.values()
            if rec.location == "gpu" and rec.module_id not in active_set
        ]
        # Oldest, largest modules first.
        candidates.sort(key=lambda r: (r.last_used, -r.total_bytes))
        ram_stats = self._ram_stats_provider()
        offloaded: Dict[str, MemoryTier] = {}

        for rec in candidates:
            if need_bytes <= 0:
                updated = self._gpu_stats_provider()
                if updated is not None:
                    free_bytes, total_bytes, free_frac = updated
                    target_bytes = (
                        int(total_bytes * target) if total_bytes and target else 0
                    )
                    if free_bytes >= target_bytes:
                        break

            tier = self._choose_tier(rec, ram_stats)
            self.offload_module(
                rec.module_id,
                target=tier,
                drop_cpu=(tier == "disk"),
                reason=reason or "evict_for_vram",
            )
            offloaded[rec.module_id] = tier
            need_bytes = max(0, need_bytes - rec.total_bytes)

        return offloaded

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _choose_tier(
        self, record: TrackedModule, ram_stats: Optional[Tuple[int, int, float]]
    ) -> MemoryTier:
        if self.force_disk_only:
            return "disk"
        if ram_stats is None:
            return "cpu"
        available, total, frac = ram_stats
        if total > 0 and frac < self.target_free_ram_fraction:
            return "disk"
        # Avoid aggressive CPU caching when a single module would consume most RAM.
        if record.total_bytes > int(available * 0.6):
            return "disk"
        return "cpu"

    def _resolve_record(self, module_or_id) -> Optional[TrackedModule]:
        if module_or_id is None:
            return None
        if isinstance(module_or_id, str):
            return self._modules.get(module_or_id)
        return self._modules.get(self._module_index.get(module_or_id, ""))

    def offload_gpu_except(
        self,
        active: Set[str] | None = None,
        *,
        target: MemoryTier = "cpu",
        reason: str = "",
    ) -> Dict[str, MemoryTier]:
        """
        Offload all GPU-resident modules except those in `active`.
        """
        active_set = set(active or set())
        try:
            active_set |= self.active_ids()
        except Exception:
            pass
        offloaded: Dict[str, MemoryTier] = {}
        with self._lock:
            candidates = [
                rec
                for rec in self._modules.values()
                if rec.location == "gpu" and rec.module_id not in active_set
            ]
        for rec in candidates:
            try:
                loc = self.offload_module(
                    rec.module_id,
                    target=target,
                    drop_cpu=(target == "disk"),
                    reason=reason or "offload_gpu_except",
                )
                if loc is not None:
                    offloaded[rec.module_id] = loc
            except Exception:
                continue
        try:
            if str(os.environ.get("APEX_MEM_DEBUG", "")).lower() in {"1", "true", "yes"}:
                print(
                    f"[weight_manager] offload_gpu_except reason={reason} target={target} "
                    f"active={active} offloaded={list(offloaded.keys())}"
                )
        except Exception:
            pass
        # After offloading, clear caches to free allocator reservations.
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        return offloaded

    def _collect_tensor_meta(self, module: torch.nn.Module) -> Dict[str, TensorMeta]:
        tensors: Dict[str, TensorMeta] = {}
        for name, p in module.named_parameters(recurse=True):
            if p is None:
                continue
            tensors[name] = TensorMeta(
                kind="param",
                name=name,
                numel=int(p.numel()),
                element_size=p.element_size(),
                dtype=p.dtype,
                device_type=p.device.type,
            )
        for name, b in module.named_buffers(recurse=True):
            if b is None:
                continue
            tensors[name] = TensorMeta(
                kind="buffer",
                name=name,
                numel=int(b.numel()),
                element_size=b.element_size(),
                dtype=b.dtype,
                device_type=b.device.type,
            )
        return tensors

    def _detect_location(self, module: torch.nn.Module) -> MemoryTier:
        for p in module.parameters():
            if p is None:
                continue
            return "gpu" if getattr(p.device, "type", "") == "cuda" else "cpu"
        for b in module.buffers():
            if b is None:
                continue
            return "gpu" if getattr(b.device, "type", "") == "cuda" else "cpu"
        return "cpu"

    def refresh_all_locations(self) -> None:
        """Best-effort: refresh location metadata for all tracked modules."""
        with self._lock:
            for rec in self._modules.values():
                try:
                    rec.location = self._detect_location(rec.module)
                except Exception:
                    continue

    def _module_disk_path(self, module_id: str) -> Path:
        h = hashlib.sha1(module_id.encode("utf-8")).hexdigest()
        return self.disk_root / f"{h}.pt"

    @staticmethod
    def _shrink_module_tensors(module: torch.nn.Module) -> None:
        # Replace parameter/buffer storage with lightweight meta tensors to release RAM.
        with torch.no_grad():
            for name, param in module.named_parameters(recurse=True):
                if param is None:
                    continue
                try:
                    param.data = torch.empty(
                        param.shape, device="meta", dtype=param.dtype
                    )
                except Exception:
                    param.data = torch.empty(
                        (0,), device="cpu", dtype=param.dtype
                    )
            for name, buf in module.named_buffers(recurse=True):
                if buf is None:
                    continue
                try:
                    empty = torch.empty(buf.shape, device="meta", dtype=buf.dtype)
                except Exception:
                    empty = torch.empty((0,), device="cpu", dtype=buf.dtype)
                try:
                    module._buffers[name] = empty  # type: ignore[attr-defined]
                except Exception:
                    pass

    # --------------------------------------------------------------------- #
    # Introspection
    # --------------------------------------------------------------------- #
    def summary(self) -> Dict[str, Dict[str, float | int]]:
        with self._lock:
            gpu_bytes = sum(
                rec.total_bytes for rec in self._modules.values() if rec.location == "gpu"
            )
            cpu_bytes = sum(
                rec.total_bytes for rec in self._modules.values() if rec.location == "cpu"
            )
            disk = sum(
                1 for rec in self._modules.values() if rec.location == "disk"
            )
            return {
                "counts": {
                    "modules": len(self._modules),
                    "gpu": len(
                        [rec for rec in self._modules.values() if rec.location == "gpu"]
                    ),
                    "cpu": len(
                        [rec for rec in self._modules.values() if rec.location == "cpu"]
                    ),
                    "disk": disk,
                },
                "bytes": {
                    "gpu": gpu_bytes,
                    "cpu": cpu_bytes,
                },
            }

    def forget(self, module_or_id) -> None:
        """Drop tracking for a module (e.g., when an engine is torn down)."""
        with self._lock:
            record = self._resolve_record(module_or_id)
            if record is None:
                return
            self._modules.pop(record.module_id, None)
            try:
                self._module_index.pop(record.module)
            except Exception:
                pass


_GLOBAL_MANAGER: Optional[GlobalWeightManager] = None


def get_global_weight_manager() -> GlobalWeightManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = GlobalWeightManager()
    return _GLOBAL_MANAGER
