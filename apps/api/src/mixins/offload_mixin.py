from __future__ import annotations

import gc
import inspect
import types
from typing import Literal
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from engine.base_engine import BaseEngine
    import torch
    import mlx.core as mx

    base_object = BaseEngine
else:
    base_object = object


def live_cuda_tensors():
    import torch

    out = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                out.append(obj)
            # also catch Parameters
            elif (
                hasattr(obj, "data") and torch.is_tensor(obj.data) and obj.data.is_cuda
            ):
                out.append(obj.data)
        except Exception:
            pass
    return out


def _tensor_bytes(t: torch.Tensor) -> int:
    # Only used at runtime; keep torch import lazy.
    return t.numel() * t.element_size()


def _varnames_in_dict(d: dict, target: object, limit: int = 8) -> list[str]:
    names: list[str] = []
    for k, v in d.items():
        try:
            if v is target:
                names.append(repr(k))
        except Exception:
            pass
        if len(names) >= limit:
            break
    return names


def _attrnames_in_obj(obj: object, target: object, limit: int = 8) -> list[str]:
    names: list[str] = []
    try:
        od = getattr(obj, "__dict__", None)
        if isinstance(od, dict):
            for k, v in od.items():
                if v is target:
                    names.append(k)
                    if len(names) >= limit:
                        break
    except Exception:
        pass
    return names


def _frame_locals_holding(
    frame: types.FrameType, target: object, limit: int = 8
) -> list[str]:
    names: list[str] = []
    try:
        for k, v in frame.f_locals.items():
            if v is target:
                names.append(k)
                if len(names) >= limit:
                    break
    except Exception:
        pass
    return names


def _indices_in_seq(seq: object, target: object, limit: int = 8) -> list[int]:
    idxs: list[int] = []
    try:
        for i, v in enumerate(seq):  # type: ignore[arg-type]
            if v is target:
                idxs.append(i)
                if len(idxs) >= limit:
                    break
    except Exception:
        pass
    return idxs


def _find_paths(
    root: object,
    target: object,
    *,
    prefix: str,
    max_depth: int = 4,
    max_paths: int = 10,
) -> list[str]:
    """
    Best-effort path finder: returns strings like `self.foo['bar'][0]` if `target`
    is reachable from `root` via common Python containers and object attributes.
    """
    paths: list[str] = []
    seen: set[int] = set()

    def walk(obj: object, path: str, depth: int) -> None:
        if len(paths) >= max_paths:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if obj is target:
            paths.append(path)
            return

        if depth <= 0:
            return

        # dict
        if isinstance(obj, dict):
            try:
                for k, v in list(obj.items())[:200]:
                    if len(paths) >= max_paths:
                        return
                    if v is target:
                        paths.append(f"{path}[{k!r}]")
                    else:
                        # avoid recursing into simple scalars
                        if isinstance(v, (str, bytes, int, float, bool, type(None))):
                            continue
                        walk(v, f"{path}[{k!r}]", depth - 1)
            except Exception:
                return
            return

        # list/tuple
        if isinstance(obj, (list, tuple)):
            try:
                for i, v in enumerate(obj[:200]):
                    if len(paths) >= max_paths:
                        return
                    if v is target:
                        paths.append(f"{path}[{i}]")
                    else:
                        if isinstance(v, (str, bytes, int, float, bool, type(None))):
                            continue
                        walk(v, f"{path}[{i}]", depth - 1)
            except Exception:
                return
            return

        # set/frozenset (no stable index)
        if isinstance(obj, (set, frozenset)):
            try:
                for v in list(obj)[:200]:
                    if len(paths) >= max_paths:
                        return
                    if v is target:
                        paths.append(f"{path}{{<set contains target>}}")
                        return
                    if isinstance(v, (str, bytes, int, float, bool, type(None))):
                        continue
                    walk(v, f"{path}{{...}}", depth - 1)
            except Exception:
                return
            return

        # generic object attributes
        try:
            od = getattr(obj, "__dict__", None)
            if isinstance(od, dict):
                for k, v in list(od.items())[:200]:
                    if len(paths) >= max_paths:
                        return
                    if v is target:
                        paths.append(f"{path}.{k}")
                    else:
                        if isinstance(v, (str, bytes, int, float, bool, type(None))):
                            continue
                        walk(v, f"{path}.{k}", depth - 1)
        except Exception:
            return

    walk(root, prefix, max_depth)
    return paths


def holders_for_tensor(
    t: torch.Tensor, *, self_obj=None, max_referrers: int = 30
) -> list[str]:
    """
    Best-effort "who is holding this tensor alive" by inspecting Python referrers.
    Note: this won't explain CUDA caching allocator 'reserved' memory.
    """
    refs: list[str] = []
    for r in gc.get_referrers(t):
        if r is refs:
            continue

        # Skip very noisy internals.
        if isinstance(
            r,
            (
                types.ModuleType,
                types.FunctionType,
                types.MethodType,
                types.CodeType,
                type,
            ),
        ):
            continue

        if isinstance(r, types.FrameType):
            info = inspect.getframeinfo(r)
            locs = _frame_locals_holding(r, t)
            refs.append(
                f"frame {info.filename}:{info.lineno} in {info.function} locals={locs}"
            )
            continue

        if isinstance(r, dict):
            owner = ""
            if self_obj is not None:
                try:
                    if r is getattr(self_obj, "__dict__", None):
                        owner = " (self.__dict__)"
                except Exception:
                    pass
            # If this looks like a module globals dict, surface the module name.
            try:
                mod_name = r.get("__name__")
                if isinstance(mod_name, str):
                    owner = f"{owner} (globals:{mod_name})"
            except Exception:
                pass
            keys = _varnames_in_dict(r, t)
            refs.append(f"dict{owner} keys={keys[:8]}")
            continue

        if isinstance(r, (list, tuple)):
            idxs = _indices_in_seq(r, t)
            refs.append(f"{type(r).__name__} idxs={idxs[:8]}")
            continue

        if isinstance(r, (set, frozenset)):
            # No stable index; just indicate membership.
            refs.append(f"{type(r).__name__} (contains target)")
            continue

        attrs = _attrnames_in_obj(r, t)
        if attrs:
            refs.append(f"{type(r).__name__} attrs={attrs[:8]}")
            continue

        refs.append(f"{type(r).__name__} id={id(r)}")

        if len(refs) >= max_referrers:
            break

    return refs


def _describe_obj(o: object) -> str:
    try:
        r = repr(o)
    except Exception:
        r = "<repr failed>"
    if len(r) > 160:
        r = r[:157] + "..."
    return f"{type(o).__name__} id={id(o)} repr={r}"


def _container_context(container: object, *, self_obj=None) -> list[str]:
    """
    Try to answer: "what is holding this container?" so list/dict hits become actionable.
    """
    ctx: list[str] = []

    if self_obj is not None:
        try:
            paths = _find_paths(
                self_obj, container, prefix="self", max_depth=4, max_paths=6
            )
            for p in paths[:6]:
                ctx.append(f"via {p}")
        except Exception:
            pass

    # try one level of referrers to label who owns this container
    try:
        for rr in gc.get_referrers(container):
            # skip very noisy internals
            if isinstance(
                rr,
                (
                    types.ModuleType,
                    types.FunctionType,
                    types.MethodType,
                    types.CodeType,
                    type,
                    types.FrameType,
                ),
            ):
                continue

            if isinstance(rr, dict):
                owner = ""
                try:
                    mod_name = rr.get("__name__")
                    if isinstance(mod_name, str):
                        owner = f"(globals:{mod_name})"
                except Exception:
                    pass
                keys = _varnames_in_dict(rr, container, limit=6)
                ctx.append(f"held by dict {owner} keys={keys[:6]}")
                if len(ctx) >= 10:
                    break
                continue

            # generic object attribute holder
            attrs = _attrnames_in_obj(rr, container, limit=6)
            if attrs:
                ctx.append(f"held by {type(rr).__name__} attrs={attrs[:6]}")
                if len(ctx) >= 10:
                    break
                continue

            # fallback
            ctx.append(f"held by {_describe_obj(rr)}")
            if len(ctx) >= 10:
                break
    except Exception:
        pass

    return ctx


class OffloadMixin(base_object):
    """
    Add to any class that owns a torch.nn.Module (e.g. your Trainer or Model
    wrapper).

    Recommended usage in Apex engines is to offload by **component name**:
        `self._offload("transformer")`
    because that allows safe "discard" semantics (dropping references on the engine).

    If you pass a module object directly, only `"cpu"` offload is supported; `"discard"`
    requires a string name so the engine can sever references.

    Example
    -------
    class MyRunner(OffloadMixin):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def teardown(self):
            # Explicitly choose a behavior for module objects.
            self._offload(self.model, offload_type="cpu")
    """

    def _offload(
        self: "BaseEngine",
        module: torch.nn.Module | str | None,
        *,
        offload_type: Literal["cpu", "discard"] | None = None
    ) -> None:
        """
        Pressure-aware offload helper.

        - If `offload_type` is explicitly set ("cpu" / "discard"), we do it immediately.
        - If `offload_type` is None (default), we keep the module warm and only offload
          when we're under RAM/VRAM pressure.

        Parameters
        ----------
        module   : torch.nn.Module
            The module whose tensors you want to off-load.
        recurse  : bool, default True
            Whether to descend into sub-modules (almost always what you want).
        delete_from_cpu : bool, default True
            If True, remove parameters and buffers from the module after off-loading,
            allowing them to be garbage-collected so the module holds no tensors.
        """

        # if isinstance(module, MlxModule):
        #     OffloadMixin._mlx_offload(module, delete_from_cpu=delete_from_cpu)
        #     if delete_from_cpu:
        #         self._null_module_refs(module)
        #     return

        import os
        import torch

        def _env_float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return float(str(raw).strip())
            except Exception:
                return default

        def _cpu_free_fraction() -> float | None:
            try:
                import psutil  # type: ignore

                vm = psutil.virtual_memory()
                if vm.total <= 0:
                    return None
                return float(vm.available) / float(vm.total)
            except Exception:
                return None

        def _cuda_free_fraction() -> float | None:
            try:
                if not torch.cuda.is_available():
                    return None
                free, total = torch.cuda.mem_get_info()
                if total <= 0:
                    return None
                return float(free) / float(total)
            except Exception:
                return None

        # Default policy: keep warm until pressure, then offload safely.
        if offload_type is None:
            min_free_vram = _env_float("APEX_OFFLOAD_MIN_FREE_VRAM_FRACTION", 0.10)
            min_free_ram = _env_float("APEX_OFFLOAD_MIN_FREE_RAM_FRACTION", 0.08)

            vram_free = _cuda_free_fraction()
            ram_free = _cpu_free_fraction()

            under_vram = vram_free is not None and vram_free < min_free_vram
            under_ram = ram_free is not None and ram_free < min_free_ram

            # If neither is under pressure, keep everything as-is (warm).
            if not under_vram and not under_ram:
                return

            # Under pressure: choose the least risky action.
            # - Prefer offload-to-CPU only when RAM headroom is healthy.
            # - Otherwise discard to avoid CPU OOM.
            if under_vram and (ram_free is None or ram_free >= (min_free_ram + 0.05)):
                offload_type = "cpu"
            else:
                offload_type = "discard"

        # Keep no_grad behavior without importing torch at module import time.
        with torch.no_grad():
            if not module:
                return

            manager = None
            try:
                from src.memory_management import get_global_weight_manager

                manager = get_global_weight_manager()
            except Exception:
                manager = None

            module_obj = None
            module_id = None

            if isinstance(module, str):
                # IMPORTANT:
                # Keep a local reference for the duration of this function, but ensure we
                # explicitly drop it BEFORE gc.collect()/empty_cache(). Otherwise, the module
                # stays alive until this function returns and VRAM won't be reclaimed yet.
                module_obj = getattr(self, module, None)
                if module_obj is None:
                    # check if module is a helper
                    if module in self._helpers:
                        module_obj = self._helpers[module]
                    else:
                        return
                try:
                    if manager is not None:
                        module_id = getattr(self, "_component_memory_ids", {}).get(
                            module
                        )
                        register_fn = getattr(self, "_register_tracked_module", None)
                        if module_id is None and callable(register_fn):
                            register_fn(module_obj, module)
                            module_id = getattr(self, "_component_memory_ids", {}).get(
                                module
                            )
                        if module_id:
                            manager.refresh(module_id)
                except Exception:
                    module_id = None

                if offload_type == "cpu":
                    if manager is not None and module_id:
                        manager.offload_module(
                            module_id, target="cpu", reason="offload_cpu"
                        )
                    else:
                        module_obj.to("cpu")
                elif offload_type == "discard":
                    if manager is not None and module_id:
                        try:
                            # Force-disk-only mode: "disk" semantics are pure discard.
                            # Do NOT serialize weights; drop tracking so the only way
                            # to access again is to reload from the original source.
                            if getattr(manager, "force_disk_only", False):
                                manager.forget(module_id)
                            else:
                                manager.offload_module(
                                    module_id,
                                    target="disk",
                                    drop_cpu=True,
                                    reason="offload_discard",
                                )
                        except Exception:
                            pass
                    component = self.get_component_by_name(module)
                    if component:
                        module_type_obj = getattr(self, component.get("type"), None)

                        if module_type_obj is module_obj:
                            self.logger.info(f"Setting {component.get('type')} to None")
                            setattr(self, component.get("type"), None)
                    else:
                        component = self.get_component_by_type(module)
                    # check if type is helper
                    if component:
                        if component.get("type") == "helper":
                            try:
                                self._helpers.pop(component.get("base"))
                            except Exception:
                                pass
                            try:
                                self._helpers.pop(component.get("name"))
                            except Exception:
                                pass
                    self.logger.info(f"Setting {module} to None")
                    setattr(self, module, None)
            else:
                module_obj = module
                try:
                    if manager is not None:
                        module_id = getattr(module_obj, "_apex_mem_id", None)
                        register_fn = getattr(self, "_register_tracked_module", None)
                        if module_id is None and callable(register_fn):
                            register_fn(
                                module_obj, getattr(module_obj, "__class__", type("X", (), {})).__name__
                            )
                            module_id = getattr(module_obj, "_apex_mem_id", None)
                        if module_id:
                            manager.refresh(module_id)
                except Exception:
                    module_id = None

                if offload_type == "cpu":
                    if manager is not None and module_id:
                        manager.offload_module(
                            module_id, target="cpu", reason="offload_cpu"
                        )
                    else:
                        module.to("cpu")
                elif offload_type == "discard":
                    raise ValueError(
                        f"Invalid offload type: {offload_type} for module."
                    )

        # Drop strong references created in this function before collecting/clearing.
        # This makes offloading effective immediately (instead of only after return).
        try:
            del module_type_obj
        except Exception:
            pass
        try:
            del component
        except Exception:
            pass
        try:
            del module_obj
        except Exception:
            pass

        gc.collect()
        # 4)  Reclaim CUDA VRAM
        if torch.cuda.is_available():
            # Helps make memory reporting deterministic; does not significantly impact runtime here.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            # Optional: reset statistics so future profiling starts fresh
            for dev_id in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev_id)
        # 5)  Reclaim Apple-silicon MPS memory
        if (
            getattr(torch, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()

        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        
        gc.collect()
