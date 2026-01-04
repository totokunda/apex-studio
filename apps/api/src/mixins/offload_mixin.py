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
    wrapper).  Call `self._offload(self.model)` when you are finished with a
    module and want to give the accelerator memory back.

    Example
    -------
    class MyRunner(OffloadMixin):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def teardown(self):
            self._offload(self.model)   # <- frees VRAM / MRAM
    """

    def _offload(
        self: "BaseEngine",
        module: torch.nn.Module | str | None,
        *,
        offload_type: Literal["cpu", "discard"] = "discard",
    ) -> None:
        """
        Move every weight/buffer to CPU **and** clear CUDA/MPS/CPU caches.
        Optionally (default) also delete the module's parameters and buffers so it no
        longer occupies CPU or accelerator memory.

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

        import torch

        # Keep no_grad behavior without importing torch at module import time.
        with torch.no_grad():
            if not module:
                return

            if isinstance(module, str):
                module_obj = getattr(self, module, None)
                if module_obj is None:
                    return
                if offload_type == "cpu":
                    module_obj.to("cpu")
                elif offload_type == "discard":
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
                                self._helpers.pop(component.get("name"))
                            except Exception:
                                pass
                    self.logger.info(f"Setting {module} to None")
                    setattr(self, module, None)
            else:
                if offload_type == "cpu":
                    module.to("cpu")
                elif offload_type == "discard":
                    raise ValueError(
                        f"Invalid offload type: {offload_type} for module."
                    )

        gc.collect()
        # 4)  Reclaim CUDA VRAM
        if torch.cuda.is_available():
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
