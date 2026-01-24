"""Compatibility shim for PyTorch builds without torch.distributed.

Some environments ship PyTorch without distributed support (or with it disabled),
but parts of MMCV import `torch.distributed` unconditionally. This module
installs a minimal stub so imports succeed, while keeping single-process
behavior correct (rank=0, world_size=1).
"""

from __future__ import annotations


def patch_torch_distributed():
    """Ensure `torch.distributed` is importable.

    If importing the real `torch.distributed` fails, this function installs a
    stub module into `sys.modules["torch.distributed"]` and attaches it to the
    `torch` package so both `import torch.distributed` and
    `from torch import distributed as dist` work.

    Returns:
        module: The real `torch.distributed` module if available, otherwise the
        installed stub.
    """

    import sys
    import types

    import torch

    try:
        import torch.distributed as dist  # noqa: F401

        return dist
    except Exception:
        # Any exception here means distributed isn't usable in this runtime.
        pass

    dist = types.ModuleType("torch.distributed")

    def _unavailable(*args, **kwargs):
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    # Basic availability/identity.
    dist.is_available = lambda: False
    dist.is_initialized = lambda: False
    dist.get_rank = lambda *args, **kwargs: 0
    dist.get_world_size = lambda *args, **kwargs: 1

    # Minimal process-group API surface.
    class _Group:
        WORLD = object()

    dist.group = _Group()
    dist.new_group = lambda *args, **kwargs: dist.group.WORLD
    dist.init_process_group = _unavailable
    dist.destroy_process_group = lambda *args, **kwargs: None

    # Collective ops become no-ops for world_size == 1.
    dist.barrier = lambda *args, **kwargs: None
    dist.broadcast = lambda tensor, src, *args, **kwargs: tensor

    def _all_gather(tensor_list, tensor, *args, **kwargs):
        for i in range(len(tensor_list)):
            try:
                tensor_list[i].copy_(tensor)
            except Exception:
                try:
                    tensor_list[i] = tensor.clone()
                except Exception:
                    tensor_list[i] = tensor

    dist.all_gather = _all_gather
    dist.all_reduce = lambda *args, **kwargs: None

    # Install for import machinery and `from torch import distributed`.
    sys.modules["torch.distributed"] = dist
    setattr(torch, "distributed", dist)

    return dist

