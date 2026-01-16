import os

import torch

from src.memory_management.weight_manager import GlobalWeightManager
from src.engine.base_engine import BaseEngine


def _tensor_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def test_register_tracks_tensor_metadata(tmp_path):
    manager = GlobalWeightManager(disk_root=tmp_path)
    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))

    module_id = manager.register_module(module, "engine:test", owner="engine")
    record = manager._modules[module_id]

    assert record.total_bytes == _tensor_bytes(module)
    names = {t.name for t in record.tensors.values()}
    assert "0.weight" in names
    assert "1.bias" in names


def test_offload_to_disk_and_restore(tmp_path):
    manager = GlobalWeightManager(
        disk_root=tmp_path, force_disk_only=True, warm_cache_enabled=False
    )
    module = torch.nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.fill_(1.23)
        module.bias.zero_()

    module_id = manager.register_module(module, "engine:linear", owner="engine")
    record = manager._modules[module_id]
    assert record.location == "disk"
    # When force_disk_only is set we do not re-serialize; a subsequent ensure will
    # return False to signal the caller to reload from the original checkpoint.
    assert record.disk_path is None
    assert manager.ensure_on_device(module_id, device=torch.device("cpu")) is False


def test_eviction_prefers_cpu_then_disk(tmp_path):
    gpu_stats = lambda: (100, 1000, 0.1)  # free / total / fraction
    ram_ok = lambda: (8_000_000_000, 16_000_000_000, 0.5)
    manager = GlobalWeightManager(
        disk_root=tmp_path,
        warm_cache_enabled=True,
        force_disk_only=False,
        gpu_stats_provider=gpu_stats,
        ram_stats_provider=ram_ok,
    )

    keep = manager.register_module(torch.nn.Linear(4, 4), "engine:keep")
    evict = manager.register_module(torch.nn.Linear(16, 16), "engine:evict")
    manager._modules[keep].location = "gpu"
    manager._modules[evict].location = "gpu"

    offloaded = manager.evict_for_vram(
        reason="unit_test", active={keep}, target_free_fraction=0.2
    )
    assert offloaded.get(evict) == "cpu"

    ram_low = lambda: (100_000_000, 16_000_000_000, 0.006)
    manager_disk = GlobalWeightManager(
        disk_root=tmp_path,
        warm_cache_enabled=True,
        force_disk_only=False,
        gpu_stats_provider=gpu_stats,
        ram_stats_provider=ram_low,
    )
    evict_disk = manager_disk.register_module(
        torch.nn.Linear(64, 64), "engine:evict_disk"
    )
    manager_disk._modules[evict_disk].location = "gpu"
    offloaded_disk = manager_disk.evict_for_vram(
        reason="unit_test", active=set(), target_free_fraction=0.2
    )
    assert offloaded_disk.get(evict_disk) == "disk"


def test_request_bytes_triggers_eviction(tmp_path):
    # Even if free fraction is healthy, a large request should trigger eviction.
    gpu_stats = lambda: (500, 1000, 0.5)  # Plenty of free fraction
    ram_ok = lambda: (8_000_000_000, 16_000_000_000, 0.5)
    manager = GlobalWeightManager(
        disk_root=tmp_path,
        warm_cache_enabled=True,
        force_disk_only=False,
        gpu_stats_provider=gpu_stats,
        ram_stats_provider=ram_ok,
    )
    big = manager.register_module(torch.nn.Linear(256, 256), "engine:big")
    manager._modules[big].location = "gpu"

    offloaded = manager.evict_for_vram(
        reason="request_bytes", active=set(), target_free_fraction=0.1, request_bytes=900
    )
    assert offloaded.get(big) in {"cpu", "disk"}


def test_preforward_hook_calls_evict(monkeypatch):
    class StubWM:
        def __init__(self):
            self.called = False
            self.args = None
            self.kw = None

        def offload_gpu_except(self, *args, **kwargs):
            self.called = True
            self.args = args
            self.kw = kwargs

    wm = StubWM()
    mod = torch.nn.Linear(4, 4)

    # Stub torch.cuda signals
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    state = {"mem": (100, 1000)}

    def _mem_get_info(dev=None):
        return state["mem"]

    monkeypatch.setattr("torch.cuda.mem_get_info", _mem_get_info)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.ipc_collect", lambda: None)

    dummy = type("Dummy", (), {})()
    dummy._component_memory_ids = {"foo": "id:foo"}
    dummy._register_tracked_module = lambda *args, **kwargs: None
    dummy._get_weight_manager = lambda: wm

    BaseEngine._install_preforward_hook(dummy, mod, "foo")
    _ = mod(torch.randn(1, 4))
    assert wm.called
    assert wm.args and wm.args[0] == {"id:foo"}

    # When free memory is high, hook should skip subsequent offloads.
    state["mem"] = (900, 1000)
    wm.called = False
    _ = mod(torch.randn(1, 4))
    assert wm.called is False
