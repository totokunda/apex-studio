import pytest
import torch


def test_preflight_component_load_force_offloads_pinned_on_cuda_pressure():
    """
    Regression test: loading a new component must not OOM just because a previously
    loaded (leased/pinned) component is still resident on GPU.

    We simulate GPU pressure and assert the manager force-offloads pinned modules
    during load-time preflight.
    """
    from src.memory_management import ComponentMemoryManager

    class DummyEngine:
        def __init__(self):
            self.device = torch.device("cuda")
            self.offload_calls = []

        def _offload(self, module, *, offload_type: str = "discard"):
            # Only record; the manager updates bookkeeping itself.
            self.offload_calls.append((module, offload_type))

    engine = DummyEngine()
    manager = ComponentMemoryManager()
    manager.install_for_engine(engine)

    # Register a "resident on GPU" component and mark it pinned/leased.
    module = torch.nn.Linear(4, 4)
    comp = manager.register_component(module, "vae", {"vae"}, engine=engine, pinned=True)
    assert comp is not None
    comp.device = torch.device("cuda")
    comp.load_pin_count = 1

    # Simulate very low free VRAM initially, then sufficient free VRAM after a few checks.
    # Note: preflight performs multiple free/total queries before it starts evicting.
    call_state = {"cuda_calls": 0}

    def fake_device_free_total(device: torch.device):
        if device.type == "cuda":
            call_state["cuda_calls"] += 1
            total = 100 * 1024**3  # 100GB total (arbitrary)
            # First check: almost no free -> triggers eviction
            if call_state["cuda_calls"] <= 10:
                return 32 * 1024**2, total  # 32MiB free
            # After eviction: lots of free -> stops evicting
            return 50 * 1024**3, total  # 50GB free
        # CPU always has room in this unit test
        return 500 * 1024**3, 1000 * 1024**3

    manager._device_free_total = fake_device_free_total  # type: ignore[method-assign]

    # Run preflight for a new transformer load.
    manager.preflight_component_load(
        engine=engine, component={"type": "transformer"}, reserve_bytes=0
    )

    # Must have forced an offload despite the component being pinned/leased.
    assert engine.offload_calls, "expected an offload call during preflight"
    assert any(offload_type in {"cpu", "discard"} for _, offload_type in engine.offload_calls)

    # Bookkeeping should reflect the forced offload and cleared lease pins.
    assert comp.device is not None and comp.device.type == "cpu"
    assert int(getattr(comp, "load_pin_count", 0)) == 0


def test_preflight_component_load_does_not_protect_same_label_other_engine():
    """
    Regression test: load preflight must not "exclude" generic labels like
    'transformer' across engines, otherwise an old transformer can stay on GPU
    and cause OOM when loading a new transformer.
    """
    from src.memory_management import ComponentMemoryManager

    class DummyEngine:
        def __init__(self, name: str):
            self.name = name
            self.device = torch.device("cuda")
            self.offload_calls = []

        def _offload(self, module, *, offload_type: str = "discard"):
            self.offload_calls.append((module, offload_type))

    engine_old = DummyEngine("old")
    engine_new = DummyEngine("new")

    manager = ComponentMemoryManager()
    manager.install_for_engine(engine_old)
    manager.install_for_engine(engine_new)

    # Old engine has a transformer resident on GPU and leased/pinned.
    old_mod = torch.nn.Linear(4, 4)
    old_comp = manager.register_component(
        old_mod, "transformer", {"transformer"}, engine=engine_old, pinned=True
    )
    assert old_comp is not None
    old_comp.device = torch.device("cuda")
    old_comp.load_pin_count = 1

    # Simulate GPU pressure so preflight must evict something.
    call_state = {"cuda_calls": 0}

    def fake_device_free_total(device: torch.device):
        if device.type == "cuda":
            call_state["cuda_calls"] += 1
            total = 100 * 1024**3
            if call_state["cuda_calls"] <= 10:
                return 32 * 1024**2, total
            return 50 * 1024**3, total
        return 500 * 1024**3, 1000 * 1024**3

    manager._device_free_total = fake_device_free_total  # type: ignore[method-assign]

    # New engine is about to load *another* transformer.
    manager.preflight_component_load(engine=engine_new, component={"type": "transformer"})

    # Old engine's transformer must have been evicted even though the label matches.
    assert engine_old.offload_calls, "expected old engine transformer to be offloaded"
    assert old_comp.device is not None and old_comp.device.type == "cpu"
    assert int(getattr(old_comp, "load_pin_count", 0)) == 0


def test_preflight_component_load_does_nothing_when_enough_free_vram():
    """
    Ensure load preflight is not eager: if we already have enough free VRAM
    headroom, it should not offload any engine attributes nor evict components.
    """
    from src.memory_management import ComponentMemoryManager

    class DummyEngine:
        def __init__(self):
            self.device = torch.device("cuda")
            self.offload_calls = []
            # Present attributes that *could* be offloaded if pressure existed.
            self.text_encoder = object()
            self.vae = object()
            self.transformer = object()

        def _offload(self, module, *, offload_type: str = "discard"):
            self.offload_calls.append((module, offload_type))

    engine = DummyEngine()
    manager = ComponentMemoryManager()
    manager.install_for_engine(engine)

    def fake_device_free_total(device: torch.device):
        if device.type == "cuda":
            total = 100 * 1024**3
            free = 90 * 1024**3  # Plenty of headroom
            return free, total
        return 500 * 1024**3, 1000 * 1024**3

    manager._device_free_total = fake_device_free_total  # type: ignore[method-assign]

    manager.preflight_component_load(engine=engine, component={"type": "transformer"}, reserve_bytes=0)

    assert engine.offload_calls == []

