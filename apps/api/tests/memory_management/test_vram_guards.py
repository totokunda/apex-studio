import torch

from src.engine.base_engine import BaseEngine
from src.mixins.to_mixin import ToMixin


def test_vae_decode_passes_timestep():
    class StubVAE:
        dtype = torch.float16

        def __init__(self):
            self.last_timestep = None

        def denormalize_latents(self, x):
            return x

        def decode(self, latents, timestep, return_dict=False):
            self.last_timestep = timestep
            return (latents + 1,)

    vae = StubVAE()
    dummy = type("Dummy", (), {})()
    dummy.device = torch.device("cpu")
    dummy.auto_memory_management = False
    dummy._decode_peak_bytes = {}
    dummy._env_float = lambda *_args, **_kwargs: 8.0
    dummy._flush_for_decode = lambda *_args, **_kwargs: None
    dummy._pre_decode_vram_guard = lambda *_args, **_kwargs: None
    dummy._relieve_vram_pressure = lambda *_args, **_kwargs: False
    dummy.enable_vae_tiling = lambda *_args, **_kwargs: None
    dummy.to_device = lambda *_args, **_kwargs: None
    dummy._offload = lambda *_args, **_kwargs: None
    dummy._get_weight_manager = lambda: None
    dummy.load_component_by_name = lambda *_args, **_kwargs: None
    dummy.load_component_by_type = lambda *_args, **_kwargs: None
    dummy.video_vae = vae

    latents = torch.zeros(2, 3)
    t = torch.tensor([0.0])
    out = BaseEngine.vae_decode(
        dummy,
        latents,
        component_name="video_vae",
        denormalize_latents=False,
        timestep=t,
    )
    assert vae.last_timestep is t
    assert torch.allclose(out, latents.to(dtype=vae.dtype) + 1)


def test_vae_decode_cpu_fallback_after_oom(monkeypatch):
    monkeypatch.setenv("APEX_VAE_DECODE_CPU_FALLBACK", "1")

    class OOMVAE:
        dtype = torch.float16

        def __init__(self):
            self.calls = 0
            self.to_calls = []

        def denormalize_latents(self, x):
            return x

        def to(self, device):
            self.to_calls.append(str(device))
            return self

        def decode(self, latents, return_dict=False):
            self.calls += 1
            if self.calls <= 2:
                raise torch.OutOfMemoryError("simulated")
            return (torch.ones_like(latents),)

    vae = OOMVAE()
    dummy = type("Dummy", (), {})()
    dummy.device = torch.device("cpu")
    dummy.auto_memory_management = False
    dummy._decode_peak_bytes = {}
    dummy._env_float = lambda name, default=0.0: default
    dummy._flush_for_decode = lambda *_args, **_kwargs: None
    dummy._pre_decode_vram_guard = lambda *_args, **_kwargs: None
    dummy._relieve_vram_pressure = lambda *_args, **_kwargs: False
    dummy.enable_vae_tiling = lambda *_args, **_kwargs: None
    dummy.to_device = lambda *_args, **_kwargs: None
    dummy._offload = lambda *_args, **_kwargs: None
    dummy._get_weight_manager = lambda: None
    dummy.load_component_by_name = lambda *_args, **_kwargs: None
    dummy.load_component_by_type = lambda *_args, **_kwargs: None
    dummy.video_vae = vae

    out = BaseEngine.vae_decode(
        dummy,
        torch.zeros(1, 4),
        component_name="video_vae",
        denormalize_latents=False,
    )
    assert vae.calls == 3
    assert "cpu" in " ".join(vae.to_calls)
    assert out.shape == (1, 4)


def test_to_device_recovers_from_oom(monkeypatch):
    class StubWM:
        def __init__(self):
            self._modules = {}
            self.evict_called = False
            self.offload_called = False

        def evict_for_vram(self, **kwargs):
            self.evict_called = True
            return {}

        def ensure_on_device(self, *_args, **_kwargs):
            return False

        def offload_gpu_except(self, *_args, **_kwargs):
            self.offload_called = True
            return {}

    wm = StubWM()
    import src.memory_management as mm

    monkeypatch.setattr(mm, "get_global_weight_manager", lambda: wm)

    class OOMToModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.calls = []

        def to(self, *args, **kwargs):
            dev = args[0] if args else kwargs.get("device")
            dev_str = str(dev)
            self.calls.append(dev_str)
            if dev_str.startswith("cuda"):
                raise torch.OutOfMemoryError("simulated")
            return self

    class Dummy(ToMixin):
        pass

    dummy = Dummy()
    mod = OOMToModule()
    dummy.to_device(mod, device=torch.device("cuda"))
    assert wm.evict_called
    assert wm.offload_called
    assert any("cpu" in c for c in mod.calls)
