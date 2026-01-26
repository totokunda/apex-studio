import tempfile

import torch
from safetensors.torch import save_file

from src.memory_management.budget_offloading import apply_budget_offloading
from src.lora.manager import LoraManager, LoraItem


class _TinyBlock(torch.nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class _TinyModel(torch.nn.Module):
    def __init__(self, dim: int = 4, blocks: int = 2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_TinyBlock(dim) for _ in range(blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_budget_offloading_basic():
    model = _TinyModel()
    manager = apply_budget_offloading(
        model,
        onload_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        block_modules=["blocks"],
        budget_mb=1,
        async_transfers=False,
        prefetch=False,
    )
    x = torch.randn(2, 4)
    _ = model(x)
    assert manager._loaded_block in manager._blocks


class _DummyLoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.peft_config = {}

    def set_adapters(self, names, weights=None):
        self.peft_config.update({name: {} for name in names})

    def load_lora_adapter(self, _state_dict, adapter_name=None, prefix=None, metadata=None):
        device = next(self.parameters()).device
        name = f"lora_{adapter_name}"
        if not hasattr(self, name):
            self.register_parameter(
                name, torch.nn.Parameter(torch.randn(4, 4, device=device))
            )


def test_lora_cpu_residency_after_load():
    manager = LoraManager()
    model = _DummyLoraModel()
    if torch.cuda.is_available():
        model = model.to("cuda")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/lora.safetensors"
        save_file({"transformer.lora_A.weight": torch.randn(2, 2)}, path)
        item = LoraItem(source="local", local_paths=[path], name="test")
        manager.load_into(model, [item], adapter_names=["test"])
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            assert param.device.type == "cpu"
