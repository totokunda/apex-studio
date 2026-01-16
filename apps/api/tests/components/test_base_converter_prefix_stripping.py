import torch

from src.converters.base_converter import BaseConverter


def test_strip_prefix_uses_model_keys_to_avoid_overstripping_model_token():
    """
    If checkpoint keys are wrapped (e.g. `base_model.model.`) but the instantiated
    model expects a `model.` prefix, we should strip only `base_model.` and keep
    `model.` (avoid overstripping).
    """
    converter = BaseConverter()
    sd = {
        "base_model.model.blocks.14.attn1.to_out.0.weight": torch.zeros(1),
        "base_model.model.blocks.14.attn1.to_out.0.bias": torch.zeros(1),
    }
    model_keys = [
        "model.blocks.14.attn1.to_out.0.weight",
        "model.blocks.14.attn1.to_out.0.bias",
    ]

    converter._strip_known_prefixes_inplace(sd, model_keys=model_keys)

    assert set(sd.keys()) == set(model_keys)


def test_strip_prefix_can_strip_all_the_way_to_blocks_when_model_keys_do_not_use_model_prefix():
    """
    When the model keys start at `blocks...`, we should be able to strip a compound
    wrapper like `base_model.model.` down to `blocks...`.
    """
    converter = BaseConverter()
    sd = {
        "base_model.model.blocks.14.attn1.to_out.0.weight": torch.zeros(1),
        "base_model.model.blocks.14.attn1.to_out.0.bias": torch.zeros(1),
    }
    model_keys = [
        "blocks.14.attn1.to_out.0.weight",
        "blocks.14.attn1.to_out.0.bias",
    ]

    converter._strip_known_prefixes_inplace(sd, model_keys=model_keys)

    assert set(sd.keys()) == set(model_keys)


def test_strip_prefix_scoring_handles_lora_A_lora_B_against_base_model_keys():
    """
    LoRA keys include an extra `.lora_A.` / `.lora_B.` (or `.lora_up` / `.lora_down`)
    segment that does not exist in base model keys. Prefix selection should still
    be able to align wrappers using LoRA-stripped variants when scoring.
    """
    converter = BaseConverter()
    sd = {
        "base_model.model.blocks.0.attn.to_q.lora_A.weight": torch.zeros(1),
        "base_model.model.blocks.0.attn.to_q.lora_B.weight": torch.zeros(1),
    }
    # Pass module keys (not full parameter keys) to exercise prefix matching.
    model_keys = ["model.blocks.0.attn.to_q"]

    converter._strip_known_prefixes_inplace(sd, model_keys=model_keys)

    assert "model.blocks.0.attn.to_q.lora_A.weight" in sd
    assert "model.blocks.0.attn.to_q.lora_B.weight" in sd
    assert not any(k.startswith("base_model.") for k in sd.keys())


