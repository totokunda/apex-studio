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
        "diffusion_model.transformer_blocks.39.attn2.to_out.0.lora_A.weight": torch.zeros(1),
        "diffusion_model.transformer_blocks.42.ff.net.2.lora_A.weight": torch.zeros(1),
    }
    model_keys = [
        "transformer_blocks.39.attn2.to_out.0.weight",
        "transformer_blocks.42.ff.net.2.weight",
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

def test_strip_prefix_can_strip_diffusion_model_for_subset_when_mixed_keys_present():
    """
    Some LoRA exports include a mix of wrapped and unwrapped keys, e.g.:
      - "proj_in.weight" (unwrapped)
      - "diffusion_model....lora_A.weight" (wrapped)

    When `model_keys` match the unwrapped layout, prefix stripping should be able
    to strip `diffusion_model.` from the wrapped subset without touching the
    already-unwrapped keys.
    """
    converter = BaseConverter()
    sd = {
        "proj_in.weight": torch.zeros(1),
        "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight": torch.zeros(1),
        "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_B.weight": torch.zeros(1),
    }
    model_keys = [
        "proj_in.weight",
        # Base model key (no LoRA segment) used for overlap scoring.
        "adaln_single.emb.timestep_embedder.linear_1.weight",
    ]

    converter._strip_known_prefixes_inplace(sd, model_keys=model_keys)

    assert "proj_in.weight" in sd
    assert "adaln_single.emb.timestep_embedder.linear_1.lora_A.weight" in sd
    assert "adaln_single.emb.timestep_embedder.linear_1.lora_B.weight" in sd
    assert not any(k.startswith("diffusion_model.") for k in sd.keys())



if __name__ == "__main__":
    test_strip_prefix_uses_model_keys_to_avoid_overstripping_model_token()
    test_strip_prefix_can_strip_all_the_way_to_blocks_when_model_keys_do_not_use_model_prefix()
    test_strip_prefix_scoring_handles_lora_A_lora_B_against_base_model_keys()
    test_strip_prefix_can_strip_diffusion_model_for_subset_when_mixed_keys_present()