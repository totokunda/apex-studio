import torch

from src.converters.transformer_converters import FluxTransformerConverter


def test_flux_converter_guidance_is_all_or_nothing():
    converter = FluxTransformerConverter()

    # Partial guidance keys should remain untouched (regression against naive rename_dict usage).
    sd_partial = {
        "guidance_in.in_layer.weight": torch.zeros(2, 2),
        "guidance_in.in_layer.bias": torch.zeros(2),
    }
    converter.convert(sd_partial)
    assert "guidance_in.in_layer.weight" in sd_partial
    assert "time_text_embed.guidance_embedder.linear_1.weight" not in sd_partial

    # Full guidance key set should be converted.
    sd_full = {
        "guidance_in.in_layer.weight": torch.zeros(2, 2),
        "guidance_in.in_layer.bias": torch.zeros(2),
        "guidance_in.out_layer.weight": torch.zeros(2, 2),
        "guidance_in.out_layer.bias": torch.zeros(2),
    }
    converter.convert(sd_full)
    assert "time_text_embed.guidance_embedder.linear_1.weight" in sd_full
    assert "time_text_embed.guidance_embedder.linear_2.bias" in sd_full
    assert not any(k.startswith("guidance_in.") for k in sd_full.keys())


def test_flux_converter_splits_double_block_qkv_without_corrupting_keys():
    converter = FluxTransformerConverter()

    sd = {
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(9, 4),
        "double_blocks.0.img_attn.qkv.bias": torch.zeros(9),
    }
    converter.convert(sd)

    assert "transformer_blocks.0.attn.to_q.weight" in sd
    assert "transformer_blocks.0.attn.to_k.weight" in sd
    assert "transformer_blocks.0.attn.to_v.weight" in sd
    assert "transformer_blocks.0.attn.to_q.bias" in sd
    assert not any("double_blocks.0.img_attn.qkv" in k for k in sd.keys())


def test_flux_converter_splits_double_block_qkv_lora_keys():
    converter = FluxTransformerConverter()

    # lora_down (A) shared; lora_up (B) split across q/k/v along dim=0.
    sd = {
        "double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros(2, 4),
        "double_blocks.0.img_attn.qkv.lora_up.weight": torch.zeros(9, 2),
    }
    converter.convert(sd)

    assert "transformer_blocks.0.attn.to_q.lora_down.weight" in sd
    assert "transformer_blocks.0.attn.to_k.lora_down.weight" in sd
    assert "transformer_blocks.0.attn.to_v.lora_down.weight" in sd

    assert "transformer_blocks.0.attn.to_q.lora_up.weight" in sd
    assert "transformer_blocks.0.attn.to_k.lora_up.weight" in sd
    assert "transformer_blocks.0.attn.to_v.lora_up.weight" in sd

    assert not any("double_blocks.0.img_attn.qkv.lora_" in k for k in sd.keys())


