import torch

from src.lora.lora_converter import convert_kohya_to_peft_state_dict


def test_convert_kohya_to_peft_preserves_double_blocks_and_single_blocks_tokens():
    """
    Regression test:

    Kohya flattens module path dots into underscores, but many models (Flux/Hunyuan/etc)
    have legitimate underscores in module names (e.g. `diffusion_model`, `double_blocks`).
    The converter must not corrupt these into dot notation (e.g. `double.blocks`).
    """
    sd = {
        # Prefix is Kohya-flattened: dots become underscores except for the last two dots
        # before `.lora_down.weight` / `.lora_up.weight`.
        "lora_unet_diffusion_model_double_blocks_0_img_attn_qkv.lora_down.weight": torch.zeros(1),
        "lora_unet_diffusion_model_single_blocks_0_img_attn_qkv.lora_up.weight": torch.zeros(1),
        # Kohya `.alpha` tensors should be dropped.
        "lora_unet_diffusion_model_double_blocks_0_img_attn_qkv.alpha": torch.zeros(1),
    }

    convert_kohya_to_peft_state_dict(sd)

    assert "unet.diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight" in sd
    assert "unet.diffusion_model.single_blocks.0.img_attn.qkv.lora_B.weight" in sd

    assert not any(k.endswith(".alpha") for k in sd.keys())
    assert not any("double.blocks" in k for k in sd.keys())
    assert not any("single.blocks" in k for k in sd.keys())


