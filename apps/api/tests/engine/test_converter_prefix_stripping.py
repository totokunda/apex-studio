import torch

from src.converters.transformer_converters import ZImageTransformerConverter


def test_zimage_converter_maps_prefixed_embedder_keys():
    converter = ZImageTransformerConverter()

    state_dict = {
        "model.diffusion_model.x_embedder.weight": torch.zeros(1, 1),
        "model.diffusion_model.x_embedder.bias": torch.zeros(1),
        "model.diffusion_model.final_layer.linear.weight": torch.zeros(1, 1),
        "model.diffusion_model.final_layer.linear.bias": torch.zeros(1),
        "model.diffusion_model.final_layer.adaLN_modulation.1.weight": torch.zeros(1, 1),
        "model.diffusion_model.final_layer.adaLN_modulation.1.bias": torch.zeros(1),
    }

    model_keys = [
        "all_x_embedder.2-1.weight",
        "all_x_embedder.2-1.bias",
        "all_final_layer.2-1.linear.weight",
        "all_final_layer.2-1.linear.bias",
        "all_final_layer.2-1.adaLN_modulation.1.weight",
        "all_final_layer.2-1.adaLN_modulation.1.bias",
    ]

    converter.convert(state_dict, model_keys=model_keys)

    assert "all_x_embedder.2-1.weight" in state_dict
    assert "all_x_embedder.2-1.bias" in state_dict
    assert "all_final_layer.2-1.linear.weight" in state_dict
    assert "all_final_layer.2-1.linear.bias" in state_dict
    assert "all_final_layer.2-1.adaLN_modulation.1.weight" in state_dict
    assert "all_final_layer.2-1.adaLN_modulation.1.bias" in state_dict

    assert "model.diffusion_model.x_embedder.weight" not in state_dict
    assert "model.diffusion_model.final_layer.linear.weight" not in state_dict

