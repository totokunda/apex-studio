import torch

from src.transformer.hunyuanvideo15.base.model import HunyuanVideo15Transformer3DModel
from src.transformer.wan.base.model import WanTransformer3DModel


def test_wan_transformer_efficiency_smoke():
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=16,
        num_layers=1,
        cross_attn_norm=False,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=32,
    )
    model.set_chunking_profile("balanced")

    hidden_states = torch.randn(1, 4, 2, 4, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([1])

    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
        rope_on_cpu=True,
    )[0]

    assert out.shape == hidden_states.shape


def test_hunyuanvideo15_efficiency_smoke():
    model = HunyuanVideo15Transformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=1,
        attention_head_dim=8,
        num_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=1,
        patch_size_t=1,
        text_embed_dim=16,
        text_embed_2_dim=8,
        image_embed_dim=8,
        rope_axes_dim=(2, 2, 2),
        target_size=8,
        chunking_profile="balanced",
    )

    hidden_states = torch.randn(1, 4, 2, 4, 4)
    encoder_hidden_states = torch.randn(1, 5, 16)
    encoder_attention_mask = torch.ones(1, 5, dtype=torch.int64)
    encoder_hidden_states_2 = torch.randn(1, 5, 8)
    encoder_attention_mask_2 = torch.ones(1, 5, dtype=torch.int64)
    image_embeds = torch.zeros(1, 3, 8)
    timestep = torch.tensor([1])

    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        encoder_hidden_states_2=encoder_hidden_states_2,
        encoder_attention_mask_2=encoder_attention_mask_2,
        image_embeds=image_embeds,
        return_dict=False,
        rope_on_cpu=True,
    )[0]

    assert out.shape == hidden_states.shape
