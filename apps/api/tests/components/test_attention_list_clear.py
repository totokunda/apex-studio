import torch

from diffusers.models.attention import Attention

from src.transformer.ltx2.base.model import LTX2Attention, LTX2AudioVideoAttnProcessor
from src.transformer.wan.base.attention import WanAttnProcessor2_0


def test_wan_attn_processor_clears_lists():
    attn = Attention(query_dim=8, heads=2, dim_head=4, bias=True)
    processor = WanAttnProcessor2_0()

    hidden_states = [torch.randn(1, 4, 8)]
    encoder_hidden_states = [torch.randn(1, 4, 8)]

    output = processor(attn, hidden_states, encoder_hidden_states)

    assert hidden_states == []
    assert encoder_hidden_states == []
    assert output.shape == (1, 4, 8)


def test_ltx2_attn_processor_clears_lists():
    attn = LTX2Attention(
        query_dim=8,
        heads=2,
        kv_heads=2,
        dim_head=4,
        qk_norm="rms_norm_across_heads",
        bias=True,
    )
    processor = LTX2AudioVideoAttnProcessor()

    hidden_states = [torch.randn(1, 4, 8)]

    output = processor(attn, hidden_states, None, None, None, None)

    assert hidden_states == []
    assert output.shape == (1, 4, 8)
