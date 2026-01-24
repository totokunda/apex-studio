import torch
import torch.nn.functional as F
from einops import rearrange
from src.attention import attention_register


class StepVideoAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StepVideoAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs,
    ) -> torch.Tensor:
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        if attn_mask is not None and attn_mask.ndim == 3:
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        q, k, v = map(lambda x: rearrange(x, "b s h d -> b h s d"), (q, k, v))

        x = attention_register.call(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, "b h s d -> b s h d")
        return x
