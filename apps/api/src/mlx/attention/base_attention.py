from mlx import nn
from typing import Optional, Tuple
import mlx.core as mx


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        bias: bool = True,
        out_bias: bool = True,
        processor=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )

        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=bias)
        self.to_out = [
            nn.Linear(self.inner_dim, dim, bias=out_bias),
            nn.Dropout(dropout),
        ]

        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.norm_added_k = nn.RMSNorm(dim_head * heads, eps=eps)

        self.set_processor(processor)

    def set_processor(self, processor):
        self.processor = processor

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        rotary_emb: Optional[Tuple[mx.array, mx.array]] = None,
        **kwargs,
    ) -> mx.array:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            **kwargs,
        )
