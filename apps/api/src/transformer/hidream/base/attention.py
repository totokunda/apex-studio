import torch
from typing import Optional, Tuple
from diffusers.models.attention import Attention
from src.attention import attention_register
from diffusers.utils.torch_utils import maybe_allow_in_graph
import torch.nn as nn


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


@maybe_allow_in_graph
class HiDreamAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
    ):
        super(Attention, self).__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        self.to_q = nn.Linear(query_dim, self.inner_dim)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, self.out_dim)
        self.q_rms_norm = nn.RMSNorm(self.inner_dim, eps)
        self.k_rms_norm = nn.RMSNorm(self.inner_dim, eps)

        if not single:
            self.to_q_t = nn.Linear(query_dim, self.inner_dim)
            self.to_k_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_v_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_out_t = nn.Linear(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)
            self.k_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)

        self.set_processor(processor)

    def forward(
        self,
        norm_hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor = None,
        norm_encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states=norm_hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: HiDreamAttention,
        hidden_states: torch.Tensor,
        hidden_states_masks: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(hidden_states)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(hidden_states)).to(dtype=dtype)
        value_i = attn.to_v(hidden_states)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if hidden_states_masks is not None:
            key_i = key_i * hidden_states_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(encoder_hidden_states)).to(
                dtype=dtype
            )
            key_t = attn.k_rms_norm_t(attn.to_k_t(encoder_hidden_states)).to(
                dtype=dtype
            )
            value_t = attn.to_v_t(encoder_hidden_states)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == image_rotary_emb.shape[-3] * 2:
            query, key = apply_rope(query, key, image_rotary_emb)

        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, image_rotary_emb)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        hidden_states = attention_register.call(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(
                hidden_states, [num_image_tokens, num_text_tokens], dim=1
            )
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states
