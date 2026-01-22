import math
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from src.utils.dtype import supports_double
from src.attention.functions import attention_register
from src.transformer.efficiency.list_clear import unwrap_single_item_list


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len]
            .to(torch.float64 if supports_double(x.device) else torch.float32)
            .reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache: dict | None = None,
        kv_cache_data: dict | None = None,
        crossattn_cache: dict | None = None,
        max_attention_size: int = 32760,
        local_attn_size: int = -1,
        sink_size: int = 0,
    ) -> torch.Tensor:
        hidden_states = unwrap_single_item_list(hidden_states)
        encoder_hidden_states = unwrap_single_item_list(encoder_hidden_states)

        current_start = 0
        grid_sizes = None
        freqs = None

        if kv_cache_data is not None:
            current_start = kv_cache_data["current_start"]
            grid_sizes = kv_cache_data["grid_sizes"]
            freqs = kv_cache_data["freqs"]

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                key = attn.to_k(encoder_hidden_states)

                if attn.norm_k is not None:
                    key = attn.norm_k(key)

                value = attn.to_v(encoder_hidden_states)
                key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                crossattn_cache["k"] = key
                crossattn_cache["v"] = value
            else:
                key = crossattn_cache["k"]
                value = crossattn_cache["v"]

            hidden_states = attention_register.call(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

            hidden_states = hidden_states.flatten(2, 3)

        else:
            key = attn.to_k(encoder_hidden_states)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            value = attn.to_v(encoder_hidden_states)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if kv_cache is not None and crossattn_cache is None:

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                query.transpose(1, 2),
                grid_sizes,
                freqs,
                start_frame=current_start_frame,
            )
            roped_key = causal_rope_apply(
                key.transpose(1, 2), grid_sizes, freqs, start_frame=current_start_frame
            )
            value = value.transpose(1, 2)
            current_end = current_start + roped_query.shape[1]
            sink_tokens = sink_size * frame_seqlen
            num_new_tokens = roped_query.shape[1]
            kv_cache_size = kv_cache["k"].shape[1]

            if (
                local_attn_size != -1
                and (current_end > kv_cache["global_end_index"].item())
                and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
                )
            ):
                num_evicted_tokens = (
                    num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                )
                num_rolled_tokens = (
                    kv_cache["local_end_index"].item()
                    - num_evicted_tokens
                    - sink_tokens
                )
                kv_cache["k"][
                    :, sink_tokens : sink_tokens + num_rolled_tokens
                ] = kv_cache["k"][
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                kv_cache["v"][
                    :, sink_tokens : sink_tokens + num_rolled_tokens
                ] = kv_cache["v"][
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                # Insert the new keys/values at the end
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = value
            else:
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens

                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = value

            hidden_states = attention_register.call(
                roped_query.transpose(1, 2),
                kv_cache["k"][
                    :,
                    max(0, local_end_index - max_attention_size) : local_end_index,
                ].transpose(1, 2),
                kv_cache["v"][
                    :,
                    max(0, local_end_index - max_attention_size) : local_end_index,
                ].transpose(1, 2),
            ).transpose(1, 2)

            hidden_states = hidden_states.flatten(2, 3)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        else:
            if rotary_emb is not None:

                def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                    dtype = (
                        torch.float32
                        if hidden_states.device.type == "mps"
                        else torch.float64
                    )
                    x_rotated = torch.view_as_complex(
                        hidden_states.to(dtype).unflatten(3, (-1, 2))
                    )
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    return x_out.type_as(hidden_states)

                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

            # I2V task
            hidden_states_img = None
            if encoder_hidden_states_img is not None:
                key_img = attn.add_k_proj(encoder_hidden_states_img)
                key_img = attn.norm_added_k(key_img)
                value_img = attn.add_v_proj(encoder_hidden_states_img)

                key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

                hidden_states_img = attention_register.call(
                    query,
                    key_img,
                    value_img,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
                hidden_states_img = hidden_states_img.type_as(query)

            hidden_states = attention_register.call(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

            if hidden_states_img is not None:
                hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
