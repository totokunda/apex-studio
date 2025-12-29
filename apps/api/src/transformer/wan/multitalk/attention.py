import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional

from src.attention.functions import attention_register


@torch.compile
def calculate_x_ref_attn_map(
    visual_q, ref_k, ref_target_masks, mode="mean", attn_bias=None
):

    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1)  # B, H, x_seqlens, ref_seqlens

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = (
            x_ref_attnmap.sum(-1) / ref_target_mask.sum()
        )  # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1)  # B, x_seqlens, H

        if mode == "mean":
            x_ref_attnmap = x_ref_attnmap.mean(-1)  # B, x_seqlens
        elif mode == "max":
            x_ref_attnmap = x_ref_attnmap.max(-1)  # B, x_seqlens

        x_ref_attn_maps.append(x_ref_attnmap)

    del attn
    del x_ref_attn_map_source

    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(
    visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False
):
    """Args:
    query (torch.tensor): B M H K
    key (torch.tensor): B M H K
    shape (tuple): (N_t, N_h, N_w)
    ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape

    x_seqlens = N_h * N_w
    ref_k = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = (
        torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)
    )

    split_chunk = heads // split_num

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_k[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_target_masks,
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


class MultiTalkWanAttnProcessor2_0:
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
        grid_sizes: Optional[torch.Tensor] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        hidden_states = hidden_states.to(attn.to_q.weight.dtype)
        encoder_hidden_states = encoder_hidden_states.to(attn.to_k.weight.dtype)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

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
            ).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = attention_register.call(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(
                query.type_as(hidden_states).transpose(1, 2),
                key.type_as(hidden_states).transpose(1, 2),
                grid_sizes,
                ref_target_masks=ref_target_masks,
            )

        return hidden_states, x_ref_attn_map
