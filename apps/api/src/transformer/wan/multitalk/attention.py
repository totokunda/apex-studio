import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional

from src.attention.functions import attention_register
from src.transformer.efficiency.ops import apply_wan_rope_inplace
from src.transformer.efficiency.list_clear import unwrap_single_item_list


@torch.compile
def calculate_x_ref_attn_map(
    visual_q,
    ref_k,
    ref_target_masks,
    mode="mean",
    attn_bias=None,
    *,
    chunk_size_x: int = 1024,
    chunk_size_ref: int = 1024,
):
    """
    Memory-efficient version of:
      softmax(q @ k^T + bias) -> masked mean over ref tokens -> reduce over heads

    This avoids allocating the full attention matrix (B, H, x_seqlens, ref_seqlens)
    by streaming a numerically-stable log-sum-exp over `ref_seqlens` in chunks, and
    additionally chunking `x_seqlens` to cap peak memory for very large spatial token
    counts.
    """

    if mode not in ("mean", "max"):
        raise ValueError(f"Unsupported mode={mode!r}; expected 'mean' or 'max'.")

    ref_k = ref_k.to(device=visual_q.device, dtype=visual_q.dtype)

    scale = 1.0 / (visual_q.shape[-1] ** 0.5)
    q = (visual_q * scale).transpose(1, 2)  # (B, H, x, D)
    k = ref_k.transpose(1, 2)  # (B, H, ref, D)

    # ref_target_masks is expected to be (C, ref_seqlens) (C="class" count)
    masks = ref_target_masks.to(device=q.device).bool()
    class_num, ref_seqlens = masks.shape

    B, H, x_seqlens, _ = q.shape
    if ref_seqlens != k.shape[2]:
        # Keep behavior obvious if caller passes mismatched masks.
        raise ValueError(
            f"ref_target_masks has ref_seqlens={ref_seqlens}, but ref_k has ref_seqlens={k.shape[2]}"
        )

    # Accumulate in fp32 for stability, especially when q/k are fp16/bf16.
    acc_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

    # Mask counts (avoid div-by-zero).
    mask_counts = masks.sum(dim=-1).clamp_min(1).to(dtype=acc_dtype)  # (C,)

    out = torch.empty((class_num, B, x_seqlens), device=q.device, dtype=acc_dtype)

    x_chunk = max(1, int(chunk_size_x))
    ref_chunk = max(1, int(chunk_size_ref))

    # Process in x blocks to cap peak memory: scores is (B,H,x_blk,ref_blk).
    for x0 in range(0, x_seqlens, x_chunk):
        x1 = min(x0 + x_chunk, x_seqlens)
        q_blk = q[:, :, x0:x1, :].to(acc_dtype)  # (B,H,x_blk,D)
        x_blk = x1 - x0

        # Streaming softmax stats over ref dimension:
        # m: running max, s: running sum(exp(scores - m))
        m = torch.full((B, H, x_blk), -torch.inf, device=q.device, dtype=acc_dtype)
        s = torch.zeros((B, H, x_blk), device=q.device, dtype=acc_dtype)
        num = torch.zeros((class_num, B, H, x_blk), device=q.device, dtype=acc_dtype)

        for r0 in range(0, ref_seqlens, ref_chunk):
            r1 = min(r0 + ref_chunk, ref_seqlens)

            k_blk = k[:, :, r0:r1, :].to(acc_dtype)  # (B,H,ref_blk,D)
            scores = q_blk @ k_blk.transpose(-2, -1)  # (B,H,x_blk,ref_blk)

            if attn_bias is not None:
                scores = scores + attn_bias[:, :, x0:x1, r0:r1].to(acc_dtype)

            chunk_max = scores.max(dim=-1).values  # (B,H,x_blk)
            m_new = torch.maximum(m, chunk_max)

            # Rescale previous sums into the new max-frame.
            rescale = torch.exp(m - m_new)  # (B,H,x_blk)
            s = s * rescale
            num = num * rescale.unsqueeze(0)

            exp_scores = torch.exp(scores - m_new.unsqueeze(-1))  # (B,H,x_blk,ref_blk)
            s = s + exp_scores.sum(dim=-1)

            # Compute masked numerator sums for all classes without 5D broadcast.
            # exp2d: (B*H*x_blk, ref_blk), masks_blk.T: (ref_blk, C) => (B*H*x_blk, C)
            exp2d = exp_scores.reshape(-1, r1 - r0)
            masks_blk = masks[:, r0:r1].to(dtype=acc_dtype)  # (C, ref_blk)
            masked_sums = exp2d @ masks_blk.transpose(0, 1)  # (B*H*x_blk, C)
            masked_sums = masked_sums.transpose(0, 1).reshape(class_num, B, H, x_blk)
            num = num + masked_sums

            m = m_new

        # Probability mass on masked tokens, averaged over mask size.
        prob = (num / s.unsqueeze(0)) / mask_counts[
            :, None, None, None
        ]  # (C,B,H,x_blk)

        # Reduce over heads.
        prob = prob.permute(0, 1, 3, 2)  # (C,B,x_blk,H)
        if mode == "mean":
            prob = prob.mean(dim=-1)  # (C,B,x_blk)
        else:  # mode == "max"
            prob = prob.max(dim=-1).values  # (C,B,x_blk)

        out[:, :, x0:x1] = prob

    out = out.to(dtype=visual_q.dtype)
    return out.reshape(class_num * B, x_seqlens)


def get_attn_map_with_target(
    visual_q,
    ref_k,
    shape,
    ref_target_masks=None,
    split_num=2,
    enable_sp=False,
    *,
    x_ref_attn_use_chunks: bool = True,
    x_ref_attn_chunk_size_x: Optional[int] = None,
    x_ref_attn_chunk_size_ref: Optional[int] = None,
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
    B, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(
        class_num * B, seq_lens, device=visual_q.device, dtype=visual_q.dtype
    )

    split_chunk = heads // split_num

    if not x_ref_attn_use_chunks:
        # Effectively disable chunking by using full lengths.
        eff_chunk_size_x = seq_lens
        eff_chunk_size_ref = ref_k.shape[1]
    else:
        eff_chunk_size_x = (
            x_ref_attn_chunk_size_x if x_ref_attn_chunk_size_x is not None else 1024
        )
        eff_chunk_size_ref = (
            x_ref_attn_chunk_size_ref if x_ref_attn_chunk_size_ref is not None else 1024
        )

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_k[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_target_masks,
            chunk_size_x=eff_chunk_size_x,
            chunk_size_ref=eff_chunk_size_ref,
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


class MultiTalkWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

        # Defaults for attention-map chunking; can be overridden by chunking_profile via
        # `set_x_ref_attn_chunking()` and/or per-call kwargs.
        self._x_ref_attn_use_chunks: bool = True
        self._x_ref_attn_chunk_size_x: Optional[int] = 1024
        self._x_ref_attn_chunk_size_ref: Optional[int] = 1024

    def set_x_ref_attn_chunking(
        self,
        *,
        use_chunks: bool = True,
        chunk_size_x: Optional[int] = None,
        chunk_size_ref: Optional[int] = None,
    ) -> None:
        self._x_ref_attn_use_chunks = bool(use_chunks)
        self._x_ref_attn_chunk_size_x = chunk_size_x
        self._x_ref_attn_chunk_size_ref = chunk_size_ref

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
        x_ref_attn_use_chunks: Optional[bool] = None,
        x_ref_attn_chunk_size_x: Optional[int] = None,
        x_ref_attn_chunk_size_ref: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = unwrap_single_item_list(hidden_states)
        encoder_hidden_states = unwrap_single_item_list(encoder_hidden_states)
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
            apply_wan_rope_inplace(
                query, rotary_emb, chunk_size=None, freqs_may_be_cpu=True
            )
            apply_wan_rope_inplace(
                key, rotary_emb, chunk_size=None, freqs_may_be_cpu=True
            )

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
            # Resolve effective chunking settings (per-call overrides > profile defaults).
            eff_use_chunks = (
                bool(x_ref_attn_use_chunks)
                if x_ref_attn_use_chunks is not None
                else self._x_ref_attn_use_chunks
            )
            eff_chunk_size_x = (
                x_ref_attn_chunk_size_x
                if x_ref_attn_chunk_size_x is not None
                else self._x_ref_attn_chunk_size_x
            )
            eff_chunk_size_ref = (
                x_ref_attn_chunk_size_ref
                if x_ref_attn_chunk_size_ref is not None
                else self._x_ref_attn_chunk_size_ref
            )
            x_ref_attn_map = get_attn_map_with_target(
                query.type_as(hidden_states).transpose(1, 2),
                key.type_as(hidden_states).transpose(1, 2),
                grid_sizes,
                ref_target_masks=ref_target_masks,
                x_ref_attn_use_chunks=eff_use_chunks,
                x_ref_attn_chunk_size_x=eff_chunk_size_x,
                x_ref_attn_chunk_size_ref=eff_chunk_size_ref,
            )

        return hidden_states, x_ref_attn_map
