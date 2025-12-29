from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn.functional as F

from einops import rearrange
from src.attention import attention_register
from diffusers.models.attention import Attention
import torch.nn as nn
import numbers
from torch import Tensor

try:
    from flash_attn.layers.rotary import apply_rotary_emb

    HAS_FLASH_ATTN = True
except:
    HAS_FLASH_ATTN = False
    from diffusers.models.embeddings import apply_rotary_emb


class FusedLayerNorm(torch.nn.Module):
    """
    Layer Norm, fused into a single CUDA kernel.
    Borrow from: https://github.com/NVIDIA/Megatron-LM/blob/6501752396e9cc360ce894cda4b2217a58c1c09d/megatron/core/fusions/fused_layer_norm.py#L30

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      model_config (ModelConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(self, zero_centered_gamma: bool, hidden_dim: int, eps: float):
        super().__init__()

        self.zero_centered_gamma = zero_centered_gamma
        if isinstance(hidden_dim, numbers.Integral):
            hidden_dim = (hidden_dim,)
        self.hidden_dim = torch.Size(hidden_dim)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(*hidden_dim))
        self.bias = nn.Parameter(torch.empty(*hidden_dim))

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        return torch.nn.functional.layer_norm(
            input, self.hidden_dim, weight, self.bias, self.eps
        )


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int = None,
        fuse_cross_attention: bool = False,
        heads: int = 8,
        num_query_groups: int = 1,
        dim_head: int = 64,
        eps: float = 1e-5,
        processor: Callable = None,
        layer_number: int = 0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.layer_number = layer_number
        self.cross_attention_dim = cross_attention_dim
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.processor = processor
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.num_query_groups = num_query_groups

        self.fuse_cross_attention = fuse_cross_attention

        self.to_kv = None
        self.to_k = None
        self.to_v = None
        self.cross_q_norm = None
        self.cross_k_norm = None
        self.norm_q = None
        self.norm_k = None

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        if fuse_cross_attention and cross_attention_dim is not None:
            self.to_kv = nn.Linear(self.inner_dim, cross_attention_dim * 2, bias=False)
        elif cross_attention_dim is not None:
            self.to_k = nn.Linear(self.inner_dim, cross_attention_dim, bias=False)
            self.to_v = nn.Linear(self.inner_dim, cross_attention_dim, bias=False)

        if fuse_cross_attention and cross_attention_dim is not None:
            self.cross_q_norm = FusedLayerNorm(
                zero_centered_gamma=True, hidden_dim=dim_head, eps=eps
            )
            self.cross_k_norm = FusedLayerNorm(
                zero_centered_gamma=True, hidden_dim=dim_head, eps=eps
            )
        else:
            self.norm_q = FusedLayerNorm(
                zero_centered_gamma=True, hidden_dim=dim_head, eps=eps
            )
            self.norm_k = FusedLayerNorm(
                zero_centered_gamma=True, hidden_dim=dim_head, eps=eps
            )

    def forward(self, *args, **kwargs):
        return self.processor(self, *args, **kwargs)


class MagiSelfAttentionProcessor:

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,  # S, B, H
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache_params: Dict[str, Any] = {},
        denoising_range_num: int = 1,
        q_range: torch.Tensor = None,
        kv_range: torch.Tensor = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
    ) -> torch.Tensor:

        sin_emb, cos_emb = rotary_emb.tensor_split(2, -1)
        batch_size = hidden_states.shape[1]

        attn_head_dim = hidden_states.shape[-1] // attn.heads

        # self-attention
        query = attn.to_q(hidden_states)  # S, B, H
        key = attn.to_k(hidden_states)  # S, B, H
        value = attn.to_v(hidden_states)  # S, B, H

        query = query.reshape(
            query.size(0), query.size(1), -1, attn_head_dim
        )  # S, B, H/heads, heads
        key = key.reshape(
            key.size(0), key.size(1), -1, attn_head_dim
        )  # S, B, H/heads, heads

        query_dtype = query.dtype

        query = query.to(dtype=torch.float32)
        key = key.to(dtype=torch.float32)

        if attn.norm_q is not None:
            query = attn.norm_q(query)  # S, B, H
        if attn.norm_k is not None:
            key = attn.norm_k(key)  # S, B, H

        # reshape query, key, value
        query = query.transpose(0, 1).contiguous()  # B, S, H/heads, heads
        key = key.transpose(0, 1).contiguous()  # B, S, H/heads, heads

        if HAS_FLASH_ATTN:
            query = apply_rotary_emb(query, cos_emb, sin_emb)
            key = apply_rotary_emb(key, cos_emb, sin_emb)
        else:
            query = apply_rotary_emb(query, (cos_emb, sin_emb))
            key = apply_rotary_emb(key, (cos_emb, sin_emb))

        query = rearrange(query, "b sq hn hd -> (sq b) hn hd").contiguous()

        key = rearrange(key, "b sq hn hd -> (sq b) hn hd").contiguous()
        value = rearrange(
            value, "sq b (hn hd) -> (sq b) hn hd", hd=attn_head_dim
        ).contiguous()

        query = query.to(dtype=query_dtype)
        key = key.to(dtype=query_dtype)
        value = value.to(dtype=query_dtype)

        if kv_cache_params is not None:
            extract_prefix_video_feature = kv_cache_params.get(
                "extract_prefix_video_feature", False
            )

            fwd_extra_1st_chunk = kv_cache_params.get("fwd_extra_1st_chunk", False)
            slice_point = kv_cache_params.get("slice_point", 0)
            max_sequence_length = kv_cache_params.get("max_sequence_length", 0)
            max_batch_size = kv_cache_params.get("max_batch_size", 0)
            key_value_memory_dict = kv_cache_params.get("key_value_memory_dict", {})
            clip_token_nums = kv_cache_params.get("clip_token_nums", 0)
            distill_nearly_clean_chunk = kv_cache_params.get(
                "distill_nearly_clean_chunk", False
            )
            update_kv_cache = kv_cache_params.get("update_kv_cache", False)
            kv_offload = kv_cache_params.get("kv_offload", False)
            key_value = torch.cat([key, value], dim=-1)

            if extract_prefix_video_feature or fwd_extra_1st_chunk or slice_point > 0:
                key_value = self._full_adjust_key_and_value(
                    max_sequence_length,
                    max_batch_size,
                    key_value,
                    attn.layer_number,
                    key_value_memory_dict,
                    clip_token_nums,
                    distill_nearly_clean_chunk,
                    update_kv_cache,
                    slice_point,
                    attn.num_query_groups,
                    attn_head_dim,
                    kv_offload=kv_offload,
                )

            key, value = torch.chunk(key_value, 2, dim=-1)
            key = key.contiguous()
            value = value.contiguous()

        query = (
            query.reshape(-1, batch_size, query.shape[1], query.shape[2])
            .transpose(0, 1)
            .contiguous()
        )
        # (sq b) hn hd -> b sq hn hd
        key = (
            key.reshape(-1, batch_size, key.shape[1], key.shape[2])
            .transpose(0, 1)
            .contiguous()
        )
        # (sq b) hn hd -> b sq hn hd
        value = (
            value.reshape(-1, batch_size, value.shape[1], value.shape[2])
            .transpose(0, 1)
            .contiguous()
        )

        if not torch.isfinite(query).all():
            raise RuntimeError(f"NaN/Inf in forward of {self.__class__.__name__}")
        if not torch.isfinite(key).all():
            raise RuntimeError(f"NaN/Inf in forward of {self.__class__.__name__}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"NaN/Inf in forward of {self.__class__.__name__}")

        core_attn_outs = []
        for i in range(denoising_range_num):
            if batch_size == 1:
                q = query[:, q_range[i, 0] : q_range[i, 1]]
                k = key[:, kv_range[i, 0] : kv_range[i, 1]]
                v = value[:, kv_range[i, 0] : kv_range[i, 1]]
            else:
                assert i == 0
                q = query[:, q_range[0, 0] : q_range[0, 1]]
                k = key[:, kv_range[0, 0] : kv_range[0, 1]]
                v = value[:, kv_range[0, 0] : kv_range[0, 1]]

            o = attention_register.call(
                q=q.transpose(1, 2),
                k=k.transpose(1, 2),
                v=v.transpose(1, 2),
                deterministic=torch.are_deterministic_algorithms_enabled(),
            ).transpose(1, 2)
            o = rearrange(o, "b sq h d -> (sq b) h d", b=batch_size)
            core_attn_outs.append(o)
        core_attn_out = torch.cat(core_attn_outs, dim=0)
        core_attn_out = rearrange(
            core_attn_out, "(sq b) hn hd -> sq b (hn hd)", b=batch_size
        )

        return core_attn_out

    def _full_adjust_key_and_value(
        self,
        max_sequence_length: int,
        max_batch_size: int,
        key_and_value: torch.Tensor,
        layer_number: int,
        key_value_memory_dict: dict,
        clip_token_nums: int,
        distill_nearly_clean_chunk: bool,
        update_kv_cache: bool,
        slice_point: int,
        attn_heads: int,
        attn_head_dim: int,
        kv_offload: bool = False,
    ):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params

        Returns a tuple: (key, value)
        """
        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        inf_max_seq_length = max_sequence_length
        inf_max_batch_size = max_batch_size

        if layer_number not in key_value_memory_dict:
            inference_key_and_value_memory = self._allocate_key_and_value_memory(
                inf_max_seq_length,
                inf_max_batch_size,
                key_and_value.dtype,
                key_and_value.device,
                attn_heads,
                attn_head_dim,
                kv_offload=kv_offload,
            )

            key_value_memory_dict[layer_number] = inference_key_and_value_memory
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_and_value_memory = key_value_memory_dict[layer_number]

        sequence_start = slice_point * clip_token_nums * inf_max_batch_size

        get_key_and_value = inference_key_and_value_memory[:sequence_start, ...].to(
            key_and_value.device
        )

        # Copy key and values.
        if update_kv_cache:
            key_and_value_total = key_and_value

            clip_size = (
                key_and_value_total.size(0) - clip_token_nums * inf_max_batch_size
                if distill_nearly_clean_chunk
                else key_and_value_total.size(0)
            )

            sequence_end = sequence_start + clip_size
            assert sequence_end <= inference_key_and_value_memory.size(0)
            # update kv cache
            inference_key_and_value_memory[sequence_start:sequence_end, ...] = (
                key_and_value_total[:clip_size]
            )

        key_and_value = torch.cat([get_key_and_value, key_and_value], dim=0)
        if not torch.isfinite(key_and_value).all():
            raise RuntimeError(f"NaN/Inf in forward of {self.__class__.__name__}")
        return key_and_value

    def _allocate_key_and_value_memory(
        self,
        sequence_length,
        batch_size,
        dtype,
        device,
        num_query_groups,
        attn_head_dim,
        kv_offload: bool = False,
    ):
        """Allocate memory to store kv cache during inference."""

        if kv_offload:
            key_and_value_memory = torch.empty(
                sequence_length * batch_size,
                num_query_groups,
                attn_head_dim * 2,
                dtype=dtype,
                pin_memory=True,
                device="cpu",
            )
        else:
            key_and_value_memory = torch.empty(
                sequence_length * batch_size,
                num_query_groups,
                attn_head_dim * 2,
                dtype=dtype,
                device=device,
            )

        return key_and_value_memory


class MagiCrossAttentionProcessor:

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,  # S, B, H
        encoder_hidden_states: Optional[torch.Tensor] = None,  # S, B, H
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        chunk_size=8,
        **kwargs,
    ) -> torch.Tensor:

        attn_head_dim = hidden_states.shape[-1] // attn.heads
        batch_size = hidden_states.shape[1]

        query = attn.to_q(hidden_states)  # S, B, H
        query = rearrange(
            query,
            "sq b (hn hd) -> (b sq) hn hd",
            hd=attn_head_dim,
        )

        query_dtype = query.dtype

        if attn.cross_q_norm is not None:
            query = attn.cross_q_norm(query)
        elif attn.norm_q is not None:
            query = attn.norm_q(query)

        mixed_key_value = torch.concat(
            [
                torch.matmul(encoder_hidden_states, w.t())
                for w in torch.chunk(attn.to_kv.weight, chunk_size, axis=0)
            ],
            axis=1,
        )

        # [y_total_token, 2*hn*hd] --> [y_total_token, hn, 2*hd]
        mixed_key_value = mixed_key_value.view(
            encoder_hidden_states.shape[0], -1, 2 * attn_head_dim
        )
        key, value = self.split_tensor_along_last_dim(mixed_key_value, 2)

        if attn.cross_k_norm is not None:
            key = attn.cross_k_norm(key)
        elif attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.to(dtype=query_dtype)
        key = key.to(dtype=query_dtype)
        value = value.to(dtype=query_dtype)

        if attention_register.is_available("flash"):
            attn_out = (
                attention_register.call(  # call directly due to implementation details
                    query,  # [b*sq, hn, hd]
                    key,  # [y_total_token, hn, hd]
                    value,  # [y_total_token, hn, hd]
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_kv,
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    key="flash",
                )
            )
        else:
            attn_out = attention_register.call(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                key="sdpa_varlen",
                is_causal=False,
            )

        attn_out = rearrange(
            attn_out, "(b sq) hn hd -> sq b (hn hd)", b=batch_size
        ).contiguous()

        return attn_out

    def split_tensor_along_last_dim(
        self,
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
    ) -> List[torch.Tensor]:
        """Split a tensor along its last dimension.

        Args:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list
