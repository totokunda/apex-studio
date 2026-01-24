# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from loguru import logger
from src.vae.seedvr.modules.utils import safe_pad_operation
from src.vae.seedvr.modules.types import MemoryState


def causal_conv_slice_inputs(x, split_size, memory_state):
    sp_size = 1
    sp_group = None
    sp_rank = 0
    if sp_group is None:
        return x

    assert memory_state != MemoryState.UNSET
    leave_out = 1 if memory_state != MemoryState.ACTIVE else 0

    # Should have at least sp_size slices.
    num_slices = (x.size(2) - leave_out) // split_size
    assert num_slices >= sp_size, f"{num_slices} < {sp_size}"

    split_sizes = [split_size + leave_out] + [split_size] * (num_slices - 1)
    split_sizes += [x.size(2) - sum(split_sizes)]
    assert sum(split_sizes) == x.size(2)

    split_sizes = torch.tensor(split_sizes)
    slices_per_rank = len(split_sizes) // sp_size
    split_sizes = split_sizes.split(
        [slices_per_rank] * (sp_size - 1)
        + [len(split_sizes) - slices_per_rank * (sp_size - 1)]
    )
    split_sizes = list(map(lambda s: s.sum().item(), split_sizes))

    return x.split(split_sizes, dim=2)[sp_rank]


def causal_conv_gather_outputs(x):
    sp_group = None
    sp_size = 1
    if sp_group is None:
        return x

    # Communicate shapes.
    unpad_lens = torch.empty((sp_size,), device=x.device, dtype=torch.long)
    local_unpad_len = torch.tensor([x.size(2)], device=x.device, dtype=torch.long)
    torch.distributed.all_gather_into_tensor(
        unpad_lens, local_unpad_len, group=sp_group
    )

    # Padding to max_len for gather.
    max_len = unpad_lens.max()
    x_pad = safe_pad_operation(x, (0, 0, 0, 0, 0, max_len - x.size(2))).contiguous()

    # Remove padding.
    x_pad_lists = list(x_pad.chunk(sp_size, dim=2))
    for i, (x_pad, unpad_len) in enumerate(zip(x_pad_lists, unpad_lens)):
        x_pad_lists[i] = x_pad[:, :, :unpad_len]

    return torch.cat(x_pad_lists, dim=2)


def get_output_len(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = (
        conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    )
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[
        dim
    ] + 1
    return output_len


def get_cache_size(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = (
        conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    )
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[
        dim
    ] + 1
    remain_len = (
        input_len
        + pad_len
        - ((output_len - 1) * conv_module.stride[dim] + dilated_kernerl_size)
    )
    overlap_len = dilated_kernerl_size - conv_module.stride[dim]
    cache_len = overlap_len + remain_len  # >= 0

    assert output_len > 0
    return cache_len


def cache_send_recv(tensor: List[Tensor], cache_size, times, memory=None):
    sp_group = None
    sp_rank = 0
    sp_size = 1
    send_dst = 1
    recv_src = 0
    recv_buffer = None
    recv_req = None

    if sp_rank == 0 or sp_group is None:
        if memory is not None:
            recv_buffer = memory.to(tensor[0])
        elif times > 0:
            tile_repeat = [1] * tensor[0].ndim
            tile_repeat[2] = times
            recv_buffer = torch.tile(tensor[0][:, :, :1], tile_repeat)

    if cache_size != 0 and sp_group is not None:
        if sp_rank > 0:
            shape = list(tensor[0].size())
            shape[2] = cache_size
            recv_buffer = torch.empty(
                *shape, device=tensor[0].device, dtype=tensor[0].dtype
            ).contiguous()
            recv_req = dist.irecv(recv_buffer, recv_src, group=sp_group)
        if sp_rank < sp_size - 1:
            if cache_size > tensor[-1].size(2) and len(tensor) == 1:
                if recv_req is not None:
                    recv_req.wait()
                tensor[0] = torch.cat([recv_buffer, tensor[0]], dim=2)
                recv_buffer = None
            assert cache_size <= tensor[-1].size(
                2
            ), f"Not enough value to cache, got {tensor[-1].size()}, cache_size={cache_size}"
            dist.isend(
                tensor[-1][:, :, -cache_size:].detach().contiguous(),
                send_dst,
                group=sp_group,
            )
        if recv_req is not None:
            recv_req.wait()

    return recv_buffer
