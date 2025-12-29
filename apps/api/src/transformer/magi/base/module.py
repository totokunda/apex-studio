# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numbers
from functools import partial
from typing import Callable, List, Optional, Tuple, Dict, Any


try:
    import flashinfer
    from flashinfer.gemm import bmm_fp8
except:
    flashinfer = None
    bmm_fp8 = None

import torch
import torch.distributed
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False
from einops import rearrange

from torch import Tensor
from torch.nn import Parameter

from .attention import (
    FusedLayerNorm,
    Attention,
    MagiCrossAttentionProcessor,
    MagiSelfAttentionProcessor,
)


##########################################################
# TimestepEmbedder
##########################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self, hidden_dim, cond_hidden_ratio=0.125, frequency_embedding_size=256
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, int(hidden_dim * cond_hidden_ratio), bias=True
            ),
            nn.SiLU(),
            nn.Linear(
                int(hidden_dim * cond_hidden_ratio),
                int(hidden_dim * cond_hidden_ratio),
                bias=True,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

        # rescale the timestep for the general transport model
        self.timestep_rescale_factor = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, timestep_rescale_factor=1):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] * timestep_rescale_factor
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t.to(torch.float32)
        t_freq = self.timestep_embedding(
            t,
            self.frequency_embedding_size,
            timestep_rescale_factor=self.timestep_rescale_factor,
        )
        t_emb = self.mlp(t_freq)
        return t_emb


##########################################################
# CaptionEmbedder
##########################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        caption_channels,
        hidden_dim,
        xattn_cond_hidden_ratio,
        cond_hidden_ratio,
        caption_max_length,
    ):
        super().__init__()

        self.y_proj_xattn = nn.Sequential(
            nn.Linear(
                caption_channels, int(hidden_dim * xattn_cond_hidden_ratio), bias=True
            ),
            nn.SiLU(),
        )

        self.y_proj_adaln = nn.Sequential(
            nn.Linear(caption_channels, int(hidden_dim * cond_hidden_ratio), bias=True)
        )

        self.null_caption_embedding = Parameter(
            torch.empty(caption_max_length, caption_channels)
        )

    def caption_drop(self, caption, caption_dropout_mask):
        """
        Drops labels to enable classifier-free guidance.
        caption.shape = (N, 1, cap_len, C)
        """
        dropped_caption = torch.where(
            caption_dropout_mask[:, None, None, None],  # (N, 1, 1, 1)
            self.null_caption_embedding[None, None, :],  # (1, 1, cap_len, C)
            caption,  # (N, 1, cap_len, C)
        )
        return dropped_caption

    def caption_drop_single_token(self, caption_dropout_mask):
        dropped_caption = torch.where(
            caption_dropout_mask[:, None, None],  # (N, 1, 1)
            self.null_caption_embedding[None, -1, :],  # (1, 1, C)
            self.null_caption_embedding[None, -2, :],  # (1, 1, C)
        )
        return dropped_caption  # (N, 1, C)

    def forward(self, caption, train, caption_dropout_mask=None):
        if train and caption_dropout_mask is not None:
            caption = self.caption_drop(caption, caption_dropout_mask)
        caption_xattn = self.y_proj_xattn(caption)
        if caption_dropout_mask is not None:
            caption = self.caption_drop_single_token(caption_dropout_mask)

        caption_adaln = self.y_proj_adaln(caption)
        return caption_xattn, caption_adaln


##########################################################
# FinalLinear
##########################################################
class FinalLinear(nn.Module):
    """
    The final linear layer of DiT.
    """

    def __init__(self, hidden_dim, patch_size, t_patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(
            hidden_dim,
            patch_size * patch_size * t_patch_size * out_channels,
            bias=False,
        )

    def forward(self, x):
        x = self.linear(x)
        return x


##########################################################
# AdaModulateLayer
##########################################################
class AdaModulateLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_hidden_ratio: float,
        cond_gating_ratio: float,
        gate_num_chunks: int = 2,
    ):
        super().__init__()
        self.cond_hidden_ratio = cond_hidden_ratio
        self.cond_gating_ratio = cond_gating_ratio

        self.gate_num_chunks = gate_num_chunks
        self.act = nn.SiLU()
        self.proj = nn.Sequential(
            nn.Linear(
                int(hidden_dim * cond_hidden_ratio),
                int(hidden_dim * cond_gating_ratio * self.gate_num_chunks),
                bias=True,
            )
        )

    def forward(self, c):
        c = self.act(c)
        return self.proj(c)


##########################################################
# bias_modulate_add
##########################################################
if TRITON_AVAILABLE:

    @triton.jit
    def range_mod_kernel_fwd(
        X,  # pointer to the input
        MAP,  # map x index to gating index
        GATINGS,  # pointer to the gatings
        Y,  # pointer to the output
        M,  # number of rows in X, unused
        N,  # number of columns in X
        stride_xm,  # how much to increase the pointer when moving by 1 row in X
        stride_xn,  # how much to increase the pointer when moving by 1 column in X
        stride_gm,  # how much to increase the pointer when moving by 1 row in GATINGS
        stride_gn,  # how much to increase the pointer when moving by 1 column in GATINGS
        stride_ym,  # how much to increase the pointer when moving by 1 row in Y
        stride_yn,  # how much to increase the pointer when moving by 1 column in Y
        BLOCK_SIZE: tl.constexpr,  # number of columns in a block
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)

        cur_X = X + row * stride_xm
        x_cols = tl.arange(0, BLOCK_SIZE) * stride_xn
        x_mask = x_cols < N * stride_xn
        x = tl.load(cur_X + x_cols, mask=x_mask, other=0.0)

        cur_MAP = MAP + row
        gating_index = tl.load(cur_MAP)
        cur_GATING = GATINGS + gating_index * stride_gm
        gating_cols = tl.arange(0, BLOCK_SIZE) * stride_gn
        gating_mask = gating_cols < N * stride_gn
        gating = tl.load(cur_GATING + gating_cols, mask=gating_mask, other=0.0)

        cur_Y = Y + row * stride_ym
        y_cols = tl.arange(0, BLOCK_SIZE) * stride_yn
        y_mask = y_cols < N * stride_yn
        tl.store(cur_Y + y_cols, x * gating, mask=y_mask)

    def range_mod_triton(x, c_mapping, gatings):
        """
        Inputs:
            x: (s, b, h). Tensor of inputs embedding (images or latent representations of images)
            c_mapping: (s, b). Tensor of condition map
            gatings: (b, denoising_range_num, h). Tensor of condition embedding
        """

        assert x.is_cuda, "x is not on cuda"
        assert c_mapping.is_cuda, "c_mapping is not on cuda"
        assert gatings.is_cuda, "gatings is not on cuda"

        # TODO: use 3D tensor for x, c_mapping, and gatings
        s, b, h = x.shape
        x = x.transpose(0, 1).flatten(0, 1)
        c_mapping = c_mapping.transpose(0, 1).flatten(0, 1)
        gatings = gatings.flatten(0, 1)

        assert x.dim() == 2, f"x must be a 2D tensor but got {x.dim()}D"
        assert (
            c_mapping.dim() == 1
        ), f"c_mapping must be a 1D tensor but got {c_mapping.dim()}D"
        assert (
            gatings.dim() == 2
        ), f"gatings must be a 2D tensor but got {gatings.dim()}D"

        M, N = x.shape
        assert (
            c_mapping.size(0) == M
        ), "c_mapping must have the same number of rows as x"

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("range_mod_triton doesn't support feature dim >= 64KB.")

        MAP = c_mapping
        y = torch.empty_like(x)

        range_mod_kernel_fwd[(M,)](
            x,
            MAP,
            gatings,
            y,
            M,
            N,
            x.stride(0),
            x.stride(1),
            gatings.stride(0),
            gatings.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        y = y.reshape(b, s, h).transpose(0, 1)

        return y


def range_mod_torch(x, c_mapping, gatings):
    """
    PyTorch fallback implementation for range_mod_triton.

    Inputs:
        x: (s, b, h). Tensor of inputs embedding (images or latent representations of images)
        c_mapping: (s, b). Tensor of condition map
        gatings: (b, denoising_range_num, h). Tensor of condition embedding
    """
    s, b, h = x.shape
    _, denoising_range_num, _ = gatings.shape

    # Flatten for easier indexing
    x_flat = x.transpose(0, 1).flatten(0, 1)  # (b*s, h)
    c_mapping_flat = c_mapping.transpose(0, 1).flatten(0, 1)  # (b*s,)
    gatings_flat = gatings.flatten(0, 1)  # (b*denoising_range_num, h)

    # Use advanced indexing to apply gating
    y_flat = x_flat * gatings_flat[c_mapping_flat]  # (b*s, h)

    # Reshape back to original dimensions
    y = y_flat.reshape(b, s, h).transpose(0, 1)  # (s, b, h)

    return y


def bias_modulate_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    condition_map: torch.Tensor,
    gate: torch.Tensor,
    post_norm: torch.nn.Module,
):
    assert gate.shape[-1] == x.shape[-1]

    original_dtype = x.dtype
    x = x.float()
    residual = residual.float()
    gate = gate.float()

    # Use Triton implementation if available, otherwise fall back to PyTorch
    if TRITON_AVAILABLE and x.is_cuda:
        x = range_mod_triton(x, condition_map, gate)
    else:
        x = range_mod_torch(x, condition_map, gate)

    x = post_norm(x)
    x = x + residual
    x = x.to(original_dtype)

    return x


##########################################################
# FusedLayerNorm
##########################################################
def make_viewless_tensor(inp, requires_grad):
    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    out = torch.empty(
        (1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad
    )
    out.data = inp.data
    return out


def softcap(x: torch.Tensor, cap: int):
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)


def div_clamp_to(x: torch.Tensor, scale: torch.Tensor):
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    prefix_shape = x.shape[:-1]
    last_shape = x.shape[-1]
    x = x.flatten().reshape(-1, last_shape)
    # Split x into 256 MB parts to avoid big memory peak
    part_size = 256 * 1024 * 1024 // last_shape
    part_num = (x.shape[0] + part_size - 1) // part_size
    return (
        torch.cat(
            [
                torch.clamp(
                    x[i * part_size : (i + 1) * part_size].float() / scale.float(),
                    fp8_min,
                    fp8_max,
                ).bfloat16()
                for i in range(part_num)
            ],
            dim=0,
        )
        .to(torch.float8_e4m3fn)
        .reshape(*prefix_shape, last_shape)
        .contiguous()
    )


class FeedForward(torch.nn.Module):
    """
    FeedForward will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    Args:
        hidden_dim: int, hidden dimension
        ffn_hidden_dim: int, hidden dimension of the feedforward network
        layernorm_epsilon: float, epsilon for the layer norm
        params_dtype: torch.dtype, dtype of the parameters
        gated_linear_unit: bool, whether to use the gated linear unit

    Returns an output and a bias to be added to the output.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int,
        layernorm_epsilon: float,
        gated_linear_unit: bool,
        layer_number: int = None,
        input_size: int = None,
    ):
        super().__init__()

        self.layer_number = layer_number

        self.input_size = input_size if input_size != None else hidden_dim
        self.gated_linear_unit = gated_linear_unit

        self.norm = torch.nn.LayerNorm(self.input_size, eps=layernorm_epsilon)

        self.act = torch.nn.GELU()

        if gated_linear_unit:
            self.proj1 = torch.nn.Linear(
                self.input_size,
                2 * ffn_hidden_dim,
                bias=False,
            )
        else:
            self.proj1 = torch.nn.Linear(
                self.input_size,
                ffn_hidden_dim,
                bias=False,
            )

        self.proj2 = torch.nn.Linear(
            ffn_hidden_dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)

        hidden_states = self.proj1(hidden_states)
        if self.gated_linear_unit:
            hidden_states = flashinfer.activation.silu_and_mul(hidden_states)
        else:
            hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return hidden_states


##########################################################
# LearnableRotaryEmbeddingCat
##########################################################
def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def pixel_freq_bands(
    num_bands: int,
    max_freq: float = 224.0,
    linear_bands: bool = True,
    device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(
            1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device
        )
    else:
        bands = 2 ** torch.linspace(
            0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device
        )
    return bands * torch.pi


def freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    step: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = (
        torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(
            torch.float32
        )
        / num_bands
    )
    bands = 1.0 / (temperature**exp)
    return bands


def build_fourier_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    num_bands: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    include_grid: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands, float(max_res), linear_bands=linear_bands, device=device
            )
        else:
            bands = freq_bands(
                num_bands, temperature=temperature, step=1, device=device
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [
            torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    else:
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
            for s in feat_shape
        ]
        # align spatial center (H/2,W/2) to (0,0)
        t[1] = t[1] - (feat_shape[1] - 1) / 2
        t[2] = t[2] - (feat_shape[2] - 1) / 2
    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        # aligning to the endpoint e.g [0,1,2] -> [0, 0.4, 0.8, 1.2, 1.6, 2]
        t_rescaled = []
        for x, f, r in zip(t, feat_shape, ref_feat_shape):
            # deal with image input
            if f == 1:
                assert r == 1, "ref_feat_shape must be 1 when feat_shape is 1"
                t_rescaled.append(x)
            else:
                t_rescaled.append(x / (f - 1) * (r - 1))
        t = t_rescaled

    grid = torch.stack(ndgrid(t), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


def build_rotary_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 8,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x

    sin_emb = sin_emb.reshape(num_spatial_dim, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1)
    return sin_emb, cos_emb


class LearnableRotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.bands = nn.Parameter(self.get_default_bands())

    def get_default_bands(self):
        if self.in_pixels:
            bands = pixel_freq_bands(
                self.dim // 8,
                float(self.max_res),
                linear_bands=self.linear_bands,
                devicse=torch.cuda.current_device(),
            )
        else:
            bands = freq_bands(
                self.dim // 8,
                temperature=self.temperature,
                step=1,
                device=torch.cuda.current_device(),
            )
        return bands

    def get_embed(
        self, shape: Optional[List[int]], ref_feat_shape: Optional[List[int]] = None
    ):
        # rebuild bands and embeddings every call, use if target shape changes
        embeds = build_rotary_pos_embed(
            feat_shape=shape,
            bands=self.bands,  # use learned bands
            dim=self.dim,
            max_res=self.max_res,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=ref_feat_shape if ref_feat_shape else self.ref_feat_shape,
            temperature=self.temperature,
            device=torch.cuda.current_device(),
        )
        return torch.cat(embeds, -1)


##########################################################
# TransformerBlock
##########################################################
class MagiTransformerBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        hidden_dim: int,
        cross_attention_dim: int,
        ffn_hidden_dim: int,
        cond_hidden_ratio: float,
        cond_gating_ratio: float,
        xattn_cond_hidden_ratio: float,
        gate_num_chunks: int = 2,
        zero_centered_gamma: bool = True,
        eps: float = 1e-5,
        layer_number: int = 1,
        gated_linear_unit: bool = True,
        num_attention_heads: int = 16,
        num_query_groups: int = 1,
    ):
        super().__init__()

        self.layer_number = layer_number
        ## [Module 1: ada_modulate_layer
        dim_head = hidden_dim // num_attention_heads
        self.adaln = AdaModulateLayer(
            hidden_dim=hidden_dim,
            cond_hidden_ratio=cond_hidden_ratio,
            cond_gating_ratio=cond_gating_ratio,
            gate_num_chunks=gate_num_chunks,
        )

        self.norm1 = torch.nn.LayerNorm(
            hidden_dim,
            eps=eps,
        )

        ## [Module 2: SelfAttention]
        self.attn1 = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            num_query_groups=num_query_groups,
            dim_head=dim_head,
            eps=eps,
            layer_number=self.layer_number,
            processor=MagiSelfAttentionProcessor(),
        )

        self.attn2 = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=cross_attention_dim,
            fuse_cross_attention=True,
            heads=num_attention_heads,
            num_query_groups=num_query_groups,
            dim_head=dim_head,
            eps=eps,
            layer_number=self.layer_number,
            processor=MagiCrossAttentionProcessor(),
        )

        self.proj = torch.nn.Linear(
            hidden_dim * 2,
            hidden_dim,
            bias=False,
        )

        ## [Module 3: SelfAttention PostNorm]
        self.norm2 = FusedLayerNorm(
            zero_centered_gamma=zero_centered_gamma,
            hidden_dim=hidden_dim,
            eps=eps,
        )

        ## [Module 4: MLP block]
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            layernorm_epsilon=eps,
            gated_linear_unit=gated_linear_unit,
            layer_number=self.layer_number,
        )

        ## [Module 5: MLP PostNorm]
        self.norm3 = FusedLayerNorm(
            zero_centered_gamma=zero_centered_gamma,
            hidden_dim=hidden_dim,
            eps=eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition: torch.Tensor,
        condition_map: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        self_attn_params: Dict[str, Any] | None = None,
        cross_attn_params: Dict[str, Any] | None = None,
        kv_cache_params: Dict[str, Any] | None = None,
    ):
        # hidden_states: [s/cp/sp, b, h]
        residual = hidden_states
        original_dtype = hidden_states.dtype

        norm_hidden_states = self.norm1(hidden_states)

        device = hidden_states.device

        # Self attention.

        attn_out = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_pos_emb,
            kv_cache_params=kv_cache_params,
            **self_attn_params,
        )

        cross_attn_out = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attn_params,
        )

        attn_out = torch.concat([attn_out, cross_attn_out], dim=2)
        # NOTE: hn=8 is hardcoded to align with TP8 traning and TP1 inference
        attn_out = rearrange(attn_out, "sq b (n hn hd) -> sq b (hn n hd)", n=2, hn=8)

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            attn_out = self.proj(attn_out)

        hidden_states = attn_out

        gate_output = self.adaln(condition)

        softcap_gate_cap = 1.0
        gate_output = softcap(gate_output, softcap_gate_cap)
        gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)

        # Residual connection for self-attention.
        hidden_states = bias_modulate_add(
            hidden_states, residual, condition_map, gate_msa, self.norm2
        ).to(original_dtype)

        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        # Residual connection for MLP.
        hidden_states = bias_modulate_add(
            hidden_states, residual, condition_map, gate_mlp, self.norm3
        ).to(original_dtype)

        return hidden_states
