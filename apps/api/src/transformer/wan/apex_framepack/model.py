from diffusers.models.transformers.transformer_wan import (
    get_1d_rotary_pos_embed,
    WanImageEmbedding,
)
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    PixArtAlphaTextProjection,
)
from typing import Tuple, Optional, Any, Dict, Union, List, Literal, Tuple
import torch
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import scale_lora_layers, USE_PEFT_BACKEND
from diffusers.utils import logger
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import List
import torch.nn as nn
from diffusers.configuration_utils import register_to_config, ConfigMixin
import math
from diffusers.models.cache_utils import CacheMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models import ModelMixin
import torch.utils.checkpoint
from diffusers.utils import scale_lora_layers, unscale_lora_layers
import einops
from src.transformer.wan.apex_framepack.module import BaseSchedule
import src.transformer.wan.apex_framepack.module
from diffusers.models.attention import Attention, FeedForward
from src.transformer.wan.base.attention import WanAttnProcessor2_0
import torch.nn.functional as F
from src.transformer import TRANSFORMERS_REGISTRY


class MoEFeedForward(nn.Module):
    """
    Drop-in replacement for a regular FFN that routes every token to one
    of `num_experts` FeedForward blocks using a linear gating network.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        num_experts: int = 8,
        activation_fn: str = "gelu-approximate",
        top_k: int = 1,  # set to 2 for Switch-Top-2
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        # pool of experts â€“ each is the vanilla FeedForward you already use
        self.experts = nn.ModuleList(
            [
                FeedForward(dim, inner_dim=inner_dim, activation_fn=activation_fn)
                for _ in range(num_experts)
            ]
        )
        # token â†’ expert scores
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,  # x : [B, T, D]
        hidden_states: torch.Tensor,  # keep name to match call-site
        style_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = hidden_states.shape
        # 1) broadcast style to [B, L, D_style]

        style_expanded = style_emb.unsqueeze(1).expand(-1, L, -1)
        gate_input = torch.cat([hidden_states, style_expanded], dim=1)
        logits = self.gate(gate_input)  # [B, T, N]
        # ---- top-k routing --------------------------------------------------
        vals, idx = torch.topk(logits, self.top_k, dim=-1)  # both [B, T, k]
        if self.top_k == 1:
            idx = idx.squeeze(-1)  # [B, T]
            out = torch.zeros_like(hidden_states)
            # route tokens to their chosen expert
            for e in range(self.num_experts):
                mask = idx == e  # bool [B, T]
                if mask.any():
                    routed = hidden_states[mask]  # [n_tok, D]
                    out[mask] = self.experts[e](routed)
            # optional: return a load-balancing histogram
            usage = torch.bincount(idx.flatten(), minlength=self.num_experts).float()
            return out, usage
        else:
            # --- simple top-k = 2 variant -------------
            weights = torch.softmax(vals, dim=-1)  # [B, T, k]
            out = torch.zeros_like(hidden_states)
            for slot in range(self.top_k):
                e_ids = idx[..., slot]  # [B, T]
                w = weights[..., slot].unsqueeze(-1)  # [B, T, 1]
                for e in range(self.num_experts):
                    mask = e_ids == e
                    if mask.any():
                        routed = hidden_states[mask]
                        out[mask] += w[mask] * self.experts[e](routed)
            usage = torch.bincount(idx.reshape(-1), minlength=self.num_experts).float()
            return out, usage


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        use_moe: bool = True,
        num_experts: int = 3,
        moe_top_k: int = 1,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.last_moe_usage = None
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        if use_moe:
            self.moe_ffn = MoEFeedForward(
                dim,
                inner_dim=ffn_dim,
                activation_fn="gelu-approximate",
                num_experts=num_experts,
                top_k=moe_top_k,
            )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        style_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb
        )

        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)

        if self.use_moe:
            style_emb = style_emb.squeeze(1)
            ff_output, usage = self.moe_ffn(norm_hidden_states, style_emb)
            self.last_moe_usage = usage
        else:
            ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        return hidden_states


# Assuming LatentEmbedPacker class exists as provided previously
class LatentEmbedPacker(nn.Module):
    def __init__(
        self,
        compressors: Dict[int, str] = {
            1: "k1h2w2",
            4: "k1h4w4",
            16: "k2h8w8",
            64: "k4h16w16",
        },
        C_latent=16,
        inner_dim=1152,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        # Ensure keys are integers for lookup consistency
        self.compressors = {int(k): v for k, v in compressors.items()}
        self.embedders = nn.ModuleDict()
        self.C_latent = C_latent
        self.inner_dim = inner_dim
        for key, value in self.compressors.items():
            # Parse kernel/stride string kThNwM -> T, N, M
            parts = value.lower().split("k")[1]
            f = int(parts.split("h")[0])
            h = int(parts.split("h")[1].split("w")[0])
            w = int(parts.split("w")[1])
            # Assume stride matches kernel size for these embedders
            self.embedders[str(key)] = nn.Conv3d(
                C_latent, inner_dim, kernel_size=(f, h, w), stride=(f, h, w)
            )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        # No need for partial methods if calling embed_fn directly

    def embed_fn(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """Applies the specified embedder with optional gradient checkpointing."""
        embedder = self.embedders[key]
        if (
            self.use_gradient_checkpointing and self.training
        ):  # Checkpointing only during training
            return torch.utils.checkpoint.checkpoint(embedder, x, use_reentrant=False)
        else:
            return embedder(x)

    def get_embedder_params(self, key: str) -> tuple[tuple, tuple]:
        """Helper to get kernel_size and stride for padding"""
        embedder = self.embedders[key]
        return getattr(embedder, "kernel_size", (1, 1, 1)), getattr(
            embedder, "stride", (1, 1, 1)
        )

    @torch.no_grad()
    def initialize_weights_from_conv3d(self, conv3d: nn.Conv3d):
        """
        Initializes the weights and biases of the internal embedders using
        an external Conv3d layer.

        Assumes the external conv3d's kernel/stride represents the smallest
        patch size, and the internal embedders have kernel/stride sizes
        that are integer multiples of the external one. Weights are repeated
        spatially using einops. Biases are copied directly.

        Args:
            conv3d (nn.Conv3d): The external Conv3d layer to copy weights/biases from.
                               Expected shape: Conv3d(C_latent, inner_dim,
                               kernel_size=patch_size, stride=patch_size)
        """
        # --- Input Validation ---
        if not isinstance(conv3d, nn.Conv3d):
            raise TypeError(f"Input layer must be nn.Conv3d, got {type(conv3d)}")

        if conv3d.in_channels != self.C_latent:
            raise ValueError(
                f"Input conv3d in_channels ({conv3d.in_channels}) "
                f"must match LatentEmbedPacker C_latent ({self.C_latent})"
            )

        if conv3d.out_channels != self.inner_dim:
            raise ValueError(
                f"Input conv3d out_channels ({conv3d.out_channels}) "
                f"must match LatentEmbedPacker inner_dim ({self.inner_dim})"
            )

        # --- Weight and Bias Initialization ---
        base_weight = conv3d.weight.clone()  # Shape: (inner_dim, C_latent, pf, ph, pw)
        base_bias = conv3d.bias.clone() if conv3d.bias is not None else None
        base_kernel_size = conv3d.kernel_size  # (pf, ph, pw)

        # Check if input layer has bias, as embedders expect it
        if base_bias is None:
            # Option 1: Raise error if embedders require bias
            raise ValueError(
                "Input conv3d layer has no bias (bias=False), but internal embedders expect bias."
            )
            # Option 2: Initialize embedder biases to zero (less ideal if trying to match behavior)
            # print("Warning: Input conv3d has no bias. Initializing embedder biases to zero.")

        for key, embedder in self.embedders.items():
            embedder_kernel_size = embedder.kernel_size  # (f, h, w)
            # to empty to get off meta device
            embedder.to_empty(device=conv3d.weight.device)

            # --- Bias Initialization ---
            if embedder.bias is not None:
                if base_bias is not None:
                    embedder.bias.copy_(base_bias)
                # else: # Base_bias is None, handled by the check above
                #     nn.init.zeros_(embedder.bias) # Option 2 if warning above is used
            elif base_bias is not None:
                # This case shouldn't happen based on default embedder creation, but good to acknowledge
                print(
                    f"Warning: Embedder {key} has no bias parameter, but input conv3d does. Skipping bias copy for this embedder."
                )

            # --- Weight Initialization ---
            # If kernel sizes match exactly, just copy
            if base_kernel_size == embedder_kernel_size:
                embedder.weight.copy_(base_weight)
            else:
                # Ensure embedder kernel is a multiple of the base kernel
                is_multiple = all(
                    ek % bk == 0
                    for ek, bk in zip(embedder_kernel_size, base_kernel_size)
                )
                if not is_multiple:
                    raise ValueError(
                        f"Embedder {key} kernel size {embedder_kernel_size} "
                        f"is not an integer multiple of input kernel size {base_kernel_size}"
                    )

                # Calculate repetition factors for each dimension (F, H, W)
                f_factor = embedder_kernel_size[0] // base_kernel_size[0]
                h_factor = embedder_kernel_size[1] // base_kernel_size[1]
                w_factor = embedder_kernel_size[2] // base_kernel_size[2]

                # Use einops.repeat to tile the base weight tensor
                # 'o i pf ph pw -> o i (pf F) (ph H) (pw W)'
                # o: out_channels (inner_dim)
                # i: in_channels (C_latent)
                # pf, ph, pw: base kernel dimensions
                # F, H, W: repetition factors
                repeated_weight = einops.repeat(
                    base_weight,
                    "o i pf ph pw -> o i (pf f_factor) (ph h_factor) (pw w_factor)",
                    f_factor=f_factor,
                    h_factor=h_factor,
                    w_factor=w_factor,
                )

                # Ensure the repeated weight has the correct target shape
                if repeated_weight.shape != embedder.weight.shape:
                    # This should not happen if logic is correct, but good for sanity check
                    raise RuntimeError(
                        f"Shape mismatch after repeating weights for embedder {key}. "
                        f"Expected {embedder.weight.shape}, got {repeated_weight.shape}"
                    )

                # Copy the repeated weights into the embedder layer
                embedder.weight.copy_(repeated_weight)


# Assuming pad_for_3d_conv exists as defined previously
def pad_for_3d_conv(x: torch.Tensor, kernel_size: tuple, stride: tuple) -> torch.Tensor:
    b, c, t, h, w = x.shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    pad_t_k = (kt - (t % kt)) % kt
    pad_h_k = (kh - (h % kh)) % kh
    pad_w_k = (kw - (w % kw)) % kw
    # Simplistic padding based on kernel only for illustration
    final_pad_t = pad_t_k
    final_pad_h = pad_h_k
    final_pad_w = pad_w_k
    padded_x = torch.nn.functional.pad(
        x, (0, final_pad_w, 0, final_pad_h, 0, final_pad_t), mode="replicate"
    )
    return padded_x


def center_down_sample_3d(x, kernel_size):
    # pt, ph, pw = kernel_size
    # cp = (pt * ph * pw) // 2
    # xp = einops.rearrange(x, 'b c (t pt) (h ph) (w pw) -> (pt ph pw) b c t h w', pt=pt, ph=ph, pw=pw)
    # xc = xp[cp]
    # return xc
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


def convert_wan_complex_to_real_2d(wan_complex_freqs: torch.Tensor) -> torch.Tensor:
    """
    Converts Wan RoPE complex frequencies (D/2) to a real representation (2D).

    Args:
        wan_complex_freqs: Tensor of complex frequencies with shape (B, D/2, T, H, W).

    Returns:
        Tensor of real frequencies with shape (B, 2D, T, H, W),
        structured like [Real_D, Imag_D].
    """
    B, C_half, T, H, W = wan_complex_freqs.shape
    D = C_half * 2

    # Check if input is complex, if not, maybe it's already converted?
    if not torch.is_complex(wan_complex_freqs):
        raise ValueError("Input tensor must be complex.")

    # Reconstruct complex representation for the full dimension D
    # Assumes pairing: freq for dim 2k+1 is same as freq for dim 2k
    # This might need adjustment depending on exact RoPE application,
    # but is standard for reconstructing full representation.
    # We need Real(D) and Imag(D).
    # Real(D) = [Real(0), Real(1), Real(2), Real(3), ...]
    # Imag(D) = [Imag(0), Imag(1), Imag(2), Imag(3), ...]
    # wan_complex_freqs has [Complex(0), Complex(2), Complex(4), ...]

    # Get Real(D/2) and Imag(D/2) corresponding to indices 0, 2, 4...
    real_part_half = wan_complex_freqs.real  # Shape (B, D/2, T, H, W)
    imag_part_half = wan_complex_freqs.imag  # Shape (B, D/2, T, H, W)

    # Create tensors for full D dimension, filling gaps
    # We need Real_D and Imag_D
    # For Real_D: Indices 0, 2, 4... come from real_part_half
    #             Indices 1, 3, 5... need Real part of Complex(1,3,5...)
    # For Imag_D: Indices 0, 2, 4... come from imag_part_half
    #             Indices 1, 3, 5... need Imag part of Complex(1,3,5...)
    # Standard RoPE applies rotation using Complex(0,2,4..) to pairs (0,1), (2,3) etc.
    # Rotation of (x_2k, x_{2k+1}) by Complex(2k) = c + is:
    # x'_2k = x_2k * c - x_{2k+1} * s
    # x'_{2k+1} = x_2k * s + x_{2k+1} * c
    # The frequencies themselves (theta, position based) are often identical for the pair.
    # So Complex(2k+1) is often considered equal to Complex(2k).

    # Let's reconstruct the full complex D tensor first, then split
    complex_D = wan_complex_freqs.repeat_interleave(2, dim=1)  # Shape (B, D, T, H, W)

    real_D = complex_D.real
    imag_D = complex_D.imag

    # Concatenate Real and Imaginary parts for all D dimensions
    real_2d_representation = torch.cat(
        [real_D, imag_D], dim=1
    )  # Shape (B, 2D, T, H, W)

    return real_2d_representation


def convert_real_2d_to_wan_complex(
    real_2d_representation: torch.Tensor,
) -> torch.Tensor:
    """
    Converts a real representation (2D) back to Wan RoPE complex frequencies (D/2).

    Args:
        real_2d_representation: Tensor of real frequencies structured like
                                 [Real_D, Imag_D] with shape (B, 2D, T, H, W).

    Returns:
        Tensor of complex frequencies with shape (B, D/2, T, H, W).
    """
    B, C_full, T, H, W = real_2d_representation.shape
    if C_full % 2 != 0:
        raise ValueError("Input feature dimension must be 2*D (even).")
    D = C_full // 2

    # Split the Real_D and Imag_D parts
    real_D = real_2d_representation[:, :D, ...]  # Shape (B, D, T, H, W)
    imag_D = real_2d_representation[:, D:, ...]  # Shape (B, D, T, H, W)

    # Form the complex tensor for the full dimension D
    complex_D = torch.complex(
        real_D.to(torch.float64), imag_D.to(torch.float64)
    )  # Shape (B, D, T, H, W)

    # Select the components corresponding to the even indices (0, 2, 4...)
    # This retrieves the original D/2 complex frequency components.
    wan_complex_freqs = complex_D[:, ::2, ...]  # Shape (B, D/2, T, H, W)

    return wan_complex_freqs


class ScheduleSelector(nn.Module):
    """
    Pick a compression schedule index 0 â€¦ N-1 from latent-frame features.
    """

    def __init__(self, c, num_sched=4):
        super().__init__()
        self.net = nn.Sequential(  # < 0.1 % params
            nn.Conv1d(c, c // 4, 1),  # (B, C, T) â†’ (B, C/4, T)
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # global pool over T
            nn.Flatten(),  # (B, C/4)
            nn.Linear(c // 4, num_sched),  # logits for N schedules
        )

    def forward(self, feats, tau=0.5, hard=False):
        """
        feats : (B, C, T)  pooled-latent features
        returns:
            y_soft : (B, N)  â€“ soft one-hot (Gumbel-Softmax)
            idx    : (B,)    â€“ argmax index if hard=True, else None
        """
        logits = self.net(feats)  # (B, N)
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # differentiable
        idx = y.argmax(-1) if hard else None
        return y, idx


class WanRotaryPosEmbedIndices(nn.Module):
    """
    Computes RoPE frequencies based on explicit frame indices and spatial dimensions,
    using pre-calculation similar to WanRoPE. Outputs complex frequencies.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,  # Max *temporal patch* index + 1
        theta: float = 10000.0,
        # Optional: Define max spatial extent if different from max_seq_len
        max_spatial_len: int = None,
    ):
        """
        Args:
            attention_head_dim (int): Dimension of the attention head (D). Must be divisible by 2.
            patch_size (Tuple[int, int, int]): Patch size (p_t, p_h, p_w).
            max_seq_len (int): Maximum sequence length for pre-calculation.
                               This should correspond to the maximum *temporal patch index*
                               you expect plus one (e.g., max_frames / p_t).
            theta (float, optional): RoPE theta value. Defaults to 10000.0.
            max_spatial_len (int, optional): Maximum spatial dimension (height or width)
                                             in patches for pre-calculation. Defaults to max_seq_len.
        """
        super().__init__()

        if attention_head_dim % 2 != 0:
            raise ValueError("attention_head_dim must be divisible by 2")

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.feature_dim_half = attention_head_dim // 2
        self._max_spatial_len = (
            max_spatial_len if max_spatial_len is not None else max_seq_len
        )

        # Calculate sub-dimensions for T, H, W RoPE components (complex dimensions)
        # Try to make them as even as possible, usually close to D/6 each
        h_complex_dim = self.feature_dim_half // 3
        w_complex_dim = self.feature_dim_half // 3
        t_complex_dim = self.feature_dim_half - h_complex_dim - w_complex_dim
        # Ensure sum is correct due to potential rounding
        if t_complex_dim + h_complex_dim + w_complex_dim != self.feature_dim_half:
            # Adjust t_dim if rounding caused mismatch
            t_complex_dim = self.feature_dim_half - h_complex_dim - w_complex_dim
            assert t_complex_dim >= 0

        self.t_complex_dim = t_complex_dim
        self.h_complex_dim = h_complex_dim
        self.w_complex_dim = w_complex_dim

        # Pre-calculate 1D RoPE frequencies for each dimension
        # Note: get_1d_rotary_pos_embed expects *real* dim (2*complex_dim) as input
        # and returns complex tensor of size complex_dim (D/2 part)

        freqs_t = get_1d_rotary_pos_embed(
            self.t_complex_dim * 2,
            max_seq_len,  # Max temporal patch index + 1
            theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,  # Use higher precision for calculation
        )  # Shape (max_seq_len, t_complex_dim)

        freqs_h = get_1d_rotary_pos_embed(
            self.h_complex_dim * 2,
            self._max_spatial_len,  # Max height patch index + 1
            theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )  # Shape (max_spatial_len, h_complex_dim)

        freqs_w = get_1d_rotary_pos_embed(
            self.w_complex_dim * 2,
            self._max_spatial_len,  # Max width patch index + 1
            theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )  # Shape (max_spatial_len, w_complex_dim)

        # Store frequencies, casting to float32 (complex64) for typical usage
        self.freqs_t = freqs_t
        self.freqs_h = freqs_h
        self.freqs_w = freqs_w

    def forward(
        self,
        frame_indices: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Calculates complex RoPE frequencies based on explicit frame indices and spatial dimensions.

        Args:
            frame_indices (torch.Tensor): Tensor containing the absolute frame indices
                                          for the sequence. Shape (T_sequence,).
                                          Indices should be non-negative.
            height (int): Height of the spatial grid corresponding to the frames.
                          Must be divisible by patch_size[1].
            width (int): Width of the spatial grid corresponding to the frames.
                         Must be divisible by patch_size[2].

        Returns:
            torch.Tensor: Complex RoPE frequencies. Shape (T_sequence, pph, ppw, D/2).
                          dtype will be complex64.
        """

        device = frame_indices.device  # Get device from input tensor
        p_t, p_h, p_w = self.patch_size

        self.freqs_t = self.freqs_t.to(device)
        self.freqs_h = self.freqs_h.to(device)
        self.freqs_w = self.freqs_w.to(device)

        # Ensure spatial dimensions are divisible by patch size
        if height % p_h != 0 or width % p_w != 0:
            raise ValueError(
                f"Height ({height}) and Width ({width}) must be divisible by patch sizes ({p_h}, {p_w})"
            )

        pph = height // p_h
        ppw = width // p_w
        T_sequence = frame_indices.shape[0]

        if T_sequence == 0:
            # Handle empty input sequence
            return torch.empty(
                (0, pph, ppw, self.feature_dim_half),
                dtype=torch.complex128,
                device=device,
            )

        # Calculate temporal *patch* indices from frame indices
        temporal_patch_indices = torch.div(frame_indices, p_t, rounding_mode="floor")

        # --- Bounds Checking ---
        max_temporal_idx = self.freqs_t.shape[0]
        max_spatial_idx = self.freqs_h.shape[
            0
        ]  # Assuming h and w buffers have same length

        if torch.any(temporal_patch_indices >= max_temporal_idx) or torch.any(
            temporal_patch_indices < 0
        ):
            raise IndexError(
                f"Frame indices map to temporal patch indices ({temporal_patch_indices.min()}..{temporal_patch_indices.max()}) out of pre-calculated bounds [0, {max_temporal_idx-1}]"
            )
        if pph > max_spatial_idx or ppw > max_spatial_idx:
            import warnings

            warnings.warn(
                f"Required spatial patch grid ({pph}x{ppw}) exceeds pre-calculated spatial bounds ({max_spatial_idx}). Adjust max_spatial_len if needed."
            )
            # Clamp spatial dimensions if exceeding bounds to avoid index error
            pph = min(pph, max_spatial_idx)
            ppw = min(ppw, max_spatial_idx)
        # ----------------------

        # Select pre-calculated frequencies based on indices
        # Buffers are already on the correct device
        freqs_t_selected = self.freqs_t[
            temporal_patch_indices
        ]  # (T_sequence, t_complex_dim)

        # Select spatial frequencies (assuming contiguous spatial grid 0..pph-1, 0..ppw-1)
        freqs_h_selected = self.freqs_h[:pph]  # (pph, h_complex_dim)
        freqs_w_selected = self.freqs_w[:ppw]  # (ppw, w_complex_dim)

        # Expand dimensions for broadcasting and concatenation
        # Target shape: (T_sequence, pph, ppw, feature_dim_half)
        freqs_t_expanded = freqs_t_selected.view(
            T_sequence, 1, 1, self.t_complex_dim
        ).expand(T_sequence, pph, ppw, -1)
        freqs_h_expanded = freqs_h_selected.view(1, pph, 1, self.h_complex_dim).expand(
            T_sequence, pph, ppw, -1
        )
        freqs_w_expanded = freqs_w_selected.view(1, 1, ppw, self.w_complex_dim).expand(
            T_sequence, pph, ppw, -1
        )

        # Concatenate along the feature dimension
        # Ensure all parts are on the same device before cat (should be true)
        freqs = torch.cat(
            [freqs_t_expanded, freqs_h_expanded, freqs_w_expanded], dim=-1
        )
        # Final shape: (T_sequence, pph, ppw, D/2)

        return freqs  # dtype complex64


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        style_names: List[str] = ["porn", "anime", "cinema"],
        num_style_tokens: int = 8,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )

        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )

        self.num_style_tokens = num_style_tokens
        self.style_names = style_names

        if num_style_tokens is not None and num_style_tokens > 0:
            self.num_styles = len(style_names)

            self.style_tokens = nn.Parameter(
                torch.randn(self.num_styles, self.num_style_tokens, dim),
                requires_grad=True,
            )
            self.style_gate = nn.Parameter(
                torch.zeros(self.num_styles, self.num_style_tokens, 1),
                requires_grad=True,
            )

        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        model_category: torch.LongTensor | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        batch_size = encoder_hidden_states.shape[0]

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        text_hidden_states = self.text_embedder(encoder_hidden_states)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        if self.num_style_tokens is not None and self.num_style_tokens > 0:
            style_embeds = self.style_tokens[model_category]
            gate = torch.exp(self.style_gate[model_category])
            style_embeds = style_embeds * gate
            style_embeds = style_embeds.view(batch_size, self.num_style_tokens, -1)
        else:
            style_embeds = None

        if style_embeds is not None:
            encoder_hidden_states = torch.cat([style_embeds, text_hidden_states], dim=1)
        else:
            encoder_hidden_states = text_hidden_states

        return (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        )


# --- Modified Compression Function ---
TailHandlingMode = Literal["none", "delete", "compress", "append"]


def apply_framepack_compression(
    hidden_states: torch.Tensor,
    hidden_states_indices: torch.Tensor,
    context: List[Tuple[torch.Tensor, torch.Tensor, int]],
    packer: LatentEmbedPacker,
    rope: WanRotaryPosEmbedIndices,
    tail_compression_factor: int,  # Compression factor for 'none', 'compress', 'append' modes
    tail_handling_mode: TailHandlingMode = "none",  # Default matches previous behavior
    debug: bool = False,
    tail_at_start: bool = True,
) -> Optional[torch.Tensor]:  # Return Optional Tensor, None if all frames deleted
    """
    Applies FramePack compression with a flexible schedule and tail handling.

    Args:
        input_latents: Tensor of shape (B, C, T_in, H, W). Oldest to newest.
        packer: An instance of LatentEmbedPacker holding the Conv3D embedders.
        schedule: List [(comp_factor, num_frames), ...], processed newest backward.
        tail_compression_factor: Compression factor key in packer for tail modes
                                ('none', 'compress', 'append').
        tail_handling_mode: How to handle frames older than schedule covers.
                            'none': Compress remaining block with tail_compression_factor.
                            'delete': Ignore remaining frames.
                            'compress': Global avg pool remaining frames -> compress pooled frame.
                            'append': Avg pool (1,32,32) each remaining frame -> compress pooled frame.

    Returns:
        A single tensor (B, D, SeqLen_packed, 1, 1) or (B, D, SeqLen_packed)
        containing concatenated, flattened, compressed features, ordered oldest to newest.
        Returns None if tail_handling_mode='delete' and the schedule is empty or
        input length is less than schedule coverage. Returns empty tensor if input T_in is 0.
    """

    compressed_latent_chunks_flattened: List[torch.Tensor] = []
    compressed_rope_flattened: List[torch.Tensor] = []
    tail = None

    for latent, frame_indices, embedding_key in context:
        B, C, T, H, W = latent.shape
        if embedding_key == -1 or embedding_key is None or embedding_key == "-1":
            tail = (latent, frame_indices)
            continue
        embedding_key = str(embedding_key)

        rotary_pos_embed = (
            rope(frame_indices=frame_indices, height=H, width=W)
            .unsqueeze(0)
            .permute(0, 4, 1, 2, 3)
        )

        kernel_size, stride = packer.get_embedder_params(embedding_key)

        rope_stride = (
            stride[0] // rope.patch_size[0],
            stride[1] // rope.patch_size[1],
            stride[2] // rope.patch_size[2],
        )

        rope_kernel_size = (
            kernel_size[0] // rope.patch_size[0],
            kernel_size[1] // rope.patch_size[1],
            kernel_size[2] // rope.patch_size[2],
        )

        padded_latent = pad_for_3d_conv(latent, kernel_size, stride)

        if (
            rope_kernel_size[0] == 1
            and rope_kernel_size[1] == 1
            and rope_kernel_size[2] == 1
        ):
            compressed_rotary_pos_embed = rotary_pos_embed.squeeze(0)
        else:
            padded_rotary_pos_embed = pad_for_3d_conv(
                rotary_pos_embed, rope_kernel_size, rope_stride
            )

            padded_rotary_pos_embed_real = convert_wan_complex_to_real_2d(
                padded_rotary_pos_embed
            )
            compressed_rotary_pos_embed_real = center_down_sample_3d(
                padded_rotary_pos_embed_real, rope_kernel_size
            )
            compressed_rotary_pos_embed = convert_real_2d_to_wan_complex(
                compressed_rotary_pos_embed_real
            ).squeeze(0)

        rotary_pos_embed = (
            einops.rearrange(compressed_rotary_pos_embed, "d t h w  -> d (t h w)")
            .unsqueeze(0)
            .unsqueeze(0)
        )
        compressed_latent = packer.embed_fn(padded_latent, embedding_key)
        compressed_latent = einops.rearrange(
            compressed_latent, "b d t h w -> b d (t h w)"
        )

        compressed_latent_chunks_flattened.append(compressed_latent)
        compressed_rope_flattened.append(rotary_pos_embed)

    # --- Handle Tail ---
    if tail is not None:
        tail_embedder_key = str(tail_compression_factor)
        tail_kernel_size, tail_stride = packer.get_embedder_params(tail_embedder_key)
        tail_latent, tail_frame_indices = tail
        H, W = tail_latent.shape[-2:]

        if tail_handling_mode == "none":
            # Compress remaining block with tail_compression_factor embedder
            padded_frames = pad_for_3d_conv(tail_latent, tail_kernel_size, tail_stride)
            compressed_chunk = packer.embed_fn(padded_frames, tail_embedder_key)
            tail_rotary_pos_embed = (
                rope(frame_indices=tail_frame_indices, height=H, width=W)
                .unsqueeze(0)
                .permute(0, 4, 1, 2, 3)
            )

            rope_stride = (
                tail_stride[0] // rope.patch_size[0],
                tail_stride[1] // rope.patch_size[1],
                tail_stride[2] // rope.patch_size[2],
            )

            rope_kernel_size = (
                tail_kernel_size[0] // rope.patch_size[0],
                tail_kernel_size[1] // rope.patch_size[1],
                tail_kernel_size[2] // rope.patch_size[2],
            )

            padded_rotary_pos_embed = pad_for_3d_conv(
                tail_rotary_pos_embed, rope_kernel_size, rope_stride
            )

            padded_rotary_pos_embed = convert_wan_complex_to_real_2d(
                padded_rotary_pos_embed
            )

            compressed_rotary_pos_embed = center_down_sample_3d(
                padded_rotary_pos_embed, rope_kernel_size
            )
            compressed_rotary_pos_embed = convert_real_2d_to_wan_complex(
                compressed_rotary_pos_embed
            ).squeeze(0)
            flattened_chunk = einops.rearrange(
                compressed_chunk, "b d t h w -> b d (t h w)"
            )
            flattened_rotary_pos_embed = (
                einops.rearrange(compressed_rotary_pos_embed, "d t h w -> d (t h w)")
                .unsqueeze(0)
                .unsqueeze(0)
            )

            if tail_at_start:
                compressed_latent_chunks_flattened.insert(0, flattened_chunk)
                compressed_rope_flattened.insert(0, flattened_rotary_pos_embed)
            else:
                compressed_latent_chunks_flattened.append(flattened_chunk)
                compressed_rope_flattened.append(flattened_rotary_pos_embed)

        elif tail_handling_mode == "delete":
            # Do nothing, skip the tail
            pass

        elif tail_handling_mode == "compress":
            # Global average pool across time, then compress the single resulting frame
            # From paper: "the â€œcompressâ€ option tc uses global average pooling for all tail frames
            #              and compresses them with the nearest kernel."

            if (
                tail_latent.shape[2] > 0
            ):  # Check if there are any tail frames to process
                print(
                    f"Compress Tail Mode: Applying Global Average Pool to {tail_latent.shape[2]} frames."
                )
                # 1. Global average pool across the time dimension (dim=2)
                pooled_tail = torch.mean(tail_latent, dim=2, keepdim=True)
                # Shape is now (B, C, 1, H, W)

                # --- Start Inserted Code ---
                # 2. Pad and compress the single pooled latent frame
                # Use tail_kernel_size, tail_stride defined earlier for this tail mode
                padded_pooled_frame = pad_for_3d_conv(
                    pooled_tail, tail_kernel_size, tail_stride
                )
                compressed_latent_chunk = packer.embed_fn(
                    padded_pooled_frame, tail_embedder_key
                )
                # Shape: (B, D_embed, T_c, H_c, W_c) - T_c depends on how tail_kernel/stride handle T=1 input

                pooled_tail_index = tail_frame_indices[0:1]  # Shape: (1,)

                # Calculate RoPE for the pooled frame's spatial dimensions
                tail_rope = rope(
                    frame_indices=pooled_tail_index,
                    height=pooled_tail.shape[3],  # H of pooled frame
                    width=pooled_tail.shape[4],  # W of pooled frame
                )  # Output shape: (T=1, H_p, W_p, D/2)

                # Reshape RoPE for processing (add batch dim, permute)
                tail_rope_proc = tail_rope.unsqueeze(0).permute(
                    0, 4, 1, 2, 3
                )  # (B=1, D/2, T=1, H_p, W_p)

                # 4. Calculate RoPE kernel/stride and downsample RoPE accordingly
                # Use tail_kernel_size, tail_stride
                rope_stride = (
                    max(1, tail_stride[0] // rope.patch_size[0]),
                    max(1, tail_stride[1] // rope.patch_size[1]),
                    max(1, tail_stride[2] // rope.patch_size[2]),
                )
                rope_kernel_size = (
                    max(1, tail_kernel_size[0] // rope.patch_size[0]),
                    max(1, tail_kernel_size[1] // rope.patch_size[1]),
                    max(1, tail_kernel_size[2] // rope.patch_size[2]),
                )

                # Pad and downsample RoPE (using real conversion)
                # Check if downsampling is actually needed (if kernel is effectively 1x1x1)
                if rope_kernel_size == (1, 1, 1):
                    compressed_rope = tail_rope_proc.squeeze(
                        0
                    )  # (D/2, T_c=1, H_pc, W_pc)
                else:
                    # Pass stride to pad_for_3d_conv if it uses it
                    padded_rope = pad_for_3d_conv(
                        tail_rope_proc, rope_kernel_size, rope_stride
                    )
                    padded_rope_real = convert_wan_complex_to_real_2d(padded_rope)
                    # Ensure downsampling handles T=1 input correctly if T kernel > 1
                    downsampled_rope_real = center_down_sample_3d(
                        padded_rope_real, rope_kernel_size
                    )
                    compressed_rope_complex = convert_real_2d_to_wan_complex(
                        downsampled_rope_real
                    )
                    compressed_rope = compressed_rope_complex.squeeze(
                        0
                    )  # (D/2, T_c, H_pc, W_pc)

                # 5. Flatten results
                flattened_latent_chunk = einops.rearrange(
                    compressed_latent_chunk, "b d t h w -> b d (t h w)"
                )
                flattened_rope = einops.rearrange(
                    compressed_rope, "d t h w -> d (t h w)"
                )
                # Reshape RoPE to match expected format (1, 1, D/2, N_chunk)
                flattened_rope = flattened_rope.unsqueeze(0).unsqueeze(0)

                # 6. Insert flattened results at the beginning (tail is oldest)
                compressed_latent_chunks_flattened.insert(0, flattened_latent_chunk)
                compressed_rope_flattened.insert(0, flattened_rope)

                # --- End Inserted Code ---

            else:  # Handle case where frames_remaining was 0 initially
                print("Warning: No tail frames to process for 'compress' mode.")

        elif tail_handling_mode == "append":
            # Pool each tail frame individually (1, 32, 32), compress each, concatenate
            temp_flattened_tail_chunks = []
            temp_flattened_tail_rope_chunks = []
            pool_kernel = (1, 32, 32)
            pool_stride = (1, 32, 32)  # Assume stride matches kernel for pooling

            # Check if pooling is feasible
            can_pool = H >= pool_kernel[1] and W >= pool_kernel[2]
            if not can_pool:
                print(
                    f"Warning: Input H/W ({H},{W}) < Pooling Kernel ({pool_kernel[1]},{pool_kernel[2]}). Cannot pool in 'append' mode. Using tail embedder directly on frames."
                )

            # Use frames_remaining which was calculated earlier for the tail size
            for t_idx in range(tail_latent.shape[2]):
                # Get the single frame latent (B, C, 1, H, W)
                single_tail_frame = tail_latent[:, :, t_idx : t_idx + 1, :, :]

                # --- Start Inserted Code ---

                # 1. Apply spatial pooling if feasible
                if can_pool:
                    try:
                        # Ensure padding is appropriate if H/W not divisible by 32
                        # For simplicity, assume exact division or minimal padding handled by avg_pool3d
                        processed_frame = nn.functional.avg_pool3d(
                            single_tail_frame,
                            kernel_size=pool_kernel,
                            stride=pool_stride,
                        )
                        # Shape might now be (B, C, 1, H_pooled, W_pooled)
                    except Exception as e:
                        print(
                            f"Warning: Pooling frame {t_idx} failed ({e}). Using original frame."
                        )
                        processed_frame = single_tail_frame  # Fallback to original
                else:
                    processed_frame = single_tail_frame  # Use original if cannot pool

                # 2. Pad and compress the (potentially pooled) single latent frame
                # Use tail_kernel_size, tail_stride defined earlier for this tail mode
                padded_single_frame = pad_for_3d_conv(
                    processed_frame, tail_kernel_size, tail_stride
                )
                compressed_single_latent = packer.embed_fn(
                    padded_single_frame, tail_embedder_key
                )
                # Shape: (B, D_embed, T_c, H_c, W_c) - T_c depends on tail_kernel/stride

                original_time_index = tail_frame_indices[
                    t_idx : t_idx + 1
                ]  # Shape: (1,)

                # Calculate RoPE for the potentially spatially pooled frame
                single_rope = rope(
                    frame_indices=original_time_index,
                    height=processed_frame.shape[3]
                    * rope.patch_size[1],  # Use H of (pooled) frame
                    width=processed_frame.shape[4]
                    * rope.patch_size[2],  # Use W of (pooled) frame
                )  # Output shape: (T=1, H_p, W_p, D/2)

                # Reshape RoPE for processing (add batch dim, permute)
                single_rope_proc = single_rope.unsqueeze(0).permute(
                    0, 4, 1, 2, 3
                )  # (B=1, D/2, T=1, H_p, W_p)

                # 4. Calculate RoPE kernel/stride and downsample RoPE
                # Use tail_kernel_size, tail_stride
                rope_stride = (
                    max(1, tail_stride[0] // rope.patch_size[0]),
                    max(1, tail_stride[1] // rope.patch_size[1]),
                    max(1, tail_stride[2] // rope.patch_size[2]),
                )

                rope_kernel_size = (
                    max(1, tail_kernel_size[0] // rope.patch_size[0]),
                    max(1, tail_kernel_size[1] // rope.patch_size[1]),
                    max(1, tail_kernel_size[2] // rope.patch_size[2]),
                )

                # Pad and downsample RoPE (using real conversion)
                if rope_kernel_size == (1, 1, 1):  # Check if downsampling is needed
                    compressed_single_rope = single_rope_proc.squeeze(
                        0
                    )  # (D/2, T_c=1, H_pc, W_pc)
                else:
                    # Pass stride to pad_for_3d_conv if it uses it
                    padded_single_rope = pad_for_3d_conv(
                        single_rope_proc, rope_kernel_size, rope_stride
                    )
                    padded_single_rope_real = convert_wan_complex_to_real_2d(
                        padded_single_rope
                    )
                    downsampled_single_rope_real = center_down_sample_3d(
                        padded_single_rope_real, rope_kernel_size
                    )
                    compressed_single_rope_complex = convert_real_2d_to_wan_complex(
                        downsampled_single_rope_real
                    )
                    compressed_single_rope = compressed_single_rope_complex.squeeze(
                        0
                    )  # (D/2, T_c, H_pc, W_pc)

                # 5. Flatten results
                flattened_single_latent = einops.rearrange(
                    compressed_single_latent, "b d t h w -> b d (t h w)"
                )
                flattened_single_rope = einops.rearrange(
                    compressed_single_rope, "d t h w -> d (t h w)"
                )
                # Reshape RoPE to match expected format (1, 1, D/2, N_chunk)
                flattened_single_rope = flattened_single_rope.unsqueeze(0).unsqueeze(0)

                # 6. Append flattened results to temporary lists
                temp_flattened_tail_chunks.append(flattened_single_latent)
                temp_flattened_tail_rope_chunks.append(flattened_single_rope)

                # --- End Inserted Code ---

            # Concatenate all processed tail frames along the sequence dimension
            if temp_flattened_tail_chunks and temp_flattened_tail_rope_chunks:
                concatenated_tail_chunk = torch.cat(
                    temp_flattened_tail_chunks, dim=2
                )  # Concatenate along sequence dim
                concatenated_tail_rope = torch.cat(
                    temp_flattened_tail_rope_chunks, dim=3
                )  # Concatenate along sequence dim

                # Insert at beginning as tail represents oldest frames
                compressed_latent_chunks_flattened.insert(0, concatenated_tail_chunk)
                compressed_rope_flattened.insert(0, concatenated_tail_rope)
            else:
                print("Warning: No tail frames processed for 'append' mode.")

        else:
            raise ValueError(f"Unknown tail_handling_mode: {tail_handling_mode}")

    # we need to add the hidden_states to the compressed_latent_chunks_flattened
    T_hidden_states = hidden_states.shape[2]
    rope_hidden_states = rope(frame_indices=hidden_states_indices, height=H, width=W)
    hidden_states = packer.embed_fn(hidden_states, str(1))
    rope_hidden_states = (
        einops.rearrange(rope_hidden_states, "t h w d -> d (t h w)")
        .unsqueeze(0)
        .unsqueeze(0)
    )
    hidden_states = einops.rearrange(hidden_states, "b d t h w -> b d (t h w)")
    # add at the end of both
    compressed_latent_chunks_flattened.append(hidden_states)
    compressed_rope_flattened.append(rope_hidden_states)

    if debug:
        for i, chunk in enumerate(compressed_latent_chunks_flattened):
            print(
                f"Latent Chunk {i}: {chunk.shape} Rope: {compressed_rope_flattened[i].shape}"
            )

    final_packed_sequence = torch.cat(
        compressed_latent_chunks_flattened, dim=2
    ).transpose(1, 2)
    final_packed_rope = torch.cat(compressed_rope_flattened, dim=3).transpose(2, 3)
    # Output shape is (B, D, SeqLen_Packed)

    return final_packed_sequence, final_packed_rope


def convert_ffn_to_moe(block: WanTransformerBlock):
    """
    Copies the weights of block.ffn into each expert of block.moe_ffn,
    then deletes block.ffn so only the MoE experts remain.

    Args:
        block: a WanTransformerBlock with `use_moe=True` and a valid .ffn and .moe_ffn.
    """
    if not getattr(block, "use_moe", False):
        raise ValueError("convert_ffn_to_moe called on a block with use_moe=False")

    # 1) Grab the state_dict of the original FeedForward
    ffn_state = block.ffn.state_dict()

    # 2) For each expert in the MoE, load the same weights
    for expert in block.moe_ffn.experts:
        expert.load_state_dict(ffn_state, strict=True)

    # 3) Delete the old .ffn module to free up memory and avoid confusion
    del block.ffn

    # 4) (Optional) enforce that any future .ffn calls go through .moe_ffn
    block.ffn = None


@TRANSFORMERS_REGISTRY("wan.apex_framepack")
class WanApexFramepackTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "condition_embedder",
        "norm",
        "latent_embed_packer",
        "patch_embedding",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        framepack_compressors: Dict[int, str] = {
            1: "k1h2w2",
            2: "k2h4w4",
            4: "k4h8w8",
            8: "k8h16w16",
        },
        framepack_schedule: str = "Schedule_BI_F1K1F2K2F16K4",
        context_decay_steps: int = 1000,
        context_decay_max: float = 0.5,
        context_decay_min: float = 0.10,
        model_categories: List[str] | None = None,
        use_moe: bool = False,
        moe_top_k: int = 1,
        num_style_tokens: int = 8,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = WanRotaryPosEmbedIndices(
            attention_head_dim, patch_size, rope_max_seq_len
        )

        self.model_categories = model_categories

        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )  # we will map this to the packer

        self.latent_embed_packer = LatentEmbedPacker(
            inner_dim=inner_dim,
            use_gradient_checkpointing=True,
            compressors=framepack_compressors,
        )

        self.framepack_schedule: BaseSchedule = getattr(
            src.transformer.wan.apex_framepack.module, framepack_schedule
        )(context_decay_steps, context_decay_max, context_decay_min)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            style_names=model_categories,
            num_style_tokens=num_style_tokens,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    use_moe,
                    len(self.model_categories),
                    moe_top_k,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = True

    def get_latent_context(
        self,
        past_latents: Optional[torch.Tensor],
        past_indices: torch.Tensor,
        future_latents: Optional[torch.Tensor],
        future_indices: torch.Tensor,
        total_frames: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:

        return self.framepack_schedule(
            past_latents, past_indices, future_latents, future_indices, total_frames
        )

    def get_model_category(self, model_category: str | List[str]) -> torch.LongTensor:
        if isinstance(model_category, str):
            cats = [
                torch.tensor(
                    [self.model_categories.index(model_category)], dtype=torch.long
                )
            ]
        else:
            cats = [
                torch.tensor([self.model_categories.index(c)], dtype=torch.long)
                for c in model_category
            ]
        return torch.stack(cats, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        model_category: torch.LongTensor | None = None,
        latent_context: List[Tuple[torch.Tensor, torch.Tensor, int]] | None = None,
        indices: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        total_frames = num_frames

        original_context_length = (
            post_patch_num_frames * post_patch_height * post_patch_width
        )

        if indices is None:
            indices = torch.arange(num_frames).to(hidden_states.device)

        if latent_context is None or len(latent_context) == 0:
            rotary_emb = (
                einops.rearrange(
                    self.rope(frame_indices=indices, height=height, width=width),
                    "t h w d -> d (t h w)",
                )
                .transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            hidden_states = self.latent_embed_packer.embed_fn(hidden_states, str(1))
            hidden_states = einops.rearrange(hidden_states, "b d t h w -> b (t h w) d")
        else:

            hidden_states, rotary_emb = apply_framepack_compression(
                hidden_states,
                indices,
                latent_context,
                self.latent_embed_packer,
                self.rope,
                self.framepack_schedule.tail_factor,
                self.framepack_schedule.tail_handling_mode,
                tail_at_start=self.framepack_schedule.tail_at_start,
            )

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, model_category
        )

        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
