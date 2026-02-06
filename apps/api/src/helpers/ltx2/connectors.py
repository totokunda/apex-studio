from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from ltx_core.model.transformer.feed_forward import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from src.helpers.helpers import helpers as helper_registry
from src.transformer.ltx2.base2.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)

from src.transformer.ltx2.base2.attention import Attention


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)



class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
        )
        

        self.ff = FeedForward(
            dim,
            dim_out=dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        norm_hidden_states_list = [norm_hidden_states]
        del norm_hidden_states
        attn_output = self.attn1(norm_hidden_states_list, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """
    Embeddings1DConnector applies a 1D transformer-based processing to sequential embeddings (e.g., for video, audio, or
    other modalities). It supports rotary positional encoding (rope), optional causal temporal positioning, and can
    substitute padded positions with learnable registers. The module is highly configurable for head size, number of
    layers, and register usage.
    Args:
        attention_head_dim (int): Dimension of each attention head (default=128).
        num_attention_heads (int): Number of attention heads (default=30).
        num_layers (int): Number of transformer layers (default=2).
        positional_embedding_theta (float): Scaling factor for position embedding (default=10000.0).
        positional_embedding_max_pos (list[int] | None): Max positions for positional embeddings (default=[1]).
        causal_temporal_positioning (bool): If True, uses causal attention (default=False).
        num_learnable_registers (int | None): Number of learnable registers to replace padded tokens. If None, disables
            register replacement. (default=128)
        rope_type (LTXRopeType): The RoPE variant to use (default=DEFAULT_ROPE_TYPE).
        double_precision_rope (bool): Use double precision rope calculation (default=False).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[1] % self.num_learnable_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}."
        )

        num_registers_duplications = hidden_states.shape[1] // self.num_learnable_registers
        learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))
        attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()

        non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
        non_zero_nums = non_zero_hidden_states.shape[1]
        pad_length = hidden_states.shape[1] - non_zero_nums
        adjusted_hidden_states = torch.nn.functional.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        hidden_states = flipped_mask * adjusted_hidden_states + (1 - flipped_mask) * learnable_registers

        attention_mask = torch.full_like(
            attention_mask,
            0.0,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Embeddings1DConnector.
        Args:
            hidden_states (torch.Tensor): Input tensor of embeddings (shape [batch, seq_len, feature_dim]).
            attention_mask (torch.Tensor|None): Optional mask for valid tokens (shape compatible with hidden_states).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed features and the corresponding (possibly modified) mask.
        """
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :]
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask
    
    
    
@helper_registry("ltx2.connectors")
class LTX2TextConnectors(ModelMixin, ConfigMixin):
    """
    Text connector stack used by LTX 2.0 to process the packed text encoder hidden states for both the video and audio
    streams.
    """

    @register_to_config
    def __init__(
        self,
        caption_channels: int,
        text_proj_in_factor: int,
        video_connector_num_attention_heads: int,
        video_connector_attention_head_dim: int,
        video_connector_num_layers: int,
        video_connector_num_learnable_registers: int | None,
        audio_connector_num_attention_heads: int,
        audio_connector_attention_head_dim: int,
        audio_connector_num_layers: int,
        audio_connector_num_learnable_registers: int | None,
        connector_rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str = "interleaved",
    ):
        super().__init__()

        self.text_proj_in = nn.Linear(
            caption_channels * text_proj_in_factor, caption_channels, bias=False
        )
        
        if rope_type == "interleaved":
            rope_type = LTXRopeType.INTERLEAVED
        elif rope_type == "split":
            rope_type = LTXRopeType.SPLIT
        else:
            raise ValueError(f"Invalid rope type: {rope_type}")
        
        self.video_connector = Embeddings1DConnector(
            num_attention_heads=video_connector_num_attention_heads,
            attention_head_dim=video_connector_attention_head_dim,
            num_layers=video_connector_num_layers,
            num_learnable_registers=video_connector_num_learnable_registers,
            positional_embedding_max_pos=[connector_rope_base_seq_len],
            positional_embedding_theta=rope_theta,
            double_precision_rope=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )
        self.audio_connector = Embeddings1DConnector(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=audio_connector_num_layers,
            num_learnable_registers=audio_connector_num_learnable_registers,
            positional_embedding_max_pos=[connector_rope_base_seq_len],
            positional_embedding_theta=rope_theta,
            double_precision_rope=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        additive_mask: bool = False,
    ):
        # Convert to additive attention mask, if necessary
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            attention_mask = (attention_mask - 1).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            )
            attention_mask = attention_mask.to(text_dtype) * torch.finfo(text_dtype).max

        text_encoder_hidden_states = self.text_proj_in(text_encoder_hidden_states)
        

        video_text_embedding, new_attn_mask = self.video_connector(
            text_encoder_hidden_states, attention_mask
        )


        attn_mask = (new_attn_mask < 1e-6).to(torch.int64)
        attn_mask = attn_mask.reshape(
            video_text_embedding.shape[0], video_text_embedding.shape[1], 1
        )
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)

        audio_text_embedding, _ = self.audio_connector(
            text_encoder_hidden_states, attention_mask
        )

        return video_text_embedding, audio_text_embedding, new_attn_mask