import math
import mlx.core as mx
from mlx import nn
from typing import Optional
from src.mlx.modules.act import get_activation


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """
    Creates sinusoidal timestep embeddings, matching the implementation in
    Denoising Diffusion Probabilistic Models.

    Args:
        timesteps (mx.array):
            A 1-D array of N indices, one per batch element.
        embedding_dim (int):
            The dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True)
            or `sin, cos` (if False).
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions.
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings.

    Returns:
        mx.array: An [N x dim] array of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(
        start=0, stop=half_dim, dtype=mx.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        # mx.pad expects a list of tuples for padding each axis
        # ((before_0, after_0), (before_1, after_1), ...)
        # We pad the last dimension by 1 on the right.
        emb = mx.pad(emb, ((0, 0), (0, 1)))

    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def __call__(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
