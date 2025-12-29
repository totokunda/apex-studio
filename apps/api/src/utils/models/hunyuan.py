import torch
from typing import Union, List, Tuple

from itertools import repeat
import collections.abc


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [D/2]
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def get_rope_freq_from_size(
    latents_size,
    ndim,
    target_ndim,
    args,
    rope_theta_rescale_factor: Union[float, List[float]] = 1.0,
    rope_interpolation_factor: Union[float, List[float]] = 1.0,
    concat_dict={},
):

    if isinstance(args.patch_size, int):
        assert all(s % args.patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({args.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // args.patch_size for s in latents_size]
    elif isinstance(args.patch_size, list):
        assert all(
            s % args.patch_size[idx] == 0 for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({args.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // args.patch_size[idx] for idx, s in enumerate(latents_size)]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = args.hidden_size // args.num_heads
    rope_dim_list = args.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(
        rope_dim_list,
        rope_sizes,
        theta=args.rope_theta,
        use_real=True,
        theta_rescale_factor=rope_theta_rescale_factor,
        interpolation_factor=rope_interpolation_factor,
        concat_dict=concat_dict,
    )
    return freqs_cos, freqs_sin


def get_nd_rotary_pos_embed_new(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
    concat_dict={},
):

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]
    if len(concat_dict) < 1:
        pass
    else:
        if concat_dict["mode"] == "timecat":
            bias = grid[:, :1].clone()
            bias[0] = concat_dict["bias"] * torch.ones_like(bias[0])
            grid = torch.cat([bias, grid], dim=1)

        elif concat_dict["mode"] == "timecat-w":
            bias = grid[:, :1].clone()
            bias[0] = concat_dict["bias"] * torch.ones_like(bias[0])
            bias[2] += start[
                -1
            ]  ## ref https://github.com/Yuanshi9815/OminiControl/blob/main/src/generate.py#L178
            grid = torch.cat([bias, grid], dim=1)
    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]

        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def get_rotary_pos_embed(
    video_length,
    height,
    width,
    patch_size,
    hidden_size,
    num_heads,
    rope_dim_list,
    concat_dict={},
    vae_scale_factor_temporal=4,
    vae_scale_factor_spatial=8,
    theta=10000.0,
):
    target_ndim = 3
    ndim = 5 - 2
    latents_size = [
        (video_length - 1) // vae_scale_factor_temporal + 1,
        height // vae_scale_factor_spatial,
        width // vae_scale_factor_spatial,
    ]

    if isinstance(patch_size, int):
        assert all(s % patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // patch_size for s in latents_size]
    elif isinstance(patch_size, list):
        assert all(s % patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // patch_size[idx] for idx, s in enumerate(latents_size)]
    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = hidden_size // num_heads
    rope_dim_list = rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(
        rope_dim_list,
        rope_sizes,
        theta=theta,
        use_real=True,
        theta_rescale_factor=1,
        concat_dict=concat_dict,
    )
    return freqs_cos, freqs_sin
