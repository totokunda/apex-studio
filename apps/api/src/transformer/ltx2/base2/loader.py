import torch
try:
    import triton
except ImportError:
    triton = None

from src.transformer.ltx2.base2.kernel import fused_add_round_kernel

BLOCK_SIZE = 1024


def fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


def calculate_weight_float8_(target_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
    result = fused_add_round_launch(target_weights, original_weights, seed=0).to(target_weights.dtype)
    target_weights.copy_(result, non_blocking=True)
    return target_weights


