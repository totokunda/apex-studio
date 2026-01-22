from torch import nn
import torch

class InplaceRMSNorm(nn.Module):
    """
    RMSNorm variant that can operate in-place on non-leaf tensors.

    This is used for Q/K normalization inside attention where the inputs are fresh linear projections,
    so in-place mutation is safe and saves memory.
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability, then apply to x in-place.
        # NOTE: In-place ops require `x` to be non-leaf (true for Q/K projections).
        y = x.float()
        y.pow_(2)
        y = y.mean(dim=-1, keepdim=True)
        y.add_(self.eps)
        y.rsqrt_()
        x.mul_(y.to(dtype=x.dtype))
        if self.weight is not None:
            x.mul_(self.weight.to(dtype=x.dtype, device=x.device))
        return x