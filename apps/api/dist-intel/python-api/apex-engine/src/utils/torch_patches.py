from __future__ import annotations

from typing import Callable

import torch
from loguru import logger


def patch_torch_linalg_solve_for_cusolver() -> None:
    """
    Patch `torch.linalg.solve` to gracefully fallback to a CPU float32 solve when cuSOLVER
    fails to initialize (CUSOLVER_STATUS_INTERNAL_ERROR on cusolverDnCreate).

    This avoids hard-crashing generation in rare/fragile CUDA environments while keeping the
    patch localized to linalg.solve only.
    """

    solve: Callable = torch.linalg.solve
    if getattr(solve, "_apex_cusolver_safe_patch", False):
        return

    def _safe_solve(A: torch.Tensor, B: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        try:
            return solve(A, B, *args, **kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cusolver" not in msg and "cusolver_status_internal_error" not in msg:
                raise

            # Preserve expected output dtype as best as possible.
            out_dtype = torch.result_type(A, B)
            out_device = B.device if isinstance(B, torch.Tensor) else A.device

            A_cpu = A.detach().to(dtype=torch.float32, device="cpu")
            B_cpu = B.detach().to(dtype=torch.float32, device="cpu")
            logger.warning(
                "cuSOLVER failed during torch.linalg.solve; falling back to CPU solve. "
                "If this happens frequently, fix the CUDA/driver/PyTorch stack alignment."
            )
            sol_cpu = torch.linalg.solve(A_cpu, B_cpu, *args, **kwargs)
            return sol_cpu.to(device=out_device, dtype=out_dtype)

    _safe_solve._apex_cusolver_safe_patch = True  # type: ignore[attr-defined]
    torch.linalg.solve = _safe_solve  # type: ignore[assignment]
