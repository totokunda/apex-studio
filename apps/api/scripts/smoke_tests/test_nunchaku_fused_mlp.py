from __future__ import annotations

from .common import SmokeContext, log, fail


def run(ctx: SmokeContext) -> None:
    log("[smoke] nunchaku fused_gelu_mlp")
    try:
        import torch  # type: ignore
    except Exception as e:
        fail(f"Failed to import torch: {e}")

    try:
        import nunchaku  # noqa: F401
    except ImportError:
        log("[smoke] nunchaku not installed; skipping")
        return

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if not cuda_ok:
        log("[smoke] cuda not available; skipping nunchaku fused_gelu_mlp")
        return

    try:
        from nunchaku.ops.fused import fused_gelu_mlp
        from nunchaku.models.linear import SVDQW4A4Linear
        from nunchaku.utils import get_precision
    except Exception as e:
        fail(f"Failed to import nunchaku fused op deps: {e}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # For int4, kernels have additional alignment constraints beyond group_size=64.
    # Use conservative dimensions (128) to match common kernel tiling/padding expectations.
    C_in = 128
    C_hidden = 128
    C_out = 128
    rank = 16
    precision = get_precision()
    if precision == "fp4":
        precision = "nvfp4"

    fc1 = SVDQW4A4Linear(
        in_features=C_in,
        out_features=C_hidden,
        rank=rank,
        bias=True,
        precision=precision,
        torch_dtype=dtype,
        device=device,
    )

    fc2 = SVDQW4A4Linear(
        in_features=C_hidden,
        out_features=C_out,
        rank=rank,
        bias=True,
        precision=precision,
        torch_dtype=dtype,
        device=device,
    )

    # Initialize to safe values (avoid uninitialized memory feeding kernels).
    with torch.no_grad():
        fc1.qweight.zero_()
        fc2.qweight.zero_()
        if fc1.bias is not None:
            fc1.bias.zero_()
        if fc2.bias is not None:
            fc2.bias.zero_()
        fc1.wscales.fill_(1)
        fc2.wscales.fill_(1)
        fc1.smooth_factor.fill_(1)
        fc2.smooth_factor.fill_(1)
        fc1.proj_down.zero_()
        fc1.proj_up.zero_()
        fc2.proj_down.zero_()
        fc2.proj_up.zero_()

    x = torch.randn((1, 1, C_in), device=device, dtype=dtype)
    y = fused_gelu_mlp(x, fc1, fc2)
    torch.cuda.synchronize()
    if not hasattr(y, "shape") or tuple(y.shape) != (1, 1, C_out):
        fail(
            f"nunchaku fused_gelu_mlp returned unexpected shape: {getattr(y, 'shape', None)}"
        )
    log("[smoke] nunchaku fused_gelu_mlp ok")
