from __future__ import annotations

import os

import torch

from src.utils.defaults import get_torch_device


def resolve_mask_device(*, model_family: str = "sam2") -> torch.device:
    """
    Resolve the torch device to use for mask models.

    Why this exists:
    - Some model stacks (notably SAM2) can hard-crash (SIGABRT) on Apple MPS due to
      MPSGraph/Metal issues inside PyTorch ops like scaled_dot_product_attention.
    - A Python try/except cannot catch a SIGABRT; we must avoid that execution path.

    Overrides:
    - MASK_DEVICE / APEX_MASK_DEVICE: explicit device string ("cpu", "mps", "cuda", "cuda:0", ...)
    - MASK_ALLOW_MPS=1: allow using MPS even for model families that default to CPU on MPS.
    """

    env_device = os.getenv("MASK_DEVICE") or os.getenv("APEX_MASK_DEVICE")
    if env_device:
        return torch.device(env_device)

    device = get_torch_device()

    # Default: avoid MPS for SAM2 unless explicitly allowed.
    if model_family.lower() in {"sam2"} and device.type == "mps":
        allow_mps = os.getenv("MASK_ALLOW_MPS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not allow_mps:
            return torch.device("cpu")

    return device
