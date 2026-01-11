from __future__ import annotations

import sys

from .common import SmokeContext, log, fail, try_import


def run(ctx: SmokeContext) -> None:
    log("[smoke] native imports")

    # Modules that commonly fail on user machines due to missing shared libs.
    modules = [
        "torch",
        "numpy",
        "PIL",
        "cv2",
        "onnxruntime",
        "torchvision",
        "torchaudio",
        "apex_download_rs",
        # optional perf backends:
        "xformers.ops",
        "flash_attn",
        "flash_attn_interface",
        "sageattention",
    ]

    optional = {"xformers.ops", "flash_attn", "flash_attn_interface", "sageattention"}

    for name in modules:
        ok, _m, err = try_import(name)
        if ok:
            log(f"[smoke] import ok: {name}")
            continue
        if name in optional:
            log(f"[smoke] import skipped/optional failed: {name} ({err})")
            continue
        fail(f"Failed to import {name}: {err}")

    # Print basic torch/cuda info for debugging
    try:
        import torch  # type: ignore

        log(f"[smoke] python={sys.executable}")
        log(f"[smoke] torch={getattr(torch, '__version__', '?')}")
        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        log(f"[smoke] cuda_available={cuda_ok}")
        if cuda_ok:
            try:
                log(f"[smoke] cuda_device={torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except Exception:
        # torch import failures are already caught above
        pass


