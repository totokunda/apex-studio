from __future__ import annotations

import inspect

from .common import SmokeContext, log, fail


def run(ctx: SmokeContext) -> None:
    log("[smoke] attention backends")
    try:
        import torch  # type: ignore
    except Exception as e:
        fail(f"Failed to import torch: {e}")

    try:
        from src.attention.functions import attention_register  # type: ignore
    except Exception as e:
        fail(f"Failed to import attention registry: {e}")

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    available = sorted(attention_register.all_available().keys())
    log(f"[smoke] attention available: {available}")

    cuda_only = {
        "flash",
        "flash3",
        "flash_padded",
        "flash_varlen",
        "sage",
        "xformers",
        "flex",
        "flex-block-attn",
    }

    def _tiny_call(key: str) -> None:
        fn = attention_register.get(key)
        device = torch.device("cuda" if cuda_ok else "cpu")
        if device.type != "cuda" and key in cuda_only:
            log(f"[smoke] attention skip (cuda-only): {key}")
            return

        B, H, S, D = 1, 2, 16, 32
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        q = torch.randn((B, H, S, D), device=device, dtype=dtype)
        k = torch.randn((B, H, S, D), device=device, dtype=dtype)
        v = torch.randn((B, H, S, D), device=device, dtype=dtype)

        kwargs = {
            "attn_mask": None,
            "attention_mask": None,
            "dropout_p": 0.0,
            "is_causal": False,
            "softmax_scale": None,
            "default_dtype": torch.float16,
        }

        try:
            out = fn(q, k, v, **kwargs)
        except TypeError:
            sig = inspect.signature(fn)
            filtered = {kk: vv for kk, vv in kwargs.items() if kk in sig.parameters}
            out = fn(q, k, v, **filtered)

        if out is None:
            fail(f"attention backend returned None: {key}")
        if key not in {"flash_varlen"} and hasattr(out, "shape"):
            if out.shape[-1] != D:
                fail(f"attention backend bad output shape for {key}: {tuple(out.shape)}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(f"[smoke] attention ok: {key}")

    for k in available:
        _tiny_call(k)


