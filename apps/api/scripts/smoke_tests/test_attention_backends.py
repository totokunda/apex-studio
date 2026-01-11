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
        gold = attention_register.get("sdpa")
        device = torch.device("cuda" if cuda_ok else "cpu")
        if device.type != "cuda" and key in cuda_only:
            log(f"[smoke] attention skip (cuda-only): {key}")
            return

        B, H, S, D = 1, 2, 16, 64
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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

        # Gold baseline: SDPA.
        try:
            gold_out = gold(q, k, v, **kwargs)
        except TypeError:
            sig = inspect.signature(gold)
            filtered = {kk: vv for kk, vv in kwargs.items() if kk in sig.parameters}
            gold_out = gold(q, k, v, **filtered)

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

        # Correctness check: compare to SDPA. Skip known mismatches / special APIs.
        if key not in {"flash_varlen"} and hasattr(out, "shape") and hasattr(gold_out, "shape"):
            if tuple(out.shape) != tuple(gold_out.shape):
                fail(
                    f"attention backend output shape mismatch vs sdpa for {key}: "
                    f"out={tuple(out.shape)} gold={tuple(gold_out.shape)}"
                )

            # Compare in fp32 for stability (many backends run in bf16/fp16).
            out_f = out.detach().float()
            gold_f = gold_out.detach().float()

            # SageAttention is intentionally quantized/approximate; max-abs diff can be
            # large on random inputs even when the result is directionally very close.
            # Use a similarity metric instead of strict elementwise tolerances.
            if key == "sage":
                o = out_f.flatten()
                g = gold_f.flatten()
                cos = torch.nn.functional.cosine_similarity(o, g, dim=0).item()
                rmse = (o - g).pow(2).mean().sqrt().item()
                gold_rms = g.pow(2).mean().sqrt().item()
                rel_rmse = rmse / (gold_rms + 1e-12)
                if cos < 0.95 or rel_rmse > 1.0:
                    fail(
                        f"attention backend differs vs sdpa for {key}: "
                        f"cos={cos:.6g} rel_rmse={rel_rmse:.6g}"
                    )
                # Skip strict max-abs checks for sage.
                if device.type == "cuda":
                    torch.cuda.synchronize()
                log(f"[smoke] attention ok: {key}")
                return

            # Loose tolerances on GPU low-precision; tighter on CPU fp32.
            if device.type == "cuda":
                atol, rtol = 2e-2, 2e-2
            else:
                atol, rtol = 1e-4, 1e-4

            max_abs = (out_f - gold_f).abs().max().item()
            denom = gold_f.abs().max().item()
            rel = max_abs / (denom + 1e-12)
            if not (max_abs <= atol or rel <= rtol):
                fail(
                    f"attention backend differs vs sdpa for {key}: "
                    f"max_abs={max_abs:.6g} rel={rel:.6g} (atol={atol} rtol={rtol})"
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(f"[smoke] attention ok: {key}")

    for k in available:
        _tiny_call(k)


