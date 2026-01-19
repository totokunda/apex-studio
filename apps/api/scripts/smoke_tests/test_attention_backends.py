from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import SmokeContext, log, fail


def run(ctx: SmokeContext) -> None:
    """
    Run each attention backend test in an isolated process.

    Some backends (or mismatched CUDA runtimes) can crash the CUDA runtime or even
    segfault the interpreter; keeping each backend in its own subprocess prevents
    a single failure from poisoning the rest of the smoke tests.
    """
    log("[smoke] attention backends (isolated subprocesses)")

    available = _list_available_backends_subprocess(ctx)
    log(f"[smoke] attention available: {available}")

    failures: list[str] = []
    for key in available:
        ok, skipped, detail = _run_one_backend_subprocess(ctx, key)
        if ok and skipped:
            log(f"[smoke] attention skip: {key}")
        elif ok:
            log(f"[smoke] attention ok: {key}")
        else:
            failures.append(f"{key}: {detail}")

    if failures:
        fail(
            "One or more attention backends failed/crashed:\n"
            + "\n".join(f"- {f}" for f in failures)
        )


def _scripts_parent_dir() -> Path:
    # `.../scripts/smoke_tests/test_attention_backends.py` -> `.../scripts`
    return Path(__file__).resolve().parents[1]


def _subprocess_env(bundle_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    # Ensure `python -m smoke_tests.*` can resolve the package in both repo + bundle.
    scripts_dir = _scripts_parent_dir()
    prior = env.get("PYTHONPATH") or ""
    env["PYTHONPATH"] = (
        str(scripts_dir) if not prior else (str(scripts_dir) + os.pathsep + prior)
    )
    # Also ensure `import src...` resolves against the bundle root.
    env["APEX_BUNDLE_ROOT"] = str(bundle_root)
    # If the bundle ships a `kernels` cache, prefer it automatically so MPS backends
    # can run offline and without depending on user-level HF caches.
    try:
        cache_dir = Path(bundle_root).resolve() / "kernels-cache"
        if cache_dir.exists() and cache_dir.is_dir() and not env.get("KERNELS_CACHE"):
            env["KERNELS_CACHE"] = str(cache_dir)
    except Exception:
        pass
    return env


def _signal_name(code: int) -> str:
    if code >= 0:
        return ""
    sig = -code
    try:
        return signal.Signals(sig).name
    except Exception:
        return f"SIG{sig}"


def _list_available_backends_subprocess(ctx: SmokeContext) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "smoke_tests.test_attention_backends",
        "--bundle-root",
        str(ctx.bundle_root),
        "--gpu-type",
        str(ctx.gpu_type or ""),
    ]
    if ctx.strict_gpu:
        cmd.append("--strict-gpu")
    cmd.append("--list")

    try:
        p = subprocess.run(
            cmd,
            env=_subprocess_env(ctx.bundle_root),
            capture_output=True,
            text=True,
            # Listing uses `verify_attention_backends()` which may need to spin up
            # multiple isolated workers (and may intentionally wait out timeouts for
            # broken backends). Keep this comfortably above that worst-case.
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        fail("Timed out (180s) while enumerating attention backends in subprocess.")
    if p.returncode != 0:
        sig = _signal_name(p.returncode)
        extra = f" ({sig})" if sig else ""
        fail(
            "Failed to enumerate attention backends in subprocess"
            + extra
            + ":\n"
            + (p.stdout or "")
            + (p.stderr or "")
        )

    try:
        # Be robust to noisy imports; parse the last JSON-looking line.
        lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
        json_line = next((ln for ln in reversed(lines) if ln.startswith("{")), "")
        data = json.loads(json_line or "{}")
        available = sorted([str(x) for x in data.get("available", [])])
    except Exception as e:
        fail(
            f"Failed to parse attention backend list JSON: {e}\n"
            f"stdout={p.stdout!r}\nstderr={p.stderr!r}"
        )

    if not available:
        fail("Attention registry reported no available backends.")
    return available


def _run_one_backend_subprocess(
    ctx: SmokeContext, key: str
) -> tuple[bool, bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "smoke_tests.test_attention_backends",
        "--bundle-root",
        str(ctx.bundle_root),
        "--gpu-type",
        str(ctx.gpu_type or ""),
        "--backend",
        key,
    ]
    if ctx.strict_gpu:
        cmd.append("--strict-gpu")

    try:
        p = subprocess.run(
            cmd,
            env=_subprocess_env(ctx.bundle_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, False, "timed out after 120s"

    out_s = (p.stdout or "").strip()
    err_s = (p.stderr or "").strip()
    skipped = "[smoke] attention skip" in out_s

    if p.returncode == 0:
        return True, skipped, out_s or err_s

    sig = _signal_name(p.returncode)
    if sig:
        detail = f"crashed with {sig}"
    else:
        detail = f"exit={p.returncode}"

    if out_s or err_s:
        detail += "\n" + "\n".join(
            x
            for x in [
                out_s and ("stdout:\n" + out_s),
                err_s and ("stderr:\n" + err_s),
            ]
            if x
        )
    return False, False, detail


def _run_one_backend_inprocess(ctx: SmokeContext, key: str) -> None:
    try:
        import torch  # type: ignore
    except Exception as e:
        fail(f"Failed to import torch: {e}")

    try:
        from src.attention.functions import attention_register  # type: ignore
    except Exception as e:
        fail(f"Failed to import attention registry: {e}")

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_ok = bool(
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

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
    mps_only = {"metal_flash", "metal_flash_varlen"}

    fn = attention_register.get(key)
    if fn is None:
        fail(f"attention backend not found in registry: {key}")

    device = torch.device("cuda" if cuda_ok else ("mps" if mps_ok else "cpu"))
    if device.type != "cuda" and key in cuda_only:
        log(f"[smoke] attention skip (cuda-only): {key}")
        return
    if device.type != "mps" and key in mps_only:
        log(f"[smoke] attention skip (mps-only): {key}")
        return

    B, H, S, D = 1, 2, 16, 64
    if device.type == "cuda":
        dtype = torch.bfloat16
    elif device.type == "mps":
        # Metal Flash-SDPA expects fp16 inputs; keep baseline in fp16 too.
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Build inputs.
    # Most backends accept padded inputs (B, H, S, D). `metal_flash_varlen` expects packed (T, H, D).
    if key == "metal_flash_varlen":
        # 2 sequences packed back-to-back.
        Bp = 2
        lens = [S // 2, S // 2]  # total T = S
        T = sum(lens)
        q = torch.randn((T, H, D), device=device, dtype=dtype)
        k = torch.randn((T, H, D), device=device, dtype=dtype)
        v = torch.randn((T, H, D), device=device, dtype=dtype)
        cu = torch.tensor([0, lens[0], T], device=device, dtype=torch.int32)
        cu_seqlens_q = cu
        cu_seqlens_k = cu
        max_seqlen_q = max(lens)
        max_seqlen_k = max(lens)
        gold = attention_register.get("sdpa_varlen") or attention_register.get("sdpa")
    else:
        q = torch.randn((B, H, S, D), device=device, dtype=dtype)
        k = torch.randn((B, H, S, D), device=device, dtype=dtype)
        v = torch.randn((B, H, S, D), device=device, dtype=dtype)
        cu_seqlens_q = None
        cu_seqlens_k = None
        max_seqlen_q = None
        max_seqlen_k = None
        gold = attention_register.get("sdpa")
    if gold is None:
        fail("attention baseline backend not found in registry (sdpa/sdpa_varlen)")

    kwargs: dict[str, Any] = {
        "attn_mask": None,
        "attention_mask": None,
        "dropout_p": 0.0,
        "is_causal": False,
        "softmax_scale": None,
        "default_dtype": torch.float16,
    }

    # Gold baseline: SDPA.
    try:
        gold_out = gold(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **kwargs,
        )
    except TypeError:
        sig = inspect.signature(gold)
        merged = dict(kwargs)
        merged.update(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
            }
        )
        filtered = {kk: vv for kk, vv in merged.items() if kk in sig.parameters}
        gold_out = gold(q, k, v, **filtered)

    try:
        out = fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **kwargs,
        )
    except TypeError:
        sig = inspect.signature(fn)
        merged = dict(kwargs)
        merged.update(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
            }
        )
        filtered = {kk: vv for kk, vv in merged.items() if kk in sig.parameters}
        out = fn(q, k, v, **filtered)

    if out is None:
        fail(f"attention backend returned None: {key}")
    if hasattr(out, "shape"):
        if out.shape[-1] != D:
            fail(f"attention backend bad output shape for {key}: {tuple(out.shape)}")

    # Correctness check: compare to SDPA. Skip known mismatches / special APIs.
    if (
        True
        and hasattr(out, "shape")
        and hasattr(gold_out, "shape")
    ):
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
            return

        # Loose tolerances on low-precision GPU; tighter on CPU fp32.
        if device.type == "cuda":
            atol, rtol = 2e-2, 2e-2
        elif device.type == "mps":
            atol, rtol = 5e-2, 5e-2
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
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _run_list_inprocess(ctx: SmokeContext) -> list[str]:
    try:
        from src.attention.functions import (  # type: ignore
            attention_register,
            verify_attention_backends,
        )
    except Exception as e:
        fail(f"Failed to import attention registry: {e}")

    # IMPORTANT:
    # `attention_register.all_available()` reflects *import-time* availability checks
    # (e.g. "module is importable"), which can be overly optimistic: some attention
    # backends can still crash the runtime or fail at first use depending on the
    # local CUDA/toolkit environment.
    #
    # `verify_attention_backends()` is our source of truth; it runs each backend in
    # an isolated process once, caches the working set, and marks failing backends
    # unavailable. The smoke test should only attempt backends from that verified set.
    candidates = sorted(attention_register.all_available().keys())
    working = verify_attention_backends(force_refresh=True)
    working_set = set(working)
    available = sorted([k for k in candidates if k in working_set])
    return available


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run attention backend smoke tests.")
    p.add_argument("--bundle-root", default=None)
    p.add_argument("--gpu-type", default="")
    p.add_argument("--strict-gpu", action="store_true")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="Print available backends as JSON.")
    g.add_argument("--backend", default=None, help="Run the smoke test for one backend key.")
    args = p.parse_args(argv)

    # Ensure `import src...` works in subprocess mode too.
    from .common import resolve_bundle_root, ensure_bundle_on_syspath

    bundle_root = resolve_bundle_root(args.bundle_root)
    ensure_bundle_on_syspath(bundle_root)
    ctx = SmokeContext(
        bundle_root=bundle_root,
        gpu_type=str(args.gpu_type or ""),
        strict_gpu=bool(args.strict_gpu),
    )

    try:
        if args.list:
            available = _run_list_inprocess(ctx)
            # Keep `available` as the stable contract for the parent process.
            print(json.dumps({"available": available}), flush=True)
            return 0

        assert args.backend is not None
        _run_one_backend_inprocess(ctx, args.backend)
        return 0
    except Exception as e:
        # Make subprocess failures concise for the parent process.
        print(f"[smoke] attention backend failed: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
