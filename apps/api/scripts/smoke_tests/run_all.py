#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE:
# This file is executed as a script from `scripts/smoke_tests/run_all.py`.
# When you run a file directly, Python sets `sys.path[0]` to the directory
# containing that file (i.e. `.../scripts/smoke_tests`). In that context,
# importing `smoke_tests.common` fails because the *parent* directory isn't on
# sys.path. Add it so `smoke_tests.*` imports are stable in both repo + bundle.
_smoke_dir = Path(__file__).resolve().parent
_parent = _smoke_dir.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from smoke_tests.common import (
    SmokeContext,
    log,
    fail,
    resolve_bundle_root,
    ensure_bundle_on_syspath,
)

from smoke_tests import (
    test_api_openapi,
    test_api_start,
    test_attention_backends,
    test_manifest_parse,
    test_native_imports,
    test_nunchaku_fused_mlp,
    test_pip_check,
    test_preprocessors_import,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Run Apex bundle smoke tests.")
    p.add_argument(
        "--bundle-root",
        default=None,
        help="Path to bundle root containing src/ and manifest/ (defaults to $APEX_BUNDLE_ROOT or cwd).",
    )
    p.add_argument(
        "--gpu-type", default="", help="GPU type label (e.g. cuda126/cpu/mps)."
    )
    p.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Fail if building a CUDA bundle but CUDA isn't available to run GPU-only smoke tests.",
    )
    p.add_argument(
        "--skip-pip-check",
        action="store_true",
        help="Skip `pip check` (not recommended).",
    )
    args = p.parse_args()

    bundle_root = resolve_bundle_root(args.bundle_root)
    ensure_bundle_on_syspath(bundle_root)

    ctx = SmokeContext(
        bundle_root=bundle_root,
        gpu_type=str(args.gpu_type or ""),
        strict_gpu=bool(args.strict_gpu),
    )

    log("[smoke] python=" + sys.executable)
    log("[smoke] bundle_root=" + str(bundle_root))

    if not (bundle_root / "src").exists():
        fail(f"bundle_root missing src/: {bundle_root}")

    # Ordered, higher-signal first.
    if not args.skip_pip_check:
        test_pip_check.run(ctx)
    test_native_imports.run(ctx)
    test_manifest_parse.run(ctx)
    test_preprocessors_import.run(ctx)
    test_attention_backends.run(ctx)
    test_api_openapi.run(ctx)
    test_api_start.run(ctx)
    test_nunchaku_fused_mlp.run(ctx)

    # Optional strictness: if building a CUDA bundle, require CUDA availability.
    if ctx.strict_gpu and str(ctx.gpu_type).startswith("cuda"):
        try:
            import torch  # type: ignore

            if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
                fail(
                    "Strict GPU mode: building a CUDA bundle but CUDA is not available; "
                    "GPU kernel smoke tests could not run."
                )
        except Exception as e:
            fail(f"Strict GPU mode: failed to verify CUDA availability: {e}")

    log("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
