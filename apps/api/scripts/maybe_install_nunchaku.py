#!/usr/bin/env python3
"""
Best-effort Nunchaku wheel selector/installer.

This is intentionally kept in sync with `BundlePython._maybe_install_nunchaku` in
`apps/api/scripts/bundle_python.py`.

We do *not* include Nunchaku in base requirements because its wheels are built against
specific PyTorch major/minor versions (e.g. +torch2.9). Installing the wrong wheel can
hard-fail an environment. Instead, we detect the *installed* torch version and install
the matching wheel if one exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


NUNCHAKU_VERSION = "1.1.0"
NUNCHAKU_RELEASE_TAG = f"v{NUNCHAKU_VERSION}"
NUNCHAKU_RELEASE_BASE_URL = f"https://github.com/nunchaku-tech/nunchaku/releases/download/{NUNCHAKU_RELEASE_TAG}"


SUPPORTED_LINUX_TORCH_MM = {"2.7", "2.8", "2.9", "2.11"}
SUPPORTED_WIN_TORCH_MM = {"2.9", "2.11"}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    platform_name: str
    python: str
    uv: Optional[str]
    compute_capability: Optional[str]
    python_tag: Optional[str]
    torch_mm: Optional[str]
    nunchaku_version: str = NUNCHAKU_VERSION
    wheel_filename: Optional[str] = None
    wheel_url: Optional[str] = None


def _run(cmd: list[str], *, timeout: int | float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _has_required_cuda_runtime_for_wheel() -> tuple[bool, str]:
    """
    Nunchaku wheels we install are currently linked against libcudart.so.13
    (see user-reported import error and `ldd nunchaku/_C*.so`).
    """
    if sys.platform == "linux":
        try:
            import ctypes

            ctypes.CDLL("libcudart.so.13")
            return True, "ok"
        except Exception as e:
            return False, "missing libcudart.so.13 (CUDA runtime 13)"
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.WinDLL("cudart64_13.dll")
            return True, "ok"
        except Exception:
            return False, "missing cudart64_13.dll (CUDA runtime 13)"
    return False, f"unsupported platform for this check: {sys.platform}"


def detect_cuda_compute_capability() -> Optional[str]:
    """
    Detect NVIDIA compute capability via nvidia-smi (e.g. "8.0", "8.6", "8.9", "9.0", "10.0").
    Returns None if not available.
    """
    try:
        result = _run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            timeout=5,
        )
        if result.returncode != 0:
            return None
        cap = (result.stdout or "").strip().split("\n")[0].strip()
        return cap or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def probe_python_and_torch(py_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (python_tag, torch_major_minor) like ("cp312", "2.9"), or (None, None) on failure.
    """
    try:
        probe = _run(
            [
                str(py_path),
                "-c",
                "import sys; "
                "import torch; "
                "v = (torch.__version__ or '').split('+')[0].strip(); "
                "mm = '.'.join(v.split('.')[:2]) if v else ''; "
                "print(f'cp{sys.version_info.major}{sys.version_info.minor}'); "
                "print(mm)",
            ]
        )
    except Exception:
        return None, None

    if probe.returncode != 0:
        return None, None

    lines = [ln.strip() for ln in (probe.stdout or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None, None
    return lines[0], lines[1]


def decide_nunchaku_install(
    *,
    py_path: Path,
    platform_name: str,
    machine_entry_name: Optional[str],
) -> Decision:
    # Only CUDA-capable platforms; skip macOS/ROCm/CPU bundles.
    if platform_name not in ("linux", "win32"):
        return Decision(
            allowed=False,
            reason=f"unsupported platform for nunchaku wheels: {platform_name}",
            platform_name=platform_name,
            python=str(py_path),
            uv=None,
            compute_capability=None,
            python_tag=None,
            torch_mm=None,
        )

    # Skip Nunchaku for SM90/Hopper bundles (if the caller knows we're building/targeting one).
    if machine_entry_name and "cuda-sm90" in machine_entry_name:
        return Decision(
            allowed=False,
            reason=f"machine entry indicates SM90/Hopper; skipping ({machine_entry_name})",
            platform_name=platform_name,
            python=str(py_path),
            uv=None,
            compute_capability=None,
            python_tag=None,
            torch_mm=None,
        )

    cap = None
    try:
        cap = detect_cuda_compute_capability()
        if cap:
            major = int(cap.split(".")[0])
            if major == 9:
                return Decision(
                    allowed=False,
                    reason=f"detected compute capability {cap} (Hopper); skipping",
                    platform_name=platform_name,
                    python=str(py_path),
                    uv=None,
                    compute_capability=cap,
                    python_tag=None,
                    torch_mm=None,
                )
    except Exception:
        # Best effort; do not fail.
        cap = cap or None

    py_tag, torch_mm = probe_python_and_torch(py_path)
    if not py_tag or not torch_mm:
        return Decision(
            allowed=False,
            reason="failed to probe python tag and torch major/minor (is torch installed?)",
            platform_name=platform_name,
            python=str(py_path),
            uv=None,
            compute_capability=cap,
            python_tag=py_tag,
            torch_mm=torch_mm,
        )

    ok_cudart, cudart_reason = _has_required_cuda_runtime_for_wheel()
    if not ok_cudart:
        return Decision(
            allowed=False,
            reason=f"{cudart_reason}; nunchaku wheel {NUNCHAKU_VERSION} is not compatible with this CUDA runtime",
            platform_name=platform_name,
            python=str(py_path),
            uv=None,
            compute_capability=cap,
            python_tag=py_tag,
            torch_mm=torch_mm,
        )

    if platform_name == "linux":
        if torch_mm not in SUPPORTED_LINUX_TORCH_MM:
            return Decision(
                allowed=False,
                reason=f"no {NUNCHAKU_VERSION} linux wheel for torch {torch_mm} ({py_tag})",
                platform_name=platform_name,
                python=str(py_path),
                uv=None,
                compute_capability=cap,
                python_tag=py_tag,
                torch_mm=torch_mm,
            )
        plat_suffix = "linux_x86_64"
    else:
        if torch_mm not in SUPPORTED_WIN_TORCH_MM:
            return Decision(
                allowed=False,
                reason=f"no {NUNCHAKU_VERSION} windows wheel for torch {torch_mm} ({py_tag})",
                platform_name=platform_name,
                python=str(py_path),
                uv=None,
                compute_capability=cap,
                python_tag=py_tag,
                torch_mm=torch_mm,
            )
        plat_suffix = "win_amd64"

    filename = f"nunchaku-{NUNCHAKU_VERSION}+torch{torch_mm}-{py_tag}-{py_tag}-{plat_suffix}.whl"
    url = f"{NUNCHAKU_RELEASE_BASE_URL}/{filename.replace('+', '%2B')}"

    return Decision(
        allowed=True,
        reason="ok",
        platform_name=platform_name,
        python=str(py_path),
        uv=None,
        compute_capability=cap,
        python_tag=py_tag,
        torch_mm=torch_mm,
        wheel_filename=filename,
        wheel_url=url,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Decide and optionally install the matching Nunchaku wheel.")
    p.add_argument(
        "--python",
        dest="python_path",
        default=sys.executable,
        help="Python interpreter to probe/install into (default: current interpreter)",
    )
    p.add_argument(
        "--uv",
        dest="uv_path",
        default=os.environ.get("UV", "uv"),
        help="uv executable path (default: $UV or 'uv')",
    )
    p.add_argument(
        "--machine-entry-name",
        default=None,
        help="Optional: machine requirements entry name (e.g. 'cuda-sm90-hopper.txt') to apply SM90 skip rule",
    )
    p.add_argument("--install", action="store_true", help="Actually install via `uv pip install --python ... URL`")
    p.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON decision payload to stdout")
    args = p.parse_args()

    py_path = Path(args.python_path)
    platform_name = sys.platform
    d = decide_nunchaku_install(
        py_path=py_path,
        platform_name=platform_name,
        machine_entry_name=args.machine_entry_name,
    )

    uv = None
    if args.install:
        uv = args.uv_path or None

    if args.as_json:
        payload = asdict(d)
        payload["uv"] = uv
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not d.allowed:
            print(f"Skipping Nunchaku: {d.reason}")
            return 0
        assert d.wheel_filename and d.wheel_url
        print(f"Nunchaku allowed: {d.wheel_filename}")
        print(d.wheel_url)

    if not args.install:
        return 0

    if not d.allowed or not d.wheel_url:
        return 0

    uv_exe = uv or "uv"
    if shutil.which(uv_exe) is None and not Path(uv_exe).exists():
        print(f"Cannot install: uv not found ({uv_exe})", file=sys.stderr)
        return 2

    print(f"Attempting Nunchaku install: {d.wheel_filename}")
    try:
        subprocess.run(
            [uv_exe, "pip", "install", "--python", str(py_path), d.wheel_url],
            check=False,
        )
    except Exception:
        # Best-effort: do not fail.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


