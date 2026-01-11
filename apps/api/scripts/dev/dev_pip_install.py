#!/usr/bin/env python3
"""
Dev/local pip installer for Apex Engine (apex-engine).

Goal: provide a simple, pip-based install path for local development that can target:
  - the current Python environment, or
  - a new virtualenv (venv)

This is intentionally lighter than `scripts/install.sh` (conda/CUDA setup) and mirrors the
high-level order used by `scripts/bundle_python.py`:
  1) ensure build tooling
  2) install honcho (used by `apex-engine dev`)
  3) install torch for the chosen backend
  4) install machine requirements (cpu/mps/cuda*/rocm)
  5) install apex-engine itself (editable, no-deps) to expose the CLI entrypoint
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm6.4",
}


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _uv() -> str | None:
    return shutil.which("uv")


def _install(
    py: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    use_no_build_isolation: bool = False,
) -> None:
    """
    Install packages using uv (fast) when available, with a pip fallback.
    `args` are pip-style arguments, e.g. ["-r", "requirements.txt"] or ["honcho"].
    """
    uv = _uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", str(py)]
        if use_no_build_isolation:
            cmd.append("--no-build-isolation")
        cmd += args
        _run(cmd, cwd=cwd)
        return

    cmd = [str(py), "-m", "pip", "install"]
    if use_no_build_isolation:
        cmd.append("--no-build-isolation")
    cmd += args
    _run(cmd, cwd=cwd)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _detect_default_machine() -> str:
    if sys.platform == "darwin":
        return "mps"
    return "cpu"


def _detect_default_torch_backend(machine: str) -> str:
    if sys.platform == "darwin":
        return "pypi"
    if machine.startswith("cuda"):
        return "cu126"
    if machine == "rocm":
        return "rocm"
    return "cpu"


def main() -> int:
    # This file lives at scripts/dev/dev_pip_install.py.
    # Project root is apps/api/.
    project_root = Path(__file__).resolve().parent.parent.parent
    machines_dir = project_root / "requirements" / "machines"

    machine_choices = sorted(p.stem for p in machines_dir.glob("*.txt"))
    default_machine = _detect_default_machine()

    parser = argparse.ArgumentParser(
        description="Install Apex Engine requirements via pip (current env or a venv)."
    )
    parser.add_argument(
        "--machine",
        default=default_machine,
        choices=machine_choices,
        help="Machine requirements entrypoint under requirements/machines/ (default: auto).",
    )
    parser.add_argument(
        "--venv",
        default="",
        help="Path to create/use a virtualenv. If omitted, uses the current Python environment.",
    )
    parser.add_argument(
        "--torch",
        default="auto",
        choices=["auto", "pypi", *sorted(TORCH_INDEX.keys())],
        help=(
            "Torch backend/index selection. "
            "Use 'pypi' for default PyPI wheels (recommended on macOS). "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--build-isolation",
        action="store_true",
        help=(
            "Force pip build isolation when installing machine requirements "
            "(overrides the default/auto behavior)."
        ),
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help=(
            "Disable pip build isolation while installing machine requirements "
            "(often required for CUDA source builds that import torch at build time)."
        ),
    )
    args = parser.parse_args()

    if args.build_isolation and args.no_build_isolation:
        raise SystemExit("Choose at most one: --build-isolation or --no-build-isolation")

    # Select interpreter (current env or venv)
    if args.venv:
        venv_dir = (project_root / args.venv).resolve()
        if not venv_dir.exists():
            _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=project_root)
        py = _venv_python(venv_dir)
    else:
        py = Path(sys.executable)

    if not py.exists():
        raise SystemExit(f"Python interpreter not found: {py}")

    req_file = machines_dir / f"{args.machine}.txt"
    if not req_file.exists():
        raise SystemExit(f"Machine requirements file not found: {req_file}")

    torch_backend = args.torch
    if torch_backend == "auto":
        torch_backend = _detect_default_torch_backend(args.machine)

    # On Linux, default to disabling build isolation unless explicitly forced on.
    # Reason: several sdists/VCS installs in our stack import runtime deps (e.g. numpy/torch)
    # during metadata/build phases without declaring them as build deps, which breaks with
    # PEP517 build isolation.
    auto_no_build_isolation = sys.platform.startswith("linux") and not args.build_isolation
    use_no_build_isolation = args.no_build_isolation or auto_no_build_isolation

    # 1) Tooling
    # Note: some packages in our requirements rely on "legacy" setup steps that
    # expect common build helpers (e.g. Cython) to already be available when build
    # isolation is disabled.
    _install(
        py,
        [
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            # Pre-install build helpers that some sdists forget to declare.
            # (Example: opendr/opendr-toolkit needs Cython/numpy at build time but doesn't specify them.)
            "Cython>=0.29.36",
            "numpy",
            "psutil",
            "enscons",
            "pytoml",
        ],
        use_no_build_isolation=False,
    )

    # 2) honcho (needed by `apex-engine dev`)
    _install(py, ["honcho"])

    # 3) torch
    if torch_backend == "pypi":
        _install(py, ["torch", "torchvision", "torchaudio"])
    else:
        index = TORCH_INDEX[torch_backend]
        _install(py, ["--index-url", index, "torch", "torchvision", "torchaudio"])

    # 4) machine requirements
    _install(
        py,
        ["-r", str(req_file)],
        cwd=project_root,
        use_no_build_isolation=use_no_build_isolation,
    )

    # 5) install apex-engine itself (expose the CLI) without re-resolving deps
    _install(py, ["-e", ".", "--no-deps"], cwd=project_root)

    print("\nDone.")
    if args.venv and sys.platform != "win32":
        # Hint only; keep output short and actionable.
        print(f"Activate: source {Path(args.venv) / 'bin' / 'activate'}")
    print("Run: apex-engine dev   (or: apex-engine start -f Procfile.dev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


