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
    project_root = Path(__file__).resolve().parent.parent
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
        "--no-build-isolation",
        action="store_true",
        help="Pass --no-build-isolation while installing requirements (matches bundler behavior).",
    )
    args = parser.parse_args()

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

    # 1) Tooling
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # 2) honcho (needed by `apex-engine dev`)
    _run([str(py), "-m", "pip", "install", "honcho"])

    # 3) torch
    if torch_backend == "pypi":
        _run([str(py), "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    else:
        index = TORCH_INDEX[torch_backend]
        _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--index-url",
                index,
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )

    # 4) machine requirements
    req_cmd = [str(py), "-m", "pip", "install", "-r", str(req_file)]
    if args.no_build_isolation:
        req_cmd.insert(5, "--no-build-isolation")
    _run(req_cmd, cwd=project_root)

    # 5) install apex-engine itself (expose the CLI) without re-resolving deps
    _run([str(py), "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=project_root)

    print("\nDone.")
    if args.venv and sys.platform != "win32":
        # Hint only; keep output short and actionable.
        print(f"Activate: source {Path(args.venv) / 'bin' / 'activate'}")
    print("Run: apex-engine dev   (or: apex-engine start -f Procfile.dev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


