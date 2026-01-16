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


def _bootstrap_uv(py: Path, *, cwd: Path | None = None) -> None:
    """
    Ensure `uv` is available in the target Python environment.

    On Windows, relying on `shutil.which("uv")` can be brittle depending on PATH
    and venv activation; installing via pip and invoking via `python -m uv` is
    reliable across platforms.
    """
    _run([str(py), "-m", "pip", "install", "--upgrade", "uv"], cwd=cwd)


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
    # Prefer uv (installed via `_bootstrap_uv`) and invoke it via `python -m uv`
    # so we don't depend on PATH/venv activation (especially on Windows).
    try:
        cmd = [str(py), "-m", "uv", "pip", "install"]
        if use_no_build_isolation:
            cmd.append("--no-build-isolation")
        cmd += args
        _run(cmd, cwd=cwd)
        return
    except subprocess.CalledProcessError:
        # Fall back to plain pip if uv isn't available or errors unexpectedly.
        pass

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
        return "mac"
    return "cpu"


def _detect_default_torch_backend(machine: str) -> str:
    if sys.platform == "darwin":
        return "pypi"
    if machine == "windows" or machine == "linux" or machine.startswith("cuda"):
        # Hopper/Blackwell and Universal entrypoints (windows.txt/linux.txt) use cu128
        return "cu128"
    if machine == "rocm":
        return "rocm"
    if machine == "rocm-windows":
        return "rocm-win-direct"
    return "cpu"


def main() -> int:
    # This file lives at scripts/dev/dev_pip_install.py.
    # Project root is apps/api/.
    project_root = Path(__file__).resolve().parent.parent.parent
    requirements_dir = project_root / "requirements"

    # Map friendly names to paths
    machine_map = {
        "cpu": requirements_dir / "cpu" / "requirements.txt",
        "mac": requirements_dir / "mps" / "requirements.txt",
        "linux": requirements_dir / "cuda" / "linux.txt",
        "windows": requirements_dir / "cuda" / "windows.txt",
        "mps": requirements_dir / "mps" / "requirements.txt",
        "rocm": requirements_dir / "rocm" / "linux.txt",
        "rocm-windows": requirements_dir / "rocm" / "windows.txt",
        # Allow cuda-linux/cuda-windows explicit selection too
        "cuda-linux": requirements_dir / "cuda" / "linux.txt",
        "cuda-windows": requirements_dir / "cuda" / "windows.txt",
    }
    # Add explicit CUDA entries if they exist (backward compat or specific files)
    cuda_dir = requirements_dir / "cuda"
    if cuda_dir.exists():
        for p in cuda_dir.glob("*.txt"):
            if p.name in ["linux.txt", "windows.txt"]:
                continue
            name = f"cuda-{p.stem}"
            machine_map[name] = p

    machine_choices = sorted(machine_map.keys())
    default_machine = _detect_default_machine()

    parser = argparse.ArgumentParser(
        description="Install Apex Engine requirements via pip (current env or a venv)."
    )
    parser.add_argument(
        "--machine",
        default=default_machine,
        choices=machine_choices,
        help="Machine requirements entrypoint (default: auto).",
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

    req_file = machine_map[args.machine]
    if not req_file.exists():
        raise SystemExit(f"Machine requirements file not found: {req_file}")

    torch_backend = args.torch
    if torch_backend == "auto":
        torch_backend = _detect_default_torch_backend(args.machine)

    # Default to disabling build isolation unless explicitly forced on.
    # Reason:
    # - Several sdists/VCS installs in our stack import runtime deps (e.g. numpy/torch)
    #   during metadata/build phases without declaring them as build deps, which breaks with
    #   PEP517 build isolation.
    # - On Windows, some sdists (notably `lmdb`/py-lmdb when a wheel isn't available) expect
    #   helper modules (e.g. `patch-ng`) to be available at build time.
    auto_no_build_isolation = (sys.platform.startswith("linux") or sys.platform == "win32") and not args.build_isolation
    use_no_build_isolation = args.no_build_isolation or auto_no_build_isolation

    # 0) Install uv first (per request), then use `uv pip` for everything else.
    _bootstrap_uv(py, cwd=project_root)

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
            # Keep packaging in sync with setuptools. Newer setuptools calls
            # `packaging.utils.canonicalize_version(..., strip_trailing_zero=...)`.
            # If packaging is too old, installs fail at metadata prep time with:
            #   TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
            "packaging>=24",
            # Pre-install build helpers that some sdists forget to declare.
            # (Example: opendr/opendr-toolkit needs Cython/numpy at build time but doesn't specify them.)
            "Cython>=0.29.36",
            "numpy",
            "psutil",
            "enscons",
            "pytoml",
            # `lmdb` (py-lmdb) source builds on Windows require `patch-ng` at build time.
            # Even when it's present in our requirements, pip can attempt to build lmdb before
            # installing patch-ng, so we ensure it's already available here.
            "patch-ng",
        ],
        use_no_build_isolation=False,
    )

    # 3) torch
    if torch_backend == "pypi":
        if sys.platform == "darwin":
            _install(py, ["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"])
        else:
            _install(py, ["torch", "torchvision", "torchaudio"])
    elif torch_backend == "rocm-win-direct":
        _install(py, [
            "torch @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl",
            "torchvision @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl",
            "torchaudio @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl",
        ])
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

    # 4a) Optional: install Nunchaku (CUDA-only) without dependencies by default.
    # This avoids dependency resolver churn (nunchaku is optional, and its deps can conflict
    # with the rest of the stack). Opt into deps with `--with-deps` on the helper or
    # `APEX_NUNCHAKU_WITH_DEPS=1`.
    req_norm = str(req_file).replace("\\", "/")
    if "/requirements/cuda/" in req_norm and sys.platform in ("linux", "win32"):
        try:
            _run(
                [
                    str(py),
                    str(project_root / "scripts" / "deps" / "maybe_install_nunchaku.py"),
                    "--python",
                    str(py),
                    "--machine-entry-name",
                    str(args.machine),
                    "--install",
                ],
                cwd=project_root,
            )
        except subprocess.CalledProcessError:
            # Best-effort; keep install flow resilient.
            pass

    # 4b) Patch third-party deps in the active env/venv (diffusers + xformers).
    # We patch by resolving the installed module paths and editing in-place, rather than
    # applying a whole patches directory (which may include optional/outdated patches).
    _run(
        [
            str(py),
            str(project_root / "scripts" / "updates" / "apply_third_party_patches.py"),
        ],
        cwd=project_root,
    )

    # 5) install apex-engine itself (expose the CLI) without re-resolving deps
    _install(py, ["-e", ".", "--no-deps"], cwd=project_root)

    print("\nDone.")
    if args.venv and sys.platform != "win32":
        # Hint only; keep output short and actionable.
        print(f"Activate: source {Path(args.venv) / 'bin' / 'activate'}")
    print("Run: python3 -m src serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


