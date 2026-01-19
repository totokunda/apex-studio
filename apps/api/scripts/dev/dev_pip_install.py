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
import platform
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TORCH_VERSION = "2.9.1"
DEFAULT_TORCHVISION_VERSION = "0.24.1"
DEFAULT_TORCHAUDIO_VERSION = "2.9.1"

# Keep numpy pinned to the same version as `requirements/requirements.txt`.
# This avoids uv selecting new numba/llvmlite releases that require numpy 2.x and
# then falling back to source builds on macOS.
PINNED_NUMPY_SPEC = "numpy==1.26.4"

# Intel macOS (x86_64): latest supported torch stack.
MACOS_INTEL_TORCH_VERSION = "2.2.2"
MACOS_INTEL_TORCHVISION_VERSION = "0.17.2"
MACOS_INTEL_TORCHAUDIO_VERSION = "2.2.2"


def _is_intel_macos() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        arch = (platform.machine() or "").lower()
    except Exception:
        arch = ""
    return arch in {"x86_64", "amd64", "i386", "i686"} or arch.startswith("x86")


def _torch_stack_versions() -> tuple[str, str, str]:
    if _is_intel_macos():
        return (
            MACOS_INTEL_TORCH_VERSION,
            MACOS_INTEL_TORCHVISION_VERSION,
            MACOS_INTEL_TORCHAUDIO_VERSION,
        )
    return (
        DEFAULT_TORCH_VERSION,
        DEFAULT_TORCHVISION_VERSION,
        DEFAULT_TORCHAUDIO_VERSION,
    )


TORCH_VERSION, TORCHVISION_VERSION, TORCHAUDIO_VERSION = _torch_stack_versions()

# setuptools pin for Intel macOS torch stack.
# We keep this conservative to avoid build/metadata toolchain changes that can break
# older PyTorch stacks and their ecosystem on x86_64 macOS.
SETUPTOOLS_SPEC = "setuptools>=61,<70" if _is_intel_macos() else "setuptools"

TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm6.4",
}


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


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
        # Use unsafe-best-match to ensure we pick the best wheel across multiple indices.
        cmd += ["--index-strategy", "unsafe-best-match"]
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


def _python_machine(py: Path) -> str:
    """
    Return the architecture reported by the given interpreter, e.g. "x86_64" or "arm64".
    Best-effort: returns empty string on failure.
    """
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import platform; print(platform.machine())"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return (out or "").strip().lower()
    except Exception:
        return ""


def _ensure_venv_arch_matches_launcher(venv_dir: Path) -> None:
    """
    On Apple Silicon, it's possible to launch this script with an x86_64-only
    interpreter (e.g. a Rosetta "intel64" launcher) but end up with a venv whose
    `bin/python` is a universal2 binary that defaults to arm64.

    That mismatch causes pip to miss x86_64 wheels (e.g. numba/llvmlite) and
    compile from source instead.

    If we detect this situation, replace the venv's `bin/python` with the exact
    launcher binary so the venv consistently runs under the requested arch.
    """
    if sys.platform != "darwin":
        return

    launcher_machine = (platform.machine() or "").lower()
    if launcher_machine not in {"x86_64", "amd64"}:
        return

    try:
        # Always copy the launcher into the venv. On Apple Silicon, venvs created from
        # framework/universal2 Pythons can "flip" arch based on how they're invoked.
        # Overwriting `bin/python` with the x86_64-only launcher makes the venv stable.
        venv_py = _venv_python(venv_dir)
        print(
            "Forcing venv python to x86_64 (copying the launcher binary) "
            "to prefer binary wheels and keep arch stable."
        )
        shutil.copy2(sys.executable, venv_py)
        venv_py.chmod(0o755)
    except Exception as e:
        print(f"Warning: Failed to force venv python architecture: {e}")


def _remove_obsolete_typing_backport(py: Path) -> None:
    """
    PyInstaller error:
      "The 'typing' package is an obsolete backport of a standard library package and is incompatible with PyInstaller."

    Even in dev mode, having this package can cause weird resolution issues or crashes.
    """
    try:
        # `pip uninstall` returns non-zero when the package isn't installed; that's fine.
        subprocess.run(
            [str(py), "-m", "pip", "uninstall", "-y", "typing"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


def _build_and_install_rust_wheels(py: Path, project_root: Path) -> None:
    rust_project = project_root / "rust" / "apex_download_rs"
    if not rust_project.exists():
        return

    # Check for cargo/rustc
    cargo = shutil.which("cargo")
    if not cargo:
        print("Warning: `cargo` not found on PATH; skipping rust wheel build.")
        return

    # Install maturin in the target environment
    _install(py, ["maturin>=1.6,<2.0"])

    print(f"Building Rust wheel (maturin): {rust_project}")
    # Use a temporary directory for the wheel to avoid noise
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            target: str | None = None
            # If we're running an x86_64 Python under Rosetta on Apple Silicon, ensure
            # we build an x86_64 wheel (otherwise maturin will default to arm64 host).
            if sys.platform == "darwin":
                host_machine = (platform.machine() or "").lower()
                py_machine = _python_machine(py)
                if host_machine == "arm64" and py_machine in {"x86_64", "amd64"}:
                    target = "x86_64-apple-darwin"

            env = None
            cmd = [
                str(py),
                "-m",
                "maturin",
                "build",
                "--release",
                "--strip",
                "--interpreter",
                str(py),
                "--out",
                tmpdir,
            ]
            if target:
                cmd += ["--target", target]
                env = dict(os.environ)
                env["CARGO_BUILD_TARGET"] = target

            _run(
                cmd,
                cwd=rust_project,
                env=env,
            )
            wheels = list(Path(tmpdir).glob("*.whl"))
            if wheels:
                _install(py, [str(wheels[0])])
        except Exception as e:
            print(f"Warning: Failed to build/install Rust wheels: {e}")


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
        raise SystemExit(
            "Choose at most one: --build-isolation or --no-build-isolation"
        )

    # Select interpreter (current env or venv)
    if args.venv:
        venv_dir = (project_root / args.venv).resolve()
        if not venv_dir.exists():
            _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=project_root)
        _ensure_venv_arch_matches_launcher(venv_dir)
        py = _venv_python(venv_dir)
    else:
        py = Path(sys.executable)

    if not py.exists():
        raise SystemExit(f"Python interpreter not found: {py}")

    req_file = machine_map[args.machine]
    if not req_file.exists():
        raise SystemExit(f"Machine requirements file not found: {req_file}")

    # macOS: treat Intel (x86_64) as CPU-only even if the user passes `--machine mac`.
    # This keeps `--machine mac` working across Apple Silicon + Intel.
    if args.machine in {"mac", "mps"} and _is_intel_macos():
        req_file = requirements_dir / "cpu" / "requirements.txt"

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
    # - On macOS, some legacy sdists (e.g. chumpy) can import `pip` during setup/metadata,
    #   which is not present in isolated build environments.
    auto_no_build_isolation = (
        sys.platform.startswith("linux")
        or sys.platform == "win32"
        or sys.platform == "darwin"
    ) and not args.build_isolation
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
            SETUPTOOLS_SPEC,
            "wheel",
            # Keep packaging in sync with setuptools. Newer setuptools calls
            # `packaging.utils.canonicalize_version(..., strip_trailing_zero=...)`.
            # If packaging is too old, installs fail at metadata prep time with:
            #   TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
            "packaging>=24",
            # Pre-install build helpers that some sdists forget to declare.
            # (Example: opendr/opendr-toolkit needs Cython/numpy at build time but doesn't specify them.)
            "Cython>=0.29.36",
            PINNED_NUMPY_SPEC,
            "psutil",
            "enscons",
            "pytoml",
            # `lmdb` (py-lmdb) source builds on Windows require `patch-ng` at build time.
            # Even when it's present in our requirements, pip can attempt to build lmdb before
            # installing patch-ng, so we ensure it's already available here.
            "patch-ng",
            # Dev tooling
            "honcho",
        ],
        use_no_build_isolation=False,
    )

    # 3) torch
    suffix = ""
    if sys.platform == "linux":
        if torch_backend == "cpu":
            suffix = "+cpu"
        elif torch_backend.startswith("cu"):
            suffix = f"+{torch_backend}"
        elif torch_backend == "rocm":
            suffix = "+rocm6.4"
    elif sys.platform == "win32":
        if torch_backend.startswith("cu"):
            suffix = f"+{torch_backend}"

    if torch_backend == "pypi":
        if sys.platform == "darwin":
            _install(
                py,
                [
                    f"torch=={TORCH_VERSION}",
                    f"torchvision=={TORCHVISION_VERSION}",
                    f"torchaudio=={TORCHAUDIO_VERSION}",
                ],
            )
        else:
            _install(py, ["torch", "torchvision", "torchaudio"])
    elif torch_backend == "rocm-win-direct":
        _install(
            py,
            [
                "torch @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl",
                "torchvision @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl",
                "torchaudio @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl",
            ],
        )
    else:
        index = TORCH_INDEX[torch_backend]
        # Map torch/vision/audio to specific versions with suffixes if needed
        specs = [
            f"torch=={TORCH_VERSION}{suffix}",
            f"torchvision=={TORCHVISION_VERSION}{suffix}",
            f"torchaudio=={TORCHAUDIO_VERSION}{suffix}",
        ]
        _install(py, ["--index-url", index, *specs])

    # 4) machine requirements
    # 4a) Build & install Rust wheels (apex_download_rs) if cargo is available.
    _build_and_install_rust_wheels(py, project_root)

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
                    str(
                        project_root / "scripts" / "deps" / "maybe_install_nunchaku.py"
                    ),
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

    # 5a) Remove obsolete typing backport if it was accidentally pulled in.
    _remove_obsolete_typing_backport(py)

    print("\nDone.")
    if args.venv and sys.platform != "win32":
        # Hint only; keep output short and actionable.
        print(f"Activate: source {Path(args.venv) / 'bin' / 'activate'}")
    print("Run: python3 -m src serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
