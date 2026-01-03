#!/usr/bin/env python3
"""
Python API Bundler for Apex Studio

This script handles bundling the Python API for distribution with the Electron app.
It supports multiple platforms and handles PyTorch/CUDA dependencies intelligently.

NOTE: We intentionally do NOT use PyInstaller. Instead, we create a self-contained
virtual environment (venv) and ship it alongside the API source/assets. The app
is started by invoking the venv's Python interpreter and running `-m src ...`.

Usage:
    python scripts/bundle_python.py --platform [darwin|linux|win32] --output ./dist
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Set, List, Tuple


class PythonBundler:
    """Bundles Python API for distribution (venv-based, no PyInstaller)."""

    SUPPORTED_PLATFORMS = ["darwin", "linux", "win32"]

    # PyTorch wheel indices for different platforms
    TORCH_INDICES = {
        "cuda126": "https://download.pytorch.org/whl/cu126",
        "cuda124": "https://download.pytorch.org/whl/cu124",
        "cuda121": "https://download.pytorch.org/whl/cu121",
        "cuda118": "https://download.pytorch.org/whl/cu118",
        "cuda128": "https://download.pytorch.org/whl/cu128",
        "cpu": "https://download.pytorch.org/whl/cpu",
        "rocm": "https://download.pytorch.org/whl/rocm6.4",
    }

    def __init__(
        self,
        platform_name: str,
        output_dir: Path,
        cuda_version: Optional[str] = None,
        sign: bool = False,
        python_executable: Optional[str] = None,
        prefer_python_312: bool = True,
    ):
        self.platform_name = platform_name
        # Resolve to an absolute path so later subprocess calls that change cwd (e.g. Rust builds)
        # still can find the venv interpreter and other bundle files reliably.
        self.output_dir = Path(output_dir).resolve()
        self.cuda_version = cuda_version
        self.sign = sign
        self.python_executable = python_executable or sys.executable
        self.prefer_python_312 = prefer_python_312

        self.project_root = Path(__file__).resolve().parent.parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.output_dir / "python-api"

        # Name of the venv directory we create & ship.
        # Users can activate it directly: `source apex-studio/bin/activate`
        self.venv_name = "apex-studio"

        # Populated after bundling
        self.last_gpu_type: Optional[str] = None
        self.last_manifest: Optional[dict] = None

        # Whether to keep the *intermediate* build venv at `<output>/<venv_name>/`.
        # The shipped bundle contains its own copy at `<output>/python-api/apex-engine/<venv_name>/`.
        self.keep_build_venv: bool = False

    def _run(self, cmd: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _ensure_uv_available(self) -> str:
        """
        Ensure the `uv` CLI is available.

        We prefer using a system-installed `uv`. If not available, we install it into the
        bundler's selected interpreter environment (the host interpreter running this script).
        """
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True)
            return "uv"
        except Exception:
            # Bootstrap uv using the selected interpreter (network access is already assumed for dependency installs).
            subprocess.run(
                [self.python_executable, "-m", "pip", "install", "--upgrade", "uv"],
                check=True,
            )
            return "uv"

    def _ensure_macos_libpython_present(self, venv_dir: Path) -> None:
        """
        macOS: ensure the venv is relocatable enough for our bundle.

        Some Python distributions (notably conda/anaconda) build the `python` executable
        to load `libpythonX.Y.dylib` from a path relative to the executable:
          @executable_path/../lib/libpythonX.Y.dylib

        When we ship/copy the venv, that dylib may not be present under `<venv>/lib/`,
        causing dyld aborts after install/move. We copy the base interpreter's libpython
        dylib into the venv `lib/` directory as a pragmatic fix.
        """
        if self.platform_name != "darwin":
            return

        if self.platform_name == "win32":
            py_path = venv_dir / "Scripts" / "python.exe"
        else:
            py_path = venv_dir / "bin" / "python"

        # If the dylib is already present, nothing to do.
        lib_dir = venv_dir / "lib"
        # Common names we might need.
        existing = list(lib_dir.glob("libpython*.dylib")) if lib_dir.exists() else []
        if existing:
            return

        # Ask the *base* interpreter where its libpython lives.
        try:
            probe = subprocess.run(
                [
                    str(self.python_executable),
                    "-c",
                    "import sys, sysconfig, pathlib; "
                    "base = getattr(sys, 'base_prefix', sys.prefix); "
                    "libdir = sysconfig.get_config_var('LIBDIR') or ''; "
                    "ldlib = sysconfig.get_config_var('LDLIBRARY') or ''; "
                    "p = pathlib.Path(libdir) / ldlib if (libdir and ldlib) else pathlib.Path(''); "
                    "print(str(base)); "
                    "print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
                    "print(str(p) if str(p) else '')",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = [ln.strip() for ln in probe.stdout.splitlines()]
            base_prefix = lines[0] if len(lines) > 0 else ""
            py_mm = lines[1] if len(lines) > 1 else ""
            reported = lines[2] if len(lines) > 2 else ""
        except Exception:
            base_prefix = ""
            py_mm = ""
            reported = ""

        candidates: list[Path] = []
        if reported:
            candidates.append(Path(reported))
        if base_prefix and py_mm:
            # Conda often reports LDLIBRARY as a static archive (.a). We want the dylib.
            candidates.append(Path(base_prefix) / "lib" / f"libpython{py_mm}.dylib")
            candidates.extend(sorted((Path(base_prefix) / "lib").glob(f"libpython{py_mm}*.dylib")))
        if base_prefix:
            candidates.extend(sorted((Path(base_prefix) / "lib").glob("libpython*.dylib")))

        src = next((p for p in candidates if p.exists() and p.suffix == ".dylib"), None)
        if src is None:
            return

        lib_dir.mkdir(parents=True, exist_ok=True)
        dst = lib_dir / src.name
        try:
            shutil.copy2(src, dst)
        except Exception:
            # Best-effort: bundling should not fail solely due to this hygiene step.
            return

    def detect_gpu_support(self) -> str:
        """Detect available GPU support"""
        if self.platform_name == "darwin":
            # macOS uses MPS (Metal Performance Shaders)
            return "mps"

        # Check for NVIDIA GPU
        try:
            result = self._run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            )
            if result.returncode == 0:
                # NVIDIA GPU found, detect CUDA version
                cuda_result = self._run(
                    ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"]
                )
                if cuda_result.returncode == 0:
                    cuda_ver = cuda_result.stdout.strip().split("\n")[0]
                    major, minor = cuda_ver.split(".")[:2]
                    return f"cuda{major}{minor}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for AMD ROCm
        if os.path.exists("/opt/rocm"):
            return "rocm"

        return "cpu"

    def detect_cuda_compute_capability(self) -> Optional[str]:
        """
        Detect NVIDIA compute capability via nvidia-smi (e.g. "8.0", "8.6", "8.9", "9.0", "10.0").
        Returns None if not available.
        """
        try:
            result = self._run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                timeout=5,
            )
            if result.returncode != 0:
                return None
            cap = result.stdout.strip().split("\n")[0].strip()
            return cap or None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def get_torch_index(self, gpu_type: str) -> str:
        """Get the appropriate PyTorch wheel index URL"""
        if gpu_type.startswith("cuda"):
            # Try to find matching CUDA version
            for key in ["cuda128", "cuda126", "cuda124", "cuda121", "cuda118"]:
                if key in gpu_type or gpu_type == key:
                    return self.TORCH_INDICES[key]
            # Default to latest CUDA
            return self.TORCH_INDICES["cuda126"]
        elif gpu_type == "rocm":
            return self.TORCH_INDICES["rocm"]
        else:
            return self.TORCH_INDICES["cpu"]

    def choose_machine_requirements_entrypoint(self, gpu_type: str) -> Path:
        """
        Choose an entrypoint under requirements/machines based on platform + GPU/arch.
        """
        machines_dir = self.project_root / "requirements" / "machines"

        # macOS / Apple Silicon
        if self.platform_name == "darwin":
            return machines_dir / "mps.txt"

        # ROCm
        if gpu_type == "rocm":
            return machines_dir / "rocm.txt"

        # CPU
        if gpu_type == "cpu":
            return machines_dir / "cpu.txt"

        # Windows CUDA: select from the standard CUDA entrypoints (they include
        # Windows-specific wheels/pins via sys_platform markers).
        if self.platform_name == "win32" and gpu_type.startswith("cuda"):
            # For the SageAttention Windows wheels we need CUDA 12.8+ (gpu_type cuda128).
            # If not on cuda128, fall back to the cu126 FlashAttention wheel stack.
            if gpu_type != "cuda128":
                return machines_dir / "cuda-sm80-ampere.txt"

            cap = self.detect_cuda_compute_capability()
            if cap:
                try:
                    major = int(cap.split(".")[0])
                    minor = int(cap.split(".")[1])
                    # Ada: 8.9
                    if major == 8 and minor == 9:
                        return machines_dir / "cuda-sm89-ada.txt"
                    # Blackwell (10.x+ / 12.x+)
                    if major >= 10:
                        return machines_dir / "cuda-sm100-blackwell.txt"
                except Exception:
                    pass
            # Fallback
            return machines_dir / "cuda-sm89-ada.txt"

        # Linux CUDA: pick by compute capability
        if gpu_type.startswith("cuda"):
            cap = self.detect_cuda_compute_capability()
            # cap examples: 8.0, 8.6, 8.9, 9.0, 10.0
            if cap:
                try:
                    major = int(cap.split(".")[0])
                    minor = int(cap.split(".")[1])
                    # Ampere: 8.0 / 8.6
                    if major == 8 and minor in (0, 6):
                        return machines_dir / "cuda-sm80-ampere.txt"
                    # Ada: 8.9
                    if major == 8 and minor == 9:
                        return machines_dir / "cuda-sm89-ada.txt"
                    # Hopper: 9.0
                    if major == 9:
                        return machines_dir / "cuda-sm90-hopper.txt"
                    # Blackwell: 10.x+
                    if major >= 10:
                        return machines_dir / "cuda-sm100-blackwell.txt"
                except Exception:
                    pass

            # Fallback if compute capability detection fails: default to Ampere-safe set.
            return machines_dir / "cuda-sm80-ampere.txt"

        # Final fallback
        return machines_dir / "cpu.txt"

    def _read_requirements_file(self, path: Path, visited: Optional[Set[Path]] = None) -> List[str]:
        """
        Read a requirements file and recursively expand `-r` includes.
        Returns raw requirement lines (including markers / direct URLs), excluding comments/empties.
        """
        visited = visited or set()
        path = path.resolve()
        if path in visited:
            return []
        visited.add(path)

        lines: List[str] = []
        if not path.exists():
            raise FileNotFoundError(f"Requirements file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("-r "):
                    include = line.split(maxsplit=1)[1].strip()
                    include_path = (path.parent / include).resolve()
                    lines.extend(self._read_requirements_file(include_path, visited))
                    continue
                lines.append(line)
        return lines

    def create_requirements(self, gpu_type: str) -> Path:
        """Create platform-specific requirements file"""
        req_file = self.output_dir / "requirements-bundle.txt"

        requirements = []

        # Add torch with appropriate backend
        torch_index = self.get_torch_index(gpu_type)
        if gpu_type != "cpu" and self.platform_name != "darwin":
            requirements.append(f"--extra-index-url {torch_index}")

        # Windows wheel stacks: pin torch to match the wheel ABI expectations.
        # (FlashAttention wheel: cu126 + torch2.6.0 + cp312; SageAttention wheel: cu128 + torch2.7.1 + cp312)
        if self.platform_name == "win32" and gpu_type.startswith("cuda"):
            if gpu_type == "cuda128":
                requirements.extend(["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"])
            else:
                requirements.extend(["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"])
        else:
            requirements.extend(["torch", "torchvision", "torchaudio"])

        # Add machine-specific requirements (expanded entrypoint)
        machine_entry = self.choose_machine_requirements_entrypoint(gpu_type)
        machine_lines = self._read_requirements_file(machine_entry)
        for line in machine_lines:
            # Skip any torch packages (we add them above, with correct index/pins)
            lower = line.lower()
            if lower.startswith("torch") or lower.startswith("torchvision") or lower.startswith("torchaudio"):
                continue
            requirements.append(line)

        # Write requirements file
        with open(req_file, "w") as f:
            f.write("\n".join(requirements))

        print(f"Created requirements file: {req_file}")
        return req_file

    def create_venv(self, requirements_file: Path) -> Path:
        """
        Create a virtual environment with all dependencies.

        The venv directory is named `apex-studio` to make manual usage predictable:
          source apex-studio/bin/activate
          python -m src start ...
        """
        venv_dir = self.output_dir / self.venv_name

        if venv_dir.exists():
            print(f"Removing existing venv: {venv_dir}")
            shutil.rmtree(venv_dir)

        print(f"Creating virtual environment: {venv_dir}")
        uv = self._ensure_uv_available()
        # Use uv to create the venv (faster + consistent across platforms).
        subprocess.run(
            [uv, "venv", str(venv_dir), "--python", str(self.python_executable)],
            check=True,
        )
        
        # Get python path within the venv
        if self.platform_name == "win32":
            py_path = venv_dir / "Scripts" / "python.exe"
        else:
            py_path = venv_dir / "bin" / "python"

        # Install with uv into the venv we just created.
        uv_path = uv

        # Ensure standard build tooling exists inside the venv.
        # Some sdists / VCS installs (e.g. diffusers from git) can assume setuptools is present.
        subprocess.run(
            [
                str(uv_path),
                "pip",
                "install",
                "--python",
                str(py_path),
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            check=True,
        )

        # Pre-install build helpers that some legacy sdists forget to declare.
        # (Example: opendr/opendr-toolkit needs Cython/numpy at build time but doesn't specify them.)
        subprocess.run(
            [
                str(uv_path),
                "pip",
                "install",
                "--python",
                str(py_path),
                "Cython>=0.29.36",
                "numpy",
            ],
            check=True,
        )
        
        # Build & install Rust wheels (apex_download_rs) into the venv so they are
        # included in the final bundle.
        if not getattr(self, "skip_rust", False):
            self._build_and_install_rust_wheels(uv_path=uv_path, py_path=py_path)

        # Install requirements
        print(f"Installing requirements from: {requirements_file}")
        subprocess.run(
            [
                str(uv_path),
                "pip",
                "install",
                "--python",
                str(py_path),
                "--no-build-isolation",
                "-r",
                str(requirements_file),
            ],
            check=True,
        )

        # Optional: install Nunchaku wheels if the current (platform, python, torch) matches
        # an available prebuilt wheel on GitHub releases.
        self._maybe_install_nunchaku(uv_path=Path(str(uv_path)), py_path=py_path)

        # Install the project itself (so the venv has the `apex-engine` console script),
        # but we still ship `src/` alongside the venv and run `python -m src ...` in production.
        subprocess.run(
            [str(uv_path), "pip", "install", "--python", str(py_path), str(self.project_root)],
            check=True,
        )

        # PyInstaller hard-fails if the obsolete PyPI backport package `typing` is installed.
        # Some environments/dependency stacks can accidentally pull it in.
        # Remove it proactively (no-op if not installed).
        self._remove_obsolete_typing_backport(py_path=py_path)

        # macOS/conda: ensure libpython is present so the venv can run after being moved/installed.
        self._ensure_macos_libpython_present(venv_dir)

        return venv_dir

    def _maybe_install_nunchaku(self, uv_path: Path, py_path: Path) -> None:
        """
        Best-effort install for Nunchaku wheels.

        We *do not* include Nunchaku in the base requirements files because its wheels are
        built against specific PyTorch major/minor versions (e.g. +torch2.9). Installing
        the wrong wheel can hard-fail the whole bundle. Instead, we detect the *installed*
        torch version in the venv and install the matching wheel if one exists.

        Release: https://github.com/nunchaku-tech/nunchaku/releases/tag/v1.1.0
        """

        # Only CUDA-capable platforms; skip macOS/ROCm/CPU bundles.
        if self.platform_name not in ("linux", "win32"):
            return

        # Determine python tag (cp310/cp311/cp312/cp313) and torch major/minor.
        try:
            probe = subprocess.run(
                [
                    str(py_path),
                    "-c",
                    "import sys; "
                    "import torch; "
                    "v = (torch.__version__ or '').split('+')[0].strip(); "
                    "mm = '.'.join(v.split('.')[:2]) if v else ''; "
                    "print(f'cp{sys.version_info.major}{sys.version_info.minor}'); "
                    "print(mm)",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return

        lines = [ln.strip() for ln in probe.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            return
        py_tag = lines[0]  # e.g. cp312
        torch_mm = lines[1]  # e.g. 2.9

        # Supported wheel builds in v1.1.0
        supported_linux = {"2.7", "2.8", "2.9", "2.11"}
        supported_win = {"2.9", "2.11"}

        if self.platform_name == "linux":
            if torch_mm not in supported_linux:
                print(f"Skipping Nunchaku wheel: no v1.1.0 linux wheel for torch {torch_mm} ({py_tag})")
                return
            plat_suffix = "linux_x86_64"
        else:
            if torch_mm not in supported_win:
                print(f"Skipping Nunchaku wheel: no v1.1.0 windows wheel for torch {torch_mm} ({py_tag})")
                return
            plat_suffix = "win_amd64"

        # Build the exact filename/URL for the matching wheel.
        # Example asset:
        #   nunchaku-1.1.0+torch2.9-cp312-cp312-win_amd64.whl
        # Note: the '+' is URL-encoded as %2B in the GitHub release URLs.
        filename = f"nunchaku-1.1.0+torch{torch_mm}-{py_tag}-{py_tag}-{plat_suffix}.whl"
        url = f"https://github.com/nunchaku-tech/nunchaku/releases/download/v1.1.0/{filename.replace('+', '%2B')}"

        print(f"Attempting Nunchaku install: {filename}")
        try:
            subprocess.run(
                [str(uv_path), "pip", "install", "--python", str(py_path), url],
                check=False,
            )
        except Exception:
            # Best-effort: bundling should not fail because Nunchaku couldn't be installed.
            return

    def _remove_obsolete_typing_backport(self, py_path: Path) -> None:
        """
        PyInstaller error:
          "The 'typing' package is an obsolete backport of a standard library package and is incompatible with PyInstaller."

        This refers to the *PyPI distribution named `typing`*, not the stdlib module.
        """
        try:
            # `pip uninstall` returns non-zero when the package isn't installed; that's fine.
            subprocess.run(
                [str(py_path), "-m", "pip", "uninstall", "-y", "typing"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            # Bundling should not fail because we couldn't perform this hygiene step.
            pass

    def _patch_diffusers_set_adapter_scale(self, venv_dir: Path) -> None:
        """
        Patch diffusers in the bundled venv.

        We intentionally remove the try/except KeyError fallback around:
          scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]

        This ensures our shipped bundle matches the expected upstream behavior.
        """
        # Allow disabling in emergencies (e.g. while bisecting build issues).
        if os.environ.get("APEX_BUNDLE_PATCH_DIFFUSERS_PEFT", "1") == "0":
            print("Skipping diffusers peft.py patch (APEX_BUNDLE_PATCH_DIFFUSERS_PEFT=0)")
            return

        if self.platform_name == "win32":
            py_path = venv_dir / "Scripts" / "python.exe"
        else:
            py_path = venv_dir / "bin" / "python"

        # Resolve the installed diffusers path inside the venv.
        probe = subprocess.run(
            [
                str(py_path),
                "-c",
                "import diffusers; from pathlib import Path; "
                "base = Path(diffusers.__file__).resolve().parent; "
                "print(str(base / 'loaders' / 'peft.py'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        peft_path = Path(probe.stdout.strip())
        if not peft_path.exists():
            raise RuntimeError(f"diffusers peft.py not found at expected path: {peft_path}")

        src = peft_path.read_text(encoding="utf-8")

        # Match the try/except block exactly (indentation-aware).
        pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)try:\n"
            r"(?P=indent)[ \t]*scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING\[self\.__class__\.__name__\]\n"
            r"(?P=indent)[ \t]*except KeyError:\n"
            r"(?P=indent)[ \t]*scale_expansion_fn = lambda model, weights: weights[ \t]*$"
        )

        def _repl(m: re.Match) -> str:
            indent = m.group("indent")
            return f"{indent}scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]"

        patched, n = pattern.subn(_repl, src, count=1)
        if n != 1:
            # If the file already has the desired one-liner, treat as success/idempotent.
            if "_SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]" in src and "except KeyError" not in src:
                print(f"diffusers peft.py already patched: {peft_path}")
                return
            raise RuntimeError(
                "Failed to apply diffusers peft.py patch (pattern not found). "
                f"File: {peft_path}"
            )

        peft_path.write_text(patched, encoding="utf-8")
        print(f"Patched diffusers peft.py: {peft_path}")

    def _build_and_install_rust_wheels(self, uv_path: Path, py_path: Path) -> None:
        rust_project = self.project_root / "rust" / "apex_download_rs"
        if not rust_project.exists():
            print(f"Rust project not found, skipping: {rust_project}")
            return

        # Ensure toolchain exists
        try:
            subprocess.run(["cargo", "--version"], check=True, capture_output=True, text=True)
            subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True)
        except Exception as e:
            raise SystemExit(
                "Rust toolchain (cargo/rustc) is required to build apex_download_rs. "
                "Install Rust, or re-run with --skip-rust."
            ) from e

        wheels_dir = self.output_dir / "wheels"
        wheels_dir.mkdir(parents=True, exist_ok=True)

        # Install maturin into the venv (pyproject build backend is maturin).
        subprocess.run(
            [str(uv_path), "pip", "install", "--python", str(py_path), "maturin>=1.6,<2.0"],
            check=True,
        )

        # Build wheel for the exact interpreter we're bundling (typically Python 3.12)
        print(f"Building Rust wheel (maturin): {rust_project}")
        subprocess.run(
            [
                str(py_path),
                "-m",
                "maturin",
                "build",
                "--release",
                "--strip",
                "--interpreter",
                str(py_path),
                "--out",
                str(wheels_dir),
            ],
            cwd=str(rust_project),
            check=True,
        )

        built_wheels = sorted(wheels_dir.glob("*.whl"))
        if not built_wheels:
            raise RuntimeError(f"No wheels produced by maturin in: {wheels_dir}")

        # Install the newest wheel produced.
        wheel_path = built_wheels[-1]
        print(f"Installing Rust wheel: {wheel_path.name}")
        subprocess.run(
            [str(uv_path), "pip", "install", "--python", str(py_path), str(wheel_path)],
            check=True,
        )

    def _create_apex_engine_entrypoint(self, bundle_dir: Path) -> None:
        """
        Ensure the bundle has a stable `apex-engine` entrypoint at the bundle root.

        We intentionally do NOT rely on venv console-script shebangs (they embed absolute
        paths from build time). Instead, we run the venv Python interpreter explicitly.
        """
        if self.platform_name == "win32":
            # Best-effort: provide a batch wrapper for manual testing.
            # (Electron will invoke the venv python directly; this is mainly for terminal usage.)
            launcher = bundle_dir / "apex-engine.bat"
            content = r"""@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%
"%SCRIPT_DIR%\apex-studio\Scripts\python.exe" -m src %*
"""
            launcher.write_text(content, encoding="utf-8")
            return

        launcher = bundle_dir / "apex-engine"
        content = """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
exec "$SCRIPT_DIR/apex-studio/bin/python" -m src "$@"
"""
        launcher.write_text(content, encoding="utf-8")
        os.chmod(launcher, 0o755)

    def create_venv_bundle(self, venv_dir: Path) -> Path:
        """
        Create a self-contained bundle built around a venv (no PyInstaller).

        Layout (bundle_dir):
          - apex-studio/        # venv (shipped)
          - src/               # API code
          - assets/, manifest/, transformer_configs/, vae_configs/
          - gunicorn.conf.py
          - apex-engine        # launcher that runs venv python -m src ...
        """
        bundle_dir = self.dist_dir / "apex-engine"
        # Make this operation idempotent (re-running bundler should not fail).
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Copy venv (avoid symlinks for portability: we want a self-contained bundle).
        dest_venv = bundle_dir / self.venv_name
        shutil.copytree(venv_dir, dest_venv, symlinks=False)

        # Copy source code
        dest_src = bundle_dir / "src"
        shutil.copytree(self.src_dir, dest_src)

        # Copy assets and configs
        for folder in ["assets", "manifest", "transformer_configs", "vae_configs"]:
            src = self.project_root / folder
            if src.exists():
                shutil.copytree(src, bundle_dir / folder)

        # Copy gunicorn config
        shutil.copy(self.project_root / "gunicorn.conf.py", bundle_dir)

        # Create launcher scripts (for manual start-api testing)
        self._create_launcher(bundle_dir)
        # Create apex-engine entrypoint at bundle root (matches Electron/runtime expectations)
        self._create_apex_engine_entrypoint(bundle_dir)

        return bundle_dir

    def _create_launcher(self, bundle_dir: Path):
        """Create platform-specific launcher scripts"""
        if self.platform_name == "win32":
            launcher = bundle_dir / "start-api.bat"
            content = """@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%
set PATH=%SCRIPT_DIR%\\apex-studio\\Scripts;%PATH%
"%SCRIPT_DIR%\\apex-studio\\Scripts\\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
"""
        else:
            launcher = bundle_dir / "start-api.sh"
            content = """#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/apex-studio/bin:$PATH"
exec "$SCRIPT_DIR/apex-studio/bin/python" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
"""

        with open(launcher, "w") as f:
            f.write(content)

        if self.platform_name != "win32":
            os.chmod(launcher, 0o755)

    def sign_bundle(self, bundle_dir: Path):
        """Sign the bundle for distribution"""
        if not self.sign:
            return

        if self.platform_name == "darwin":
            identity = os.environ.get("APPLE_IDENTITY")
            if not identity:
                print("Warning: APPLE_IDENTITY not set, skipping code signing")
                return

            # Sign all .dylib and .so files
            for ext in ["*.dylib", "*.so", "*.app"]:
                for f in bundle_dir.rglob(ext):
                    print(f"Signing: {f}")
                    subprocess.run(
                        [
                            "codesign",
                            "--force",
                            "--deep",
                            "--sign",
                            identity,
                            "--options",
                            "runtime",
                            "--entitlements",
                            str(self.project_root / "entitlements.plist"),
                            str(f),
                        ],
                        check=True,
                    )

        elif self.platform_name == "win32":
            # Windows code signing
            cert_file = os.environ.get("WINDOWS_CERT_FILE")
            cert_pass = os.environ.get("WINDOWS_CERT_PASSWORD")

            if not cert_file:
                print("Warning: WINDOWS_CERT_FILE not set, skipping code signing")
                return

            for exe in bundle_dir.rglob("*.exe"):
                print(f"Signing: {exe}")
                subprocess.run(
                    [
                        "signtool",
                        "sign",
                        "/f",
                        cert_file,
                        "/p",
                        cert_pass,
                        "/tr",
                        "http://timestamp.digicert.com",
                        "/td",
                        "sha256",
                        "/fd",
                        "sha256",
                        str(exe),
                    ],
                    check=True,
                )

    def create_manifest(self, bundle_dir: Path, gpu_type: str) -> dict:
        """Create a manifest file with bundle information"""
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        manifest = {
            "version": "0.1.0",
            "platform": self.platform_name,
            "arch": platform.machine(),
            "gpu_support": gpu_type,
            "python_tag": py_tag,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "bundle_type": "venv",
            "venv_dirname": self.venv_name,
            "rust_wheels": not getattr(self, "skip_rust", False),
            "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            "signed": self.sign,
        }

        manifest_file = bundle_dir / "apex-engine-manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Created manifest: {manifest_file}")
        self.last_manifest = manifest
        return manifest

    def _safe_filename_component(self, value: str) -> str:
        """
        Convert arbitrary strings into something safe for filenames.
        Keeps: alnum, '.', '-', '_'
        """
        s = (value or "").strip()
        if not s:
            return "unknown"
        out = []
        for ch in s:
            if ch.isalnum() or ch in "._-":
                out.append(ch)
            elif ch.isspace():
                out.append("-")
            else:
                out.append("-")
        collapsed = re.sub(r"-{2,}", "-", "".join(out)).strip("-")
        return collapsed or "unknown"

    def default_zip_name(self, zip_scope: str) -> str:
        """
        Compute a descriptive default zip name based on the *actual* bundled config.
        """
        m = self.last_manifest or {}
        prefix = "python-api" if zip_scope == "python-api" else "apex-engine"
        version = self._safe_filename_component(str(m.get("version", "0.0.0")))
        plat = self._safe_filename_component(str(m.get("platform", self.platform_name)))
        arch = self._safe_filename_component(str(m.get("arch", platform.machine())))
        gpu = self._safe_filename_component(str(m.get("gpu_support", self.last_gpu_type or "unknown")))
        py_tag = self._safe_filename_component(
            str(m.get("python_tag", f"cp{sys.version_info.major}{sys.version_info.minor}"))
        )

        extras: List[str] = []
        if m.get("signed"):
            extras.append("signed")
        if not m.get("rust_wheels", True):
            extras.append("norust")

        extra = ("-" + "-".join(extras)) if extras else ""
        return f"{prefix}-{version}-{plat}-{arch}-{gpu}-{py_tag}{extra}.zip"

    def _should_exclude_from_zip(self, rel_path: Path) -> bool:
        """
        Exclude files that don't belong in release zips (noise / cache artifacts).

        Keep this conservative: the unzipped bundle should still be runnable.
        """
        parts = set(rel_path.parts)
        if "__pycache__" in parts:
            return True
        name = rel_path.name
        if name.endswith(".pyc") or name.endswith(".pyo"):
            return True
        if name == ".DS_Store":
            return True
        return False

    def _zip_dir(
        self,
        src_dir: Path,
        zip_path: Path,
        *,
        include_root_dir: bool = True,
    ) -> Path:
        """
        Create a zip of src_dir.

        - include_root_dir=True makes the zip contain a single top-level folder named like src_dir.name
          (recommended for release artifacts to avoid scattering files on unzip).
        - Best-effort preserve Unix permission bits for executables (important for `apex-engine`).
        """
        src_dir = src_dir.resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()

        arc_prefix = src_dir.name if include_root_dir else ""

        files: List[Tuple[Path, Path]] = []
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(src_dir)
            if self._should_exclude_from_zip(rel):
                continue
            files.append((p, rel))

        files.sort(key=lambda t: str(t[1]).replace("\\", "/"))

        with zipfile.ZipFile(
            zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as zf:
            for abs_path, rel in files:
                arcname = str(Path(arc_prefix) / rel) if arc_prefix else str(rel)
                arcname = arcname.replace("\\", "/")

                zi = zipfile.ZipInfo.from_file(abs_path, arcname=arcname)
                try:
                    st_mode = abs_path.stat().st_mode
                    # Preserve permission bits (e.g. executable flags) for Unix unzip tools.
                    zi.external_attr = (st_mode & 0xFFFF) << 16
                except Exception:
                    pass

                with open(abs_path, "rb") as f:
                    zf.writestr(zi, f.read())

        print(f"Created zip: {zip_path} (from {src_dir})")
        return zip_path

    def _tar_zst_dir(
        self,
        src_dir: Path,
        out_path: Path,
        *,
        include_root_dir: bool = True,
        level: int = 19,
    ) -> Path:
        """
        Create a .tar.zst of src_dir using system `tar` + `zstd`.

        Notes:
        - This avoids a huge intermediate `.tar` file by streaming tar -> zstd.
        - Requires `zstd` to be installed and available on PATH.
        """
        tar_bin = shutil.which("tar")
        zstd_bin = shutil.which("zstd")
        if not tar_bin:
            raise RuntimeError("Cannot create .tar.zst: `tar` not found on PATH.")
        if not zstd_bin:
            raise RuntimeError(
                "Cannot create .tar.zst: `zstd` not found on PATH. "
                "Install it (macOS: `brew install zstd`)."
            )

        src_dir = src_dir.resolve()
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

        # We keep the same exclude semantics as zip to avoid cache noise.
        excludes = [
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=*.pyo",
            "--exclude=.DS_Store",
        ]

        if include_root_dir:
            tar_cmd = [
                tar_bin,
                *excludes,
                "-C",
                str(src_dir.parent),
                "-cf",
                "-",
                str(src_dir.name),
            ]
        else:
            tar_cmd = [tar_bin, *excludes, "-C", str(src_dir), "-cf", "-", "."]

        # zstd reads from stdin and writes to file.
        # -T0 uses all cores.
        zstd_cmd = [
            zstd_bin,
            "-T0",
            f"-{int(level)}",
            "-o",
            str(out_path),
        ]

        print(f"Creating tar.zst: {out_path.name} (from {src_dir})")
        p1 = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            p2 = subprocess.run(
                zstd_cmd,
                stdin=p1.stdout,
                capture_output=True,
                text=True,
            )
        finally:
            # Ensure tar's stdout pipe is closed so tar can receive SIGPIPE if zstd exits early.
            try:
                if p1.stdout:
                    p1.stdout.close()
            except Exception:
                pass

        tar_stderr = ""
        try:
            _, tar_stderr = p1.communicate(timeout=120)
            tar_stderr = (tar_stderr or b"").decode("utf-8", errors="replace") if isinstance(tar_stderr, (bytes, bytearray)) else (tar_stderr or "")
        except Exception:
            # If tar hangs for any reason, terminate best-effort.
            try:
                p1.kill()
            except Exception:
                pass

        if p1.returncode not in (0, None):
            raise RuntimeError(f"`tar` failed (exit {p1.returncode}). stderr:\n{tar_stderr}")
        if p2.returncode != 0:
            raise RuntimeError(f"`zstd` failed (exit {p2.returncode}). stderr:\n{p2.stderr}")

        print(f"Created tar.zst: {out_path}")
        return out_path

    def bundle(self) -> Path:
        """Run the full bundling process"""
        print(f"Bundling Python API for platform: {self.platform_name}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect GPU support
        gpu_type = self.cuda_version or self.detect_gpu_support()
        self.last_gpu_type = gpu_type
        print(f"GPU support: {gpu_type}")

        # Create requirements
        req_file = self.create_requirements(gpu_type)

        # Create virtual environment
        venv_dir = self.create_venv(req_file)

        # Patch third-party deps inside the venv before we copy it into the shipped bundle.
        self._patch_diffusers_set_adapter_scale(venv_dir)
        
        # Build venv-based bundle (no PyInstaller)
        bundle_dir = self.create_venv_bundle(venv_dir)

        # IMPORTANT: avoid doubling output size.
        # We build the venv under `<output>/<venv_name>/` and then copy it into the shipped bundle.
        # If we keep both, release zips (or any packaging of the whole output dir) will include Python twice.
        if not getattr(self, "keep_build_venv", False):
            try:
                if venv_dir.exists():
                    shutil.rmtree(venv_dir)
            except Exception:
                # Best-effort: don't fail bundling just because cleanup didn't work.
                pass

        # Sign the bundle
        self.sign_bundle(bundle_dir)

        # Create manifest
        self.create_manifest(bundle_dir, gpu_type=gpu_type)

        print(f"Bundle created: {bundle_dir}")
        return bundle_dir


def main():
    parser = argparse.ArgumentParser(description="Bundle Python API for Apex Studio")
    parser.add_argument(
        "--platform",
        choices=["darwin", "linux", "win32", "auto"],
        default="auto",
        help="Target platform (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./dist"),
        help="Output directory for the bundle",
    )
    parser.add_argument(
        "--cuda",
        choices=["cuda128", "cuda126", "cuda124", "cuda121", "cuda118", "cpu", "rocm", "auto"],
        default="auto",
        help="CUDA version to bundle (default: auto-detect)",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help="Python interpreter to use for the venv (recommended: Python 3.12). Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--require-python312",
        action="store_true",
        help="Fail fast if the selected interpreter is not Python 3.12.x",
    )
    parser.add_argument(
        "--sign", action="store_true", help="Sign the bundle for distribution"
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip building/installing Rust wheels (apex_download_rs).",
    )
    parser.add_argument(
        "--keep-build-venv",
        action="store_true",
        help="Keep the intermediate build venv at <output>/apex-studio (default: delete it to avoid doubling bundle size).",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a release zip artifact after bundling (zip is written under --output unless --zip-output is provided).",
    )
    parser.add_argument(
        "--zip-scope",
        choices=["bundle", "python-api"],
        default="python-api",
        help="What to zip: 'bundle' zips the apex-engine folder; 'python-api' zips the python-api folder containing apex-engine.",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=None,
        help="Path to write the zip artifact (default: <output>/<zip-name>).",
    )
    parser.add_argument(
        "--zip-name",
        default=None,
        help="Zip filename to use when --zip-output is not provided (default: python-api-<platform>.zip or apex-engine-<platform>.zip).",
    )
    parser.add_argument(
        "--tar-zst",
        action="store_true",
        help="Create a release .tar.zst artifact after bundling (requires `zstd` on PATH).",
    )
    parser.add_argument(
        "--tar-zst-scope",
        choices=["bundle", "python-api"],
        default="python-api",
        help="What to tar.zst: 'bundle' tars the apex-engine folder; 'python-api' tars the python-api folder containing apex-engine.",
    )
    parser.add_argument(
        "--tar-zst-output",
        type=Path,
        default=None,
        help="Path to write the .tar.zst artifact (default: <output>/<tar-zst-name>).",
    )
    parser.add_argument(
        "--tar-zst-name",
        default=None,
        help="Filename for the .tar.zst when --tar-zst-output is not provided (default: same as zip naming but with .tar.zst).",
    )
    parser.add_argument(
        "--tar-zst-level",
        type=int,
        default=19,
        help="Zstd compression level (default: 19).",
    )

    args = parser.parse_args()

    # Auto-detect platform
    platform_name = args.platform
    if platform_name == "auto":
        platform_name = sys.platform

    # Auto-detect CUDA
    cuda_version = args.cuda if args.cuda != "auto" else None

    py_exe = args.python_executable or sys.executable
    if args.require_python312:
        if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
            # If they didn't pass --python, this is definitely wrong.
            # If they did pass --python, we can't inspect that interpreter without executing it,
            # so keep this a conservative guard for the common case.
            raise SystemExit(
                "Python 3.12.x is required. Re-run with a Python 3.12 interpreter or pass --python <path-to-python3.12>."
            )
    else:
        if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
            print(
                "Warning: bundler is not running under Python 3.12.x. "
                "For best compatibility with Windows wheels (cp312) and newer stacks, use Python 3.12."
            )

    bundler = PythonBundler(
        platform_name=platform_name,
        output_dir=args.output,
        cuda_version=cuda_version,
        sign=args.sign,
        python_executable=py_exe,
    )
    bundler.skip_rust = args.skip_rust
    bundler.keep_build_venv = args.keep_build_venv

    bundle_dir = bundler.bundle()

    if args.zip:
        if args.zip_scope == "bundle":
            zip_src = bundle_dir
        else:
            zip_src = bundler.dist_dir

        zip_name = args.zip_name or bundler.default_zip_name(args.zip_scope)
        zip_path = args.zip_output or (bundler.output_dir / zip_name)
        bundler._zip_dir(zip_src, zip_path, include_root_dir=True)

    if args.tar_zst:
        if args.tar_zst_scope == "bundle":
            tar_src = bundle_dir
        else:
            tar_src = bundler.dist_dir

        default_name = bundler.default_zip_name(args.tar_zst_scope).replace(".zip", ".tar.zst")
        tar_name = args.tar_zst_name or default_name
        tar_path = args.tar_zst_output or (bundler.output_dir / tar_name)
        bundler._tar_zst_dir(tar_src, tar_path, include_root_dir=True, level=args.tar_zst_level)

    print(f"\nBundle complete: {bundle_dir}")
    if platform_name == "win32":
        print(f"To test: cd {bundle_dir} && apex-engine.bat start --daemon --port 8765")
    else:
        print(f"To test: cd {bundle_dir} && ./apex-engine start --daemon --port 8765")


if __name__ == "__main__":
    main()
