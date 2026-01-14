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
import shlex
import shutil
import subprocess
import sys
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
        bundle_version: Optional[str] = None,
    ):
        self.platform_name = platform_name
        # Resolve to an absolute path so later subprocess calls that change cwd (e.g. Rust builds)
        # still can find the venv interpreter and other bundle files reliably.
        self.output_dir = Path(output_dir).resolve()
        self.cuda_version = cuda_version
        self.sign = sign
        self.python_executable = python_executable or sys.executable
        self.prefer_python_312 = prefer_python_312
        self.bundle_version = (bundle_version or "").strip() or None

        # This file lives at scripts/bundling/bundle_python.py.
        # Project root is apps/api/.
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.output_dir / "python-api"
        # Code-only update bundle output (no venv). Intended for frequent updates:
        # ship updated `src/` + small configs/assets + a requirements spec for dependency syncing.
        self.code_dist_dir = self.output_dir / "python-code"

        # Name of the venv directory we create & ship.
        # Users can activate it directly: `source apex-studio/bin/activate`
        self.venv_name = "apex-studio"

        # Populated after bundling
        self.last_gpu_type: Optional[str] = None
        self.last_machine_entry: Optional[Path] = None
        self.last_manifest: Optional[dict] = None
        self.last_code_manifest: Optional[dict] = None

        # Whether to keep the *intermediate* build venv at `<output>/<venv_name>/`.
        # The shipped bundle contains its own copy at `<output>/python-api/apex-engine/<venv_name>/`.
        self.keep_build_venv: bool = False
        # Whether to run smoke tests against the shipped bundle.
        # This is strongly recommended to avoid shipping broken bundles.
        self.run_smoke_tests: bool = True
        # If True, fail bundling when optional GPU-only smoke tests are skipped
        # due to missing CUDA at build time.
        self.smoke_tests_strict: bool = False

    def _bundle_python_path_env(self, bundle_dir: Path, env: dict) -> dict:
        """
        Ensure the bundle root is on PYTHONPATH, matching runtime launch behavior.
        """
        bundle_dir = Path(bundle_dir).resolve()
        env = dict(env or {})
        sep = ";" if self.platform_name == "win32" else ":"
        existing = env.get("PYTHONPATH", "")
        parts = [str(bundle_dir)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = sep.join(parts)
        # Provide an explicit separator hint for in-venv smoke scripts (useful if the
        # build host OS differs from the target platform).
        env["APEX_PYTHONPATH_SEP"] = sep
        return env

    def _smoke_py_path(self, bundle_dir: Path) -> Path:
        if self.platform_name == "win32":
            return bundle_dir / self.venv_name / "Scripts" / "python.exe"
        return bundle_dir / self.venv_name / "bin" / "python"

    def _run_smoke_tests(self, *, bundle_dir: Path, gpu_type: str) -> None:
        """
        Run smoke tests inside the *shipped* bundle venv.

        Goal: catch "imports fine locally, broken on user machines" failures early:
          - broken dependency graphs (`pip check`)
          - native extension import failures (missing shared libs)
          - manifest parse/import regressions
          - attention backend unusable despite being "installed"
          - optional Nunchaku presence sanity
        """
        bundle_dir = Path(bundle_dir).resolve()
        py_path = self._smoke_py_path(bundle_dir)
        if not py_path.exists():
            raise RuntimeError(f"Smoke tests: bundle venv python not found: {py_path}")

        env = self._bundle_python_path_env(bundle_dir, os.environ.copy())
        env["APEX_BUNDLE_SMOKE_TEST"] = "1"
        env["APEX_BUNDLE_GPU_TYPE"] = str(gpu_type or "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        print("\n--- Running bundle smoke tests (inside shipped venv) ---")
        print(f"Bundle dir: {bundle_dir}")
        print(f"Python: {py_path}")

        smoke_runner = bundle_dir / "scripts" / "smoke_tests" / "run_all.py"
        if not smoke_runner.exists():
            raise RuntimeError(f"Smoke tests runner not found in bundle: {smoke_runner}")

        args = [
            str(py_path),
            str(smoke_runner),
            "--bundle-root",
            str(bundle_dir),
            "--gpu-type",
            str(gpu_type or ""),
        ]
        if self.smoke_tests_strict:
            args.append("--strict-gpu")

        res = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(bundle_dir),
        )
        if res.stdout:
            print(res.stdout)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            raise RuntimeError(f"Bundle smoke tests failed (exit {res.returncode}). stderr:\n{stderr}")

        # If building a CUDA bundle but CUDA isn't available on the build machine, optionally fail fast.
        if self.smoke_tests_strict and str(gpu_type or "").startswith("cuda"):
            # The in-venv harness will report cuda_available=False; enforce here for clarity.
            try:
                probe = subprocess.run(
                    [str(py_path), "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=str(bundle_dir),
                )
                if probe.returncode != 0:
                    raise RuntimeError(
                        "Smoke tests strict mode: building a CUDA bundle but CUDA was not available "
                        "on the build machine, so GPU kernel smoke tests could not run."
                    )
            except Exception:
                raise

        print("--- Bundle smoke tests passed ---\n")

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
        Choose an entrypoint under requirements/{cpu,mps,cuda} based on platform + GPU/arch.
        """
        req_root = self.project_root / "requirements"

        # macOS / Apple Silicon
        if self.platform_name == "darwin":
            return req_root / "mps" / "requirements.txt"

        # ROCm
        if gpu_type == "rocm":
            if self.platform_name == "win32":
                return req_root / "rocm" / "windows.txt"
            return req_root / "rocm" / "linux.txt"

        # CPU
        if gpu_type == "cpu":
            return req_root / "cpu" / "requirements.txt"

        # Windows CUDA: select from the standard CUDA entrypoints (they include
        # Windows-specific wheels/pins via sys_platform markers).
        if self.platform_name == "win32" and gpu_type.startswith("cuda"):
            # Universal Windows Bundle
            return req_root / "cuda" / "windows.txt"

        # Linux CUDA: pick by compute capability
        if gpu_type.startswith("cuda"):
            # Universal Linux Bundle
            return req_root / "cuda" / "linux.txt"

        # Final fallback
        return req_root / "cpu" / "requirements.txt"

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

        # Choose machine-specific requirements entrypoint early so we can apply
        # correct Windows wheel-stack torch pins (they depend on the chosen entry).
        machine_entry = self.choose_machine_requirements_entrypoint(gpu_type)
        self.last_machine_entry = machine_entry

        # Add torch with appropriate backend
        torch_index = self.get_torch_index(gpu_type)
        if gpu_type == "rocm" and self.platform_name == "win32":
            # Windows ROCm: Use specific community wheels (TheRock)
            pass  # We append the direct URLs below
        elif gpu_type != "cpu" and self.platform_name != "darwin":
            requirements.append(f"--extra-index-url {torch_index}")

        # Windows wheel stacks: pin torch to match the wheel ABI expectations.
        # (FlashAttention wheel: cu126 + torch2.6.0 + cp312; cu128 stacks vary by machine entry.)
        if self.platform_name == "win32":
            if gpu_type.startswith("cuda"):
                if gpu_type == "cuda128":
                    # Unified Stack: CUDA 12.8 + Torch 2.7.1
                    # This covers Ampere, Ada, Hopper, Blackwell with FA2/FA3 support.
                    requirements.extend(
                        ["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"]
                    )
                else:
                    requirements.extend(["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"])
            elif gpu_type == "rocm":
                # ROCm 6.5 Windows (TheRock)
                requirements.extend([
                    "torch @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl",
                    "torchvision @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl",
                    "torchaudio @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl",
                ])
            else:
                requirements.extend(["torch", "torchvision", "torchaudio"])
        elif self.platform_name == "darwin":
            # MPS requires Torch 2.7.1+
            requirements.extend(["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"])
        else:
            requirements.extend(["torch", "torchvision", "torchaudio"])

        # Add machine-specific requirements (expanded entrypoint)
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

    def create_lockfile(self, *, requirements_file: Path, venv_dir: Path) -> Path:
        """
        Create a deterministic lockfile (`requirements.lock`) for update-time syncing.

        We build the lockfile by:
          - preserving resolver/index options from the generated requirements file
            (e.g. --extra-index-url for CUDA PyTorch wheels),
          - freezing the *actual* installed environment via `uv pip freeze`,
          - filtering out local-path / editable artifacts that would be un-installable on end user machines
            (notably our own `apex-engine` install done from a local directory during bundling).

        The resulting file is suitable for:
          - `uv pip sync -p <venv_python> requirements.lock`
        """
        lock_path = self.output_dir / "requirements.lock"

        # Determine venv python
        if self.platform_name == "win32":
            py_path = venv_dir / "Scripts" / "python.exe"
        else:
            py_path = venv_dir / "bin" / "python"

        uv = self._ensure_uv_available()

        # Capture any pip option lines from the requirements file (indexes, find-links, etc.).
        header_lines: list[str] = []
        try:
            for raw in requirements_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("--"):
                    header_lines.append(line)
        except Exception:
            header_lines = []

        # Freeze installed packages from the venv.
        res = subprocess.run(
            [str(uv), "pip", "freeze", "--python", str(py_path), "--exclude-editable"],
            check=True,
            capture_output=True,
            text=True,
        )
        frozen = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]

        def _keep(line: str) -> bool:
            lower = line.lower()
            # Exclude our own package, since it's installed from a local path during bundling
            # and may not be publishable on an index.
            if lower.startswith("apex-engine"):
                return False
            # Exclude local file references (un-installable on end-user machines).
            # Example: package @ file:///... or package @ /abs/path
            if " @ file:" in lower or lower.startswith("-e "):
                return False
            return True

        filtered = [ln for ln in frozen if _keep(ln)]

        out_lines: list[str] = []
        out_lines.extend(header_lines)
        if header_lines:
            out_lines.append("")  # spacer
        out_lines.extend(filtered)

        lock_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"Created lockfile: {lock_path}")
        return lock_path

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
                "psutil",
                "enscons",
                "pytoml",
            ],
            check=True,
        )

        # Pre-install torch *before* installing the full requirements set.
        #
        # Why: some packages (notably `sageattention` from git) import `torch` at
        # build/metadata time, but do not declare it as a build dependency. `uv pip`
        # (like pip) can evaluate/build metadata for requirements before installing
        # anything, so `torch` must already exist in the environment to avoid:
        #   ModuleNotFoundError: No module named 'torch'
        #
        # We derive both the index options and the torch package specs from the
        # generated requirements file so CUDA/CPU + Windows pins stay consistent.
        try:
            req_lines = [
                ln.strip()
                for ln in requirements_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            option_args: list[str] = []
            for ln in req_lines:
                if ln.startswith("--"):
                    # Requirements files store options as a single line like:
                    #   --extra-index-url https://...
                    # but subprocess argv must be split into separate args.
                    option_args.extend(shlex.split(ln))
            torch_specs: list[str] = []
            for ln in req_lines:
                lower = ln.lower()
                if lower.startswith("torch") or lower.startswith("torchvision") or lower.startswith("torchaudio"):
                    # Exclude stray option lines, which also start with '-' in requirements.
                    if ln.startswith("--"):
                        continue
                    torch_specs.append(ln)
            # Only run if torch is part of this bundle's requirements.
            if torch_specs:
                subprocess.run(
                    [
                        str(uv_path),
                        "pip",
                        "install",
                        "--python",
                        str(py_path),
                        "--no-build-isolation",
                        "--index-strategy",
                        "unsafe-best-match",
                        *option_args,
                        *torch_specs,
                    ],
                    check=True,
                )
        except Exception:
            # Best-effort: don't fail bundling solely due to this preinstall step.
            pass
        
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
                "--index-strategy",
                "unsafe-best-match",
                "-r",
                str(requirements_file),
            ],
            check=True,
        )

        # Optional: Nunchaku (CUDA-only) — install separately and default to --no-deps.
        # This prevents resolver churn from nunchaku's dependency metadata while still
        # enabling nunchaku-backed models on supported platforms.
        try:
            if self.last_machine_entry and str(self.last_machine_entry).replace("\\", "/").endswith("/requirements/cuda/linux.txt"):
                subprocess.run(
                    [
                        str(py_path),
                        str(self.project_root / "scripts" / "deps" / "maybe_install_nunchaku.py"),
                        "--python",
                        str(py_path),
                        "--machine-entry-name",
                        str(self.last_machine_entry.name),
                        "--install",
                    ],
                    check=False,
                )
            elif self.last_machine_entry and str(self.last_machine_entry).replace("\\", "/").endswith("/requirements/cuda/windows.txt"):
                subprocess.run(
                    [
                        str(py_path),
                        str(self.project_root / "scripts" / "deps" / "maybe_install_nunchaku.py"),
                        "--python",
                        str(py_path),
                        "--machine-entry-name",
                        str(self.last_machine_entry.name),
                        "--install",
                    ],
                    check=False,
                )
        except Exception:
            # Best-effort; do not fail bundling.
            pass

        # Install local universal wheels if available (overrides requirements)
        # We look for wheels in: apps/api/<pkg>/wheelhouse/universal/*.whl
        local_wheel_dirs = [
            self.project_root / "flash-attention" / "wheelhouse" / "universal",
            self.project_root / "SageAttention" / "wheelhouse" / "universal",
        ]
        local_wheels = []
        for wd in local_wheel_dirs:
            if wd.exists():
                for whl in wd.glob("*.whl"):
                    local_wheels.append(whl)
        
        if local_wheels:
            print(f"Installing local universal wheels: {[w.name for w in local_wheels]}")
            subprocess.run(
                [
                    str(uv_path),
                    "pip",
                    "install",
                    "--python",
                    str(py_path),
                    "--force-reinstall",
                    "--no-deps",
                    *[str(w) for w in local_wheels],
                ],
                check=True,
            )

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

        # Final hygiene: remove cache/artifact files that should never ship in bundles.
        # (Best-effort; don't fail bundling if cleanup has issues.)
        try:
            removed_files, removed_dirs = self._cleanup_bundle_tree(venv_dir)
            if removed_files or removed_dirs:
                print(f"Cleaned venv artifacts: removed_files={removed_files}, removed_dirs={removed_dirs}")
        except Exception:
            pass

        return venv_dir

    def _cleanup_bundle_tree(self, root: Path) -> tuple[int, int]:
        """
        Remove filesystem junk / cache artifacts that should not be shipped.

        This is intentionally defensive: even if copy/packaging excludes are configured,
        these files can sneak in via platform-specific tooling and can break imports
        (notably AppleDouble files like `._*.py` on macOS).
        """
        root = Path(root).resolve()
        removed_files = 0
        removed_dirs = 0

        # Remove directories first (deepest-first) to avoid traversing deleted trees.
        dir_names = {"__pycache__", "__MACOSX"}
        dirs: list[Path] = []
        try:
            for p in root.rglob("*"):
                try:
                    if p.is_dir() and p.name in dir_names:
                        dirs.append(p)
                except Exception:
                    continue
        except Exception:
            dirs = []

        for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                shutil.rmtree(d, ignore_errors=True)
                removed_dirs += 1
            except Exception:
                pass

        # Remove files (AppleDouble, Finder junk, bytecode).
        try:
            for p in root.rglob("*"):
                try:
                    if not p.is_file():
                        continue
                    name = p.name
                    if name == ".DS_Store" or name.startswith("._") or name.endswith((".pyc", ".pyo")):
                        try:
                            p.unlink(missing_ok=True)  # py3.8+
                        except TypeError:
                            # older python fallback
                            if p.exists():
                                p.unlink()
                        removed_files += 1
                except Exception:
                    continue
        except Exception:
            pass

        return removed_files, removed_dirs

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

        # Build wheels into a Python-ABI-specific directory so switching between
        # Python versions (e.g. cp312 vs cp313) cannot accidentally install an
        # incompatible stale wheel from a previous run.
        try:
            probe = subprocess.run(
                [str(py_path), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
                check=True,
                capture_output=True,
                text=True,
            )
            py_tag = (probe.stdout or "").strip() or "unknown"
        except Exception:
            py_tag = "unknown"

        wheels_dir = self.output_dir / "wheels" / py_tag
        if wheels_dir.exists():
            # Make this step idempotent and avoid picking up old wheels.
            for p in wheels_dir.glob("*.whl"):
                try:
                    p.unlink()
                except Exception:
                    pass
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

        built_wheels = sorted(wheels_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
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

    def _choose_venv_libdir_to_ignore(self, venv_dir: Path) -> Optional[str]:
        """
        Linux venvs often contain both `lib/` and `lib64/`, where one is a symlink to the other.
        Because we copy the venv with `symlinks=False` (dereference), we would otherwise duplicate
        the same files in the shipped bundle. We keep the "real" directory and ignore the other.
        """
        lib = venv_dir / "lib"
        lib64 = venv_dir / "lib64"
        if not lib.exists() or not lib64.exists():
            return None

        try:
            lib_is_link = lib.is_symlink()
            lib64_is_link = lib64.is_symlink()
        except Exception:
            lib_is_link = False
            lib64_is_link = False

        # Prefer keeping the non-symlink directory.
        if lib64_is_link and not lib_is_link:
            return "lib64"
        if lib_is_link and not lib64_is_link:
            return "lib"

        # If both are symlinks or both are real dirs, fall back to keeping `lib/`.
        return "lib64"

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

        # Avoid shipping filesystem metadata / cache artifacts into the bundle.
        # - macOS: AppleDouble files like `._*.py` can crash libraries that scan `*.py` (e.g. transformers).
        # - Python: `__pycache__` / `*.pyc` / `*.pyo` are noisy and not needed in release bundles.
        ignore_junk = shutil.ignore_patterns(
            ".DS_Store",
            "._*",
            "__MACOSX",
            "__pycache__",
            "*.pyc",
            "*.pyo",
        )

        # Copy venv (avoid symlinks for portability: we want a self-contained bundle).
        dest_venv = bundle_dir / self.venv_name
        ignore_libdir = self._choose_venv_libdir_to_ignore(venv_dir)

        def _ignore_with_libdir(src: str, names: list[str]) -> set[str]:
            ignored = set(ignore_junk(src, names) or [])
            # Only apply the lib/lib64 exclusion at the venv root.
            try:
                if ignore_libdir and Path(src).resolve() == Path(venv_dir).resolve():
                    ignored.add(ignore_libdir)
            except Exception:
                pass
            return ignored

        shutil.copytree(venv_dir, dest_venv, symlinks=False, ignore=_ignore_with_libdir)

        # Copy source code
        dest_src = bundle_dir / "src"
        shutil.copytree(self.src_dir, dest_src, ignore=ignore_junk)

        # Copy assets and configs
        for folder in ["assets", "manifest"]:
            src = self.project_root / folder
            if src.exists():
                shutil.copytree(src, bundle_dir / folder, ignore=ignore_junk)

        # Copy gunicorn config
        shutil.copy(self.project_root / "gunicorn.conf.py", bundle_dir)

        # Copy maintenance/update scripts into the bundle so installs can self-update without
        # needing access to the repo checkout.
        scripts_dir = bundle_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        # Copy known helper scripts.
        #
        # IMPORTANT: the repo keeps compatibility wrappers at scripts/*.py, but bundles must
        # ship the real implementations (so they work standalone after extraction).
        helper_scripts: list[tuple[Path, str]] = [
            (self.project_root / "scripts" / "updates" / "apply_code_update.py", "apply_code_update.py"),
            (self.project_root / "scripts" / "setup" / "setup.py", "setup.py"),
        ]
        for src, dest_name in helper_scripts:
            try:
                if src.exists():
                    shutil.copy(src, scripts_dir / dest_name)
            except Exception:
                pass

        # Copy smoke tests (directory) for post-install diagnostics and for bundling-time execution.
        try:
            smoke_src = self.project_root / "scripts" / "smoke_tests"
            smoke_dst = scripts_dir / "smoke_tests"
            if smoke_src.exists() and smoke_src.is_dir():
                if smoke_dst.exists():
                    shutil.rmtree(smoke_dst)
                shutil.copytree(smoke_src, smoke_dst, ignore=ignore_junk)
        except Exception:
            pass

        # Create launcher scripts (for manual start-api testing)
        self._create_launcher(bundle_dir)
        # Create apex-engine entrypoint at bundle root (matches Electron/runtime expectations)
        self._create_apex_engine_entrypoint(bundle_dir)

        # Defensive cleanup (in case platform tooling injected artifacts after the copy).
        try:
            self._cleanup_bundle_tree(bundle_dir)
        except Exception:
            pass

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
        version = self.bundle_version or "0.0.0"
        manifest = {
            "version": version,
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
        if zip_scope == "python-api":
            prefix = "python-api"
        elif zip_scope == "python-code":
            prefix = "python-code"
        else:
            prefix = "apex-engine"
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

    def _write_code_update_manifest(self, bundle_dir: Path, gpu_type: str, requirements_file: Path) -> dict:
        """
        Create a small manifest describing the code-only update bundle.

        The goal is to give the installer/updater enough metadata to:
          - validate platform/arch/gpu/python compatibility,
          - decide whether dependency syncing is needed (based on requirements file hash),
          - display version info.
        """
        import hashlib

        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        req_bytes = requirements_file.read_bytes() if requirements_file.exists() else b""
        req_sha256 = hashlib.sha256(req_bytes).hexdigest() if req_bytes else ""

        lock_file = self.output_dir / "requirements.lock"
        lock_bytes = lock_file.read_bytes() if lock_file.exists() else b""
        lock_sha256 = hashlib.sha256(lock_bytes).hexdigest() if lock_bytes else ""

        base = self.last_manifest or {}
        manifest = {
            "kind": "code-update",
            # Keep versioning consistent with the full bundle when available.
            "version": str(base.get("version", "0.1.0")),
            "platform": self.platform_name,
            "arch": platform.machine(),
            "gpu_support": gpu_type,
            "python_tag": py_tag,
            "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            "requirements": {
                "filename": requirements_file.name,
                "sha256": req_sha256,
            },
            "lockfile": (
                {
                    "filename": lock_file.name,
                    "sha256": lock_sha256,
                }
                if lock_sha256
                else None
            ),
        }

        manifest_file = bundle_dir / "apex-code-update-manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self.last_code_manifest = manifest
        print(f"Created code update manifest: {manifest_file}")
        return manifest

    def create_code_only_bundle(self, requirements_file: Path, gpu_type: str) -> Path:
        """
        Create a code-only bundle intended for incremental updates.

        Layout (code_bundle_dir):
          - src/               # API code
          - assets/, manifest/ # small static assets/configs
          - gunicorn.conf.py
          - requirements-bundle.txt (or whichever file was generated)
          - apex-code-update-manifest.json

        NOTE: This bundle intentionally does NOT contain the venv. The updater is expected
        to apply this bundle on top of an existing installed runtime env and optionally
        sync dependencies if the requirements file changed.
        """
        code_bundle_root = self.code_dist_dir
        bundle_dir = code_bundle_root / "apex-engine"

        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        ignore_junk = shutil.ignore_patterns(
            ".DS_Store",
            "._*",
            "__MACOSX",
            "__pycache__",
            "*.pyc",
            "*.pyo",
        )

        # Copy source code
        dest_src = bundle_dir / "src"
        shutil.copytree(self.src_dir, dest_src, ignore=ignore_junk)

        # Copy assets/configs (match the full bundle semantics today)
        for folder in ["assets", "manifest"]:
            src = self.project_root / folder
            if src.exists():
                shutil.copytree(src, bundle_dir / folder, ignore=ignore_junk)

        # Copy gunicorn config
        try:
            shutil.copy(self.project_root / "gunicorn.conf.py", bundle_dir)
        except Exception:
            pass

        # Copy maintenance/update scripts (no venv in this bundle, but we want the updater/setup scripts
        # available alongside the updated code when extracted).
        try:
            scripts_dir = bundle_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            helper_scripts: list[tuple[Path, str]] = [
                (self.project_root / "scripts" / "updates" / "apply_code_update.py", "apply_code_update.py"),
                (self.project_root / "scripts" / "setup" / "setup.py", "setup.py"),
            ]
            for src, dest_name in helper_scripts:
                if src.exists():
                    shutil.copy(src, scripts_dir / dest_name)

            # Copy smoke tests (directory) for update bundles as well.
            smoke_src = self.project_root / "scripts" / "smoke_tests"
            smoke_dst = scripts_dir / "smoke_tests"
            if smoke_src.exists() and smoke_src.is_dir():
                if smoke_dst.exists():
                    shutil.rmtree(smoke_dst)
                shutil.copytree(smoke_src, smoke_dst, ignore=ignore_junk)
        except Exception:
            pass

        # Copy the generated requirements spec into the bundle root for update-time comparison.
        try:
            if requirements_file.exists():
                shutil.copy(requirements_file, bundle_dir / requirements_file.name)
        except Exception:
            pass

        # Copy lockfile (if present)
        try:
            lock_file = self.output_dir / "requirements.lock"
            if lock_file.exists():
                shutil.copy(lock_file, bundle_dir / lock_file.name)
        except Exception:
            pass

        # Create a small manifest that includes a hash of the requirements spec.
        self._write_code_update_manifest(bundle_dir=bundle_dir, gpu_type=gpu_type, requirements_file=requirements_file)

        # Defensive cleanup (code-only bundles should also be free of cache/artifact files).
        try:
            self._cleanup_bundle_tree(bundle_dir)
        except Exception:
            pass

        print(f"Code-only bundle created: {bundle_dir}")
        return bundle_dir

    def _should_exclude_from_zip(self, rel_path: Path) -> bool:
        """
        Exclude files that don't belong in release zips (noise / cache artifacts).

        Keep this conservative: the unzipped bundle should still be runnable.
        """
        parts = set(rel_path.parts)
        if "__pycache__" in parts:
            return True
        if "__MACOSX" in parts:
            return True
        name = rel_path.name
        if name.endswith(".pyc") or name.endswith(".pyo"):
            return True
        if name == ".DS_Store":
            return True
        # AppleDouble files (Finder/zip/tar metadata). These can masquerade as .py files and break imports.
        if name.startswith("._"):
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
            "--exclude=*/__pycache__",
            "--exclude=*.pyc",
            "--exclude=*.pyo",
            "--exclude=.DS_Store",
            "--exclude=__MACOSX",
            "--exclude=*/__MACOSX",
            "--exclude=._*",
            "--exclude=*/._*",
        ]
        
        env = os.environ.copy()
        env["COPYFILE_DISABLE"] = "1"

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
            tar_cmd = [
                tar_bin,
                *excludes,
                "-C",
                str(src_dir),
                "-cf",
                "-",
                ".",
            ]

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
        p1 = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            p2 = subprocess.run(
                zstd_cmd,
                stdin=p1.stdout,
                capture_output=True,
                text=True,
                env=env,
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
        """Run the full bundling process (full env bundle + code-only update bundle)."""
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
        # Patch xformers (if installed) in the venv, by resolving the installed module
        # path and editing in-place (avoids applying a broad patches directory).
        if self.platform_name == "win32":
            py_path = venv_dir / "Scripts" / "python.exe"
        else:
            py_path = venv_dir / "bin" / "python"
        subprocess.run(
            [
                str(py_path),
                str(self.project_root / "scripts" / "updates" / "patch_xformers_flash3.py"),
            ],
            check=True,
        )

        # Create a lockfile for update-time dependency syncing.
        # Do this after all installs/patches so the lock reflects the final environment.
        try:
            self.create_lockfile(requirements_file=req_file, venv_dir=venv_dir)
        except Exception as e:
            print(f"Warning: failed to create requirements.lock: {e}")
        
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

        # Smoke test the shipped bundle (strongly recommended).
        smoke_env = os.environ.get("APEX_BUNDLE_SMOKE_TESTS")
        if self.run_smoke_tests and (smoke_env is None or str(smoke_env).strip().lower() not in ("0", "false", "no", "off")):
            self._run_smoke_tests(bundle_dir=bundle_dir, gpu_type=gpu_type)

        # Sign the bundle
        self.sign_bundle(bundle_dir)

        # Create manifest
        self.create_manifest(bundle_dir, gpu_type=gpu_type)

        # Create code-only update bundle (small; used for frequent updates)
        try:
            self.create_code_only_bundle(requirements_file=req_file, gpu_type=gpu_type)
        except Exception as e:
            # Best-effort: code-only bundle should not break full bundle creation.
            print(f"Warning: failed to create code-only update bundle: {e}")

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
        "--bundle-version",
        default=None,
        help=(
            "Bundle version to write into manifests and artifact names. "
            "Precedence: --bundle-version > APEX_BUNDLE_VERSION env var > pyproject.toml [project].version."
        ),
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
        "--skip-smoke-tests",
        action="store_true",
        help="Skip running smoke tests against the shipped bundle venv (NOT recommended).",
    )
    parser.add_argument(
        "--smoke-tests-strict",
        action="store_true",
        help="Fail if building a CUDA bundle but CUDA is not available to run GPU kernel smoke tests.",
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
        choices=["bundle", "python-api", "python-code"],
        default="python-api",
        help="What to zip: 'bundle' zips the apex-engine folder; 'python-api' zips the python-api folder containing apex-engine; 'python-code' zips the python-code folder containing apex-engine (code-only update bundle).",
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
        choices=["bundle", "python-api", "python-code"],
        default="python-api",
        help="What to tar.zst: 'bundle' tars the apex-engine folder; 'python-api' tars the python-api folder containing apex-engine; 'python-code' tars the python-code folder containing apex-engine (code-only update bundle).",
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

    def _read_project_version_from_pyproject(pyproject_path: Path) -> Optional[str]:
        """
        Best-effort parse of `pyproject.toml` to read `[project].version`.
        Prefers stdlib `tomllib` (py3.11+), falls back to `tomli` if installed.
        """
        try:
            data = pyproject_path.read_bytes()
        except Exception:
            return None

        toml_loads = None
        try:
            import tomllib  # type: ignore

            toml_loads = tomllib.loads
        except Exception:
            try:
                import tomli  # type: ignore

                toml_loads = tomli.loads
            except Exception:
                toml_loads = None

        if not toml_loads:
            return None

        try:
            obj = toml_loads(data.decode("utf-8"))
            v = obj.get("project", {}).get("version")
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            return None
        return None

    # Auto-detect platform
    platform_name = args.platform
    if platform_name == "auto":
        platform_name = sys.platform

    # Auto-detect CUDA
    cuda_version = args.cuda if args.cuda != "auto" else None
 
    py_exe = args.python_executable or sys.executable
    # If the caller didn't specify a venv interpreter, prefer Python 3.12 when available.
    # Many binary deps in our stack (e.g. ray) do not provide cp313 wheels yet.
    if args.python_executable is None and not (sys.version_info.major == 3 and sys.version_info.minor == 12):
        py312 = shutil.which("python3.12")
        if py312:
            py_exe = py312
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

    # Resolve bundle version (CLI > env > pyproject).
    bundle_version = (args.bundle_version or "").strip() or None
    if not bundle_version:
        bundle_version = (os.environ.get("APEX_BUNDLE_VERSION", "") or "").strip() or None
    if not bundle_version:
        bundle_version = _read_project_version_from_pyproject(
            Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        )

    bundler = PythonBundler(
        platform_name=platform_name,
        output_dir=args.output,
        cuda_version=cuda_version,
        sign=args.sign,
        python_executable=py_exe,
        bundle_version=bundle_version,
    )
    bundler.skip_rust = args.skip_rust
    bundler.keep_build_venv = args.keep_build_venv
    bundler.run_smoke_tests = not args.skip_smoke_tests
    bundler.smoke_tests_strict = bool(args.smoke_tests_strict)

    bundle_dir = bundler.bundle()

    if args.zip:
        if args.zip_scope == "bundle":
            zip_src = bundle_dir
        elif args.zip_scope == "python-code":
            zip_src = bundler.code_dist_dir
        else:
            zip_src = bundler.dist_dir

        zip_name = args.zip_name or bundler.default_zip_name(args.zip_scope)
        zip_path = args.zip_output or (bundler.output_dir / zip_name)
        bundler._zip_dir(zip_src, zip_path, include_root_dir=True)

        # If the caller zipped the full python-api artifact, also emit a code-only zip by default.
        if args.zip_scope == "python-api" and bundler.code_dist_dir.exists():
            code_zip_name = bundler.default_zip_name("python-code")
            code_zip_path = bundler.output_dir / code_zip_name
            bundler._zip_dir(bundler.code_dist_dir, code_zip_path, include_root_dir=True)

    if args.tar_zst:
        if args.tar_zst_scope == "bundle":
            tar_src = bundle_dir
        elif args.tar_zst_scope == "python-code":
            tar_src = bundler.code_dist_dir
        else:
            tar_src = bundler.dist_dir

        default_name = bundler.default_zip_name(args.tar_zst_scope).replace(".zip", ".tar.zst")
        tar_name = args.tar_zst_name or default_name
        tar_path = args.tar_zst_output or (bundler.output_dir / tar_name)
        bundler._tar_zst_dir(tar_src, tar_path, include_root_dir=True, level=args.tar_zst_level)

        # If the caller created the full python-api tar.zst artifact, also emit a code-only tar.zst by default.
        if args.tar_zst_scope == "python-api" and bundler.code_dist_dir.exists():
            code_tar_name = bundler.default_zip_name("python-code").replace(".zip", ".tar.zst")
            code_tar_path = bundler.output_dir / code_tar_name
            bundler._tar_zst_dir(bundler.code_dist_dir, code_tar_path, include_root_dir=True, level=args.tar_zst_level)

    print(f"\nBundle complete: {bundle_dir}")
    if platform_name == "win32":
        print(f"To test: cd {bundle_dir} && apex-engine.bat start --daemon --port 8765")
    else:
        print(f"To test: cd {bundle_dir} && ./apex-engine start --daemon --port 8765")


if __name__ == "__main__":
    main()
