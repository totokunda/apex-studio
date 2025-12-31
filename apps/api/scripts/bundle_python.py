#!/usr/bin/env python3
"""
Python API Bundler for Apex Studio

This script handles bundling the Python API for distribution with the Electron app.
It supports multiple platforms and handles PyTorch/CUDA dependencies intelligently.

Usage:
    python scripts/bundle_python.py --platform [darwin|linux|win32] --output ./dist
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Set, List


class PythonBundler:
    """Bundles Python API for distribution"""

    SUPPORTED_PLATFORMS = ["darwin", "linux", "win32"]

    # PyTorch wheel indices for different platforms
    TORCH_INDICES = {
        "cuda126": "https://download.pytorch.org/whl/cu126",
        "cuda124": "https://download.pytorch.org/whl/cu124",
        "cuda121": "https://download.pytorch.org/whl/cu121",
        "cuda118": "https://download.pytorch.org/whl/cu118",
        # Newer stacks (used by some Windows wheels / newer GPUs)
        "cuda128": "https://download.pytorch.org/whl/cu128",
        "cpu": "https://download.pytorch.org/whl/cpu",
        "rocm": "https://download.pytorch.org/whl/rocm6.4",
    }

    def __init__(
        self,
        platform_name: str,
        output_dir: Path,
        cuda_version: Optional[str] = None,
        use_pyinstaller: bool = True,
        sign: bool = False,
        python_executable: Optional[str] = None,
        prefer_python_312: bool = True,
    ):
        self.platform_name = platform_name
        self.output_dir = Path(output_dir)
        self.cuda_version = cuda_version
        self.use_pyinstaller = use_pyinstaller
        self.sign = sign
        self.python_executable = python_executable or sys.executable
        self.prefer_python_312 = prefer_python_312

        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.output_dir / "python-api"

    def _run(self, cmd: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

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

        # Add PyInstaller
        requirements.append("pyinstaller>=6.0.0")

        # Write requirements file
        with open(req_file, "w") as f:
            f.write("\n".join(requirements))

        print(f"Created requirements file: {req_file}")
        return req_file

    def create_venv(self, requirements_file: Path) -> Path:
        """Create a virtual environment with all dependencies"""
        venv_dir = self.output_dir / "venv"

        if venv_dir.exists():
            print(f"Removing existing venv: {venv_dir}")
            shutil.rmtree(venv_dir)

        print(f"Creating virtual environment: {venv_dir}")
        subprocess.run([self.python_executable, "-m", "venv", str(venv_dir)], check=True)

        # Get pip path
        if self.platform_name == "win32":
            pip_path = venv_dir / "Scripts" / "pip.exe"
            py_path = venv_dir / "Scripts" / "python.exe"
        else:
            pip_path = venv_dir / "bin" / "pip"
            py_path = venv_dir / "bin" / "python"

        # Upgrade pip
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
        )

        # install uv
        subprocess.run([str(pip_path), "install", "uv"], check=True)

        # install with uv
        if self.platform_name == "win32":
            uv_path = venv_dir / "Scripts" / "uv.exe"
        else:
            uv_path = venv_dir / "bin" / "uv"

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

        # Install the project itself
        subprocess.run(
            [str(uv_path), "pip", "install", "--python", str(py_path), "-e", str(self.project_root)],
            check=True,
        )

        return venv_dir

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

    def build_with_pyinstaller(self, venv_dir: Path) -> Path:
        """Build the bundle using PyInstaller"""
        if self.platform_name == "win32":
            pyinstaller_path = venv_dir / "Scripts" / "pyinstaller.exe"
        else:
            pyinstaller_path = venv_dir / "bin" / "pyinstaller"

        spec_file = self.project_root / "apex_engine.spec"

        print(f"Building with PyInstaller: {spec_file}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)

        # Add code signing identity if available
        if self.sign and self.platform_name == "darwin":
            identity = os.environ.get("APPLE_IDENTITY")
            if identity:
                env["APPLE_IDENTITY"] = identity

        subprocess.run(
            [
                str(pyinstaller_path),
                str(spec_file),
                "--distpath",
                str(self.dist_dir),
                "--workpath",
                str(self.output_dir / "build"),
                "--clean",
            ],
            cwd=str(self.project_root),
            env=env,
            check=True,
        )

        return self.dist_dir / "apex-engine"

    def create_standalone_bundle(self, venv_dir: Path) -> Path:
        """Create a standalone bundle without PyInstaller (portable Python)"""
        bundle_dir = self.dist_dir / "apex-engine"
        # Make this operation idempotent (re-running bundler should not fail).
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Copy Python installation
        if self.platform_name == "win32":
            python_dir = venv_dir / "Scripts"
            lib_dir = venv_dir / "Lib" / "site-packages"
        else:
            python_dir = venv_dir / "bin"
            # Find site-packages
            for sp in (venv_dir / "lib").glob("python*/site-packages"):
                lib_dir = sp
                break

        # Copy essential Python files
        dest_python = bundle_dir / "python"
        shutil.copytree(venv_dir, dest_python, symlinks=True)

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

        # Create launcher script
        self._create_launcher(bundle_dir)

        return bundle_dir

    def _create_launcher(self, bundle_dir: Path):
        """Create platform-specific launcher scripts"""
        if self.platform_name == "win32":
            launcher = bundle_dir / "start-api.bat"
            content = """@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PYTHONPATH=%SCRIPT_DIR%
set PATH=%SCRIPT_DIR%\\python\\Scripts;%PATH%
"%SCRIPT_DIR%\\python\\Scripts\\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
"""
        else:
            launcher = bundle_dir / "start-api.sh"
            content = """#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/python/bin:$PATH"
exec "$SCRIPT_DIR/python/bin/python" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
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

    def create_manifest(self, bundle_dir: Path):
        """Create a manifest file with bundle information"""
        manifest = {
            "version": "0.1.0",
            "platform": self.platform_name,
            "gpu_support": self.cuda_version or "auto",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "created_at": __import__("datetime").datetime.utcnow().isoformat(),
            "signed": self.sign,
        }

        manifest_file = bundle_dir / "apex-engine-manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Created manifest: {manifest_file}")

    def bundle(self) -> Path:
        """Run the full bundling process"""
        print(f"Bundling Python API for platform: {self.platform_name}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect GPU support
        gpu_type = self.cuda_version or self.detect_gpu_support()
        print(f"GPU support: {gpu_type}")

        # Create requirements
        req_file = self.create_requirements(gpu_type)

        # Create virtual environment
        venv_dir = self.create_venv(req_file)

        # Build bundle
        if self.use_pyinstaller:
            bundle_dir = self.build_with_pyinstaller(venv_dir)
        else:
            bundle_dir = self.create_standalone_bundle(venv_dir)

        # Sign the bundle
        self.sign_bundle(bundle_dir)

        # Create manifest
        self.create_manifest(bundle_dir)

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
        "--no-pyinstaller",
        action="store_true",
        help="Create a portable bundle instead of using PyInstaller",
    )
    parser.add_argument(
        "--sign", action="store_true", help="Sign the bundle for distribution"
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip building/installing Rust wheels (apex_download_rs).",
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
        use_pyinstaller=not args.no_pyinstaller,
        sign=args.sign,
        python_executable=py_exe,
    )
    bundler.skip_rust = args.skip_rust

    bundle_dir = bundler.bundle()
    print(f"\nBundle complete: {bundle_dir}")
    print(f"To test: cd {bundle_dir} && ./apex-engine start --daemon")


if __name__ == "__main__":
    main()
