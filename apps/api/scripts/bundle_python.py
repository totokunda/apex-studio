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
from typing import Optional


class PythonBundler:
    """Bundles Python API for distribution"""

    SUPPORTED_PLATFORMS = ["darwin", "linux", "win32"]

    # PyTorch wheel indices for different platforms
    TORCH_INDICES = {
        "cuda126": "https://download.pytorch.org/whl/cu126",
        "cuda124": "https://download.pytorch.org/whl/cu124",
        "cuda121": "https://download.pytorch.org/whl/cu121",
        "cuda118": "https://download.pytorch.org/whl/cu118",
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
    ):
        self.platform_name = platform_name
        self.output_dir = Path(output_dir)
        self.cuda_version = cuda_version
        self.use_pyinstaller = use_pyinstaller
        self.sign = sign

        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.dist_dir = self.output_dir / "python-api"

    def detect_gpu_support(self) -> str:
        """Detect available GPU support"""
        if self.platform_name == "darwin":
            # macOS uses MPS (Metal Performance Shaders)
            return "mps"

        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # NVIDIA GPU found, detect CUDA version
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
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

    def get_torch_index(self, gpu_type: str) -> str:
        """Get the appropriate PyTorch wheel index URL"""
        if gpu_type.startswith("cuda"):
            # Try to find matching CUDA version
            for key in ["cuda126", "cuda124", "cuda121", "cuda118"]:
                if key in gpu_type or gpu_type == key:
                    return self.TORCH_INDICES[key]
            # Default to latest CUDA
            return self.TORCH_INDICES["cuda126"]
        elif gpu_type == "rocm":
            return self.TORCH_INDICES["rocm"]
        else:
            return self.TORCH_INDICES["cpu"]

    def create_requirements(self, gpu_type: str) -> Path:
        """Create platform-specific requirements file"""
        req_file = self.output_dir / "requirements-bundle.txt"

        # Read base requirements
        base_req = self.project_root / "requirements" / "requirements.txt"
        proc_req = self.project_root / "requirements" / "processors.requirements.txt"

        requirements = []

        # Add torch with appropriate backend
        torch_index = self.get_torch_index(gpu_type)
        if gpu_type == "cpu" or self.platform_name == "darwin":
            requirements.append("torch")
            requirements.append("torchvision")
            requirements.append("torchaudio")
        else:
            requirements.append(f"--extra-index-url {torch_index}")
            requirements.append("torch")
            requirements.append("torchvision")
            requirements.append("torchaudio")

        # Read and filter base requirements
        if base_req.exists():
            with open(base_req) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-r"):
                        continue
                    # Skip torch packages (handled above)
                    if any(
                        pkg in line.lower()
                        for pkg in ["torch", "torchvision", "torchaudio"]
                    ):
                        continue
                    requirements.append(line)

        # Add processor requirements with platform filtering
        if proc_req.exists():
            with open(proc_req) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Handle platform-specific packages
                    if "sys_platform" in line:
                        # Parse conditional syntax
                        if self.platform_name == "darwin" and "darwin" in line:
                            pkg = line.split(";")[0].strip()
                            requirements.append(pkg)
                        elif self.platform_name == "linux" and "linux" in line:
                            pkg = line.split(";")[0].strip()
                            requirements.append(pkg)
                        elif self.platform_name == "win32" and "win32" in line:
                            pkg = line.split(";")[0].strip()
                            requirements.append(pkg)
                    else:
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
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

        # Get pip path
        if self.platform_name == "win32":
            pip_path = venv_dir / "Scripts" / "pip.exe"
        else:
            pip_path = venv_dir / "bin" / "pip"

        # Upgrade pip
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
        )

        # install uv
        subprocess.run([str(pip_path), "install", "uv"], check=True)

        # install with uv
        uv_path = venv_dir / "bin" / "uv"
        # Install requirements
        print(f"Installing requirements from: {requirements_file}")
        subprocess.run([str(uv_path), "sync", "-r", str(requirements_file)], check=True)

        # Install the project itself
        subprocess.run([str(uv_path), "sync", "-e", str(self.project_root)], check=True)

        return venv_dir

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
        choices=["cuda126", "cuda124", "cuda121", "cuda118", "cpu", "rocm", "auto"],
        default="auto",
        help="CUDA version to bundle (default: auto-detect)",
    )
    parser.add_argument(
        "--no-pyinstaller",
        action="store_true",
        help="Create a portable bundle instead of using PyInstaller",
    )
    parser.add_argument(
        "--sign", action="store_true", help="Sign the bundle for distribution"
    )

    args = parser.parse_args()

    # Auto-detect platform
    platform_name = args.platform
    if platform_name == "auto":
        platform_name = sys.platform

    # Auto-detect CUDA
    cuda_version = args.cuda if args.cuda != "auto" else None

    bundler = PythonBundler(
        platform_name=platform_name,
        output_dir=args.output,
        cuda_version=cuda_version,
        use_pyinstaller=not args.no_pyinstaller,
        sign=args.sign,
    )

    bundle_dir = bundler.bundle()
    print(f"\nBundle complete: {bundle_dir}")
    print(f"To test: cd {bundle_dir} && ./apex-engine start --daemon")


if __name__ == "__main__":
    main()
