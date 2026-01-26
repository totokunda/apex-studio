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
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Set, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class PythonBundler:
    """Bundles Python API for distribution (venv-based, no PyInstaller)."""

    SUPPORTED_PLATFORMS = ["darwin", "linux", "win32"]

    # Default Torch stack version (most bundles).
    #
    # We intentionally pin torch/torchvision/torchaudio together to avoid ABI mismatches.
    # For Linux CUDA/CPU wheels, PyTorch uses local version suffixes (e.g. +cu128, +cpu)
    # on the official wheel indices, so pins for those platforms must include the suffix.
    TORCH_VERSION = "2.9.1"
    TORCHVISION_VERSION = "0.24.1"
    TORCHAUDIO_VERSION = "2.9.1"

    # Intel macOS (x86_64): latest supported PyTorch stack.
    # These wheels are installed from PyPI (no +cpu suffix on macOS).
    TORCH_WHEEL_MACOS_INTEL = "https://huggingface.co/datasets/totoku/universal-macos-torch/resolve/main/torch-2.11.0a0%2Bgit51e04aa-cp311-cp311-macosx_15_0_universal2.whl"
    TORCHVISION_WHEEL_MACOS_INTEL = "https://huggingface.co/datasets/totoku/universal-macos-torch/resolve/main/torchvision-0.17.2%2Bc1d70fe-cp311-cp311-macosx_10_9_universal2.whl"
    TORCHAUDIO_WHEEL_MACOS_INTEL = "https://huggingface.co/datasets/totoku/universal-macos-torch/resolve/main/torchaudio-2.11.0a0%2Be123269-cp311-cp311-macosx_10_9_universal2.whl"

    # PyTorch wheel indices for different platforms
    TORCH_INDICES = {
        "cuda": "https://download.pytorch.org/whl/cu128",
        "mps": "https://download.pytorch.org/whl/cpu",
        "cpu": "https://download.pytorch.org/whl/cpu",
        "rocm": "https://download.pytorch.org/whl/rocm6.4",
    }

    def __init__(
        self,
        platform_name: str,
        output_dir: Path,
        cuda_version: Optional[str] = None,
        python_executable: Optional[str] = None,
        prefer_python_312: bool = True,
        bundle_version: Optional[str] = None,
        target_arch: Optional[str] = None,
        use_portable_python: bool = True,
    ):
        self.platform_name = platform_name
        # Resolve to an absolute path so later subprocess calls that change cwd (e.g. Rust builds)
        # still can find the venv interpreter and other bundle files reliably.
        self.output_dir = Path(output_dir).resolve()
        self.cuda_version = cuda_version
        # NOTE: `python_executable` is the interpreter we will use to create the bundle venv.
        # By default we will bootstrap a portable Python (python-build-standalone) and use that.
        self.python_executable = python_executable or sys.executable
        self.prefer_python_312 = prefer_python_312
        self.bundle_version = (bundle_version or "").strip() or None
        self.use_portable_python: bool = bool(use_portable_python)

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

        # Target arch/triple (controls portable Python selection and macOS Intel special-cases).
        self.target_arch: str = self._normalize_arch(target_arch or platform.machine())
        if not self.target_arch:
            self.target_arch = self._normalize_arch(platform.machine())
        # Normalize common Apple naming.
        if self.target_arch in {"arm64", "aarch64"}:
            self.target_arch = "arm64"
        if self.target_arch in {"x86_64", "amd64", "x64"}:
            self.target_arch = "x86_64"

        # Portable Python bootstrap cache root.
        self.portable_python_root = self.output_dir / "_portable-python"
        self.portable_python_exe: Optional[Path] = None
        self.portable_python_version: Optional[str] = None
        self.portable_python_tag: Optional[str] = None

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

        # Populated after we create the venv (this is the Python version that ships).
        self.venv_python_tag: Optional[str] = None
        self.venv_python_version: Optional[str] = None

    @staticmethod
    def _normalize_arch(arch: str) -> str:
        """
        Normalize architecture strings for bundle naming/metadata.

        Goal: keep bundle artifacts consistent across platforms/toolchains.
        In particular, Windows commonly reports `AMD64` but we want `x86_64`.
        """
        low = (arch or "").strip().lower()
        if not low:
            return ""
        if low in {"amd64", "x64"}:
            return "x86_64"
        return low

    def _probe_python_info(self, python_exe: Path) -> tuple[Optional[str], Optional[str]]:
        """
        Return (python_tag, python_version) for the given interpreter.

        - python_tag: like "cp312"
        - python_version: like "3.12.6"
        """
        python_exe = Path(python_exe).resolve()
        try:
            res = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    (
                        "import sys; "
                        "print(f'cp{sys.version_info.major}{sys.version_info.minor}'); "
                        "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
            tag = lines[0] if len(lines) >= 1 else None
            ver = lines[1] if len(lines) >= 2 else None
            if tag and not re.fullmatch(r"cp\d{2,3}", tag, flags=re.IGNORECASE):
                tag = None
            return (tag, ver)
        except Exception:
            return (None, None)

    # ----------------------------
    # Portable Python bootstrap
    # ----------------------------

    PYTHON_BUILD_STANDALONE_LATEST_JSON = (
        "https://raw.githubusercontent.com/astral-sh/python-build-standalone/latest-release/latest-release.json"
    )

    def _portable_python_major_minor(self) -> str:
        """
        Policy:
          - macOS Intel (x86_64): Python 3.11.x
          - everything else we bundle: Python 3.12.x
        """
        if self.platform_name == "darwin" and self.target_arch == "x86_64":
            return "3.11"
        return "3.12"

    def _portable_python_target_triple(self) -> str:
        """
        Return the LLVM target triple we use to select python-build-standalone artifacts.

        We intentionally pick conservative targets (no x86_64_v2/v3/v4) so the bundle runs on
        the widest range of machines for a given OS.
        """
        if self.platform_name == "darwin":
            if self.target_arch == "x86_64":
                return "x86_64-apple-darwin"
            return "aarch64-apple-darwin"
        if self.platform_name == "win32":
            return "x86_64-pc-windows-msvc"
        # linux
        return "x86_64-unknown-linux-gnu"

    def _http_get_json(self, url: str, *, timeout: int = 60) -> dict:
        req = Request(
            url,
            headers={
                "User-Agent": "apex-studio-bundler/1.0",
                "Accept": "application/json",
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))

    def _http_download(self, url: str, dest: Path, *, timeout: int = 600) -> Path:
        dest = Path(dest).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = Request(
            url,
            headers={
                "User-Agent": "apex-studio-bundler/1.0",
                "Accept": "*/*",
            },
        )
        # Stream download so we can show progress (GitHub release assets can be large).
        with urlopen(req, timeout=timeout) as resp:
            try:
                total = int(resp.headers.get("Content-Length") or "0") or None
            except Exception:
                total = None

            def _fmt_bytes(n: int) -> str:
                if n < 1024:
                    return f"{n} B"
                kb = n / 1024.0
                if kb < 1024:
                    return f"{kb:.1f} KiB"
                mb = kb / 1024.0
                if mb < 1024:
                    return f"{mb:.1f} MiB"
                gb = mb / 1024.0
                return f"{gb:.2f} GiB"

            # Only render a bar if we're in an interactive terminal.
            is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
            chunk_size = 1024 * 1024  # 1 MiB
            downloaded = 0
            last_print = 0

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Update progress.
                    if total and is_tty:
                        pct = min(1.0, downloaded / float(total))
                        width = 30
                        filled = int(width * pct)
                        bar = "=" * filled + "-" * (width - filled)
                        msg = (
                            f"\r  [{bar}] {pct*100:6.2f}% "
                            f"({_fmt_bytes(downloaded)} / {_fmt_bytes(total)})"
                        )
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    else:
                        # Non-tty or unknown total: print occasionally (every ~16 MiB).
                        if downloaded - last_print >= 16 * 1024 * 1024:
                            if total:
                                pct = min(100.0, 100.0 * downloaded / float(total))
                                print(
                                    f"  Downloaded {_fmt_bytes(downloaded)} / {_fmt_bytes(total)} ({pct:.1f}%)"
                                )
                            else:
                                print(f"  Downloaded {_fmt_bytes(downloaded)}")
                            last_print = downloaded

            if total and is_tty:
                sys.stdout.write("\n")
                sys.stdout.flush()

        tmp.replace(dest)
        return dest

    def _select_python_build_asset(
        self, assets: list[dict], *, major_minor: str, target_triple: str
    ) -> tuple[str, str, str]:
        """
        Choose the best python-build-standalone asset for (major_minor, target_triple).

        Returns (asset_name, download_url, python_version_str).
        """
        # Prefer formats that we can extract with stdlib only (no zstd needed).
        if self.platform_name == "win32":
            prefer_exts = [".zip"]
        else:
            prefer_exts = [".tar.gz", ".tgz", ".zip", ".tar.zst"]

        mm = str(major_minor).strip()
        triple = str(target_triple).strip()

        def _parse_ver(name: str) -> Optional[tuple[int, int, int]]:
            # Try common patterns like "python-3.12.7" or "cpython-3.12.7".
            m = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", name)
            if not m:
                return None
            try:
                return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except Exception:
                return None

        def _score(name: str) -> tuple:
            # Higher is better.
            ver = _parse_ver(name) or (0, 0, 0)
            ext_rank = 0
            for i, ext in enumerate(prefer_exts):
                if name.endswith(ext):
                    ext_rank = len(prefer_exts) - i
                    break
            flavor_rank = 2 if "install_only" in name else (1 if "full" in name else 0)
            shared_rank = 1 if "windows-msvc" in name else 0
            return (ver[0], ver[1], ver[2], ext_rank, flavor_rank, shared_rank)

        candidates: list[tuple[tuple, dict]] = []
        for a in assets or []:
            name = str(a.get("name") or "")
            url = str(a.get("browser_download_url") or "")
            if not name or not url:
                continue
            if triple not in name:
                continue
            ver = _parse_ver(name)
            if not ver:
                continue
            if f"{ver[0]}.{ver[1]}" != mm:
                continue
            # Avoid static linux/musl builds (can't load .so extensions).
            if "musl" in name:
                continue
            candidates.append((_score(name), a))

        if not candidates:
            raise RuntimeError(
                "Could not find a python-build-standalone asset for "
                f"python {mm}.x and target {triple}."
            )

        candidates.sort(key=lambda t: t[0], reverse=True)
        best = candidates[0][1]
        best_name = str(best.get("name"))
        best_url = str(best.get("browser_download_url"))
        best_ver = _parse_ver(best_name) or (0, 0, 0)
        return (best_name, best_url, f"{best_ver[0]}.{best_ver[1]}.{best_ver[2]}")

    def _extract_archive(self, archive_path: Path, dest_dir: Path) -> Path:
        """
        Extract supported archives into dest_dir and return the extraction root.
        """
        archive_path = Path(archive_path).resolve()
        dest_dir = Path(dest_dir).resolve()
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        name = archive_path.name.lower()
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                # Python 3.12+ supports `filter=...` and warns that the default behavior
                # will change in Python 3.14. Use the "data" filter to be explicit and
                # future-proof while still extracting normal archives.
                try:
                    tf.extractall(dest_dir, filter="data")
                except TypeError:
                    # Older Python: no filter parameter.
                    tf.extractall(dest_dir)
        elif name.endswith(".tar.zst"):
            # Decompress via the `zstandard` Python module (bootstrap it into the runner env if needed).
            try:
                import zstandard as zstd  # type: ignore
            except Exception:
                # Install into the *runner* environment (the Python executing this script),
                # using uv (no direct pip usage).
                uv = self._ensure_uv_available()
                subprocess.run(
                    [uv, "pip", "install", "--python", str(sys.executable), "--upgrade", "zstandard"],
                    check=True,
                )
                import zstandard as zstd  # type: ignore

            with open(archive_path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    with tarfile.open(fileobj=reader, mode="r|") as tf:
                        tf.extractall(dest_dir)
        else:
            raise RuntimeError(f"Unsupported portable Python archive format: {archive_path.name}")

        # Heuristic: prefer a single top-level directory if present.
        children = [p for p in dest_dir.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return dest_dir

    def _find_python_executable(self, root: Path) -> Path:
        root = Path(root).resolve()
        if self.platform_name == "win32":
            # Common layouts include:
            # - <root>/python/install/python.exe
            # - <root>/install/python.exe
            # - <root>/python.exe
            candidates: list[Path] = []
            for pat in ["python.exe"]:
                for p in root.rglob(pat):
                    if p.is_file() and p.name.lower() == "python.exe":
                        candidates.append(p)
            if not candidates:
                raise RuntimeError(f"Portable Python extraction did not contain python.exe under: {root}")

            def _rank(p: Path) -> tuple[int, int]:
                parts = [x.lower() for x in p.parts]
                score = 0
                if "install" in parts:
                    score += 2
                if "python" in parts:
                    score += 1
                return (score, -len(parts))

            candidates.sort(key=_rank, reverse=True)
            return candidates[0]

        # POSIX (macOS/Linux)
        preferred_names = [
            "python3",
            f"python{self._portable_python_major_minor()}",
            "python",
        ]
        candidates: list[Path] = []
        for nm in preferred_names:
            for p in root.rglob(nm):
                try:
                    if p.is_file() and p.parent.name == "bin":
                        candidates.append(p)
                except Exception:
                    continue
        if not candidates:
            raise RuntimeError(f"Portable Python extraction did not contain a bin/python under: {root}")

        def _rank(p: Path) -> tuple[int, int]:
            parts = [x.lower() for x in p.parts]
            score = 0
            if "install" in parts:
                score += 2
            if "bin" in parts:
                score += 1
            return (score, -len(parts))

        candidates.sort(key=_rank, reverse=True)
        return candidates[0]

    def _ensure_pip(self, python_exe: Path) -> None:
        """
        Legacy helper retained for compatibility.

        We intentionally do NOT invoke pip here. All dependency installation should go through `uv`.
        """
        _ = python_exe
        return

    def ensure_portable_python(self) -> Path:
        """
        Download and extract a portable Python distribution and return its python executable path.

        This uses python-build-standalone. We always pick a conservative target triple so the
        resulting bundles run on as many machines as possible for a given platform.
        """
        if not self.use_portable_python:
            return Path(self.python_executable).resolve()

        mm = self._portable_python_major_minor()
        triple = self._portable_python_target_triple()

        self.portable_python_root.mkdir(parents=True, exist_ok=True)
        meta_path = self.portable_python_root / "portable-python.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            meta = {}

        # If cache matches, reuse.
        if (
            meta.get("major_minor") == mm
            and meta.get("target_triple") == triple
            and meta.get("python_exe")
        ):
            cached = self.portable_python_root / str(meta.get("python_exe"))
            if cached.exists():
                self.portable_python_exe = cached
                self.portable_python_version = str(meta.get("python_version") or "") or None
                self.portable_python_tag = str(meta.get("python_tag") or "") or None
                self.python_executable = str(cached)
                return cached

        # Fetch latest tag
        try:
            latest = self._http_get_json(self.PYTHON_BUILD_STANDALONE_LATEST_JSON)
        except (URLError, HTTPError) as e:
            raise RuntimeError(
                "Failed to fetch python-build-standalone latest-release.json. "
                f"url={self.PYTHON_BUILD_STANDALONE_LATEST_JSON}"
            ) from e

        tag = str(latest.get("tag") or "").strip()
        release_url = str(latest.get("release_url") or "").strip()
        if not tag and release_url:
            tag = release_url.rstrip("/").split("/")[-1]
        if not tag:
            raise RuntimeError(
                "python-build-standalone latest-release.json did not contain a tag/release_url."
            )

        # Query GitHub release assets for that tag.
        api_url = f"https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/{tag}"
        release = self._http_get_json(api_url)
        assets = list(release.get("assets") or [])

        asset_name, asset_url, py_ver = self._select_python_build_asset(
            assets, major_minor=mm, target_triple=triple
        )

        print(f"Bootstrapping portable Python: {py_ver} ({triple})")
        archive_path = self.portable_python_root / asset_name
        if not archive_path.exists():
            print(f"Downloading portable Python asset: {asset_name}")
            self._http_download(asset_url, archive_path)

        extract_root = self.portable_python_root / "extracted"
        extracted = self._extract_archive(archive_path, extract_root)
        py_exe = self._find_python_executable(extracted)

        # Avoid pip bootstrapping; all installs are done via `uv` elsewhere.

        # Probe tag/version.
        tag_str, ver_str = self._probe_python_info(py_exe)
        self.portable_python_exe = py_exe
        self.portable_python_version = ver_str or py_ver
        self.portable_python_tag = tag_str
        self.python_executable = str(py_exe)

        meta_out = {
            "tag": tag,
            "release_url": release_url,
            "major_minor": mm,
            "target_triple": triple,
            "asset_name": asset_name,
            "python_version": self.portable_python_version or "",
            "python_tag": self.portable_python_tag or "",
            "python_exe": str(py_exe.relative_to(self.portable_python_root)),
        }
        meta_path.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")

        return py_exe

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
        """
        Return the Python interpreter path inside the *shipped bundle*.

        NOTE: We no longer ship a venv. Instead, we ship a portable CPython distribution
        under `<bundleRoot>/<venv_name>/` (kept name for backwards compatibility).
        """
        base = (Path(bundle_dir).resolve() / self.venv_name).resolve()
        if self.platform_name == "win32":
            candidates = [
                base / "python.exe",
                base / "Scripts" / "python.exe",  # legacy venv layout (if present)
                base / "install" / "python.exe",  # some archive layouts
            ]
        else:
            candidates = [
                base / "bin" / "python",
                base / "bin" / "python3",
            ]
        for p in candidates:
            if p.exists():
                return p
        # Fallback: return the first candidate for clearer error messages upstream.
        return candidates[0]



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
        # Ensure smoke test runner logs stream immediately (avoid Python stdio buffering).
        env["PYTHONUNBUFFERED"] = "1"
        # Prefer bundle-local cache for kernels fetched via `kernels.get_kernel(...)`.
        # This is especially important for MPS bundles where we ship pre-fetched kernels.

        print("\n--- Running bundle smoke tests (inside shipped venv) ---")
        print(f"Bundle dir: {bundle_dir}")
        print(f"Python: {py_path}")

        smoke_runner = bundle_dir / "scripts" / "smoke_tests" / "run_all.py"
        if not smoke_runner.exists():
            raise RuntimeError(
                f"Smoke tests runner not found in bundle: {smoke_runner}"
            )

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

        # Stream output as it is produced. We merge stderr into stdout so we can stream a
        # single ordered log, while still capturing enough output for error reporting.
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=str(bundle_dir),
            bufsize=1,
        )
        assert proc.stdout is not None

        output_lines: List[str] = []
        for line in proc.stdout:
            # Preserve the child process' newlines, and flush so logs show up immediately.
            print(line, end="", flush=True)
            output_lines.append(line)
        proc.wait()

        if proc.returncode != 0:
            output = ("".join(output_lines) or "").strip()
            # Cap error context so exceptions stay readable.
            if len(output) > 20000:
                output = output[-20000:]
            raise RuntimeError(
                f"Bundle smoke tests failed (exit {proc.returncode}). output:\n{output}"
            )

        # If building a CUDA bundle but CUDA isn't available on the build machine, optionally fail fast.
        if self.smoke_tests_strict and str(gpu_type or "").startswith("cuda"):
            # The in-venv harness will report cuda_available=False; enforce here for clarity.
            try:
                probe = subprocess.run(
                    [
                        str(py_path),
                        "-c",
                        "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)",
                    ],
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

        We require a system-installed `uv` (or otherwise available on PATH).
        This bundler intentionally avoids bootstrapping uv via pip.
        """
        try:
            subprocess.run(
                ["uv", "--version"], check=True, capture_output=True, text=True
            )
            return "uv"
        except Exception:
            raise RuntimeError(
                "`uv` was not found on PATH. Install uv first, then re-run bundling.\n"
                "macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                "Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex\n"
                "Or install via your package manager and ensure `uv` is on PATH."
            )

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
            candidates.extend(
                sorted((Path(base_prefix) / "lib").glob(f"libpython{py_mm}*.dylib"))
            )
        if base_prefix:
            candidates.extend(
                sorted((Path(base_prefix) / "lib").glob("libpython*.dylib"))
            )

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
            # macOS:
            # - Apple Silicon can use MPS (Metal Performance Shaders)
            # - Intel macOS (x86_64) cannot use MPS; treat as CPU-only
            arch = (self.target_arch or "").lower()
            if arch in {"x86_64", "amd64", "i386", "i686"} or arch.startswith("x86"):
                return "cpu"
            return "mps"

        # Check for NVIDIA GPU
        try:
            result = self._run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            )
            if result.returncode == 0:
                # NVIDIA GPU found: we intentionally bundle a single CUDA wheel stack.
                # Keep this stable as "cuda" so requirement pins and indices stay consistent.
                return "cuda"
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
            for key in ["cuda"]:
                if key in gpu_type or gpu_type == key:
                    return self.TORCH_INDICES[key]
            # Default to latest CUDA
            return self.TORCH_INDICES["cuda"]
        elif gpu_type == "rocm":
            return self.TORCH_INDICES["rocm"]
        elif gpu_type == "mps":
            return self.TORCH_INDICES["mps"]
        else:
            return self.TORCH_INDICES["cpu"]

    def _torch_local_suffix_for_gpu(self, gpu_type: str) -> Optional[str]:
        """
        Return the local version suffix for official PyTorch wheels for this gpu_type.

        Examples:
          - cpu -> "cpu"
          - cuda -> "cu128"
          - mps -> "cpu"
          - rocm -> "rocm6.4" (matches TORCH_INDICES["rocm"])
        """
        if gpu_type == "cpu":
            return "cpu"
        if gpu_type.startswith("cuda"):
            # We intentionally ship a single CUDA wheel stack.
            # Even if a caller passes something like "cuda124", keep the suffix stable.
            return "cu128"
        if gpu_type == "rocm":
            # Keep in sync with TORCH_INDICES["rocm"]
            return "rocm6.4"
        if gpu_type == "mps":
            return "cpu"
        return None

    def _torch_specs_for_bundle(self, gpu_type: str) -> list[str]:
        """
        Produce torch/torchvision/torchaudio requirement specs for the bundle.

        Requirement:
          - macOS Intel (x86_64): torch==2.2.2 (latest supported)
          - everything else: torch==2.9.1
        """
        # macOS: Apple Silicon uses MPS, Intel uses CPU-only.
        if self.platform_name == "darwin":
            if self._is_intel_macos():
                return [
                    f"torch @ {self.TORCH_WHEEL_MACOS_INTEL}",
                    f"torchvision @ {self.TORCHVISION_WHEEL_MACOS_INTEL}",
                    f"torchaudio @ {self.TORCHAUDIO_WHEEL_MACOS_INTEL}",
                ]
            return [
                f"torch=={self.TORCH_VERSION}",
                f"torchvision=={self.TORCHVISION_VERSION}",
                f"torchaudio=={self.TORCHAUDIO_VERSION}",
            ]

        # Windows ROCm is a special-case (direct TheRock wheels) handled elsewhere.
        if self.platform_name == "win32" and gpu_type == "rocm":
            return []

        # Linux/Windows CPU: on Windows, CPU wheels are typically plain versions on PyPI;
        # on Linux, CPU wheels come from the CPU index and are published with +cpu.
        suffix = self._torch_local_suffix_for_gpu(gpu_type)
        if self.platform_name == "linux" and suffix:
            return [
                f"torch=={self.TORCH_VERSION}+{suffix}",
                f"torchvision=={self.TORCHVISION_VERSION}+{suffix}",
                f"torchaudio=={self.TORCHAUDIO_VERSION}+{suffix}",
            ]

        # Windows CUDA wheels also typically use local suffixes on the PyTorch indices.
        if self.platform_name == "win32" and gpu_type.startswith("cuda") and suffix:
            return [
                f"torch=={self.TORCH_VERSION}+{suffix}",
                f"torchvision=={self.TORCHVISION_VERSION}+{suffix}",
                f"torchaudio=={self.TORCHAUDIO_VERSION}+{suffix}",
            ]

        # Default: plain pins (works for Windows CPU; also fine if a platform publishes
        # non-suffixed wheels on the configured extra index).
        return [
            f"torch=={self.TORCH_VERSION}",
            f"torchvision=={self.TORCHVISION_VERSION}",
            f"torchaudio=={self.TORCHAUDIO_VERSION}",
        ]

    def _is_intel_macos(self) -> bool:
        """
        Best-effort check for Intel macOS (x86_64).

        Note: On Apple Silicon, running an x86_64 interpreter under Rosetta will
        report platform.machine() == "x86_64", which is exactly what we want when
        building/testing an Intel-targeted bundle.
        """
        if self.platform_name != "darwin":
            return False
        arch = (self.target_arch or "").lower()
        return arch in {"x86_64", "amd64", "i386", "i686"} or arch.startswith("x86")

    def _setuptools_spec_for_venv(self) -> str:
        # Pin setuptools on Intel macOS for older torch stacks.
        return "setuptools>=61,<70" if self._is_intel_macos() else "setuptools"

    def choose_machine_requirements_entrypoint(self, gpu_type: str) -> Path:
        """
        Choose an entrypoint under requirements/{cpu,mps,cuda} based on platform + GPU/arch.
        """
        req_root = self.project_root / "requirements"

        # macOS:
        # - Apple Silicon should use MPS requirements
        # - Intel macOS (x86_64) should use CPU requirements (no MPS)
        if self.platform_name == "darwin":
            arch = (self.target_arch or "").lower()
            if arch in {"x86_64", "amd64", "i386", "i686"} or arch.startswith("x86"):
                return req_root / "cpu" / "requirements.txt"
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

    def _read_requirements_file(
        self, path: Path, visited: Optional[Set[Path]] = None
    ) -> List[str]:
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

        requirements: list[str] = []

        # Choose machine-specific requirements entrypoint early so we can apply
        # correct Windows wheel-stack torch pins (they depend on the chosen entry).
        machine_entry = self.choose_machine_requirements_entrypoint(gpu_type)
        self.last_machine_entry = machine_entry

        # Add torch with appropriate backend.
        #
        # IMPORTANT: for CPU bundles on Linux/Windows, we MUST ensure the PyTorch CPU
        # wheel index is available; otherwise `torch` can resolve to a CUDA wheel
        # depending on resolver/index behavior.
        torch_index = self.get_torch_index(gpu_type)
        if self.platform_name != "darwin":
            if gpu_type == "rocm" and self.platform_name == "win32":
                # Windows ROCm: Use specific community wheels (TheRock). We append
                # direct URLs below, so no index is needed here.
                pass
            else:
                requirements.append(f"--extra-index-url {torch_index}")

        # Windows wheel stacks: pin torch to match wheel ABI expectations.
        if self.platform_name == "win32":
            if gpu_type.startswith("cuda"):
                requirements.extend(self._torch_specs_for_bundle(gpu_type))
            elif gpu_type == "rocm":
                # ROCm 6.5 Windows (TheRock)
                requirements.extend(
                    [
                        "torch @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl",
                        "torchvision @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl",
                        "torchaudio @ https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl",
                    ]
                )
            else:
                requirements.extend(self._torch_specs_for_bundle(gpu_type))
        elif self.platform_name == "darwin":
            # MPS requires Torch 2.7.1+
            requirements.extend(self._torch_specs_for_bundle(gpu_type))
        else:
            # Linux: always pin to the requested global Torch stack version.
            requirements.extend(self._torch_specs_for_bundle(gpu_type))

        # Add machine-specific requirements (expanded entrypoint)
        machine_lines = self._read_requirements_file(machine_entry)
        for line in machine_lines:
            # Skip any torch packages (we add them above, with correct index/pins)
            lower = line.lower()
            if (
                lower.startswith("torch")
                or lower.startswith("torchvision")
                or lower.startswith("torchaudio")
            ):
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

        # Determine python inside the runtime dir (venv_dir is now a portable runtime root).
        py_path = self._find_python_executable(Path(venv_dir))

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
            [uv, "pip", "freeze", "--python", str(py_path)],
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

    def _stage_portable_python_runtime(self, *, base_prefix: Path) -> Path:
        """
        Stage the portable Python installation prefix into `<output>/<venv_name>/`.

        Previous behavior:
          - extract portable python into `<output>/_portable-python/extracted/...`
          - copytree that prefix into `<output>/apex-studio`
          - later copytree `<output>/apex-studio` into the shipped bundle

        That means two full filesystem copies of a large Python tree. On Windows this is
        particularly slow (Defender/AV scans + many small files).

        New behavior:
          - if the extracted prefix lives under `<output>/_portable-python/`, we *move/rename*
            it into `<output>/apex-studio` (fast, same-volume rename),
          - otherwise we fall back to copytree (safe for unexpected layouts).
        """
        base_prefix = Path(base_prefix).resolve()
        runtime_dir = (self.output_dir / self.venv_name).resolve()

        if runtime_dir.exists():
            print(f"Removing existing portable runtime: {runtime_dir}")
            shutil.rmtree(runtime_dir)

        print(f"Staging portable Python runtime: {runtime_dir}")

        # Prefer a fast rename/move when the extracted portable python lives inside our
        # output-dir cache (same drive).
        try:
            cache_root = Path(self.portable_python_root).resolve()
        except Exception:
            cache_root = Path(self.portable_python_root)

        try:
            _ = base_prefix.relative_to(cache_root)
            shutil.move(str(base_prefix), str(runtime_dir))
            return runtime_dir
        except Exception:
            pass

        # Fallback: copytree (previous behavior).
        ignore_junk = shutil.ignore_patterns(
            ".DS_Store",
            "._*",
            "__MACOSX",
            "__pycache__",
            "*.pyc",
            "*.pyo",
        )
        ignore_libdir = self._choose_venv_libdir_to_ignore(base_prefix)

        def _ignore_with_libdir(src: str, names: list[str]) -> set[str]:
            ignored = set(ignore_junk(src, names) or [])
            # Only apply the lib/lib64 exclusion at the runtime root.
            try:
                if ignore_libdir and Path(src).resolve() == base_prefix.resolve():
                    ignored.add(ignore_libdir)
            except Exception:
                pass
            return ignored

        shutil.copytree(
            base_prefix, runtime_dir, symlinks=False, ignore=_ignore_with_libdir
        )
        return runtime_dir

    def create_portable_python_runtime(self, requirements_file: Path) -> Path:
        """
        Create a *portable* Python runtime directory that contains:
          - the python-build-standalone CPython distribution
          - all dependencies installed directly into that Python (NO venv)

        The runtime directory is named `apex-studio` (self.venv_name) for backwards
        compatibility with the Electron app's bundle discovery.
        """
        # Ensure we have a base portable python downloaded.
        base_py = self.ensure_portable_python()

        # Probe the base prefix (installation root) so we copy the correct subtree.
        probe = subprocess.run(
            [
                str(base_py),
                "-c",
                "import json, sys; print(json.dumps({'prefix': sys.prefix, 'exe': sys.executable}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads((probe.stdout or "").strip() or "{}")
        base_prefix = Path(str(info.get("prefix") or "")).resolve()
        if not base_prefix.exists():
            raise RuntimeError(f"Portable Python prefix not found: {base_prefix}")
        # Stage into `<output>/apex-studio`. Prefer a fast move/rename to avoid copying
        # the portable Python tree twice.
        runtime_dir = self._stage_portable_python_runtime(base_prefix=base_prefix)

        # Resolve python exe within the staged runtime.
        py_path = self._find_python_executable(runtime_dir)

        uv = self._ensure_uv_available()

        # Record shipped Python tag/version based on the runtime interpreter we ship.
        tag, ver = self._probe_python_info(py_path)
        self.venv_python_tag = tag
        self.venv_python_version = ver

        # Ensure pip tooling exists and is up to date inside the runtime.
        self._ensure_pip(py_path)

        # Install standard build tooling inside the runtime (use uv, not pip).
        subprocess.run(
            [uv, "pip", "install", "--python", str(py_path), "--upgrade", self._setuptools_spec_for_venv(), "wheel"],
            check=True,
        )

        # Pre-install build helpers that some legacy sdists forget to declare.
        subprocess.run(
            [uv, "pip", "install", "--python", str(py_path), "Cython>=0.29.36", "numpy", "psutil", "enscons", "pytoml"],
            check=True,
        )

        # Pre-install torch (same reasoning as before: some packages import torch at build time).
        try:
            req_lines = [
                ln.strip()
                for ln in requirements_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            option_args: list[str] = []
            torch_specs: list[str] = []
            for ln in req_lines:
                if ln.startswith("--"):
                    option_args.extend(shlex.split(ln))
            for ln in req_lines:
                lower = ln.lower()
                if (
                    lower.startswith("torch")
                    or lower.startswith("torchvision")
                    or lower.startswith("torchaudio")
                ):
                    if ln.startswith("--"):
                        continue
                    torch_specs.append(ln)
            if torch_specs:
                env = os.environ.copy()
                env["PYTHONNOUSERSITE"] = "1"
                subprocess.run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--index-strategy",
                        "unsafe-best-match",
                        "--python",
                        str(py_path),
                        "--no-build-isolation",
                        *option_args,
                        *torch_specs,
                    ],
                    check=True,
                    env=env,
                )
        except Exception:
            # Best-effort: don't fail bundling solely due to this preinstall step.
            pass

        # Build & install Rust wheels into the runtime.
        if not getattr(self, "skip_rust", False):
            self._build_and_install_rust_wheels(py_path=py_path)

        # Install requirements directly into this portable runtime (NO venv).
        print(f"Installing requirements into portable runtime from: {requirements_file}")
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--index-strategy",
                "unsafe-best-match",
                "--python",
                str(py_path),
                "--no-build-isolation",
                "-r",
                str(requirements_file),
            ],
            check=True,
            env=env,
        )

        # Optional: Nunchaku (CUDA-only) — best-effort.
        try:
            if self.last_machine_entry and str(self.last_machine_entry).replace(
                "\\", "/"
            ).endswith("/requirements/cuda/linux.txt"):
                subprocess.run(
                    [
                        str(py_path),
                        str(
                            self.project_root
                            / "scripts"
                            / "deps"
                            / "maybe_install_nunchaku.py"
                        ),
                        "--python",
                        str(py_path),
                        "--machine-entry-name",
                        str(self.last_machine_entry.name),
                        "--install",
                    ],
                    check=False,
                )
            elif self.last_machine_entry and str(self.last_machine_entry).replace(
                "\\", "/"
            ).endswith("/requirements/cuda/windows.txt"):
                subprocess.run(
                    [
                        str(py_path),
                        str(
                            self.project_root
                            / "scripts"
                            / "deps"
                            / "maybe_install_nunchaku.py"
                        ),
                        "--python",
                        str(py_path),
                        "--machine-entry-name",
                        str(self.last_machine_entry.name),
                        "--install",
                    ],
                    check=False,
                )
        except Exception:
            pass

        # Install local universal wheels if available (overrides requirements)
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
            print(
                f"Installing local universal wheels: {[w.name for w in local_wheels]}"
            )
            subprocess.run(
                [uv, "pip", "install", "--python", str(py_path), "--force-reinstall", "--no-deps", *[str(w) for w in local_wheels]],
                check=True,
            )

        # Install the project itself so the runtime has the `apex-engine` console script,
        # but we still ship `src/` alongside and run `python -m src ...` in production.
        subprocess.run(
            [uv, "pip", "install", "--python", str(py_path), str(self.project_root)],
            check=True,
        )

        self._remove_obsolete_typing_backport(py_path=py_path)

        # Final hygiene: remove cache/artifact files that should never ship in bundles.
        try:
            removed_files, removed_dirs = self._cleanup_bundle_tree(runtime_dir)
            if removed_files or removed_dirs:
                print(
                    f"Cleaned runtime artifacts: removed_files={removed_files}, removed_dirs={removed_dirs}"
                )
        except Exception:
            pass

        # Windows: ship VC++ runtime DLLs app-locally so torch can import without any
        # system vc_redist install/upgrade (avoids UAC prompts for end users).
        if self.platform_name == "win32":
            self._bundle_windows_vc_runtime(dest_dir=Path(py_path).resolve().parent)

        return runtime_dir

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
                    if (
                        name == ".DS_Store"
                        or name.startswith("._")
                        or name.endswith((".pyc", ".pyo"))
                    ):
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

    def _bundle_windows_vc_runtime(self, *, dest_dir: Path) -> None:
        """
        Windows: ship the VC++ runtime DLLs *app-locally* alongside the shipped Python.

        Why:
          - End users may have an old 14.x VC runtime installed (registry Installed=1),
            which can still fail to load modern PyTorch DLLs (c10.dll dependencies).
          - Installing/upgrading vc_redist requires elevation/UAC; app-local DLLs avoid that.

        IMPORTANT:
          - To keep builds reproducible, we do NOT copy DLLs from the build machine's System32.
            Instead, we extract the official redist payload:
              vc_redist.x64.exe /layout -> MSI payloads
              msiexec /a (administrative install) -> extracted DLLs

        This step only runs when the bundler itself is running on Windows.
        """
        if self.platform_name != "win32":
            return

        dest_dir = Path(dest_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Only supported when bundling on a Windows host.
        if os.name != "nt":
            print(
                f"Skipping VC++ runtime extraction: host os.name={os.name!r} (platform_name={self.platform_name})"
            )
            return

        # If the required DLLs are already present, don't re-extract.
        required = ["vcruntime140_1.dll", "msvcp140.dll", "vcruntime140.dll"]
        if all((dest_dir / name).exists() for name in required):
            print(f"VC++ runtime DLLs already present next to python.exe: {dest_dir}")
            return

        VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

        # Prefer a repo-local copy (downloaded by apps/app/scripts/ensure-vc-redist.js),
        # but fall back to downloading into the bundler output cache.
        repo_root = self.project_root.parent.parent
        candidate = repo_root / "apps" / "app" / "buildResources" / "vc_redist.x64.exe"
        vc_redist_dir = (self.output_dir / "_vc-redist").resolve()
        vc_redist_dir.mkdir(parents=True, exist_ok=True)
        vc_redist_exe = (vc_redist_dir / "vc_redist.x64.exe").resolve()

        try:
            if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
                vc_redist_exe = candidate.resolve()
            else:
                # Download into the bundler output cache (local builds).
                if not vc_redist_exe.exists() or vc_redist_exe.stat().st_size < 1024 * 1024:
                    print(f"Downloading vc_redist.x64.exe for extraction: {VC_REDIST_URL}")
                    self._http_download(VC_REDIST_URL, vc_redist_exe)
        except Exception as e:
            raise RuntimeError(f"Failed to obtain vc_redist.x64.exe: {e}") from e

        layout_dir = (vc_redist_dir / "layout").resolve()
        msi_out_dir = (vc_redist_dir / "msi").resolve()

        # Clean previous extractions to avoid stale/corrupt outputs.
        try:
            if layout_dir.exists():
                shutil.rmtree(layout_dir)
            if msi_out_dir.exists():
                shutil.rmtree(msi_out_dir)
        except Exception:
            pass

        ps1 = (vc_redist_dir / "extract-vc-runtime.ps1").resolve()
        ps1.write_text(
            r"""param(
  [Parameter(Mandatory=$true)][string]$VcRedist,
  [Parameter(Mandatory=$true)][string]$LayoutDir,
  [Parameter(Mandatory=$true)][string]$MsiOutDir,
  [Parameter(Mandatory=$true)][string]$DestDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LayoutDir, $MsiOutDir, $DestDir | Out-Null

Write-Host "Extracting vc_redist payload (layout)..." -ForegroundColor Cyan
& $VcRedist /layout $LayoutDir /quiet | Out-Null

$msiMin = Get-ChildItem -Recurse -File -Path $LayoutDir -Filter "vc_runtimeMinimum_x64.msi" | Select-Object -First 1
$msiAdd = Get-ChildItem -Recurse -File -Path $LayoutDir -Filter "vc_runtimeAdditional_x64.msi" | Select-Object -First 1
if (-not $msiMin) { throw "Could not find vc_runtimeMinimum_x64.msi under layout dir: $LayoutDir" }
if (-not $msiAdd) { throw "Could not find vc_runtimeAdditional_x64.msi under layout dir: $LayoutDir" }

Write-Host "Extracting MSI payloads (administrative install)..." -ForegroundColor Cyan
Start-Process msiexec -Wait -ArgumentList "/a `"$($msiMin.FullName)`" /qn TARGETDIR=`"$MsiOutDir`""
Start-Process msiexec -Wait -ArgumentList "/a `"$($msiAdd.FullName)`" /qn TARGETDIR=`"$MsiOutDir`""

$sys32 = Join-Path $MsiOutDir "Windows\System32"
if (-not (Test-Path $sys32)) { throw "Expected extracted System32 folder not found: $sys32" }

Write-Host "Copying VC runtime DLLs into bundle..." -ForegroundColor Cyan
Copy-Item -Force -Path (Join-Path $sys32 "vcruntime140*.dll") -Destination $DestDir
Copy-Item -Force -Path (Join-Path $sys32 "msvcp140*.dll") -Destination $DestDir
Copy-Item -Force -Path (Join-Path $sys32 "concrt140.dll") -Destination $DestDir -ErrorAction SilentlyContinue
Copy-Item -Force -Path (Join-Path $sys32 "vcomp140.dll") -Destination $DestDir -ErrorAction SilentlyContinue
Copy-Item -Force -Path (Join-Path $sys32 "vcamp140.dll") -Destination $DestDir -ErrorAction SilentlyContinue

if (-not (Test-Path (Join-Path $DestDir "vcruntime140_1.dll"))) { throw "vcruntime140_1.dll missing after extraction" }
if (-not (Test-Path (Join-Path $DestDir "msvcp140.dll"))) { throw "msvcp140.dll missing after extraction" }
if (-not (Test-Path (Join-Path $DestDir "vcruntime140.dll"))) { throw "vcruntime140.dll missing after extraction" }

Write-Host "VC runtime DLLs bundled successfully." -ForegroundColor Green
""",
            encoding="utf-8",
        )

        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ps1),
                "-VcRedist",
                str(vc_redist_exe),
                "-LayoutDir",
                str(layout_dir),
                "-MsiOutDir",
                str(msi_out_dir),
                "-DestDir",
                str(dest_dir),
            ],
            check=True,
        )

        copied = sorted({p.name for p in dest_dir.glob("*.dll")})
        print(f"Bundled VC++ runtime DLLs next to python.exe: dest={dest_dir} dlls={copied}")

    def _remove_obsolete_typing_backport(self, py_path: Path) -> None:
        """
        PyInstaller error:
          "The 'typing' package is an obsolete backport of a standard library package and is incompatible with PyInstaller."

        This refers to the *PyPI distribution named `typing`*, not the stdlib module.
        """
        try:
            uv = self._ensure_uv_available()
            # `pip uninstall` returns non-zero when the package isn't installed; that's fine.
            subprocess.run(
                [uv, "pip", "uninstall", "--python", str(py_path), "-y", "typing"],
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
        v = os.environ.get("APEX_PATCH_DIFFUSERS_PEFT")
        if v is None:
            v = os.environ.get("APEX_BUNDLE_PATCH_DIFFUSERS_PEFT", "1")
        if str(v).strip().lower() in ("0", "false", "no", "off"):
            print("Skipping diffusers peft.py patch (APEX_PATCH_DIFFUSERS_PEFT=0)")
            return

        # We no longer ship a venv; `venv_dir` is the portable runtime root.
        py_path = self._find_python_executable(Path(venv_dir))

        # Patch diffusers first so it can be imported by subsequent patchers.
        # Some diffusers versions reference `torch.xpu` at import time; our Intel macOS
        # torch stack (torch==2.2.2) does not provide `torch.xpu`, so imports can crash.
        subprocess.run(
            [
                str(py_path),
                str(
                    self.project_root
                    / "scripts"
                    / "updates"
                    / "patch_diffusers_torch_xpu.py"
                ),
            ],
            check=True,
        )

        # Run the shared patcher in the target venv so bundling and dev installs stay consistent.
        subprocess.run(
            [
                str(py_path),
                str(
                    self.project_root
                    / "scripts"
                    / "updates"
                    / "patch_diffusers_peft.py"
                ),
            ],
            check=True,
        )

    def _build_and_install_rust_wheels(self, *, py_path: Path) -> None:
        rust_project = self.project_root / "rust" / "apex_download_rs"
        if not rust_project.exists():
            print(f"Rust project not found, skipping: {rust_project}")
            return

        # Ensure toolchain exists
        try:
            subprocess.run(
                ["cargo", "--version"], check=True, capture_output=True, text=True
            )
            subprocess.run(
                ["rustc", "--version"], check=True, capture_output=True, text=True
            )
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
                [
                    str(py_path),
                    "-c",
                    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')",
                ],
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

        uv = self._ensure_uv_available()

        # Install maturin into the environment (pyproject build backend is maturin), using uv.
        subprocess.run(
            [uv, "pip", "install", "--python", str(py_path), "maturin>=1.6,<2.0"],
            check=True,
        )

        # Build wheel for the exact interpreter we're bundling.
        #
        # IMPORTANT (macOS Apple Silicon): even when bundling an Intel (x86_64) Python
        # under Rosetta, a universal/arm64 `cargo` may default to building arm64 wheels
        # unless we explicitly pass a target. We therefore derive the Rust target from
        # the selected Python interpreter, not from the host machine.
        print(f"Building Rust wheel (maturin): {rust_project}")
        target: Optional[str] = None
        env: Optional[dict[str, str]] = None
        if self.platform_name == "darwin":
            try:
                probe = subprocess.run(
                    [str(py_path), "-c", "import platform; print(platform.machine())"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                py_machine = (probe.stdout or "").strip().lower()
            except Exception:
                py_machine = ""

            if py_machine in {"x86_64", "amd64"}:
                target = "x86_64-apple-darwin"
            elif py_machine in {"arm64", "aarch64"}:
                target = "aarch64-apple-darwin"

        cmd = [
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
        ]
        if target:
            cmd += ["--target", target]
            env = dict(os.environ)
            env["CARGO_BUILD_TARGET"] = target

        subprocess.run(cmd, cwd=str(rust_project), check=True, env=env)

        built_wheels = sorted(wheels_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
        if not built_wheels:
            raise RuntimeError(f"No wheels produced by maturin in: {wheels_dir}")

        # Install the newest wheel produced.
        wheel_path = built_wheels[-1]
        print(f"Installing Rust wheel: {wheel_path.name}")
        subprocess.run(
            [uv, "pip", "install", "--python", str(py_path), str(wheel_path)],
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
"%SCRIPT_DIR%\apex-studio\python.exe" -m src %*
"""
            launcher.write_text(content, encoding="utf-8")
            return

        launcher = bundle_dir / "apex-engine"
        content = """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
if [ -x "$SCRIPT_DIR/apex-studio/bin/python" ]; then
  exec "$SCRIPT_DIR/apex-studio/bin/python" -m src "$@"
fi
exec "$SCRIPT_DIR/apex-studio/bin/python3" -m src "$@"
"""
        launcher.write_text(content, encoding="utf-8")
        os.chmod(launcher, 0o755)

        # We no longer override an internal venv console-script because we don't ship a venv.

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

    def create_portable_python_bundle(self, venv_dir: Path) -> Path:
        """
        Create a self-contained bundle built around a venv (no PyInstaller).

        Layout (bundle_dir):
          - apex-studio/        # venv (shipped)
          - src/               # API code
          - assets/, manifest/, transformer_configs/, vae_configs/
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

        # Copy portable runtime (avoid symlinks for portability: we want a self-contained bundle).
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

        # Copy maintenance/update scripts into the bundle so installs can self-update without
        # needing access to the repo checkout.
        scripts_dir = bundle_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        # Copy known helper scripts.
        #
        # IMPORTANT: the repo keeps compatibility wrappers at scripts/*.py, but bundles must
        # ship the real implementations (so they work standalone after extraction).
        helper_scripts: list[tuple[Path, str]] = [
            (
                self.project_root / "scripts" / "updates" / "apply_code_update.py",
                "apply_code_update.py",
            ),
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
set PATH=%SCRIPT_DIR%\\apex-studio;%PATH%
if exist "%SCRIPT_DIR%\\apex-studio\\python.exe" (
  "%SCRIPT_DIR%\\apex-studio\\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
) else (
  "%SCRIPT_DIR%\\apex-studio\\Scripts\\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
)
"""
        else:
            launcher = bundle_dir / "start-api.sh"
            content = """#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/apex-studio/bin:$PATH"
if [ -x "$SCRIPT_DIR/apex-studio/bin/python" ]; then
  exec "$SCRIPT_DIR/apex-studio/bin/python" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
fi
exec "$SCRIPT_DIR/apex-studio/bin/python3" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765
"""

        with open(launcher, "w") as f:
            f.write(content)

        if self.platform_name != "win32":
            os.chmod(launcher, 0o755)


    def create_manifest(self, bundle_dir: Path, gpu_type: str) -> dict:
        """Create a manifest file with bundle information"""
        py_tag = self.venv_python_tag or f"cp{sys.version_info.major}{sys.version_info.minor}"
        py_ver = self.venv_python_version or f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        version = self.bundle_version or "0.0.0"
        manifest = {
            "version": version,
            "platform": self.platform_name,
            "arch": self._normalize_arch(self.target_arch),
            "gpu_support": gpu_type,
            "python_tag": py_tag,
            "python_version": py_ver,
            "bundle_type": "portable-python",
            "python_dirname": self.venv_name,
            "rust_wheels": not getattr(self, "skip_rust", False),
            "created_at": __import__("datetime").datetime.utcnow().isoformat()
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
        arch_raw = str(m.get("arch", platform.machine()))
        arch = self._safe_filename_component(self._normalize_arch(arch_raw))
        gpu = self._safe_filename_component(
            str(m.get("gpu_support", self.last_gpu_type or "unknown"))
        )
        py_tag = self._safe_filename_component(
            str(
                m.get(
                    "python_tag", f"cp{sys.version_info.major}{sys.version_info.minor}"
                )
            )
        )

        extras: List[str] = []
        if not m.get("rust_wheels", True):
            extras.append("norust")

        extra = ("-" + "-".join(extras)) if extras else ""
        return f"{prefix}-{version}-{plat}-{arch}-{gpu}-{py_tag}{extra}.zip"

    def _write_code_update_manifest(
        self, bundle_dir: Path, gpu_type: str, requirements_file: Path
    ) -> dict:
        """
        Create a small manifest describing the code-only update bundle.

        The goal is to give the installer/updater enough metadata to:
          - validate platform/arch/gpu/python compatibility,
          - decide whether dependency syncing is needed (based on requirements file hash),
          - display version info.
        """
        import hashlib

        # Code-only bundles should still be tagged with the Python ABI expected by the
        # installed runtime env, which is the venv we bundle in the full artifact.
        py_tag = self.venv_python_tag or f"cp{sys.version_info.major}{sys.version_info.minor}"
        req_bytes = (
            requirements_file.read_bytes() if requirements_file.exists() else b""
        )
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
            "arch": self._normalize_arch(self.target_arch),
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

        # Copy maintenance/update scripts (no venv in this bundle, but we want the updater/setup scripts
        # available alongside the updated code when extracted).
        try:
            scripts_dir = bundle_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            helper_scripts: list[tuple[Path, str]] = [
                (
                    self.project_root / "scripts" / "updates" / "apply_code_update.py",
                    "apply_code_update.py",
                ),
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
        self._write_code_update_manifest(
            bundle_dir=bundle_dir,
            gpu_type=gpu_type,
            requirements_file=requirements_file,
        )

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
        Create a .tar.zst of src_dir.

        This works cross-platform (including Windows) without requiring system `tar` or `zstd`.
        We intentionally do NOT depend on the portable-python cache here, since bundling cleans
        it up by default after producing the final artifacts.

        Strategy:
          - Prefer native `tar` + `zstd` if available (much faster; enables multithreaded zstd).
          - Fallback to a pure-Python implementation using `tarfile` + `zstandard`.
        """
        src_dir = Path(src_dir).resolve()
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

        def _should_exclude(rel_parts: list[str], name: str) -> bool:
            parts = set(rel_parts)
            if "__pycache__" in parts or "__MACOSX" in parts:
                return True
            if name in (".DS_Store",):
                return True
            if name.startswith("._"):
                return True
            if name.endswith((".pyc", ".pyo")):
                return True
            return False

        def _try_native_tar_zst() -> bool:
            """
            Try `tar | zstd` streaming archiving.
            Returns True on success, False if unavailable or failed.
            """
            tar_exe = shutil.which("tar")
            zstd_exe = shutil.which("zstd")
            if not tar_exe or not zstd_exe:
                return False

            # Create the archive from the parent dir so we can control whether the root
            # dir is included.
            if include_root_dir:
                cwd = src_dir.parent
                tar_target = src_dir.name
            else:
                cwd = src_dir
                tar_target = "."

            # Exclusions (keep conservative; mirrors Python fallback).
            exclude_patterns = [
                "__pycache__",
                "*/__pycache__",
                "__MACOSX",
                "*/__MACOSX",
                ".DS_Store",
                "*/.DS_Store",
                "._*",
                "*/._*",
                "*.pyc",
                "*.pyo",
            ]
            exclude_args: list[str] = []
            for pat in exclude_patterns:
                # Most tar implementations support both `--exclude PAT` and `--exclude=PAT`.
                exclude_args.extend(["--exclude", pat])

            tar_cmd = [tar_exe, "-cf", "-", *exclude_args, tar_target]

            # zstd:
            # -T0: use all cores
            # -<level>: compression level
            # -q: quieter logs (we print our own status)
            zstd_cmd = [
                zstd_exe,
                "-q",
                f"-{int(level)}",
                "-T0",
                "-o",
                str(out_path),
                "-",
            ]

            print(f"Creating tar.zst (native): {out_path.name} (from {src_dir})")

            try:
                tar_proc = subprocess.Popen(
                    tar_cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                )
                assert tar_proc.stdout is not None
                zstd_proc = subprocess.Popen(
                    zstd_cmd,
                    stdin=tar_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Allow tar to receive SIGPIPE if zstd exits.
                tar_proc.stdout.close()

                z_out, z_err = zstd_proc.communicate()
                t_err = tar_proc.stderr.read() if tar_proc.stderr else b""
                t_rc = tar_proc.wait()
                z_rc = zstd_proc.returncode

                if t_rc != 0 or z_rc != 0:
                    # Best-effort cleanup of partial output.
                    try:
                        out_path.unlink(missing_ok=True)  # py3.8+
                    except TypeError:
                        if out_path.exists():
                            out_path.unlink()

                    # Don't raise; fall back to Python archiving for maximum compatibility.
                    t_msg = (t_err or b"").decode(errors="ignore").strip()
                    z_msg = ((z_err or b"") + (z_out or b"")).decode(errors="ignore").strip()
                    if t_msg or z_msg:
                        print(
                            "Warning: native tar|zstd failed; falling back to Python tar.zst.\n"
                            f"tar: {t_msg}\n"
                            f"zstd: {z_msg}\n"
                        )
                    return False

                return True
            except Exception:
                try:
                    out_path.unlink(missing_ok=True)  # py3.8+
                except TypeError:
                    if out_path.exists():
                        out_path.unlink()
                return False

        # Fast path: native tar + zstd if available.
        if _try_native_tar_zst():
            print(f"Created tar.zst: {out_path}")
            return out_path

        # Fallback: pure Python (tarfile + zstandard).
        # Ensure the runner Python can import zstandard.
        runner_py = Path(sys.executable).resolve()
        try:
            subprocess.run(
                [str(runner_py), "-c", "import zstandard as zstd; print(zstd.__version__)"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            uv = self._ensure_uv_available()
            subprocess.run(
                [uv, "pip", "install", "--python", str(runner_py), "--upgrade", "zstandard"],
                check=True,
            )

        # Create tar.zst using a small helper script executed by the runner Python.
        # We stream tar -> zstd to avoid a massive intermediate .tar file.
        helper = r"""
import os, sys, tarfile

def should_exclude(rel_parts, name):
    parts = set(rel_parts)
    if "__pycache__" in parts or "__MACOSX" in parts:
        return True
    if name in (".DS_Store",):
        return True
    if name.startswith("._"):
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    return False

def main():
    import zstandard as zstd
    src_dir = os.path.abspath(sys.argv[1])
    out_path = os.path.abspath(sys.argv[2])
    include_root = sys.argv[3] == "1"
    level = int(sys.argv[4])

    root_name = os.path.basename(src_dir.rstrip(os.sep)) if include_root else ""

    os.environ["COPYFILE_DISABLE"] = "1"
    with open(out_path, "wb") as f:
        # Prefer multi-threaded zstd when supported.
        try:
            cctx = zstd.ZstdCompressor(level=level, threads=-1)
        except TypeError:
            cctx = zstd.ZstdCompressor(level=level)
        with cctx.stream_writer(f) as zfh:
            with tarfile.open(fileobj=zfh, mode="w|") as tf:
                for dirpath, dirnames, filenames in os.walk(src_dir):
                    # Ensure stable traversal order.
                    dirnames.sort()
                    filenames.sort()
                    rel_dir = os.path.relpath(dirpath, src_dir)
                    rel_parts = [] if rel_dir in (".", "") else rel_dir.split(os.sep)

                    # Exclude entire dirs early.
                    dirnames[:] = [
                        d for d in dirnames
                        if not should_exclude(rel_parts + [d], d)
                    ]

                    for fn in filenames:
                        if should_exclude(rel_parts + [fn], fn):
                            continue
                        abs_path = os.path.join(dirpath, fn)
                        rel_path = os.path.relpath(abs_path, src_dir)
                        arc = os.path.join(root_name, rel_path) if root_name else rel_path
                        arc = arc.replace(os.sep, "/")
                        tf.add(abs_path, arcname=arc, recursive=False)

if __name__ == "__main__":
    main()
"""
        print(f"Creating tar.zst: {out_path.name} (from {src_dir})")
        subprocess.run(
            [
                str(runner_py),
                "-c",
                helper,
                str(src_dir),
                str(out_path),
                "1" if include_root_dir else "0",
                str(int(level)),
            ],
            check=True,
        )
        print(f"Created tar.zst: {out_path}")
        return out_path

    def _cleanup_intermediate_portable_python(self) -> None:
        """
        Remove intermediate portable-python artifacts under the output directory.

        Deletes:
          - `<output>/<venv_name>` (intermediate runtime root, e.g. `<output>/apex-studio`)
          - `<output>/_portable-python` (portable Python download/extraction cache)
        """
        runtime_dir = (self.output_dir / self.venv_name).resolve()
        cache_dir = Path(self.portable_python_root).resolve()

        try:
            if runtime_dir.exists():
                shutil.rmtree(runtime_dir)
        except Exception:
            pass

        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        except Exception:
            pass

    def bundle(self) -> Path:
        """Run the full bundling process (full env bundle + code-only update bundle)."""
        print(f"Bundling Python API for platform: {self.platform_name}")

        # Create output directory
        # if path exists remove it
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)

        # Detect GPU support
        gpu_type = self.cuda_version or self.detect_gpu_support()
        self.last_gpu_type = gpu_type
        print(f"GPU support: {gpu_type}")

        # Create requirements
        req_file = self.create_requirements(gpu_type)

        # Create portable runtime (NO venv) and install dependencies directly into it.
        runtime_dir = self.create_portable_python_runtime(req_file)

        # Patch third-party deps inside the portable runtime before we copy it into the shipped bundle.
        self._patch_diffusers_set_adapter_scale(runtime_dir)
        # Patch xformers (if installed) in the runtime, by resolving the installed module
        # path and editing in-place (avoids applying a broad patches directory).
        py_path = self._find_python_executable(runtime_dir)
        subprocess.run(
            [
                str(py_path),
                str(
                    self.project_root
                    / "scripts"
                    / "updates"
                    / "patch_xformers_flash3.py"
                ),
            ],
            check=True,
        )

        # Create a lockfile for update-time dependency syncing.
        # Do this after all installs/patches so the lock reflects the final environment.
        try:
            self.create_lockfile(requirements_file=req_file, venv_dir=runtime_dir)
        except Exception as e:
            print(f"Warning: failed to create requirements.lock: {e}")

        # Build portable-python-based bundle (no venv, no PyInstaller)
        bundle_dir = self.create_portable_python_bundle(runtime_dir)

        # Smoke test the shipped bundle (strongly recommended).
        smoke_env = os.environ.get("APEX_BUNDLE_SMOKE_TESTS")
        if self.run_smoke_tests and (
            smoke_env is None
            or str(smoke_env).strip().lower() not in ("0", "false", "no", "off")
        ):
            self._run_smoke_tests(bundle_dir=bundle_dir, gpu_type=gpu_type)

 
        # Create manifest
        self.create_manifest(bundle_dir, gpu_type=gpu_type)

        # Create code-only update bundle (small; used for frequent updates)
        try:
            self.create_code_only_bundle(requirements_file=req_file, gpu_type=gpu_type)
        except Exception as e:
            # Best-effort: code-only bundle should not break full bundle creation.
            print(f"Warning: failed to create code-only update bundle: {e}")

        # Cleanup intermediate portable-python artifacts unless explicitly requested.
        if not self.keep_build_venv:
            self._cleanup_intermediate_portable_python()

        print(f"Bundle created: {bundle_dir}")
        return bundle_dir


def main():
    parser = argparse.ArgumentParser(description="Bundle Python API for Apex Studio")
    parser.add_argument(
        "--arch",
        choices=["auto", "x86_64", "arm64"],
        default="auto",
        help=(
            "Target CPU architecture for the bundle. "
            "macOS supports x86_64 (Intel) and arm64 (Apple Silicon). "
            "Windows/Linux bundles are built for x86_64. Default: auto-detect."
        ),
    )
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
        "--gpu",
        choices=[
            "cuda",
            "cpu",
            "mps",
            "rocm",
            "auto",
        ],
        default="auto",
        help=(
            "GPU backend to bundle. "
            "Use 'auto' to detect this machine; use 'cuda' for NVIDIA, 'rocm' for AMD, or 'cpu'. "
            "Use 'mps' for Apple Silicon."
        ),
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
        help=(
            "Python interpreter to use for the venv ONLY when --no-portable-python is set. "
            "By default the bundler downloads a portable Python and uses it for all installs/tests."
        ),
    )
    parser.add_argument(
        "--no-portable-python",
        action="store_true",
        help="Disable portable-Python bootstrap and use --python / the current interpreter instead (not recommended).",
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
        help="Fail if building a CUDA GPU bundle but CUDA is not available to run GPU kernel smoke tests.",
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
        "--no-tar-zst",
        action="store_true",
        help="Disable writing .tar.zst artifacts after bundling (not recommended).",
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

    # Auto-detect GPU backend
    gpu = str(getattr(args, "gpu", "auto") or "auto")
    if gpu == "auto":
        gpu_version = None
    elif gpu == "cuda":
        # Treat generic "cuda" as "latest supported" CUDA stack for bundling.
        gpu_version = "cuda"
    elif gpu == "mps":
        gpu_version = "mps"
    else:
        gpu_version = gpu

    # Target arch (controls portable-Python selection and macOS Intel special-case).
    def _norm_arch(a: str) -> str:
        low = (a or "").strip().lower()
        if low in {"amd64", "x64"}:
            return "x86_64"
        if low in {"aarch64"}:
            return "arm64"
        return low

    arch = args.arch
    if arch == "auto":
        arch = _norm_arch(platform.machine())
    else:
        arch = _norm_arch(arch)

    # We only build conservative universal bundles for these platforms.
    if platform_name in ("win32", "linux"):
        arch = "x86_64"
    if platform_name == "darwin" and arch not in ("x86_64", "arm64"):
        arch = "arm64"

    use_portable_python = not bool(args.no_portable_python)
    py_exe = args.python_executable or sys.executable

    # Resolve bundle version (CLI > env > pyproject).
    bundle_version = (args.bundle_version or "").strip() or None
    if not bundle_version:
        bundle_version = (
            os.environ.get("APEX_BUNDLE_VERSION", "") or ""
        ).strip() or None
    if not bundle_version:
        bundle_version = _read_project_version_from_pyproject(
            Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        )

    bundler = PythonBundler(
        platform_name=platform_name,
        output_dir=args.output,
        cuda_version=gpu_version,
        python_executable=py_exe,
        bundle_version=bundle_version,
        target_arch=arch,
        use_portable_python=use_portable_python,
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
            bundler._zip_dir(
                bundler.code_dist_dir, code_zip_path, include_root_dir=True
            )

    if not args.no_tar_zst:
        if args.tar_zst_scope == "bundle":
            tar_src = bundle_dir
        elif args.tar_zst_scope == "python-code":
            tar_src = bundler.code_dist_dir
        else:
            tar_src = bundler.dist_dir

        default_name = bundler.default_zip_name(args.tar_zst_scope).replace(
            ".zip", ".tar.zst"
        )
        tar_name = args.tar_zst_name or default_name
        tar_path = args.tar_zst_output or (bundler.output_dir / tar_name)
        bundler._tar_zst_dir(
            tar_src, tar_path, include_root_dir=True, level=args.tar_zst_level
        )

        # If the caller created the full python-api tar.zst artifact, also emit a code-only tar.zst by default.
        if args.tar_zst_scope == "python-api" and bundler.code_dist_dir.exists():
            code_tar_name = bundler.default_zip_name("python-code").replace(
                ".zip", ".tar.zst"
            )
            code_tar_path = bundler.output_dir / code_tar_name
            bundler._tar_zst_dir(
                bundler.code_dist_dir,
                code_tar_path,
                include_root_dir=True,
                level=args.tar_zst_level,
            )

    print(f"\nBundle complete: {bundle_dir}")
    if platform_name == "win32":
        print(f"To test: cd {bundle_dir} && apex-engine.bat start --daemon --port 8765")
    else:
        print(f"To test: cd {bundle_dir} && ./apex-engine start --daemon --port 8765")


if __name__ == "__main__":
    main()
