from __future__ import annotations
import os, subprocess, signal, psutil, sys, shlex, importlib.util, time, shutil
import re
import json
import urllib.request
import urllib.parse
import platform as _platform
from dataclasses import dataclass
from typing import Any, Optional

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
from pathlib import Path
import typer
import torch
import multiprocessing
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency for local/dev convenience
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()


def _num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def create_procfile(procfile: Path, mode="dev"):
    # Ray manages workers internally, so we only need the API server
    host = os.getenv("APEX_HOST", "127.0.0.1")
    port = os.getenv("APEX_PORT", "8765")
    workers = os.getenv("APEX_UVICORN_WORKERS", "4")
    loglevel = os.getenv("APEX_LOG_LEVEL", "info")
    if mode == "dev":
        # Default to loopback so the desktop app can connect without extra config.
        # Users who need LAN access can set APEX_HOST=0.0.0.0 or pass --host.

        start = (
            "api: uvicorn src.api.main:app " f"--host {host} --port {port} --reload\n"
        )
    else:
        # Procfiles are parsed line-by-line as "<name>: <command>". Avoid using shell-specific
        # line continuations (like Windows '^') and always include the process name.
        start = (
            "api: uvicorn src.api.main:app "
            f"--host {host} --port {port} "
            f"--workers {workers} "
            '--proxy-headers --forwarded-allow-ips="*" '
            f"--timeout-keep-alive 5 --log-level {loglevel}\n"
        )
    with open(procfile, "w") as f:
        f.write(start)


def create_envfile(envfile: Path, mode="dev"):
    if mode == "dev":
        with open(envfile, "w") as f:
            f.write(f"NUM_GPUS={_num_gpus()}")
    elif mode == "prod":
        with open(envfile, "w") as f:
            f.write(f"NUM_GPUS={_num_gpus()}")


app = typer.Typer(help="Apex command line")


def _honcho_available() -> bool:
    return importlib.util.find_spec("honcho") is not None


def _is_frozen() -> bool:
    # PyInstaller sets sys.frozen = True in the bundled executable.
    return bool(getattr(sys, "frozen", False))


def _load_envfile_if_present(envfile: Path | None) -> None:
    if envfile is None:
        return
    if envfile.exists():
        load_dotenv(dotenv_path=envfile, override=True)


def _proc_cmd_from_procfile(procfile_path: Path, name: str = "api") -> list[str]:
    """
    Parse a Heroku-style Procfile and return the command for the requested process name.
    Only minimal parsing is needed for our use-case.
    """
    if not procfile_path.exists():
        raise FileNotFoundError(f"Procfile not found: {procfile_path}")

    for raw in procfile_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        proc_name, cmd = line.split(":", 1)
        if proc_name.strip() != name:
            continue
        cmd = cmd.strip()
        if not cmd:
            raise ValueError(f"Empty command for process '{name}' in {procfile_path}")
        parts = shlex.split(cmd)
        # Prefer running via the active interpreter to avoid PATH/env mismatch.
        #
        # IMPORTANT: In a PyInstaller frozen build, sys.executable is the *apex-engine binary*,
        # not a Python interpreter. So `sys.executable -m uvicorn ...` would become:
        #   apex-engine -m uvicorn ...
        # which Typer then treats as CLI flags and errors ("No such option: -m").
        if parts and parts[0] in {"uvicorn"}:
            if _is_frozen():
                # We only need to support the API server start; use our internal serve command.
                # We ignore most Procfile flags and rely on APEX_HOST/APEX_PORT.
                parts = [sys.executable, "serve"]
            else:
                parts = [sys.executable, "-m", parts[0], *parts[1:]]
        return parts

    raise ValueError(f"Process '{name}' not found in {procfile_path}")


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    daemon: bool = False,
    log_path: Path | None = None,
):
    if daemon:
        # Run in background as daemon
        stdout = None
        stderr = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = open(log_path, "a", buffering=1)
            stdout = log_f
            stderr = log_f
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=stdout if stdout is not None else subprocess.DEVNULL,
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        print(f"Started daemon process with PID: {proc.pid}")
        if log_path is not None:
            print(f"Logs: {log_path}")
        return proc.pid
    else:
        # Run in foreground
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        raise SystemExit(proc.returncode)


@app.command()
def start(
    procfile: Path = typer.Option(
        Path("Procfile"), "--procfile", "-f", help="Path to Procfile"
    ),
    envfile: Path | None = typer.Option(None, "--env", "-e", help=".env file to load"),
    cwd: Path = typer.Option(
        Path("."), "--cwd", help="Working directory where processes run"
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Host to bind the API to (sets APEX_HOST).",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Port to bind the API to (sets APEX_PORT).",
    ),
    daemon: bool = typer.Option(
        False, "--daemon", "-d", help="Run as daemon in background"
    ),
):
    """
    Start FastAPI (and anything else in your Procfile) via Honcho.
    Equivalent to: honcho start -f Procfile [-e .env]
    """
    cwd = cwd.resolve()

    # Allow callers (Electron bundle / CLI users) to override host/port without needing to
    # manage environment variables themselves.
    if host is not None:
        os.environ["APEX_HOST"] = str(host)
    if port is not None:
        os.environ["APEX_PORT"] = str(port)

    # Resolve procfile/envfile relative to cwd (matches how honcho interprets -f/-e)
    procfile_path = procfile if procfile.is_absolute() else (cwd / procfile)
    envfile_path = None
    if envfile is not None:
        envfile_path = envfile if envfile.is_absolute() else (cwd / envfile)

    mode = "dev" if procfile_path.name.endswith(".dev") else "prod"
    create_procfile(procfile_path, mode)

    # Helpful connection hint: 0.0.0.0 is a *bind address* (server-side), not a client URL.
    # Most users should connect via 127.0.0.1 (or localhost).
    effective_host = os.getenv("APEX_HOST") or (
        "127.0.0.1" if mode in {"dev", "prod"} else "127.0.0.1"
    )
    effective_port = os.getenv("APEX_PORT", "8765")
    client_host = (
        "127.0.0.1" if effective_host in {"0.0.0.0", "::", "[::]"} else effective_host
    )
    print(f"API should be reachable at: http://{client_host}:{effective_port}")

    log_path = (cwd / "apex-engine-start.log") if daemon else None
    _load_envfile_if_present(envfile_path)

    # In frozen builds we cannot use `sys.executable -m honcho ...` because sys.executable
    # points to this binary. Use our internal server runner instead.
    if _is_frozen():
        _run([sys.executable, "serve"], cwd=cwd, daemon=daemon, log_path=log_path)
        return

    if _honcho_available():
        args = [sys.executable, "-m", "honcho", "start", "-f", str(procfile_path)]
        if envfile_path is not None:
            args += ["-e", str(envfile_path)]
        _run(args, cwd=cwd, daemon=daemon, log_path=log_path)
        return

    # Fallback: procfile only contains `api:` in our project, so we can run it directly.
    cmd = _proc_cmd_from_procfile(procfile_path, name="api")
    if daemon and log_path is not None:
        with open(log_path, "a", buffering=1) as f:
            f.write("Honcho not available; running Procfile 'api' command directly.\n")
    _run(cmd, cwd=cwd, daemon=daemon, log_path=log_path)


@app.command(name="serve", hidden=True)
def internal_serve(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Host to bind the API to (overrides APEX_HOST).",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Port to bind the API to (overrides APEX_PORT).",
    ),
    dual_stack: bool = typer.Option(
        False,
        "--dual-stack",
        help="Enable dual-stack binding (IPv4 + IPv6). Default is IPv4 only.",
    ),
):
    """
    Internal command used by frozen builds to start the API without relying on
    `python -m ...` module execution.
    """
    # Resolve from CLI first, then environment (start() sets these when flags are passed).
    if host is None:
        host = os.getenv("APEX_HOST", "127.0.0.1")
    if port is None:
        port = int(os.getenv("APEX_PORT", "8765"))

    # Mirror key production defaults, but for uvicorn.
    backlog = 2048
    keepalive = 2
    # Desktop/local stability: do NOT terminate the server after N requests by default.
    # This can still be enabled (e.g. to mitigate long-running memory leaks) by setting:
    #   APEX_MAX_REQUESTS=1000
    # Note: Uvicorn expects "None" (or CLI flag omitted) to disable.
    max_requests = int(os.getenv("APEX_MAX_REQUESTS", "0") or "0")

    # Default to single-stack (IPv4 only) to avoid Ray race conditions with multiple processes.
    # Enable dual-stack via --dual-stack flag or APEX_DUAL_STACK=1 env var.
    enable_dual_stack = dual_stack or bool(os.getenv("APEX_DUAL_STACK"))

    def _binds_for(host_value: str) -> list[str]:
        if host_value in {"127.0.0.1", "localhost"}:
            return ["127.0.0.1", "::1"] if enable_dual_stack else ["127.0.0.1"]
        if host_value in {"0.0.0.0", "::", "[::]"}:
            return ["0.0.0.0", "::"] if enable_dual_stack else ["0.0.0.0"]
        return [host_value]

    binds = _binds_for(host)
    if enable_dual_stack:
        print(
            f"Dual-stack enabled; binding to: {', '.join(f'{b}:{port}' for b in binds)}"
        )

    # Worker count + loglevel follow semantics.
    env = os.getenv("ENVIRONMENT")
    workers = int(
        os.getenv("APEX_UVICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1"
    )
    loglevel = "info"
    reload = False
    if env == "development":
        reload = True
        workers = 1
        loglevel = "debug"
    elif env == "staging":
        workers = int(
            os.getenv("APEX_UVICORN_WORKERS")
            or os.getenv("WEB_CONCURRENCY")
            or str(multiprocessing.cpu_count())
        )
        loglevel = "warning"
    elif env == "production":
        workers = int(
            os.getenv("APEX_UVICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1"
        )
        loglevel = "error"

    def _uvicorn_cmd(h: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            h,
            "--port",
            str(port),
            "--log-level",
            loglevel,
            "--backlog",
            str(backlog),
            "--timeout-keep-alive",
            str(keepalive),
        ]
        if max_requests > 0:
            cmd += ["--limit-max-requests", str(max_requests)]
        if reload:
            cmd.append("--reload")
        else:
            # Uvicorn workers are only supported in non-reload mode.
            if workers > 1:
                cmd += ["--workers", str(workers)]
        return cmd

    # Run Uvicorn. On macOS we support dual-stack loopback binds by running two servers.
    procs: list[subprocess.Popen] = []
    try:
        for h in binds:
            procs.append(subprocess.Popen(_uvicorn_cmd(h)))

        # Wait: if any server exits, terminate the rest and bubble up the exit code.
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    for other in procs:
                        if other is not p and other.poll() is None:
                            try:
                                other.terminate()
                            except Exception:
                                pass
                    raise SystemExit(rc)
            time.sleep(0.25)
    except KeyboardInterrupt:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass


def _find_apex_processes():
    """Find running processes related to apex engine (uvicorn, ray, honcho)"""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            # Look for our specific processes
            if any(
                pattern in cmdline
                for pattern in [
                    "uvicorn src.api.main:app",
                    "ray::",
                    "honcho start",
                    " -m honcho start",
                ]
            ):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return processes


@app.command()
def stop(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force kill processes (SIGKILL instead of SIGTERM)"
    ),
):
    """
    Stop running Apex Engine processes (uvicorn, ray, honcho).
    Finds and terminates any existing server processes.
    """
    processes = _find_apex_processes()

    if not processes:
        print("No running Apex Engine processes found.")
        return

    print(f"Found {len(processes)} running process(es):")
    for proc in processes:
        try:
            cmdline = " ".join(proc.cmdline())
            print(f"  PID {proc.pid}: {cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    signal_type = signal.SIGKILL if force else signal.SIGTERM
    signal_name = "SIGKILL" if force else "SIGTERM"

    killed_count = 0
    for proc in processes:
        try:
            print(f"Sending {signal_name} to PID {proc.pid}...")
            proc.send_signal(signal_type)
            killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Could not kill PID {proc.pid}: {e}")

    if killed_count > 0:
        print(f"Successfully sent {signal_name} to {killed_count} process(es).")
        if not force:
            print(
                "Processes should shutdown gracefully. Use --force if they don't stop."
            )
    else:
        print("No processes were terminated.")


# Optional sugar: `apex dev` alias
@app.command()
def dev(cwd: Path = Path(".")):
    """Convenience alias for apex start -f Procfile.dev"""
    create_procfile(cwd / "Procfile.dev", "dev")
    _run([sys.executable, "-m", "honcho", "start", "-f", "Procfile.dev"], cwd=cwd)


@app.command()
def bundle(
    platform: str = typer.Option(
        "auto",
        "--platform",
        help="Target platform for bundle (darwin|linux|win32|auto). Default: auto (this machine).",
    ),
    gpu: str = typer.Option(
        "auto",
        "--gpu",
        help="GPU backend for bundle (cuda|mps|cpu|rocm|auto). Default: auto.",
    ),
    output: Path = typer.Option(
        Path("./dist"),
        "--output",
        help="Output directory for bundle artifacts (relative to apps/api/ by default).",
    ),
    bundle_version: str | None = typer.Option(
        None,
        "--bundle-version",
        help="Bundle version to write into manifests/artifact names (default: env/pyproject).",
    ),
    nightly: bool = typer.Option(
        False,
        "--nightly/--no-nightly",
        help="Mark the bundle as a nightly (adds a -nightly.<timestamp> suffix to the version).",
    ),
    tar_zst: bool = typer.Option(
        True,
        "--tar-zst/--no-tar-zst",
        help="Create a .tar.zst artifact after bundling (default: on).",
    ),
    tar_zst_level: int = typer.Option(
        12,
        "--tar-zst-level",
        help="Zstd compression level for .tar.zst (default: 12).",
    ),
    venv_python: str | None = typer.Option(
        None,
        "--python",
        help=(
            "Python interpreter to use for the bundled venv (forwarded to bundler). "
            "If omitted, bundler chooses a recommended interpreter automatically."
        ),
    ),
    runner_python: str = typer.Option(
        "python3.12",
        "--runner-python",
        help=(
            "Python interpreter used to run the bundling script itself (default: python3.12). "
            "Falls back to the current interpreter if not found on PATH."
        ),
    ),
):
    """
    Bundle the Python API for distribution.

    This is a thin wrapper around `scripts/bundle_python.py` with sensible defaults:
    - platform/gpu default to auto-detect for this machine
    - tar.zst enabled with level 12
    - signing enabled
    """ 
    
    project_root = Path(__file__).resolve().parent.parent  # apps/api/
    script = project_root / "scripts" / "bundle_python.py"
    if not script.exists():
        raise typer.BadParameter(f"Bundling script not found: {script}")
    


    runner = shutil.which(runner_python) or runner_python
    if shutil.which(runner_python) is None and runner_python == "python3.12":
        # Best-effort fallback for dev environments where python3.12 isn't installed.
        runner = sys.executable
        print(
            "Warning: `python3.12` not found on PATH; "
            f"running bundler with current interpreter: {runner}"
        )

    cmd: list[str] = [
        runner,
        str(script),
        "--platform",
        str(platform),
        "--gpu",
        str(gpu),
        "--output",
        str(output),
    ]

    # Resolve effective version (CLI > env > pyproject), then apply nightly suffix if requested.
    effective_version = (bundle_version or "").strip() or None
    if not effective_version:
        effective_version = (os.environ.get("APEX_BUNDLE_VERSION", "") or "").strip() or None
    if not effective_version:
        effective_version = _read_project_version(project_root / "pyproject.toml")
    effective_version = _with_nightly_version(str(effective_version or "0.0.0"), nightly)
    if effective_version:
        cmd += ["--bundle-version", str(effective_version)]
    if venv_python:
        cmd += ["--python", str(venv_python)]
    if tar_zst:
        cmd.append("--tar-zst")
        cmd += ["--tar-zst-level", str(int(tar_zst_level))]

    # Run from the API project root so relative paths (like ./dist) behave as expected.
    proc = subprocess.Popen(cmd, cwd=str(project_root))
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    raise SystemExit(proc.returncode)


def _safe_filename_component(value: str) -> str:
    """
    Match the bundler's filename sanitization: keep [A-Za-z0-9._-], map others to '-'.
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


def _pick_newest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No paths provided")
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _nightly_suffix() -> str:
    """
    Generate a unique-ish nightly suffix.
    Format: nightly.YYYYMMDDHHMMSS (UTC).
    """
    return "nightly." + time.strftime("%Y%m%d%H%M%S", time.gmtime())


def _with_nightly_version(version: str, nightly: bool) -> str:
    v = (version or "").strip()
    if not nightly:
        return v
    if not v:
        v = "0.0.0"
    # If the caller already provided a prerelease/nightly version string, keep it.
    if "nightly" in v.lower():
        return v
    return f"{v}-{_nightly_suffix()}"


def _read_project_version(pyproject_path: Path) -> str | None:
    """
    Best-effort parse of `pyproject.toml` to read `[project].version`.
    """
    if not pyproject_path.exists():
        return None
    try:
        import tomllib  # pyright: ignore[reportMissingImports]
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Fallback to simple regex if no toml parser is available
            try:
                content = pyproject_path.read_text()
                match = re.search(r'\[project\].*?version\s*=\s*["\']([^"\']+)["\']', content, re.DOTALL)
                if match:
                    return match.group(1)
            except:
                pass
            return None

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version")
    except:
        return None


@app.command()
def publish(
    version: str | None = typer.Option(
        None,
        "--version",
        help="Release version folder/name (default: from pyproject.toml). Used for bundle version and HF upload folder.",
    ),
    nightly: bool = typer.Option(
        False,
        "--nightly/--no-nightly",
        help="Publish as a nightly (adds a -nightly.<timestamp> suffix to the version/folder name).",
    ),
    platform: str = typer.Option(
        "auto",
        "--platform",
        help="Target platform for bundle (darwin|linux|win32|auto). Default: auto (this machine).",
    ),
    gpu: str = typer.Option(
        "auto",
        "--gpu",
        help="GPU backend for bundle (cuda|mps|cpu|rocm|auto). Default: auto.",
    ),
    output: Path = typer.Option(
        Path("./dist"),
        "--output",
        help="Output directory for bundle artifacts (relative to apps/api/ by default).",
    ),
    tar_zst_level: int = typer.Option(
        12,
        "--tar-zst-level",
        help="Zstd compression level for .tar.zst (default: 12).",
    ),
    venv_python: str | None = typer.Option(
        None,
        "--python",
        help=(
            "Python interpreter to use for the bundled venv (forwarded to bundler). "
            "If omitted, bundler chooses a recommended interpreter automatically."
        ),
    ),
    runner_python: str = typer.Option(
        "python3.12",
        "--runner-python",
        help=(
            "Python interpreter used to run the bundling script itself (default: python3.12). "
            "Falls back to the current interpreter if not found on PATH."
        ),
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help="Hugging Face repo id (default: uploader script default).",
    ),
    repo_type: str | None = typer.Option(
        None,
        "--repo-type",
        help="Hugging Face repo type: model|dataset|space (default: uploader script default).",
    ),
    commit_message: str = typer.Option(
        "Upload Apex Studio server release artifacts",
        "--commit-message",
        help="Commit message to use for the upload",
    ),
):
    """
    Bundle + create .tar.zst artifacts, then upload them to Hugging Face.

    Requirements:
    - version is optional (default: from pyproject.toml)
    - uploads the generated tarballs (python-api + python-code)
    - if HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set, you will be prompted before upload
    """
    if _is_frozen():
        raise typer.BadParameter("publish is not supported in frozen builds.")

    project_root = Path(__file__).resolve().parent.parent  # apps/api/
    if version is None:
        version = _read_project_version(project_root / "pyproject.toml")

    version = _with_nightly_version((version or "").strip(), nightly)
    if not version:
        raise typer.BadParameter("version is required (could not be determined from pyproject.toml).")

    # Enforce signing requirements where we actually support it.
    effective_platform = platform
    if effective_platform == "auto":
        effective_platform = sys.platform
    
    bundle_script = project_root / "scripts" / "bundle_python.py"
    if not bundle_script.exists():
        raise typer.BadParameter(f"Bundling script not found: {bundle_script}")

    runner = shutil.which(runner_python) or runner_python
    if shutil.which(runner_python) is None and runner_python == "python3.12":
        runner = sys.executable
        print(
            "Warning: `python3.12` not found on PATH; "
            f"running bundler with current interpreter: {runner}"
        )

    started_at = time.time()

    bundle_cmd: list[str] = [
        runner,
        str(bundle_script),
        "--platform",
        str(platform),
        "--gpu",
        str(gpu),
        "--output",
        str(output),
        "--bundle-version",
        str(version),
        "--tar-zst",
        "--tar-zst-level",
        str(int(tar_zst_level)),
    ]
    if venv_python:
        bundle_cmd += ["--python", str(venv_python)]

    # Run from the API project root so relative paths (like ./dist) behave as expected.
    proc = subprocess.Popen(bundle_cmd, cwd=str(project_root))
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    # Resolve the bundler output directory relative to apps/api/ (since we ran with cwd=project_root).
    out_dir = output if output.is_absolute() else (project_root / output)
    out_dir = out_dir.resolve()

    safe_version = _safe_filename_component(version)
    api_candidates = list(out_dir.glob(f"python-api-{safe_version}-*.tar.zst"))
    code_candidates = list(out_dir.glob(f"python-code-{safe_version}-*.tar.zst"))

    # Prefer tarballs created by *this* publish run.
    api_recent = [p for p in api_candidates if p.stat().st_mtime >= started_at - 2]
    code_recent = [p for p in code_candidates if p.stat().st_mtime >= started_at - 2]
    api_tar = _pick_newest(api_recent or api_candidates)
    code_tar = _pick_newest(code_recent or code_candidates)

    upload_script = project_root / "scripts" / "release" / "upload_release_artifacts.py"
    if not upload_script.exists():
        raise typer.BadParameter(f"Upload script not found: {upload_script}")

    upload_cmd: list[str] = [
        sys.executable,
        str(upload_script),
        "--version",
        str(version),
        "--api-tar",
        str(api_tar),
        "--code-tar",
        str(code_tar),
        "--commit-message",
        str(commit_message),
    ]
    if repo_id:
        upload_cmd += ["--repo-id", str(repo_id)]
    if repo_type:
        upload_cmd += ["--repo-type", str(repo_type)]

    # Let the uploader prompt for a token if needed.
    res = subprocess.run(upload_cmd, cwd=str(project_root))
    raise SystemExit(res.returncode)


# ---------------------------------------------------------------------------
# Update commands (used by the desktop app)
# ---------------------------------------------------------------------------


def _semver_triplet_prefix(v: str) -> Optional[tuple[int, int, int]]:
    """
    Parse an X.Y.Z semver triplet from the start of a string.
    Accepts suffixes like "1.2.3-nightly.20260119".
    """
    # Also accept a leading "v" (common in release tags / folders): "v1.2.3".
    m = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)", (v or "").strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _host_platform_candidates() -> set[str]:
    base = sys.platform.lower()
    out = {base}
    if base.startswith("win"):
        out |= {"win32", "windows"}
    if base == "darwin":
        out |= {"macos"}
    return out


def _host_arch_candidates() -> set[str]:
    m = (_platform.machine() or "").lower()
    out: set[str] = {m} if m else set()
    if m in {"x86_64", "amd64"}:
        out |= {"x64", "x86_64", "amd64"}
    if m in {"aarch64", "arm64"}:
        out |= {"arm64", "aarch64"}
    return out


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_installed_manifest(target_dir: Path) -> dict:
    # Prefer the code-update manifest if present.
    #
    # Reason: code-only updates replace `src/` + related assets, but do NOT update the
    # original bundle manifest. If we always prefer apex-engine-manifest.json, the
    # updater can think it's still on the old version and keep reporting an update
    # available even after applying it.
    m2 = target_dir / "apex-code-update-manifest.json"
    if m2.exists():
        data = _read_json_file(m2)
        v = str(data.get("version") or "").strip()
        if _semver_triplet_prefix(v) is not None:
            return data

    # Fallback: full bundle manifest.
    m = target_dir / "apex-engine-manifest.json"
    if m.exists():
        return _read_json_file(m)
    return {}


@app.command()
def version(
    format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
        show_default=True,
    ),
    target_dir: Path = typer.Option(
        None,
        "--target-dir",
        help="Path to the installed apex-engine directory (defaults to best-effort auto-detect).",
    ),
):
    """
    Print the installed Apex Engine version.

    This is resolved from the installed manifest (prefers the code-update manifest if present).
    """
    target = (target_dir or _default_target_dir()).expanduser().resolve()
    m = _read_installed_manifest(target)
    v = _current_installed_version(target)

    if format.strip().lower() == "json":
        payload = {
            "version": v,
            "target_dir": str(target),
            "manifest_kind": str(m.get("kind") or "bundle"),
            "platform": str(m.get("platform") or ""),
            "arch": str(m.get("arch") or ""),
            "gpu_support": str(m.get("gpu_support") or ""),
            "python_tag": str(m.get("python_tag") or ""),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(v)


def _current_installed_version(target_dir: Path) -> str:
    m = _read_installed_manifest(target_dir)
    v = str(m.get("version") or "").strip()
    if _semver_triplet_prefix(v):
        return v

    # Dev/checkout fallback: allow `python -m src.__main__ version` to be useful even
    # without a bundle manifest.
    try:
        pv = _read_project_version(target_dir / "pyproject.toml")
        if pv and _semver_triplet_prefix(str(pv)):
            return str(pv)
    except Exception:
        pass
    return "0.0.0"


def _current_gpu_support(target_dir: Path) -> str:
    m = _read_installed_manifest(target_dir)
    gpu = str(m.get("gpu_support") or "").strip().lower()
    return gpu or "cpu"


def _default_target_dir() -> Path:
    """
    Best-effort guess of the installed apex-engine directory.
    - In a bundled install, this file lives at <root>/src/__main__.py => target_dir=<root>.
    - In a dev checkout, this will resolve to apps/api/ (we refuse to mutate it unless allowed).
    """
    env = (
        os.environ.get("APEX_ENGINE_ROOT")
        or os.environ.get("APEX_ENGINE_DIR")
        or os.environ.get("APEX_ENGINE_TARGET_DIR")
    )
    if env:
        return Path(env).expanduser().resolve()
    try:
        here = Path(__file__).resolve()
        candidate = here.parent.parent
        if (candidate / "src").is_dir():
            return candidate
    except Exception:
        pass
    return Path.cwd().resolve()


def _safe_to_update(target_dir: Path, *, allow_dev: bool) -> None:
    """
    Avoid mutating a dev checkout unless explicitly enabled.
    """
    target_dir = Path(target_dir).resolve()
    if not (target_dir / "src").is_dir():
        raise typer.BadParameter(f"Not an apex-engine directory (missing src/): {target_dir}")

    if (target_dir / "apex-engine-manifest.json").exists():
        return

    if allow_dev:
        return

    if os.environ.get("APEX_ALLOW_DEV_UPDATE", "").strip().lower() in {"1", "true", "yes"}:
        return

    raise typer.BadParameter(
        "Refusing to update a dev checkout (no apex-engine-manifest.json found). "
        "Pass --target-dir pointing to an installed bundle, or pass --allow-dev."
    )


def _encode_hf_path(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p]
    return "/".join(urllib.parse.quote(p, safe="") for p in parts)


def _hf_json(url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "apex-engine",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8"))


def _hf_list_tree(owner: str, repo: str, path: str | None = None) -> list[dict]:
    base = f"https://huggingface.co/api/models/{owner}/{repo}/tree/main"
    url = f"{base}/{_encode_hf_path(path)}" if path else base
    data = _hf_json(url)
    return data if isinstance(data, list) else []


@dataclass(frozen=True)
class UpdateAsset:
    tag: str
    tag_version: str
    prerelease: bool
    asset_name: str
    asset_version: str
    platform: str
    arch: str
    device: str
    python_tag: str
    download_url: str


def _is_prerelease_tag(tag: str) -> bool:
    t = (tag or "").strip().lower()
    if "nightly" in t or "prerelease" in t:
        return True
    # Only treat strings that *start* with a semver triplet as candidates for
    # semver prerelease detection. (Many non-version strings like
    # "python-code-0.1.1-darwin-..." contain '-' but are not prerelease tags.)
    if _semver_triplet_prefix(t) is None:
        return False
    # Semver prerelease is indicated by a '-' immediately after X.Y.Z.
    return bool(re.match(r"^\s*v?\d+\.\d+\.\d+-", t, flags=re.IGNORECASE))


def _parse_python_code_asset(asset_name: str) -> Optional[dict]:
    """
    Accept:
      python-code-<version>-<platform>-<arch>-<device>-<pythonTag>[-extras].tar.zst

    Robust to <version> containing '-' (nightlies).
    """
    name = (asset_name or "").strip()
    if not name.lower().endswith(".tar.zst"):
        return None
    if not name.lower().startswith("python-code-"):
        return None

    core = name[: -len(".tar.zst")]
    rest = core[len("python-code-") :]
    tokens = [t for t in rest.split("-") if t]
    if len(tokens) < 5:
        return None

    # Find python tag from the end to tolerate version strings containing '-'.
    py_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        if re.fullmatch(r"cp\d{2,3}", tokens[i], flags=re.IGNORECASE):
            py_idx = i
            break
    if py_idx is None or py_idx < 3:
        return None

    plat = tokens[py_idx - 3]
    arch = tokens[py_idx - 2]
    device = tokens[py_idx - 1]
    py = tokens[py_idx]
    version_tokens = tokens[: py_idx - 3]
    if not version_tokens:
        return None
    version = "-".join(version_tokens)

    return {
        "asset_version": version,
        "platform": plat,
        "arch": arch,
        "device": device,
        "python_tag": py,
    }


def _list_remote_python_code_assets(
    *, owner: str, repo: str, allow_nightly: bool
) -> list[UpdateAsset]:
    plat_ok = _host_platform_candidates()
    arch_ok = _host_arch_candidates()
    py_ok = _python_tag().lower()

    root = _hf_list_tree(owner, repo)

    version_dirs: list[tuple[str, str, bool]] = []
    for item in root:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "directory":
            continue
        p = str(item.get("path") or "")
        if not p:
            continue
        base = (p.split("/")[-1] or "").strip()
        if re.fullmatch(r"v?\d+\.\d+\.\d+", base, flags=re.IGNORECASE):
            version_dirs.append((p, base.lstrip("vV"), False))
            continue
        if allow_nightly:
            if base.lower() in {"nightly", "nightlies"}:
                try:
                    nested = _hf_list_tree(owner, repo, p)
                except Exception:
                    nested = []
                for child in nested:
                    if not isinstance(child, dict):
                        continue
                    if str(child.get("type") or "") != "directory":
                        continue
                    cp = str(child.get("path") or "")
                    if not cp:
                        continue
                    cbase = (cp.split("/")[-1] or "").strip()
                    ver = cbase.lstrip("vV")
                    if _semver_triplet_prefix(ver) is not None or _is_prerelease_tag(cbase):
                        version_dirs.append((cp, ver, True))
            else:
                ver = base.lstrip("vV")
                if _semver_triplet_prefix(ver) is not None or _is_prerelease_tag(base):
                    version_dirs.append((p, ver, True))

    def _dir_key(d: tuple[str, str, bool]) -> tuple[tuple[int, int, int], int, str]:
        path, ver, pre = d
        sem = _semver_triplet_prefix(ver) or (0, 0, 0)
        # Prefer stable over prerelease when equal semver.
        pr = 0 if not pre else 1
        return (sem, pr, path)

    version_dirs.sort(key=_dir_key, reverse=True)

    out: list[UpdateAsset] = []
    for dir_path, ver, is_pre in version_dirs:
        tag = (dir_path.split("/")[-1] or "").strip()
        entries = _hf_list_tree(owner, repo, dir_path)
        for e in entries:
            if not isinstance(e, dict):
                continue
            if str(e.get("type") or "") != "file":
                continue
            full_path = str(e.get("path") or "")
            if not full_path:
                continue
            asset_name = full_path.split("/")[-1] or ""
            if not asset_name:
                continue
            parsed = _parse_python_code_asset(asset_name)
            if not parsed:
                continue

            if str(parsed["python_tag"]).lower() != py_ok:
                continue
            if str(parsed["platform"]).lower() not in plat_ok:
                continue
            if str(parsed["arch"]).lower() not in arch_ok:
                continue

            dl = (
                f"https://huggingface.co/{owner}/{repo}/resolve/main/"
                f"{_encode_hf_path(full_path)}?download=true"
            )
            out.append(
                UpdateAsset(
                    tag=tag,
                    tag_version=ver,
                    prerelease=bool(is_pre),
                    asset_name=asset_name,
                    asset_version=str(parsed["asset_version"]),
                    platform=str(parsed["platform"]),
                    arch=str(parsed["arch"]),
                    device=str(parsed["device"]),
                    python_tag=str(parsed["python_tag"]),
                    download_url=dl,
                )
            )

    def _asset_key(a: UpdateAsset) -> tuple:
        sem = _semver_triplet_prefix(a.asset_version) or _semver_triplet_prefix(a.tag_version) or (0, 0, 0)
        stable = 1 if not a.prerelease else 0
        # Try to sort nightlies by a numeric suffix if present (best-effort).
        nightly_num = 0
        m = re.search(r"(?:nightly[._-]?)(\d{6,})", a.asset_version.lower())
        if m:
            try:
                nightly_num = int(m.group(1))
            except Exception:
                nightly_num = 0
        return (sem, stable, nightly_num, a.asset_name)

    out.sort(key=_asset_key, reverse=True)
    if not allow_nightly:
        out = [a for a in out if not a.prerelease and "nightly" not in a.asset_version.lower()]
    return out


def _cache_dir() -> Path:
    # Cross-platform-ish default; can be overridden by --download-dir.
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "apex-engine" / "updates"
    return Path.home().expanduser().resolve() / ".cache" / "apex-engine" / "updates"


def _is_within_dir(path: Path, root: Path) -> bool:
    """
    Return True if `path` resolves under `root`.
    """
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except Exception:
        try:
            p = str(Path(path).resolve())
            r = str(Path(root).resolve())
            return p == r or p.startswith(r + os.sep)
        except Exception:
            return False


def _cleanup_cached_update_archive(archive_path: Path, *, cache_root: Path, quiet: bool) -> None:
    """
    Delete the downloaded update archive from the cache after a successful install.
    Only deletes if the archive lives under `cache_root` (to avoid deleting user files).
    """
    archive_path = Path(archive_path).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    if not _is_within_dir(archive_path, cache_root):
        return

    # Also remove any leftover partial download (best-effort).
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    for p in (archive_path, part):
        try:
            p.unlink(missing_ok=True)  # py3.8+
        except TypeError:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    if not quiet:
        print(f"Removed cached update archive: {archive_path}")


def _download_to(url: str, dest: Path, *, quiet: bool) -> None:
    dest = Path(dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    req = urllib.request.Request(url, headers={"User-Agent": "apex-engine"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = None
        try:
            total = int(r.headers.get("Content-Length") or "0") or None
        except Exception:
            total = None

        tmp = dest.with_suffix(dest.suffix + ".part")
        wrote = 0
        started = time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                wrote += len(chunk)
                if not quiet and total:
                    pct = (wrote / total) * 100.0
                    elapsed = max(0.001, time.time() - started)
                    mbps = (wrote / 1024.0 / 1024.0) / elapsed
                    print(
                        f"Downloading… {pct:5.1f}% ({wrote}/{total} bytes, {mbps:.1f} MiB/s)",
                        end="\r",
                        flush=True,
                    )
        os.replace(tmp, dest)
        if not quiet:
            print(f"Downloaded: {dest}")


def _apply_code_update(archive_path: Path, target_dir: Path, *, quiet: bool) -> None:
    archive_path = Path(archive_path).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    if not archive_path.exists():
        raise typer.BadParameter(f"Archive not found: {archive_path}")

    # Prefer scripts shipped with the bundle root (installed scenario).
    script = target_dir / "scripts" / "apply_code_update.py"
    if not script.exists():
        # Dev fallback: repo layout (apps/api/scripts)
        repo_script = Path(__file__).resolve().parent.parent / "scripts" / "apply_code_update.py"
        if repo_script.exists():
            script = repo_script
    if not script.exists():
        raise typer.BadParameter(f"apply_code_update.py not found under {target_dir}/scripts")

    cmd = [
        sys.executable,
        str(script),
        "--code-archive",
        str(archive_path),
        "--target-dir",
        str(target_dir),
    ]
    if quiet:
        cmd.append("--quiet")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


@app.command(name="check-updates")
def check_updates(
    format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
        show_default=True,
    ),
    allow_nightly: bool = typer.Option(
        False,
        "--allow-nightly",
        help="Also include nightly/prerelease update artifacts.",
    ),
    target_dir: Path = typer.Option(
        None,
        "--target-dir",
        help="Path to the installed apex-engine directory (defaults to best-effort auto-detect).",
    ),
    repo_owner: str = typer.Option(
        "totoku",
        "--repo-owner",
        help="Update repo owner (Hugging Face).",
        show_default=True,
    ),
    repo_name: str = typer.Option(
        "apex-studio-server",
        "--repo-name",
        help="Update repo name (Hugging Face).",
        show_default=True,
    ),
    any_device: bool = typer.Option(
        False,
        "--any-device",
        help="List updates for any device backend (cpu/mps/cuda/rocm). Default is to prefer the installed device backend.",
    ),
):
    """Check for updates to the apex-engine.

    This command checks the remote Hugging Face repo for available updates to the apex-engine.
    It can list updates for any device backend (cpu/mps/cuda/rocm) or prefer the installed device backend.
    It can also include nightly/prerelease update artifacts.
    It can output the results in text or json format.
    """
    target = (target_dir or _default_target_dir()).expanduser().resolve()
    current_version = _current_installed_version(target)
    current_gpu = _current_gpu_support(target)
    current_sem = _semver_triplet_prefix(current_version) or (0, 0, 0)
    current_is_prerelease = _is_prerelease_tag(current_version)

    assets = _list_remote_python_code_assets(
        owner=repo_owner, repo=repo_name, allow_nightly=allow_nightly
    )
 
    if not any_device:
        preferred = [a for a in assets if a.device.lower() == current_gpu.lower()]
    else:
        preferred = assets

    def _is_newer(a: UpdateAsset) -> bool:
        a_sem = _semver_triplet_prefix(a.asset_version) or (0, 0, 0)
        if a_sem > current_sem:
            return True
        # If the user is on a prerelease/nightly of X.Y.Z, then the stable X.Y.Z
        # should be offered in the regular (non-nightly) channel.
        if (
            a_sem == current_sem
            and current_is_prerelease
            and (not allow_nightly)
            and (not a.prerelease)
            and (not _is_prerelease_tag(a.asset_version))
            and ("nightly" not in str(a.asset_version).lower())
            and str(a.asset_version).strip() != str(current_version).strip()
        ):
            return True
        if allow_nightly and a_sem == current_sem:
            # Nightlies/prereleases within the same base version are only "newer"
            # if the current install is ALSO a prerelease/nightly. This prevents
            # offering e.g. 0.1.1-nightly.* as an "update" when already on stable 0.1.1.
            a_is_pre = bool(a.prerelease) or _is_prerelease_tag(a.asset_version) or (
                "nightly" in str(a.asset_version).lower()
            )
            if not current_is_prerelease and a_is_pre:
                return False
            return str(a.asset_version).strip() != str(current_version).strip()
        return False

    updates = [a for a in preferred if _is_newer(a)]

    if format.strip().lower() == "json":
        payload = {
            "current": {
                "version": current_version,
                "device": current_gpu,
                "python_tag": _python_tag(),
                "platform": sys.platform,
                "arch": _platform.machine(),
            },
            "updates": [
                {
                    "version": a.asset_version,
                    "tag": a.tag,
                    "prerelease": bool(a.prerelease),
                    "asset": a.asset_name,
                    "download_url": a.download_url,
                    "platform": a.platform,
                    "arch": a.arch,
                    "device": a.device,
                    "python_tag": a.python_tag,
                }
                for a in updates
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    # Text output
    if not updates:
        print(f"No updates available. Current version: {current_version}")
        return
    print(f"Current version: {current_version} (device={current_gpu}, python={_python_tag()})")
    print("Available updates:")
    for a in updates:
        pre = " (nightly)" if (a.prerelease or "nightly" in a.asset_version.lower()) else ""
        print(f"- {a.asset_version}{pre}  [{a.asset_name}]")


@app.command()
def update(
    from_tar: Path | None = typer.Option(
        None,
        "--from-tar",
        help="Apply an update from a local python-code-*.tar.zst file.",
    ),
    allow_nightly: bool = typer.Option(
        False,
        "--allow-nightly",
        help="Also allow installing nightly/prerelease update artifacts.",
    ),
    target_dir: Path = typer.Option(
        None,
        "--target-dir",
        help="Path to the installed apex-engine directory (defaults to best-effort auto-detect).",
    ),
    download_dir: Path | None = typer.Option(
        None,
        "--download-dir",
        help="Directory to cache downloaded update artifacts.",
    ),
    repo_owner: str = typer.Option(
        "totoku",
        "--repo-owner",
        help="Update repo owner (Hugging Face).",
        show_default=True,
    ),
    repo_name: str = typer.Option(
        "apex-studio-server",
        "--repo-name",
        help="Update repo name (Hugging Face).",
        show_default=True,
    ),
    any_device: bool = typer.Option(
        False,
        "--any-device",
        help="Allow installing an update for any device backend. Default prefers the installed device backend.",
    ),
    allow_dev: bool = typer.Option(
        False,
        "--allow-dev",
        help="Allow applying updates to a dev checkout (dangerous).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Reduce output.",
    ),
):
    """Apply an update to the apex-engine.

    This command applies an update to the apex-engine from a local python-code-*.tar.zst file or from a remote Hugging Face repo.
    It can also apply an update from a local python-code-*.tar.zst file.
    It can also apply an update from a remote Hugging Face repo.
    It can also apply an update to a dev checkout.
    It can output the results in text or json format.
    """
    target = (target_dir or _default_target_dir()).expanduser().resolve()
    _safe_to_update(target, allow_dev=allow_dev)

    cache_root = (download_dir or _cache_dir()).expanduser().resolve()
    if from_tar is not None:
        _apply_code_update(from_tar, target, quiet=quiet)
        _cleanup_cached_update_archive(from_tar, cache_root=cache_root, quiet=quiet)
        if not quiet:
            print("Update applied from local archive. Restart the app/engine to use it.")
        return

    current_version = _current_installed_version(target)
    current_gpu = _current_gpu_support(target)
    current_sem = _semver_triplet_prefix(current_version) or (0, 0, 0)
    current_is_prerelease = _is_prerelease_tag(current_version)

    assets = _list_remote_python_code_assets(
        owner=repo_owner, repo=repo_name, allow_nightly=allow_nightly
    )
    if not any_device:
        preferred = [a for a in assets if a.device.lower() == current_gpu.lower()]
        candidates = preferred or assets
    else:
        candidates = assets

    def _is_newer(a: UpdateAsset) -> bool:
        a_sem = _semver_triplet_prefix(a.asset_version) or (0, 0, 0)
        if a_sem > current_sem:
            return True
        if (
            a_sem == current_sem
            and current_is_prerelease
            and (not allow_nightly)
            and (not a.prerelease)
            and (not _is_prerelease_tag(a.asset_version))
            and ("nightly" not in str(a.asset_version).lower())
            and str(a.asset_version).strip() != str(current_version).strip()
        ):
            return True
        if allow_nightly and a_sem == current_sem:
            a_is_pre = bool(a.prerelease) or _is_prerelease_tag(a.asset_version) or (
                "nightly" in str(a.asset_version).lower()
            )
            if not current_is_prerelease and a_is_pre:
                return False
            return str(a.asset_version).strip() != str(current_version).strip()
        return False

    candidates = [a for a in candidates if _is_newer(a)]
    if not candidates:
        if not quiet:
            print(f"No updates available. Current version: {current_version}")
        return

    chosen = candidates[0]
    archive_path = cache_root / chosen.asset_name
    if not quiet:
        print(f"Updating {current_version} -> {chosen.asset_version}")
        print(f"Downloading: {chosen.asset_name}")
    _download_to(chosen.download_url, archive_path, quiet=quiet)
    _apply_code_update(archive_path, target, quiet=quiet)
    _cleanup_cached_update_archive(archive_path, cache_root=cache_root, quiet=quiet)
    if not quiet:
        print("Update applied. Restart the app/engine to use it.")


def main():
    """Module entrypoint: enables `python -m src.__main__ ...`."""
    app()


if __name__ == "__main__":
    main()
