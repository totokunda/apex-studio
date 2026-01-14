from __future__ import annotations
import os, subprocess, signal, psutil, sys, shlex, importlib.util, time
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
from pathlib import Path
import typer
import torch
import multiprocessing
from dotenv import load_dotenv

load_dotenv()


def _num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def create_procfile(procfile: Path, mode="dev"):
    # Ray manages workers internally, so we only need the API server
    if mode == "dev":
        # Default to loopback so the desktop app can connect without extra config.
        # Users who need LAN access can set APEX_HOST=0.0.0.0 or pass --host.
        host = os.getenv("APEX_HOST", "127.0.0.1")
        port = os.getenv("APEX_PORT", "8765")
        start = (
            "api: uvicorn src.api.main:app " f"--host {host} --port {port} --reload\n"
        )
    elif mode == "prod":
        start = "api: gunicorn src.api.main:app --config gunicorn.conf.py\n"
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
        if parts and parts[0] in {"uvicorn", "gunicorn"}:
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
    effective_host = os.getenv("APEX_HOST") or ("127.0.0.1" if mode in {"dev", "prod"} else "127.0.0.1")
    effective_port = os.getenv("APEX_PORT", "8765")
    client_host = "127.0.0.1" if effective_host in {"0.0.0.0", "::", "[::]"} else effective_host
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
    Internal command used by frozen (PyInstaller) builds to start the API without relying on
    `python -m ...` module execution.
    """
    # Resolve from CLI first, then environment (start() sets these when flags are passed).
    if host is None:
        host = os.getenv("APEX_HOST", "127.0.0.1")
    if port is None:
        port = int(os.getenv("APEX_PORT", "8765"))

    # Mirror key production defaults from `apps/api/gunicorn.conf.py`, but for uvicorn.
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
        print(f"Dual-stack enabled; binding to: {', '.join(f'{b}:{port}' for b in binds)}")

    # Worker count + loglevel follow `gunicorn.conf.py` semantics.
    env = os.getenv("ENVIRONMENT")
    workers = int(os.getenv("APEX_UVICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1")
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
        workers = int(os.getenv("APEX_UVICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1")
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


def main():
    """Module entrypoint: enables `python -m src.__main__ ...`."""
    app()


if __name__ == "__main__":
    main()
