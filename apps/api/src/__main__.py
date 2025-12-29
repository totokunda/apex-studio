from __future__ import annotations
import os, subprocess, signal, psutil, sys, shlex, importlib.util
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


def _num_cpus():
    return multiprocessing.cpu_count()


def create_procfile(procfile: Path, mode="dev"):
    # Ray manages workers internally, so we only need the API server
    if mode == "dev":
        host = os.getenv("APEX_HOST", "0.0.0.0")
        port = os.getenv("APEX_PORT", "8765")
        start = (
            "api: uvicorn src.api.main:app "
            f"--host {host} --port {port} --reload\n"
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
        if parts and parts[0] in {"uvicorn", "gunicorn"}:
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
    daemon: bool = typer.Option(
        False, "--daemon", "-d", help="Run as daemon in background"
    ),
):
    """
    Start FastAPI + Celery (and anything else in your Procfile) via Honcho.
    Equivalent to: honcho start -f Procfile [-e .env]
    """
    cwd = cwd.resolve()

    # Resolve procfile/envfile relative to cwd (matches how honcho interprets -f/-e)
    procfile_path = procfile if procfile.is_absolute() else (cwd / procfile)
    envfile_path = None
    if envfile is not None:
        envfile_path = envfile if envfile.is_absolute() else (cwd / envfile)

    mode = "dev" if procfile_path.name.endswith(".dev") else "prod"
    create_procfile(procfile_path, mode)

    log_path = (cwd / "apex-engine-start.log") if daemon else None
    _load_envfile_if_present(envfile_path)

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
            f.write(
                "Honcho not available; running Procfile 'api' command directly.\n"
            )
    _run(cmd, cwd=cwd, daemon=daemon, log_path=log_path)


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
