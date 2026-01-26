from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .common import SmokeContext, log, fail


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get(url: str, *, timeout: float) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def run(ctx: SmokeContext) -> None:
    """
    Start the API (uvicorn) and verify it's reachable.

    This catches issues that "import works" won't:
    - event loop / uvicorn startup issues
    - missing runtime deps required at startup
    - import cycles triggered by ASGI loading path
    """
    log("[smoke] api start (uvicorn)")

    port = _pick_free_port()
    host = "127.0.0.1"

    env = os.environ.copy()
    # Ensure bundle code is importable in the subprocess (matches shipped launchers).
    env["PYTHONPATH"] = str(ctx.bundle_root)
    env["APEX_BUNDLE_SMOKE_TEST"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    p = subprocess.Popen(
        cmd,
        cwd=str(ctx.bundle_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for /openapi.json to become reachable.
        deadline = time.time() + 45.0
        last_err: Exception | None = None
        while time.time() < deadline:
            if p.poll() is not None:
                out = (p.stdout.read() if p.stdout else "") if p.stdout else ""
                err = (p.stderr.read() if p.stderr else "") if p.stderr else ""
                fail(
                    "uvicorn exited early during smoke test.\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout:\n{out}\n"
                    f"stderr:\n{err}"
                )
            try:
                data = _http_get(f"http://{host}:{port}/openapi.json", timeout=5.0)
                if not data:
                    fail("API started but /openapi.json returned empty response")
                log("[smoke] api start ok")
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_err = e
                time.sleep(0.25)

        fail(f"Timed out waiting for API to start (last error: {last_err})")
    finally:
        # Terminate uvicorn
        try:
            if p.poll() is None:
                if os.name == "nt":
                    p.terminate()
                else:
                    p.send_signal(signal.SIGTERM)
        except Exception:
            pass
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
