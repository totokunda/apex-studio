from fastapi import FastAPI
from .ws import router as ws_router
from .manifest import router as manifest_router
from .config import router as config_router
from .engine import router as engine_router
from .preprocessor import router as preprocessor_router
from .postprocessor import router as postprocessor_router
from .jobs import router as jobs_router
from .mask import router as mask_router
from .components import router as components_router
from .system import router as system_router
from .files import router as files_router
from .download import router as download_router
from .ray import router as ray_router
from fastapi.middleware.cors import CORSMiddleware
from .ray_app import get_ray_app, shutdown_ray
from contextlib import asynccontextmanager
import asyncio
from typing import Optional
import os
import threading
import time
from .stability import install_stability_middleware
from .log_suppression import install_http_log_suppression

_ray_ready: bool = False
_ray_start_error: Optional[str] = None


def _start_parent_watchdog() -> None:
    """
    Ownership-safe shutdown: if this server was launched by Apex Studio (Electron),
    it will set APEX_PARENT_PID. If that parent process goes away, we exit.
    This avoids orphaned API servers without ever killing unrelated processes.
    """
    # Allow manual/CLI runs to opt out of parent ownership checks.
    # Any non-empty value enables disablement (e.g. "1", "true").
    if os.getenv("APEX_DISABLE_PARENT_WATCHDOG"):
        return
    raw = os.getenv("APEX_PARENT_PID")
    if not raw:
        return
    try:
        parent_pid = int(raw)
    except Exception:
        return

    def _exists(pid: int) -> bool:
        try:
            # signal 0: existence check (works on Unix; on Windows it raises if missing)
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _loop() -> None:
        # small delay so parent fully initializes
        time.sleep(1.0)
        while True:
            if not _exists(parent_pid):
                os._exit(0)
            time.sleep(1.0)

    t = threading.Thread(target=_loop, name="apex-parent-watchdog", daemon=True)
    t.start()


async def _start_background_services() -> None:
    """
    Start Ray + dependent background services without blocking API startup.
    Keeps `/health` responsive quickly while Ray spins up.
    """
    global _ray_ready, _ray_start_error
    try:
        await asyncio.to_thread(get_ray_app)

        # Initialize the Ray websocket bridge
        from .ws_manager import get_ray_ws_bridge

        get_ray_ws_bridge()

        # Initialize preprocessor download tracking
        from .preprocessor_registry import initialize_download_tracking

        initialize_download_tracking()

        _ray_ready = True
    except Exception as e:
        _ray_start_error = repr(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_parent_watchdog()
    # Startup: initialize Ray and related services in the background (non-blocking)
    startup_task = asyncio.create_task(_start_background_services())

    # Start background task for polling Ray updates
    from .preprocessor import poll_ray_updates

    poll_task = asyncio.create_task(poll_ray_updates())

    # Start background task for automatic code updates (non-blocking)
    from .auto_update import auto_update_loop

    auto_update_task = asyncio.create_task(auto_update_loop())

    yield

    # Shutdown: Cancel polling task and close Ray
    startup_task.cancel()
    poll_task.cancel()
    auto_update_task.cancel()
    try:
        await startup_task
    except asyncio.CancelledError:
        pass
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
    try:
        await auto_update_task
    except asyncio.CancelledError:
        pass
    shutdown_ray()


install_http_log_suppression()

app = FastAPI(name="Apex Engine", lifespan=lifespan)

# Keep the local API stable even if the desktop app becomes "chatty" (polling loops, retries, etc.).
install_stability_middleware(app)

app.include_router(ws_router)
app.include_router(manifest_router)
app.include_router(config_router)
app.include_router(preprocessor_router)
app.include_router(postprocessor_router)
app.include_router(mask_router)
app.include_router(components_router)
app.include_router(jobs_router)
app.include_router(system_router)
app.include_router(engine_router)
app.include_router(files_router)
app.include_router(download_router)
app.include_router(ray_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def read_root():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # Readiness probe: indicates whether Ray + dependent services finished starting.
    if _ray_ready:
        return {"status": "ready"}
    return {"status": "starting", "error": _ray_start_error}
