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


_ray_ready: bool = False
_ray_start_error: Optional[str] = None


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
    # Startup: initialize Ray and related services in the background (non-blocking)
    startup_task = asyncio.create_task(_start_background_services())

    # Start background task for polling Ray updates
    from .preprocessor import poll_ray_updates

    poll_task = asyncio.create_task(poll_ray_updates())

    yield

    # Shutdown: Cancel polling task and close Ray
    startup_task.cancel()
    poll_task.cancel()
    try:
        await startup_task
    except asyncio.CancelledError:
        pass
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
    shutdown_ray()

app = FastAPI(name="Apex Engine", lifespan=lifespan)
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
