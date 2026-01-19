import os
import json
import signal
import threading
import ray
from .settings import settings
from loguru import logger

# Suppress noisy Ray/abseil stack traces on SIGTERM (must be set before ray.init)
# These are cosmetic logs from Ray's C++ layer during shutdown
os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")
# Reduce raylet verbosity during shutdown
os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "warning")
# Disable abseil failure signal handler stack traces
os.environ.setdefault("ABSL_FLAGS_symbolize_stacktrace", "0")
os.environ.setdefault("GLOG_minloglevel", "2")  # Only log errors, not warnings

# Lock to prevent concurrent ray.init() calls (causes "core worker already initialized" crash)
_ray_init_lock = threading.Lock()

# Track whether we've installed the shutdown handler
_shutdown_handler_installed = False


def _init_ray() -> None:
    """
    Initialize Ray (once) with sane defaults for this service.

    IMPORTANT: do NOT initialize Ray at import-time. When running under uvicorn,
    import-time side effects interact badly with multi-worker preload/forking and
    can lead to very slow startup or hung workers.
    """
    # Fast path: already initialized (no lock needed)
    if ray.is_initialized():
        return

    with _ray_init_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if ray.is_initialized():
            return

        # Prepare spilling config directory
        spill_dir = settings.results_dir / "ray_spill"
        spill_dir.mkdir(parents=True, exist_ok=True)

        # Try to initialize with dashboard, fall back without if packages missing
        try:
            ray.init(
                ignore_reinit_error=True,
                num_cpus=os.cpu_count(),
                dashboard_host="0.0.0.0",
                dashboard_port=settings.ray_dashboard_port,
                include_dashboard=True,
                _metrics_export_port=0,  # Disable metrics agent
                _system_config={
                    "automatic_object_spilling_enabled": True,
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {"directory_path": str(spill_dir)},
                        }
                    ),
                },
            )
            _install_shutdown_handler()
            logger.info(f"Ray initialized with {ray.available_resources()}")
            logger.info(
                f"Ray dashboard available at http://localhost:{settings.ray_dashboard_port}"
            )
        except Exception as e:
            if "Cannot include dashboard" in str(e):
                logger.warning(
                    "Ray dashboard packages not installed, starting without dashboard"
                )
                logger.info(
                    "To enable dashboard: pip install 'ray[default,dashboard]' aiohttp aiohttp-cors grpcio"
                )
                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=os.cpu_count(),
                    include_dashboard=False,
                    _metrics_export_port=0,  # Disable metrics agent
                    _system_config={
                        "automatic_object_spilling_enabled": True,
                        "object_spilling_config": json.dumps(
                            {
                                "type": "filesystem",
                                "params": {"directory_path": str(spill_dir)},
                            }
                        ),
                    },
                )
                _install_shutdown_handler()
                logger.info(f"Ray initialized with {ray.available_resources()}")
            else:
                raise


def get_ray_app():
    """Get the Ray runtime"""
    if not ray.is_initialized():
        _init_ray()
    return ray


def shutdown_ray():
    """Shutdown Ray runtime"""
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")


def _graceful_shutdown_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully to avoid noisy Ray/JAX stack traces."""
    logger.info(f"Received signal {signum}, shutting down Ray gracefully...")
    shutdown_ray()
    # Re-raise as SystemExit to let uvicorn/FastAPI handle the rest
    raise SystemExit(0)


def _install_shutdown_handler():
    """Install signal handlers for graceful Ray shutdown (once)."""
    global _shutdown_handler_installed
    if _shutdown_handler_installed:
        return
    _shutdown_handler_installed = True

    # Only install in main thread (signal handlers must be set from main thread)
    try:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _graceful_shutdown_handler)
            # Don't override SIGINT - let uvicorn handle Ctrl+C normally
            logger.debug("Installed graceful shutdown handler for SIGTERM")
    except Exception as e:
        logger.debug(f"Could not install shutdown handler: {e}")
