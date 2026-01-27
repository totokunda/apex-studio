import os
import json
import signal
import threading
import socket
import ray
from .settings import settings
from loguru import logger

# Suppress noisy Ray/abseil stack traces on SIGTERM (must be set before ray.init)
# These are cosmetic logs from Ray's C++ layer during shutdown
os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")
# Reduce raylet verbosity - set to "error" to suppress file system monitor warnings
os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "error")
# Disable abseil failure signal handler stack traces
os.environ.setdefault("ABSL_FLAGS_symbolize_stacktrace", "0")
os.environ.setdefault("GLOG_minloglevel", "3")  # Only log fatal errors (3 = FATAL, 2 = ERROR)
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

# Reduce peak CUDA init pressure in spawned Ray workers (helps avoid Windows
# WinError 1455 "paging file too small" during torch CUDA DLL/module loading).
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Safer allocator defaults (no-op on non-CUDA builds).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# Ray init is not thread-safe. We must prevent concurrent init *and* provide a way for
# other threads to wait until init is fully finished (ray.is_initialized() can become
# True before the core worker is ready).
_ray_init_cond = threading.Condition()
_ray_init_state: str = "not_started"  # not_started | starting | ready | failed
_ray_init_error: Exception | None = None

# Track whether we've installed the shutdown handler
_shutdown_handler_installed = False


def _port_available(host: str, port: int) -> bool:
    """
    Best-effort check whether we can bind to (host, port).
    Used to avoid Ray dashboard startup failures when the port is already taken.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
        return True
    except Exception:
        return False


def _dashboard_deps_available() -> bool:
    """
    Ray's dashboard requires extra dependencies in many environments.
    We preflight-import them so we don't call `ray.init(include_dashboard=True)` and
    then try to "fallback" with a second `ray.init()` (which can crash the process).
    """
    try:
        import aiohttp  # noqa: F401
        import aiohttp_cors  # noqa: F401
        # grpc is used by parts of ray dashboard stack on some installs.
        import grpc  # noqa: F401

        return True
    except Exception:
        return False


def _init_ray_once() -> None:
    """
    Initialize Ray (once) with sane defaults for this service.

    IMPORTANT: do NOT initialize Ray at import-time. When running under uvicorn,
    import-time side effects interact badly with multi-worker preload/forking and
    can lead to very slow startup or hung workers.
    """
    # Prepare spilling config directory
    spill_dir = settings.results_dir / "ray_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)

    # Decide dashboard behavior up-front.
    # Calling ray.init() twice in the same process (even as a "fallback") can crash Ray's
    # C++ core worker ("The process is already initialized for core worker.").
    want_dashboard = os.getenv("APEX_RAY_DASHBOARD", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    dashboard_ok = _dashboard_deps_available()
    # Ray binds the dashboard to 0.0.0.0; preflight that exact bind to catch
    # conflicts on any interface (not just loopback).
    port_ok = _port_available("0.0.0.0", int(settings.ray_dashboard_port))
    include_dashboard = bool(want_dashboard and dashboard_ok and port_ok)
    if want_dashboard and not dashboard_ok:
        logger.warning(
            "Ray dashboard deps missing; starting Ray without dashboard. "
            "To enable: pip install 'ray[default,dashboard]' aiohttp aiohttp-cors grpcio"
        )
    if want_dashboard and dashboard_ok and not port_ok:
        logger.warning(
            f"Ray dashboard port {settings.ray_dashboard_port} is already in use; "
            "starting Ray without dashboard."
        )

    # Start local Ray by default. This intentionally ignores Ray's own `RAY_ADDRESS`
    # env var unless the user explicitly opts into cluster mode via `APEX_RAY_ADDRESS`.
    # In some containerized environments Ray may pick an IP that isn't reachable
    # from within the same namespace; allow overriding node IP as needed.
    node_ip = os.getenv("APEX_RAY_NODE_IP_ADDRESS", "127.0.0.1")
    init_kwargs = dict(
        address=settings.ray_address,
        ignore_reinit_error=True,
        num_cpus=os.cpu_count(),
        include_dashboard=include_dashboard,
        _node_ip_address=node_ip,
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
    if include_dashboard:
        init_kwargs.update(
            dict(
                dashboard_host="0.0.0.0",
                dashboard_port=settings.ray_dashboard_port,
            )
        )

    ray.init(**init_kwargs)
    _install_shutdown_handler()
    logger.info(f"Ray initialized with {ray.available_resources()}")
    if include_dashboard:
        logger.info(
            f"Ray dashboard available at http://localhost:{settings.ray_dashboard_port}"
        )


def is_ray_ready() -> bool:
    """True only after our init sequence fully completed."""
    with _ray_init_cond:
        return _ray_init_state == "ready"


def _ensure_ray_ready() -> None:
    """
    Ensure Ray is fully initialized, in a way that is safe under concurrency.

    This guards against a rare but nasty case where `ray.is_initialized()` becomes True
    while Ray's core worker is still initializing; using Ray at that moment can trigger
    a second init path and crash the process.
    """
    global _ray_init_state, _ray_init_error
    while True:
        with _ray_init_cond:
            # If Ray is already up (maybe started elsewhere), treat as ready.
            try:
                if ray.is_initialized():
                    _ray_init_state = "ready"
                    _ray_init_error = None
                    _ray_init_cond.notify_all()
                    return
            except Exception:
                pass

            if _ray_init_state == "ready":
                return
            if _ray_init_state == "failed":
                raise RuntimeError("Ray failed to initialize") from _ray_init_error
            if _ray_init_state == "starting":
                _ray_init_cond.wait(timeout=0.5)
                continue

            # not_started -> we become the initializer
            _ray_init_state = "starting"
            _ray_init_error = None

        # Perform initialization outside the condition lock so other threads can wait.
        try:
            _init_ray_once()
        except Exception as e:
            with _ray_init_cond:
                _ray_init_state = "failed"
                _ray_init_error = e
                _ray_init_cond.notify_all()
            raise
        else:
            with _ray_init_cond:
                _ray_init_state = "ready"
                _ray_init_error = None
                _ray_init_cond.notify_all()
            return


def get_ray_app():
    """Get the Ray runtime"""
    _ensure_ray_ready()
    return ray


def shutdown_ray():
    """Shutdown Ray runtime"""
    global _ray_init_state, _ray_init_error
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")
    with _ray_init_cond:
        _ray_init_state = "not_started"
        _ray_init_error = None
        _ray_init_cond.notify_all()


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
