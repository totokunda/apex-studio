import multiprocessing
import os

# Server socket
_host = os.getenv("APEX_HOST", "127.0.0.1")
_port = os.getenv("APEX_PORT", "8765")


def _format_bind(host: str, port: str) -> str:
    """
    Gunicorn expects IPv6 binds in bracket form: [::1]:8765.
    """
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


# Bind defaults are tuned for desktop/local usage:
# - Prefer loopback (matches Electron default backend URL)
# - Bind both IPv4 and IPv6 so `localhost` / `::1` works on macOS
if _host in {"127.0.0.1", "localhost"}:
    bind = [_format_bind("127.0.0.1", _port), _format_bind("::1", _port)]
elif _host in {"0.0.0.0", "::", "[::]"}:
    bind = [_format_bind("0.0.0.0", _port), _format_bind("::", _port)]
else:
    bind = _format_bind(_host, _port)
backlog = 2048

# Worker processes
#
# NOTE: This service initializes Ray. Ray does not like being initialized in many
# gunicorn workers (and especially not with preload_app). Default to ONE worker.
# If you truly need more, set WEB_CONCURRENCY / APEX_GUNICORN_WORKERS explicitly.
workers = int(os.getenv("APEX_GUNICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1")
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
# Desktop/local stability: do NOT terminate workers after N requests by default.
# Enable (e.g. to mitigate long-running memory leaks) by setting:
#   APEX_MAX_REQUESTS=1000
max_requests = int(os.getenv("APEX_MAX_REQUESTS", "0") or "0")
max_requests_jitter = 100 if max_requests > 0 else 0

# Timeout
timeout = 120
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%h %l %u %t "%r" %s %b "%{Referer}i" "%{User-Agent}i" %D'

# Process naming
proc_name = "apex-engine-api"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
preload_app = False
reuse_port = False

# SSL (uncomment and configure for HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Environment specific settings
if os.getenv("ENVIRONMENT") == "development":
    reload = True
    workers = 1
    loglevel = "debug"
elif os.getenv("ENVIRONMENT") == "staging":
    workers = int(
        os.getenv("APEX_GUNICORN_WORKERS")
        or os.getenv("WEB_CONCURRENCY")
        or str(multiprocessing.cpu_count())
    )
    loglevel = "warning"
elif os.getenv("ENVIRONMENT") == "production":
    workers = int(
        os.getenv("APEX_GUNICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1"
    )
    loglevel = "error"
    preload_app = False
