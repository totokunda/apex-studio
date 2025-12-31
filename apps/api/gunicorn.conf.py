import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8765"
backlog = 2048

# Worker processes
#
# NOTE: This service initializes Ray. Ray does not like being initialized in many
# gunicorn workers (and especially not with preload_app). Default to ONE worker.
# If you truly need more, set WEB_CONCURRENCY / APEX_GUNICORN_WORKERS explicitly.
workers = int(os.getenv("APEX_GUNICORN_WORKERS") or os.getenv("WEB_CONCURRENCY") or "1")
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100

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
