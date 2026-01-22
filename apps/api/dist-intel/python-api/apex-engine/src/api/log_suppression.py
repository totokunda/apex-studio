from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Iterable, List, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_csv(name: str) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    parts: List[str] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        parts.append(p)
    return parts


_REQUEST_LINE_RE = re.compile(r'"[A-Z]+ (?P<path>/[^ ]*) HTTP/[0-9.]+"')


def _default_suppressed_path_prefixes() -> List[str]:
    # Keep defaults conservative; users can extend via env.
    return [
        "/health",
        "/ready",
        "/system/memory",
        "/jobs/status/",
        "/download/status/",
        "/ray/jobs",
        "/preprocessor/ray-status",
        "/mask/health",
    ]


@lru_cache(maxsize=1)
def suppressed_path_prefixes() -> List[str]:
    """
    Returns path prefixes that should be hidden from noisy/access logs.

    Env:
      - APEX_HIDE_POLLING_LOGS (default: true): enable built-in suppression defaults
      - APEX_HIDE_LOG_PATH_PREFIXES: additional comma-separated prefixes to suppress
    """
    prefixes: List[str] = []
    if _env_bool("APEX_HIDE_POLLING_LOGS", True):
        prefixes.extend(_default_suppressed_path_prefixes())

    prefixes.extend(_env_csv("APEX_HIDE_LOG_PATH_PREFIXES"))
    # Normalize: ensure leading slash for path prefixes.
    normalized: List[str] = []
    for p in prefixes:
        p = p.strip()
        if not p:
            continue
        if not p.startswith("/"):
            p = "/" + p
        normalized.append(p)
    return normalized


def is_suppressed_http_path(path: str) -> bool:
    # Strip query string if present.
    if "?" in path:
        path = path.split("?", 1)[0]
    for prefix in suppressed_path_prefixes():
        if path == prefix or path.startswith(prefix):
            return True
    return False


def _path_from_access_log_record(record: logging.LogRecord) -> Optional[str]:
    """
    Try to extract the request path from common access log records (uvicorn).
    Returns None if we can't confidently parse.
    """
    args = getattr(record, "args", None)
    try:
        # Uvicorn access logger typically uses:
        #   logger.info('%s - "%s %s HTTP/%s" %d', client_addr, method, full_path, http_version, status_code)
        if isinstance(args, tuple) and len(args) >= 3:
            maybe_path = args[2]
            if isinstance(maybe_path, str) and maybe_path.startswith("/"):
                return maybe_path
    except Exception:
        pass

    try:
        msg = record.getMessage()
        m = _REQUEST_LINE_RE.search(msg)
        if m:
            return m.group("path")
    except Exception:
        pass

    return None


class AccessLogPathFilter(logging.Filter):
    def __init__(self, name: str = "") -> None:
        super().__init__(name)

    def filter(self, record: logging.LogRecord) -> bool:
        path = _path_from_access_log_record(record)
        if path and is_suppressed_http_path(path):
            return False
        return True


_INSTALLED = False


def install_http_log_suppression(
    logger_names: Iterable[str] = ("uvicorn.access"),
) -> None:
    """
    Attach a filter to common access loggers so polling endpoints don't spam logs.

    This is safe to call multiple times.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    flt = AccessLogPathFilter()
    all_loggers = list(logging.root.manager.loggerDict.keys())
    for name in all_loggers:
        try:
            logging.getLogger(name).addFilter(flt)
        except Exception:
            # Best-effort: never break app startup due to logging tweaks.
            pass
    _INSTALLED = True
