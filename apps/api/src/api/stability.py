from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from fastapi import Request
from starlette.responses import Response


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass
class _TokenBucket:
    tokens: float
    last_ts: float


class SimpleRateLimiter:
    """
    Tiny in-memory token bucket limiter.
    Intended for *local desktop* stability (single-user) to avoid runaway polling loops.
    """

    def __init__(self, rate_per_sec: float, burst: float) -> None:
        self.rate_per_sec = float(rate_per_sec)
        self.burst = float(burst)
        self._buckets: Dict[str, _TokenBucket] = {}

    def allow(self, key: str, now: float) -> bool:
        b = self._buckets.get(key)
        if b is None:
            self._buckets[key] = _TokenBucket(tokens=self.burst - 1.0, last_ts=now)
            return True

        # refill
        elapsed = max(0.0, now - b.last_ts)
        b.last_ts = now
        b.tokens = min(self.burst, b.tokens + elapsed * self.rate_per_sec)

        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True
        return False


class ResponseCoalescer:
    """
    Very short TTL response cache for noisy polling endpoints.
    This is *not* a general HTTP cache; it's a best-effort coalescer to keep the API stable.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self.ttl_seconds = float(ttl_seconds)
        # cache_key -> (expires_at, status_code, media_type, headers, body_bytes)
        self._cache: Dict[str, Tuple[float, int, Optional[str], Dict[str, str], bytes]] = {}

    def get(self, cache_key: str, now: float) -> Optional[Response]:
        hit = self._cache.get(cache_key)
        if not hit:
            return None
        expires_at, status_code, media_type, headers, body = hit
        if now > expires_at:
            self._cache.pop(cache_key, None)
            return None
        return Response(content=body, status_code=status_code, media_type=media_type, headers=headers)

    def put(self, cache_key: str, now: float, response: Response) -> None:
        # Only cache non-streaming, fully-materialized responses
        body = getattr(response, "body", None)
        if body is None:
            return
        if not isinstance(body, (bytes, bytearray)):
            return
        media_type = getattr(response, "media_type", None)
        # Cache mainly JSON responses (status/progress polling)
        if media_type and "json" not in media_type:
            return
        headers: Dict[str, str] = {}
        # avoid copying hop-by-hop / length; Starlette will compute Content-Length
        for k, v in (response.headers or {}).items():
            lk = k.lower()
            if lk in {"content-length", "transfer-encoding", "connection"}:
                continue
            headers[k] = v

        self._cache[cache_key] = (now + self.ttl_seconds, int(response.status_code), media_type, headers, bytes(body))


def _is_noisy_poll_path(path: str) -> bool:
    # Keep this list narrow to avoid surprising semantics changes.
    return (
        path == "/health"
        or path == "/ready"
        or path.startswith("/jobs/status/")
        or path.startswith("/download/status/")
    )


async def _cache_key_for_request(request: Request) -> str:
    # For our use (polling endpoints), query params matter; body generally doesn't (GET).
    base = f"{request.method}:{request.url.path}?{request.url.query}"
    if request.method not in {"GET", "HEAD"}:
        # Best effort: include a short body hash for idempotent-ish POSTs if we ever add them.
        try:
            body = await request.body()
            if body:
                base += ":" + hashlib.sha256(body).hexdigest()
        except Exception:
            pass
    return base


def install_stability_middleware(app: Any) -> None:
    """
    Installs:
    - Short-TTL response coalescing for noisy polling endpoints (/health, /ready, /jobs/status/*, /download/status/*)
    - Token-bucket rate limiting for those endpoints (returns 429 instead of letting the server thrash)
    - Periodic request stats logging (to identify runaway clients/endpoints)
    """

    enabled = os.getenv("APEX_STABILITY_MIDDLEWARE", "1") not in {"0", "false", "False"}
    if not enabled:
        return

    ttl = _env_float("APEX_POLL_COALESCE_TTL_SECONDS", 0.25)
    rate = _env_float("APEX_POLL_RATE_LIMIT_PER_SEC", 20.0)
    burst = _env_float("APEX_POLL_RATE_LIMIT_BURST", 40.0)
    log_every = _env_float("APEX_REQUEST_STATS_LOG_SECONDS", 30.0)
    top_n = _env_int("APEX_REQUEST_STATS_TOP_N", 10)

    limiter = SimpleRateLimiter(rate_per_sec=rate, burst=burst)
    coalescer = ResponseCoalescer(ttl_seconds=ttl)

    counts: Dict[str, int] = {}
    last_log_ts = time.time()

    @app.middleware("http")
    async def _stability(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        nonlocal last_log_ts
        now = time.time()

        key = f"{request.method} {request.url.path}"
        counts[key] = counts.get(key, 0) + 1
        if now - last_log_ts >= log_every:
            last_log_ts = now
            # log without importing loguru; rely on uvicorn/gunicorn std logging
            try:
                import logging

                logger = logging.getLogger("apex.stability")
                top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_n)]
                total = sum(counts.values())
                logger.info("request-stats total=%s top=%s", total, top)
            except Exception:
                pass
            counts.clear()

        path = request.url.path
        if _is_noisy_poll_path(path):
            cache_key = await _cache_key_for_request(request)
            cached = coalescer.get(cache_key, now=now)
            if cached is not None:
                return cached

            bucket_key = f"poll:{request.method}:{path}"
            if not limiter.allow(bucket_key, now=now):
                # If we don't have a cached response, degrade gracefully.
                return Response(
                    content=b'{"detail":"Too Many Requests"}',
                    status_code=429,
                    media_type="application/json",
                    headers={"Retry-After": "0.25"},
                )

            response = await call_next(request)
            coalescer.put(cache_key, now=now, response=response)
            return response

        return await call_next(request)


