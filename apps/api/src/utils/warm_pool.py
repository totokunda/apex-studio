from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from loguru import logger


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except Exception:
        return default


def stable_hash_dict(payload: Dict[str, Any]) -> str:
    """
    Stable hash for dict-like payloads used in warm pool keys.
    Avoids capturing huge objects by JSON encoding with sorted keys.
    """
    try:
        s = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        s = repr(payload)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cuda_free_fraction() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        if total <= 0:
            return None
        return float(free) / float(total)
    except Exception:
        return None


def _cpu_free_fraction() -> Optional[float]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        if vm.total <= 0:
            return None
        return float(vm.available) / float(vm.total)
    except Exception:
        return None


@dataclass
class WarmPoolEntry:
    engine: Any
    last_used_ts: float
    in_use: int = 0


class EngineWarmPool:
    """
    Per-process warm pool for engines.

    Goals:
    - Keep engines warm *after* runs (fast subsequent runs)
    - Evict only when needed (TTL / max size / VRAM pressure)
    - Never interfere with a running engine (no concurrent sharing)
      NOTE: Ray GPU scheduling already serializes GPU tasks when num_gpus=1.0,
      but we still track `in_use` defensively.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        max_entries: int,
        ttl_seconds: int,
        min_free_vram_fraction: float,
        min_free_ram_fraction: float,
    ):
        self.enabled = bool(enabled)
        self.max_entries = max(0, int(max_entries))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.min_free_vram_fraction = float(min_free_vram_fraction)
        self.min_free_ram_fraction = float(min_free_ram_fraction)

        self._lock = threading.RLock()
        self._entries: Dict[str, WarmPoolEntry] = {}

    @classmethod
    def from_env(cls) -> "EngineWarmPool":
        return cls(
            enabled=_env_bool("APEX_WARM_POOL_ENABLED", True),
            max_entries=_env_int("APEX_WARM_POOL_MAX_ENGINES", 1),
            ttl_seconds=_env_int("APEX_WARM_POOL_TTL_SECONDS", 20 * 60),
            min_free_vram_fraction=_env_float("APEX_WARM_POOL_MIN_FREE_VRAM_FRACTION", 0.12),
            min_free_ram_fraction=_env_float("APEX_WARM_POOL_MIN_FREE_RAM_FRACTION", 0.10),
        )

    def _now(self) -> float:
        return time.time()

    def _should_evict_for_vram(self) -> bool:
        frac = _cuda_free_fraction()
        if frac is None:
            return False
        return frac < self.min_free_vram_fraction

    def _should_evict_for_ram(self) -> bool:
        frac = _cpu_free_fraction()
        if frac is None:
            return False
        return frac < self.min_free_ram_fraction

    def _evict_one_locked(self) -> bool:
        """
        Evict one idle entry using LRU (oldest last_used_ts), returns True if evicted.
        """
        idle = [(k, e) for k, e in self._entries.items() if e.in_use <= 0]
        if not idle:
            return False
        idle.sort(key=lambda kv: kv[1].last_used_ts)
        key, ent = idle[0]
        eng = ent.engine
        del self._entries[key]

        try:
            if hasattr(eng, "offload_engine"):
                eng.offload_engine()
        except Exception as e:
            logger.warning(f"Warm pool eviction offload failed for key={key}: {e}")

        try:
            # Drop reference immediately.
            del eng
        except Exception:
            pass
        return True

    def _evict_locked(self) -> None:
        # TTL eviction
        if self.ttl_seconds > 0:
            cutoff = self._now() - self.ttl_seconds
            for k in list(self._entries.keys()):
                ent = self._entries.get(k)
                if ent is None:
                    continue
                if ent.in_use <= 0 and ent.last_used_ts < cutoff:
                    try:
                        if hasattr(ent.engine, "offload_engine"):
                            ent.engine.offload_engine()
                    except Exception:
                        pass
                    del self._entries[k]

        # Size eviction
        while self.max_entries > 0 and len(self._entries) > self.max_entries:
            if not self._evict_one_locked():
                break

        # VRAM/RAM pressure eviction (evict until we're above thresholds or no idle entries)
        if self._should_evict_for_vram() or self._should_evict_for_ram():
            for _ in range(max(1, len(self._entries))):
                if (not self._should_evict_for_vram()) and (not self._should_evict_for_ram()):
                    break
                if not self._evict_one_locked():
                    break

    def acquire(
        self, key: str, factory: Callable[[], Any], *, allow_pool: bool = True
    ) -> Tuple[Any, bool]:
        """
        Acquire an engine for exclusive use.
        Returns: (engine, pooled_flag)
        """
        if not self.enabled or not allow_pool or self.max_entries <= 0:
            return factory(), False

        with self._lock:
            self._evict_locked()
            ent = self._entries.get(key)
            if ent is not None and ent.in_use <= 0:
                ent.in_use = 1
                ent.last_used_ts = self._now()
                return ent.engine, True

        # Slow path: construct outside lock
        eng = factory()

        with self._lock:
            # If another thread created it first, we keep this one as non-pooled
            # to avoid interfering with the in-use engine.
            if key in self._entries:
                return eng, False
            self._entries[key] = WarmPoolEntry(engine=eng, last_used_ts=self._now(), in_use=1)
            self._evict_locked()
            # If we got evicted immediately (pressure/size), treat as non-pooled.
            if key not in self._entries:
                return eng, False
            return eng, True

    def release(self, key: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            ent = self._entries.get(key)
            if ent is None:
                return
            ent.in_use = max(0, int(ent.in_use) - 1)
            ent.last_used_ts = self._now()
            self._evict_locked()

    def discard(self, key: str, *, offload: bool = True) -> None:
        """
        Remove an entry from the pool (and offload) regardless of TTL.
        Safe no-op if missing or in-use.
        """
        if not self.enabled:
            return
        with self._lock:
            ent = self._entries.get(key)
            if ent is None:
                return
            if ent.in_use > 0:
                return
            eng = ent.engine
            del self._entries[key]
        if offload:
            try:
                if hasattr(eng, "offload_engine"):
                    eng.offload_engine()
            except Exception:
                pass

    def snapshot_entries(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a shallow snapshot of current pool entries.

        This is intended for best-effort cleanup/diagnostics in the *same process*.
        Callers must still respect `in_use` (never offload/evict in-use engines).
        """
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for k, ent in self._entries.items():
                out[str(k)] = {
                    "engine": ent.engine,
                    "in_use": int(ent.in_use),
                    "last_used_ts": float(ent.last_used_ts),
                }
            return out

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "min_free_vram_fraction": self.min_free_vram_fraction,
                "min_free_ram_fraction": self.min_free_ram_fraction,
                "entries": len(self._entries),
                "keys": list(self._entries.keys())[:50],
            }


