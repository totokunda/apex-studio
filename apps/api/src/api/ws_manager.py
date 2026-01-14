"""
Websocket manager for handling job status updates
"""

from typing import Dict, Set, Optional, Any, Callable
from fastapi.websockets import WebSocket
import ray
from collections import defaultdict, deque
import os
import json


class WebSocketManager:
    """Manages websocket connections for job updates"""

    def __init__(self):
        # Map job_id to set of connected websockets
        self.connections: Dict[str, Set[WebSocket]] = {}
        # Store latest updates for each job (for clients connecting late)
        self.latest_updates: Dict[str, dict] = {}

    def clear_latest(self, job_id: str):
        """Clear the cached latest update for a job_id to avoid replaying stale state."""
        try:
            if job_id in self.latest_updates:
                del self.latest_updates[job_id]
        except Exception:
            pass

    async def connect(self, websocket: WebSocket, job_id: str):
        """Register a websocket connection for a job"""
        await websocket.accept()
        if job_id not in self.connections:
            self.connections[job_id] = set()
        self.connections[job_id].add(websocket)

        # Send latest update if available
        if job_id in self.latest_updates:
            try:
                await websocket.send_json(self.latest_updates[job_id])
            except Exception:
                pass

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Unregister a websocket connection"""
        if job_id in self.connections:
            self.connections[job_id].discard(websocket)
            if not self.connections[job_id]:
                del self.connections[job_id]

    async def send_update(self, job_id: str, data: dict):
        """
        Send update to all websockets listening to a job.

        We also cache the *merged* latest state for the job so that fields like
        `preview_path` (emitted during render_step/preview events) are not lost
        when subsequent non-preview progress updates arrive.
        """
        # Merge with any previously cached latest update so we don't drop fields
        # such as preview_path that may only be present on some events.
        try:
            existing = self.latest_updates.get(job_id)
            if existing:
                merged = dict(existing)
                # Always take the newest scalar fields when present
                for key in ("progress", "message", "status"):
                    if key in data and data[key] is not None:
                        merged[key] = data[key]
                # Merge metadata dictionaries (existing keys can be overridden)
                existing_meta = existing.get("metadata") or {}
                new_meta = data.get("metadata") or {}
                if isinstance(existing_meta, dict) and isinstance(new_meta, dict):
                    merged["metadata"] = {**existing_meta, **new_meta}
                elif "metadata" in data:
                    merged["metadata"] = data["metadata"]
                # Use merged object as the cached latest
                data = merged
        except Exception:
            # Best-effort only; fall back to storing the raw update
            pass

        # Store latest (possibly merged) update
        self.latest_updates[job_id] = data

        if job_id in self.connections:
            disconnected = set()
            for websocket in self.connections[job_id]:
                try:
                    await websocket.send_json(data)
                except Exception:
                    disconnected.add(websocket)

            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket, job_id)

    def get_latest_update(self, job_id: str) -> Optional[dict]:
        """Get the latest update for a job"""
        return self.latest_updates.get(job_id)


# Global websocket manager instance
websocket_manager = WebSocketManager()


@ray.remote
class RayWebSocketBridge:
    """Ray actor that bridges Ray workers to the websocket manager"""

    def __init__(self):
        self._max_pending_per_job = int(
            os.environ.get("RAY_WS_MAX_PENDING_UPDATES_PER_JOB", "500")
        )
        self._max_message_chars = int(os.environ.get("RAY_WS_MAX_MESSAGE_CHARS", "4096"))
        self._max_metadata_chars = int(
            os.environ.get("RAY_WS_MAX_METADATA_CHARS", "20000")
        )
        self._max_items_per_pull = int(
            os.environ.get("RAY_WS_MAX_UPDATES_PER_PULL", "200")
        )

        def _new_queue() -> deque:
            return deque(maxlen=self._max_pending_per_job)

        self._new_queue: Callable[[], deque] = _new_queue
        self.updates: Dict[str, deque] = defaultdict(self._new_queue)
        print("RayWebSocketBridge initialized")

    def _truncate_str(self, s: str, limit: int) -> str:
        if not isinstance(s, str):
            s = str(s)
        if limit <= 0:
            return ""
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 12)] + "…(truncated)"

    def _sanitize(self, obj: Any, *, depth: int = 0, max_depth: int = 5) -> Any:
        """
        Best-effort conversion to a small JSON-serializable structure.

        This is defensive: Ray uses msgpack internally for some messages and can fail
        on Windows when deserializing very large buffers (contiguous allocation).
        """
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj

        if isinstance(obj, str):
            return self._truncate_str(obj, self._max_message_chars)

        if depth >= max_depth:
            return self._truncate_str(repr(obj), 256)

        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            # Cap dict fanout
            for i, (k, v) in enumerate(obj.items()):
                if i >= 200:
                    out["__truncated__"] = True
                    break
                try:
                    key = k if isinstance(k, str) else str(k)
                except Exception:
                    key = "<unprintable_key>"
                out[key] = self._sanitize(v, depth=depth + 1, max_depth=max_depth)
            return out

        if isinstance(obj, (list, tuple, set)):
            out_list = []
            for i, item in enumerate(obj):
                if i >= 200:
                    out_list.append("__truncated__")
                    break
                out_list.append(self._sanitize(item, depth=depth + 1, max_depth=max_depth))
            return out_list

        # Numpy / torch tensors: represent compactly without importing heavy deps.
        try:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                meta = {
                    "__type__": type(obj).__name__,
                    "shape": getattr(obj, "shape", None),
                    "dtype": str(getattr(obj, "dtype", "")),
                }
                if hasattr(obj, "device"):
                    meta["device"] = str(getattr(obj, "device"))
                return meta
        except Exception:
            pass

        # Fallback to small string
        return self._truncate_str(repr(obj), 512)

    def _shrink_metadata(self, metadata: Any) -> Dict[str, Any]:
        safe = self._sanitize(metadata)
        if not isinstance(safe, dict):
            safe = {"value": safe}
        # Hard cap by JSON-encoded size (best-effort)
        try:
            raw = json.dumps(safe, ensure_ascii=False, default=str)
            if len(raw) <= self._max_metadata_chars:
                return safe
            # If too big, keep only a few common keys and a note
            keep = {}
            for k in ("status", "preview_path", "type", "index", "stage", "input_id", "error"):
                if k in safe:
                    keep[k] = safe[k]
            keep["__metadata_truncated__"] = True
            return keep
        except Exception:
            return {"__metadata_unserializable__": True}

    def send_update(
        self,
        job_id: str,
        progress: float,
        message: str,
        metadata: Optional[Dict] = None,
    ):
        """Store update to be pulled by main process"""
        metadata = metadata or {}
        # status should be a small scalar
        try:
            status = metadata.pop("status", "processing")
        except Exception:
            status = "processing"

        # Sanitize aggressively to avoid huge msgpack payloads.
        try:
            safe_message = self._truncate_str(message, self._max_message_chars)
        except Exception:
            safe_message = "<unprintable_message>"

        safe_metadata = self._shrink_metadata(metadata)

        update = {
            "progress": progress,
            "message": safe_message,
            "status": status if isinstance(status, (str, int, float, bool)) else str(status),
            "metadata": safe_metadata,
        }

        q = self.updates[job_id]
        # Coalesce noisy progress updates (keep latest), but preserve preview/error updates.
        is_preview = isinstance(safe_metadata, dict) and ("preview_path" in safe_metadata)
        is_error = isinstance(safe_metadata, dict) and (
            safe_metadata.get("status") == "error" or "error" in safe_metadata
        )
        if q and not is_preview and not is_error and update.get("status") == "processing":
            try:
                q[-1] = update
            except Exception:
                q.append(update)
        else:
            q.append(update)
        return True

    def get_updates(self, job_id: str, max_items: Optional[int] = None) -> list:
        """Get up to max_items pending updates for a job (default capped)."""
        q = self.updates.get(job_id)
        if not q:
            return []
        limit = self._max_items_per_pull if max_items is None else int(max_items)
        if limit <= 0:
            return []
        out = []
        for _ in range(min(limit, len(q))):
            try:
                out.append(q.popleft())
            except Exception:
                break
        # If queue becomes empty, delete key to keep the job list small.
        if not q:
            try:
                del self.updates[job_id]
            except Exception:
                pass
        return out

    def clear_updates(self, job_id: str) -> bool:
        """Clear any pending updates for a job_id (used when restarting a job)."""
        try:
            if job_id in self.updates:
                try:
                    self.updates[job_id].clear()
                except Exception:
                    self.updates[job_id] = self._new_queue()
            return True
        except Exception:
            return False

    def get_all_job_ids(self, max_items: int = 5000) -> list:
        """Get job IDs that have updates (capped)."""
        try:
            keys = list(self.updates.keys())
            return keys[: int(max_items)]
        except Exception:
            return []

    def has_updates(self, job_id: str) -> bool:
        """Check if there are pending updates"""
        q = self.updates.get(job_id)
        try:
            return bool(q) and len(q) > 0
        except Exception:
            return False


# Global Ray actor for websocket bridge
_ray_ws_bridge = None


def get_ray_ws_bridge():
    """Get or create the Ray websocket bridge actor"""
    global _ray_ws_bridge
    if _ray_ws_bridge is None:
        import ray

        if not ray.is_initialized():
            # Lazily initialize Ray if not already done (avoids race on first request)
            from .ray_app import get_ray_app

            get_ray_app()
        _ray_ws_bridge = RayWebSocketBridge.remote()
    return _ray_ws_bridge
