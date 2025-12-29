"""
Websocket manager for handling job status updates
"""

from typing import Dict, Set, Optional
from fastapi.websockets import WebSocket
import ray
from collections import defaultdict


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
        self.updates: Dict[str, list] = defaultdict(list)
        print("RayWebSocketBridge initialized")

    def send_update(
        self,
        job_id: str,
        progress: float,
        message: str,
        metadata: Optional[Dict] = None,
    ):
        """Store update to be pulled by main process"""
        metadata = metadata or {}
        status = metadata.pop("status", "processing")

        update = {
            "progress": progress,
            "message": message,
            "status": status,
            "metadata": metadata,
        }

        self.updates[job_id].append(update)
        return True

    def get_updates(self, job_id: str) -> list:
        """Get all pending updates for a job"""
        updates = self.updates.get(job_id, [])
        self.updates[job_id] = []  # Clear after retrieving
        return updates

    def clear_updates(self, job_id: str) -> bool:
        """Clear any pending updates for a job_id (used when restarting a job)."""
        try:
            if job_id in self.updates:
                self.updates[job_id] = []
            return True
        except Exception:
            return False

    def get_all_job_ids(self) -> list:
        """Get all job IDs that have updates"""
        return list(self.updates.keys())

    def has_updates(self, job_id: str) -> bool:
        """Check if there are pending updates"""
        return len(self.updates.get(job_id, [])) > 0


# Global Ray actor for websocket bridge
_ray_ws_bridge = None


def get_ray_ws_bridge():
    """Get or create the Ray websocket bridge actor"""
    global _ray_ws_bridge
    if _ray_ws_bridge is None:
        import ray

        if not ray.is_initialized():
            raise RuntimeError(
                "Ray must be initialized before creating websocket bridge"
            )
        _ray_ws_bridge = RayWebSocketBridge.remote()
    return _ray_ws_bridge
