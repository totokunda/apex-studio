import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileDeletedEvent, FileModifiedEvent
from src.manifest.paths import local_manifest_base_path
from src.utils.defaults import get_components_path
from src.api.manifest import invalidate_manifest_caches
from src.manifest.db import setup_manifest_db
import time
from loguru import logger
from typing import Callable

class ComponentsFolderHandler(FileSystemEventHandler):
    def __init__(self, on_event_callback: Callable[[str, str], None], debounce_time: float = 1.0):
        self._on_event = on_event_callback
        self.last_fired = time.time()
        self.debounce_time = debounce_time

    def on_created(self, event: FileCreatedEvent):
        logger.debug(f"File created: {event.src_path}")
        if time.time() - self.last_fired < self.debounce_time:
            return
        self.last_fired = time.time()
        invalidate_manifest_caches()
        setup_manifest_db()

    def on_deleted(self, event: FileDeletedEvent):
        logger.debug(f"File deleted: {event.src_path}")    
        if time.time() - self.last_fired < self.debounce_time:
            return
        self.last_fired = time.time()
        invalidate_manifest_caches()
        setup_manifest_db()
        
    def on_modified(self, event: FileModifiedEvent):
        # only fire if we modify a yaml file
        if not event.src_path.endswith('.yml'):
            return
        logger.debug(f"File modified: {event.src_path}")
        if time.time() - self.last_fired < self.debounce_time:
            return
        self.last_fired = time.time()
        invalidate_manifest_caches()
        setup_manifest_db()
        
_observer: Observer | None = None

def _handle_components_event(event_type: str, path: str) -> None:
    """Called from watchdog thread when a file is created or deleted."""
    # Option 1: Broadcast via Ray WebSocket bridge
    try:
        from .ws_manager import get_ray_ws_bridge
        bridge = get_ray_ws_bridge()
        import ray
        ray.get(bridge.send_update.remote(
            "components-folder-watch",
            {"event": event_type, "path": path, "status": "update"}
        ))
    except Exception:
        pass

def start_components_watchdog() -> None:
    global _observer
    if _observer is not None:
        return
    components_path = get_components_path()
    # get local manifest path
    local_manifest_path = local_manifest_base_path()
    if not Path(components_path).exists():
        return
    if not Path(local_manifest_path).exists():
        return
    handler = ComponentsFolderHandler(on_event_callback=_handle_components_event)
    _observer = Observer()
    _observer.schedule(handler, components_path, recursive=True)
    _observer.schedule(handler, local_manifest_path, recursive=True)
    _observer.start()


def stop_components_watchdog() -> None:
    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join(timeout=5.0)
        _observer = None