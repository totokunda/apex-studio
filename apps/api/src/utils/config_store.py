from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


_CONFIG_STORE_THREAD_LOCK = threading.RLock()


class ConfigStoreLockTimeout(RuntimeError):
    pass


@contextmanager
def config_store_lock(config_store_path: Path, timeout_s: float = 10.0) -> Iterator[None]:
    """
    Cross-thread + cross-process lock for the persisted config store JSON.

    - Thread-safety: in-process serialized via `_CONFIG_STORE_THREAD_LOCK`.
    - Process-safety: best-effort via `fcntl.flock` (POSIX). If `fcntl` is not
      available (e.g. Windows), we still serialize within the process.
    """
    with _CONFIG_STORE_THREAD_LOCK:
        if fcntl is None:
            yield
            return

        lock_path = Path(str(config_store_path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_f = lock_path.open("a+", encoding="utf-8")
        start = time.time()
        try:
            while True:
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if timeout_s is not None and (time.time() - start) >= float(timeout_s):
                        raise ConfigStoreLockTimeout(
                            f"Timed out waiting for config-store lock: {lock_path}"
                        )
                    time.sleep(0.05)

            yield
        finally:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_f.close()
            except Exception:
                pass


def read_json_dict(path: Path) -> dict:
    """
    Best-effort JSON dict read. Returns {} on any failure.
    """
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def write_json_dict_atomic(path: Path, data: dict, *, indent: int = 2) -> None:
    """
    Atomically write JSON to `path` (write temp file, fsync, then os.replace).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unique-ish temp file name to avoid collisions.
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time() * 1000)}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # Some filesystems may not support fsync; best-effort only.
            pass

    os.replace(str(tmp), str(path))


