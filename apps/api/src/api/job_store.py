from __future__ import annotations
from typing import Dict, Optional, Any, Callable, Iterable
import os
import shutil
import time
import ray
from loguru import logger


_JOB_STORE_ACTOR_NAME = os.getenv("APEX_JOB_STORE_ACTOR_NAME", "apex-unified-job-store")


def _cancel_ray_ref(ref: Any, *, job_id: Optional[str] = None) -> None:
    """
    Best-effort cancel of a Ray task/actor call.

    Ray does not support `force=True` for actor tasks, and will raise:
      "force=True is not supported for actor tasks."

    We attempt a forced cancel first (useful for regular tasks), then fall back to a
    non-forced cancel when the ref is an actor task.
    """

    if ref is None or isinstance(ref, dict):
        return
    if isinstance(ref, list) and len(ref) == 1:
        ref = ref[0]

    try:
        ray.cancel(ref, force=True, recursive=True)
        return
    except Exception as e:
        msg = str(e)
        if "not supported for actor tasks" in msg:
            # Actor calls can't be force-cancelled; retry without force.
            if job_id:
                logger.debug(
                    f"Job {job_id}: Ray ref looks like an actor task; retrying cancel without force"
                )
            try:
                ray.cancel(ref, recursive=True)
                return
            except Exception as e2:
                raise e2 from e
        raise


@ray.remote
class UnifiedJobStoreActor:
    """
    A single source-of-truth for all Ray job refs/status across API workers.

    IMPORTANT:
    - This actor is created as a *named detached* actor so it survives API restarts
      (as long as the Ray cluster is still alive).
    - All API worker processes talk to the same actor, eliminating "missing" jobs
      due to per-process in-memory state.
    """

    def __init__(self) -> None:
        # job_id -> record
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def exists(self, job_id: str) -> bool:
        return job_id in self._jobs

    def create(
        self, job_id: str, job_type: str, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        now = time.time()
        rec = self._jobs.get(job_id)
        if rec is None:
            rec = {
                "job_id": job_id,
                "type": job_type,
                "meta": meta or {},
                "ref": None,
                "created_at": now,
                "submitted_at": None,
                "status_override": None,
                "error": None,
            }
            self._jobs[job_id] = rec
        else:
            # Update type/meta for callers that re-register the same job_id
            rec["type"] = job_type or rec.get("type")
            if meta is not None:
                try:
                    # merge (shallow) to preserve any previous fields
                    merged = dict(rec.get("meta") or {})
                    merged.update(meta)
                    rec["meta"] = merged
                except Exception:
                    rec["meta"] = meta or {}
        return rec

    def attach_ref(self, job_id: str, ref: Any) -> Dict[str, Any]:
        rec = self._jobs.get(job_id)
        if rec is None:
            # Fail closed: we do not want untracked work.
            raise KeyError(f"Job not created: {job_id}")
        # NOTE: Ray auto-dereferences ObjectRef arguments passed into actor methods.
        # Callers should pass a boxed form (via ray.put([ref])) so we can store the
        # actual ObjectRef object without Ray replacing it with the task result.
        if isinstance(ref, list) and len(ref) == 1:
            rec["ref"] = ref[0]
        else:
            rec["ref"] = ref
        rec["submitted_at"] = time.time()
        # Clear any previous override/error on resubmission
        rec["status_override"] = None
        rec["error"] = None
        return rec

    def register(
        self,
        job_id: str,
        ref: Any,
        job_type: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.create(job_id, job_type, meta)
        return self.attach_ref(job_id, ref)

    def mark_failed(self, job_id: str, error: str) -> Dict[str, Any]:
        rec = self._jobs.get(job_id)
        if rec is None:
            rec = self.create(job_id, "unknown", {})
        rec["status_override"] = "error"
        rec["error"] = str(error)
        return rec

    def mark_cancelled(self, job_id: str) -> Dict[str, Any]:
        rec = self._jobs.get(job_id)
        if rec is None:
            rec = self.create(job_id, "unknown", {})
        rec["status_override"] = "cancelled"
        return rec

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def all_job_ids(self) -> list[str]:
        return list(self._jobs.keys())


def _get_actor() -> Any:
    """
    Get/create the named detached job store actor.

    NOTE: We intentionally do *not* call ray.init() here. Callers that need Ray should
    ensure it's initialized (see `src/api/ray_app.py`).
    """
    try:
        return ray.get_actor(_JOB_STORE_ACTOR_NAME)
    except Exception:
        # Create as detached so it outlives API process restarts.
        # Multiple API workers may race to create this; handle "already exists".
        try:
            return UnifiedJobStoreActor.options(
                name=_JOB_STORE_ACTOR_NAME, lifetime="detached"
            ).remote()
        except Exception:
            return ray.get_actor(_JOB_STORE_ACTOR_NAME)


class UnifiedJobStoreProxy:
    def __init__(self) -> None:
        self._actor = None

    def _handle(self):
        if self._actor is None:
            self._actor = _get_actor()
        return self._actor

    # ---- core registry methods ----
    def create(self, job_id: str, job_type: str, meta: Optional[Dict[str, Any]] = None):
        return ray.get(self._handle().create.remote(job_id, job_type, meta))

    def attach_ref(self, job_id: str, ref: ray.ObjectRef):
        # CRITICAL:
        # - Ray forbids ray.put(ObjectRef) directly.
        # - Ray auto-dereferences ObjectRef *arguments* passed into actor methods.
        # Workaround: box the ref inside a list, then ray.put the list.
        # The actor will receive the list and extract the contained ObjectRef.
        ref_box = ray.put([ref])
        return ray.get(self._handle().attach_ref.remote(job_id, ref_box))

    def register(
        self,
        job_id: str,
        ref: ray.ObjectRef,
        job_type: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        ref_box = ray.put([ref])
        ray.get(self._handle().register.remote(job_id, ref_box, job_type, meta))
        logger.info(f"Registered job {job_id} of type {job_type}")

    def exists(self, job_id: str) -> bool:
        try:
            return bool(ray.get(self._handle().exists.remote(job_id)))
        except Exception:
            return False

    def mark_failed(self, job_id: str, error: str) -> None:
        try:
            ray.get(self._handle().mark_failed.remote(job_id, str(error)))
        except Exception:
            pass

    def get_ref(self, job_id: str) -> Optional[ray.ObjectRef]:
        data = self.get(job_id)
        if not data:
            return None
        ref = data.get("ref")
        if isinstance(ref, list) and len(ref) == 1:
            return ref[0]
        # If legacy/corrupt entries stored a dict result, treat as no ref.
        if isinstance(ref, dict):
            return None
        return ref

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            return ray.get(self._handle().get.remote(job_id))
        except Exception:
            return None

    def all_job_ids(self) -> set[str]:
        try:
            return set(ray.get(self._handle().all_job_ids.remote()))
        except Exception:
            return set()

    # ---- status/cancel (frontend-facing semantics) ----
    def status(self, job_id: str) -> Dict[str, Any]:
        """
        Unified status lookup for any registered job.

        Behaviour:
        - If a job isn't in the store, falls back entirely to websocket state (if present).
        - If a job exists but has no Ray ref yet (created-before-submit), reports `queued`.
        - If a job has a Ray ref, reports `running` unless complete/error.
        - Always merges in latest websocket update (progress/message/metadata) when available.
        """
        # Best-effort fetch of the latest websocket-published update for this job
        latest = None
        try:
            from .ws_manager import websocket_manager

            latest = websocket_manager.get_latest_update(job_id)
        except Exception:
            latest = None

        data = self.get(job_id)

        # No registered job – fall back entirely to websocket state if present
        if not data:
            if latest:
                response: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": latest.get("status", "unknown"),
                    "latest": latest,
                }
                if "progress" in latest:
                    response["progress"] = latest.get("progress")
                if "message" in latest:
                    response["message"] = latest.get("message")
                if "metadata" in latest and latest.get("metadata") is not None:
                    response["metadata"] = latest.get("metadata")
                return response
            return {"job_id": job_id, "status": "unknown", "message": "Job not found"}

        # Respect explicit override states
        override = (data.get("status_override") or "").strip().lower()
        if override in {"cancelled", "canceled"}:
            response: Dict[str, Any] = {"job_id": job_id, "status": "cancelled"}
            if latest:
                response["latest"] = latest
            return response
        if override == "error":
            response = {
                "job_id": job_id,
                "status": "error",
                "error": data.get("error") or "Job failed",
            }
            if latest:
                response["latest"] = latest
            return response

        ref = data.get("ref")
        if ref is None:
            # Job exists but hasn't been submitted yet (we create-before-submit to prevent rogues).
            response = {"job_id": job_id, "status": "queued"}
            if latest:
                response["latest"] = latest
                if "progress" in latest:
                    response["progress"] = latest.get("progress")
                if "message" in latest:
                    response["message"] = latest.get("message")
                if "metadata" in latest and latest.get("metadata") is not None:
                    response["metadata"] = latest.get("metadata")
            return response

        # Back-compat: a previous version could have accidentally stored the *result dict*
        # instead of an ObjectRef (due to Ray auto-dereferencing ObjectRef args).
        # Treat that as a completed job rather than erroring.
        if isinstance(ref, dict):
            response = {
                "job_id": job_id,
                "status": (ref or {}).get("status", "complete"),
                "result": ref,
            }
            if latest:
                response["latest"] = latest
                if "progress" in latest:
                    response["progress"] = latest.get("progress")
                if "message" in latest:
                    response["message"] = latest.get("message")
                if "metadata" in latest and latest.get("metadata") is not None:
                    response["metadata"] = latest.get("metadata")
            return response

        # Some deployments may have stored a boxed ref as [ObjectRef]; unwrap it.
        if isinstance(ref, list) and len(ref) == 1:
            ref = ref[0]

        # We have a registered Ray task; check if it's finished
        try:
            ready, _ = ray.wait([ref], timeout=0)
        except Exception as e:
            # If we cannot inspect the ref, surface as error (safer than "unknown").
            response = {"job_id": job_id, "status": "error", "error": str(e)}
            if latest:
                response["latest"] = latest
            return response

        if ready:
            try:
                result = ray.get(ready[0])
                response = {
                    "job_id": job_id,
                    "status": (result or {}).get("status", "complete"),
                    "result": result,
                }
                if latest:
                    response["latest"] = latest
                    if "progress" in latest:
                        response["progress"] = latest.get("progress")
                    if "message" in latest:
                        response["message"] = latest.get("message")
                    if "metadata" in latest and latest.get("metadata") is not None:
                        response["metadata"] = latest.get("metadata")
                return response
            except Exception as e:
                response = {"job_id": job_id, "status": "error", "error": str(e)}
                if latest:
                    response["latest"] = latest
                return response

        # Still running – surface live websocket-driven progress/preview data
        response = {"job_id": job_id, "status": "running"}
        if latest:
            response["latest"] = latest
            response["status"] = latest.get("status", response["status"])
            if "progress" in latest:
                response["progress"] = latest.get("progress")
            if "message" in latest:
                response["message"] = latest.get("message")
            if "metadata" in latest and latest.get("metadata") is not None:
                response["metadata"] = latest.get("metadata")
        return response

    def cancel(self, job_id: str) -> Dict[str, Any]:
        data = self.get(job_id)
        if not data:
            return {"job_id": job_id, "status": "unknown", "message": "Job not found"}

        ref = data.get("ref")
        job_type = data.get("type")
        meta = data.get("meta") or {}

        # Try cancel
        try:
            _cancel_ray_ref(ref, job_id=job_id)
        except Exception as e:
            msg = str(e)
            # When a ref is an actor task, Ray doesn't support force-cancel; we already
            # retry without force above, so only warn if cancellation still fails.
            logger.warning(f"Failed to cancel Ray task for job {job_id}: {msg}")

        # If this is an engine generation job, hard-stop the runner process so we do not
        # keep the model warm after a stop request. This also guarantees the GPU context
        # and warm pool are torn down even if Ray cannot cooperatively cancel the actor call.
        if job_type == "engine":
            try:
                from .ray_tasks import kill_engine_runner_actor

                device_type = (meta or {}).get("device_type", None)
                device_index = (meta or {}).get("device_index", None)
                if device_type is not None:
                    dev = str(device_type).strip().lower()
                    idx = None
                    try:
                        idx = int(device_index) if device_index is not None else None
                    except Exception:
                        idx = None
                    if dev in {"cuda", "mps"}:
                        killed = bool(
                            kill_engine_runner_actor(device_index=idx, device_type=dev)
                        )
                        if killed:
                            logger.info(
                                f"Killed EngineRunner actor after cancelling engine job {job_id}"
                            )
            except Exception as e:
                logger.warning(
                    f"Failed to kill EngineRunner actor after cancelling job {job_id}: {e}"
                )

        # Type-specific cleanup
        if job_type == "components":
            try:
                from src.utils.defaults import get_components_path
                from src.mixins.download_mixin import DownloadMixin

                paths = meta.get("paths") or []
                save_path = meta.get("save_path") or get_components_path()
                for p in paths:
                    try:
                        local_path = DownloadMixin.is_downloaded(p, save_path)
                        if local_path and os.path.exists(local_path):
                            logger.info(f"Cleaning up partial download: {local_path}")
                            if os.path.isdir(local_path):
                                shutil.rmtree(local_path, ignore_errors=True)
                            else:
                                os.unlink(local_path)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to cleanup {p}: {cleanup_err}")
            except Exception as e:
                logger.warning(f"Components cleanup failed for job {job_id}: {e}")

        # Mark cancelled (keep record so polling can observe it)
        try:
            ray.get(self._handle().mark_cancelled.remote(job_id))
        except Exception:
            pass

        # Notify listeners that the job was canceled
        try:
            from .ws_manager import get_ray_ws_bridge

            bridge = get_ray_ws_bridge()
            ray.get(
                bridge.send_update.remote(
                    job_id, 0.0, "Cancelled", {"status": "canceled"}
                )
            )
        except Exception:
            pass

        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job has been cancelled",
        }


job_store = UnifiedJobStoreProxy()


def register_job(
    job_id: str,
    ref: ray.ObjectRef,
    job_type: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    job_store.register(job_id, ref, job_type, meta)


def submit_tracked_job(
    *,
    job_id: str,
    job_type: str,
    meta: Optional[Dict[str, Any]] = None,
    submit: Callable[[], ray.ObjectRef],
) -> ray.ObjectRef:
    """
    Submit Ray work in a way that makes "rogue/untracked" jobs impossible.

    Contract:
    - We *create* the job record first (so it exists even if the API worker restarts).
    - Only after the record exists do we submit Ray work.
    - We then attach the Ray ObjectRef back onto the record.
    - If submission fails, we mark the job as failed in the store.
    """
    job_store.create(job_id, job_type, meta or {})
    try:
        ref = submit()
    except Exception as e:
        job_store.mark_failed(job_id, str(e))
        raise
    try:
        job_store.attach_ref(job_id, ref)
    except Exception as e:
        # Fail closed: don't allow work to continue untracked.
        try:
            _cancel_ray_ref(ref, job_id=job_id)
        except Exception:
            pass
        job_store.mark_failed(job_id, f"Failed to attach ref to job store: {e}")
        raise
    return ref


def require_tracked_job(
    job_id: str, *, allowed_types: Optional[Iterable[str]] = None
) -> None:
    """
    Fail-closed guard for Ray workers: ensure the job_id is registered before doing work.
    """
    info = job_store.get(job_id)
    if info is None:
        raise RuntimeError(
            f"Untracked job_id '{job_id}' started work without registration"
        )
    if allowed_types:
        try:
            allowed = {str(t) for t in allowed_types}
            actual = str((info or {}).get("type") or "")
            if actual and actual not in allowed:
                raise RuntimeError(
                    f"Job '{job_id}' has unexpected type '{actual}' (allowed: {sorted(allowed)})"
                )
        except RuntimeError:
            raise
        except Exception:
            # If type comparison fails for any reason, do not block execution.
            pass
