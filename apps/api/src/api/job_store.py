from __future__ import annotations
from typing import Dict, Optional, Any
import os
import shutil
import ray
from loguru import logger


class UnifiedJobStore:
    def __init__(self) -> None:
        # job_id -> { 'ref': ray.ObjectRef, 'type': str, 'meta': dict }
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        job_id: str,
        ref: ray.ObjectRef,
        job_type: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._jobs[job_id] = {
            "ref": ref,
            "type": job_type,
            "meta": meta or {},
        }
        logger.info(f"Registered job {job_id} of type {job_type}")

    def get_ref(self, job_id: str) -> Optional[ray.ObjectRef]:
        data = self._jobs.get(job_id)
        return data.get("ref") if data else None

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def all_job_ids(self) -> set[str]:
        return set(self._jobs.keys())

    def status(self, job_id: str) -> Dict[str, Any]:
        """
        Unified status lookup for any registered job.

        Behaviour:
        - Always merges in the latest websocket update (progress / previews / metadata)
          if available, so callers can inspect render-step and preview information.
        - If the Ray task has completed, returns its final result as before, with any
          websocket "latest" data attached for additional context.
        """
        # Best-effort fetch of the latest websocket-published update for this job
        latest = None
        try:
            from .ws_manager import websocket_manager

            latest = websocket_manager.get_latest_update(job_id)
        except Exception:
            latest = None

        data = self._jobs.get(job_id)

        # No registered Ray ref – fall back entirely to websocket state if present
        if not data:
            if latest:
                response: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": latest.get("status", "unknown"),
                    "latest": latest,
                }
                # Promote commonly used fields for easier polling
                if "progress" in latest:
                    response["progress"] = latest.get("progress")
                if "message" in latest:
                    response["message"] = latest.get("message")
                if "metadata" in latest and latest.get("metadata") is not None:
                    response["metadata"] = latest.get("metadata")
                return response

            return {"job_id": job_id, "status": "unknown", "message": "Job not found"}

        # We have a registered Ray task; check if it's finished
        ref = data["ref"]
        ready, _ = ray.wait([ref], timeout=0)
        if ready:
            try:
                result = ray.get(ready[0])
                response = {
                    "job_id": job_id,
                    "status": result.get("status", "complete"),
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
            # Prefer websocket's notion of status if present (e.g. "preview", "processing")
            response["status"] = latest.get("status", response["status"])
            if "progress" in latest:
                response["progress"] = latest.get("progress")
            if "message" in latest:
                response["message"] = latest.get("message")
            if "metadata" in latest and latest.get("metadata") is not None:
                response["metadata"] = latest.get("metadata")
        return response

    def cancel(self, job_id: str) -> Dict[str, Any]:
        data = self._jobs.get(job_id)
        if not data:
            return {"job_id": job_id, "status": "unknown", "message": "Job not found"}

        ref = data.get("ref")
        job_type = data.get("type")
        meta = data.get("meta") or {}

        # Try cancel
        try:
            if ref is not None:
                ray.cancel(ref, force=True, recursive=True)
        except Exception as e:
            logger.warning(f"Failed to cancel Ray task for job {job_id}: {e}")

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

        # Remove from store
        try:
            del self._jobs[job_id]
        except KeyError:
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


job_store = UnifiedJobStore()


def register_job(
    job_id: str,
    ref: ray.ObjectRef,
    job_type: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    job_store.register(job_id, ref, job_type, meta)
