from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from .job_store import job_store as unified_job_store
from .ws_manager import websocket_manager


router = APIRouter(prefix="/ray", tags=["ray"])


def _job_summary(job_id: str) -> Dict[str, Any]:
    """
    Helper to normalize a job status payload.

    The unified job store already returns a dict with at least:
      { "job_id": ..., "status": ... }
    We augment that with:
      - `category`: one of {"download", "processor", "engine", "other"}
      - `latest`: last websocket update (includes progress/message/metadata) if available
      - `progress`: convenience top-level 0..1 progress hint when running/processing
      - queued detection: jobs that have no websocket updates yet are exposed as `status="queued"`
    """
    # Inspect underlying job record (if present) to derive type/category
    job_record = unified_job_store.get(job_id) or {}
    job_type = job_record.get("type")
    category = "other"
    if job_type in {"preprocessor", "postprocessor"}:
        category = "processor"
    elif job_type in {"download", "components"}:
        category = "download"
    elif job_type == "engine":
        category = "engine"

    data = unified_job_store.status(job_id)

    # Attach latest websocket update (includes progress/message/metadata)
    try:
        latest = websocket_manager.get_latest_update(job_id)
    except Exception:
        latest = None
    if latest is not None:
        # Do not overwrite explicit fields coming from the job store, just augment
        data.setdefault("latest", latest)
        # If job is still running, expose a top-level progress hint for convenience
        if data.get("status") in {"running", "processing"} and "progress" in latest:
            data.setdefault("progress", latest.get("progress"))
            # Prefer latest message if not already present
            data.setdefault("message", latest.get("message"))
        # If we have no category yet, attempt to infer from websocket metadata
        try:
            meta = latest.get("metadata") or {}
            bucket = (meta.get("bucket") or "").lower()
            stage = (meta.get("stage") or "").lower()
            if category == "other":
                if bucket in {"component", "lora", "preprocessor"}:
                    category = "download"
                elif stage in {"preprocessor", "postprocessor"}:
                    category = "processor"
                elif stage == "engine":
                    category = "engine"
        except Exception:
            pass
    else:
        # If the Ray task is not yet running (no websocket events) but Ray reports "running",
        # treat this as "queued" so the UI can distinguish and hide progress bars.
        if data.get("status") == "running":
            data["status"] = "queued"

    # Ensure job_id, status, and category are always present and consistent
    data.setdefault("job_id", job_id)
    data.setdefault("status", "unknown")
    data.setdefault("category", category)
    return data


@router.get("/jobs", response_model=dict)
def list_jobs() -> Dict[str, Any]:
    """
    List all jobs currently tracked in the unified Ray job store, with their status.

    Response:
      {
        "jobs": [
          { "job_id": "...", "status": "running" | "complete" | "error" | "unknown", ... },
          ...
        ]
      }
    """
    job_ids = sorted(unified_job_store.all_job_ids())
    jobs: List[Dict[str, Any]] = [_job_summary(job_id) for job_id in job_ids]
    return {"jobs": jobs}


@router.get("/jobs/{job_id}", response_model=dict)
def get_job(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a specific job.

    This mirrors the unified status returned elsewhere and always includes a `status` field.
    """
    data = unified_job_store.status(job_id)
    status = data.get("status", "unknown")
    # If the underlying store says "Job not found", surface that as 404
    if status == "unknown" and data.get("message") == "Job not found":
        raise HTTPException(status_code=404, detail=data.get("message"))
    data.setdefault("job_id", job_id)
    data.setdefault("status", status)
    return data


@router.post("/jobs/{job_id}/cancel", response_model=dict)
def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a specific running job.

    Uses the unified job store's cancellation logic and ensures a `status` is returned.
    """
    result = unified_job_store.cancel(job_id)
    status = result.get("status", "unknown")
    message = result.get("message")

    if status in {"cancelled", "canceled"}:
        return {
            "job_id": job_id,
            "status": status,
            "message": message or "Job has been cancelled",
        }

    # Surface "Job not found" as 404, keep other errors as 400
    if message == "Job not found" or status == "unknown":
        raise HTTPException(status_code=404, detail=message or "Job not found")

    raise HTTPException(status_code=400, detail=message or "Unable to cancel job")


@router.post("/jobs/cancel_all", response_model=dict)
def cancel_all_jobs() -> Dict[str, Any]:
    """
    Cancel all currently tracked jobs.

    Returns per-job results, each with a `status`, mirroring the behavior of the
    single-job cancellation endpoint.
    """
    job_ids = sorted(unified_job_store.all_job_ids())
    results: List[Dict[str, Any]] = []
    for job_id in job_ids:
        res = unified_job_store.cancel(job_id)
        res.setdefault("job_id", job_id)
        res.setdefault("status", "unknown")
        results.append(res)

    return {
        "status": "complete",
        "cancelled": [
            r for r in results if r.get("status") in {"cancelled", "canceled"}
        ],
        "failed": [
            r for r in results if r.get("status") not in {"cancelled", "canceled"}
        ],
        "results": results,
    }
