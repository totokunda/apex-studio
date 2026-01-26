from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import ray

from .ws_manager import websocket_manager
from .job_store import job_store as unified_job_store

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/status/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    """
    Unified job status endpoint for any Ray job tracked by the engine.
    It checks known job stores (preprocessor, components) and falls back to the latest
    websocket update if the job ref is not available locally.
    """
    return unified_job_store.status(job_id)


@router.post("/cancel/{job_id}")
def job_cancel(job_id: str) -> Dict[str, Any]:
    """
    Unified job cancellation endpoint. Delegates to the appropriate subsystem if known.
    """
    result = unified_job_store.cancel(job_id)
    status = result.get("status", "unknown")
    message = result.get("message")
    if status in ["cancelled", "canceled"]:
        return {"job_id": job_id, "status": status, "message": message}
    raise HTTPException(status_code=404, detail=message or "Job not found")
