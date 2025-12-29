from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import os
import shutil
from pathlib import Path
import ray
from loguru import logger
from src.utils.defaults import get_components_path
from .ws_manager import get_ray_ws_bridge
from .job_store import register_job, job_store as unified_job_store

router = APIRouter(prefix="/components", tags=["components"])


class ComponentsDownloadRequest(BaseModel):
    paths: List[str]
    save_path: Optional[str] = None
    job_id: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


# Legacy/auxiliary tracking. Unified store is the source of truth for ref and cancel.
job_store: Dict[str, ray.ObjectRef] = {}
job_paths: Dict[str, list] = {}


@router.post("/download", response_model=JobResponse)
def start_components_download(request: ComponentsDownloadRequest):
    if not request.paths:
        raise HTTPException(status_code=400, detail="paths must be a non-empty list")

    job_id = request.job_id or str(uuid.uuid4())
    save_path = request.save_path or get_components_path()

    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    bridge = get_ray_ws_bridge()

    from .ray_tasks import download_components  # lazy import to avoid circulars

    try:
        ref = download_components.remote(request.paths, job_id, bridge, save_path)
        # Register in unified store
        register_job(
            job_id, ref, "components", {"paths": request.paths, "save_path": save_path}
        )
        # Keep legacy store for compatibility with existing status endpoint
        job_store[job_id] = ref
        job_paths[job_id] = request.paths
        logger.info(
            f"Started components download job {job_id} with {len(request.paths)} items"
        )
        return JobResponse(
            job_id=job_id, status="queued", message="Download job created"
        )
    except Exception as e:
        logger.error(f"Failed to start components download: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteRequest(BaseModel):
    path: str  # file or directory relative to components path or absolute within it


@router.delete("/delete", response_model=Dict[str, str])
def delete_component(request: DeleteRequest):
    base = Path(get_components_path()).resolve()
    target = Path(request.path)
    target = (base / target).resolve() if not target.is_absolute() else target.resolve()

    # Safety: ensure deletion stays within components directory
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="path must be within components directory"
        )

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"status": "deleted", "path": str(target)}
    except Exception as e:
        logger.error(f"Failed to delete: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@router.get("/status/{job_id}")
def components_job_status(job_id: str):
    return unified_job_store.status(job_id)


@router.post("/cancel/{job_id}", response_model=JobResponse)
def cancel_components_download(job_id: str):
    """Cancel a running components download job.

    The cancellation is best-effort and will propagate to child tasks.
    An update is emitted so connected websocket clients can reflect the canceled state.
    Partial downloaded files are cleaned up to prevent incomplete files from being detected as downloaded.
    """
    try:
        result = unified_job_store.cancel(job_id)
        status = result.get("status", "unknown")
        message = result.get("message")
        if status in ["cancelled", "canceled"]:
            try:
                del job_store[job_id]
            except KeyError:
                pass
            try:
                del job_paths[job_id]
            except KeyError:
                pass
            return JobResponse(job_id=job_id, status=status, message=message)
        raise HTTPException(status_code=404, detail=message or "Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel: {e}")
