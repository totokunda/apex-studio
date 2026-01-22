"""
API endpoints for postprocessor operations (e.g., frame interpolation)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import uuid
import ray
from loguru import logger

from .ray_tasks import run_frame_interpolation
from .ray_resources import get_best_gpu, get_ray_resources
from .job_store import submit_tracked_job
from .ws_manager import get_ray_ws_bridge
from .engine_resource_guard import maybe_release_warm_engine_for_non_engine_request

router = APIRouter(prefix="/postprocessor", tags=["postprocessor"])


class FrameInterpolateRequest(BaseModel):
    input_path: str
    target_fps: float
    job_id: Optional[str] = None
    exp: Optional[int] = None
    scale: Optional[float] = 1.0


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


@router.post("/frame-interpolate", response_model=JobResponse)
def frame_interpolate(request: FrameInterpolateRequest):
    """
    Submit a RIFE frame interpolation job for a given video input.
    Returns a job_id; progress and completion are streamed over websocket /ws/job/{job_id}.
    """
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Input file not found: {request.input_path}"
        )

    if not (request.target_fps and request.target_fps > 0):
        raise HTTPException(status_code=400, detail="target_fps must be > 0")

    # If a warm engine is idling on the GPU, release it so this non-engine job can run.
    # If an engine job is active/queued, we do nothing (Ray will queue naturally).
    try:
        maybe_release_warm_engine_for_non_engine_request()
    except Exception:
        pass

    # Choose resources (prefer GPU if available)
    device_index, device_type = get_best_gpu()
    resources = get_ray_resources(device_index, device_type, load_profile="medium")

    try:
        job_id = request.job_id or str(uuid.uuid4())
        bridge = get_ray_ws_bridge()
        logger.info(
            f"Submitting RIFE frame interpolation for {input_path} -> {request.target_fps} fps with resources {resources}"
        )

        task_ref = submit_tracked_job(
            job_id=job_id,
            job_type="postprocessor",
            meta={
                "method": "frame-interpolate",
                "input_path": str(input_path),
                "target_fps": float(request.target_fps),
            },
            submit=lambda: run_frame_interpolation.options(**resources).remote(
                str(input_path),
                float(request.target_fps),
                job_id,
                bridge,
                request.exp,
                request.scale if request.scale is not None else 1.0,
            ),
        )

        return JobResponse(
            job_id=job_id, status="queued", message="RIFE interpolation job submitted"
        )
    except Exception as e:
        logger.error(f"Failed to submit RIFE job: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit Ray task: {str(e)}"
        )
