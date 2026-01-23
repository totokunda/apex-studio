from __future__ import annotations

"""
Engine-agnostic API for executing v1 manifest YAMLs with provided inputs.

This exposes endpoints to:
- Submit a run job for a manifest (by id or path)
- Query status and result
- Cancel a running job
"""

from pathlib import Path
from typing import Any, Dict, Optional

import os
import uuid
import ray
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from .ws_manager import get_ray_ws_bridge
from .job_store import submit_tracked_job, job_store
from .ray_resources import get_best_gpu, get_ray_resources
from .manifest import get_manifest, MANIFEST_BASE_PATH

router = APIRouter(prefix="/engine", tags=["engine"])


class RunEngineRequest(BaseModel):
    # Identify the manifest to run. One of these must be provided.
    manifest_id: Optional[str] = (
        None  # matches metadata.id discovered via /manifest/list
    )
    yaml_path: Optional[str] = None  # absolute path to a YAML file
    # Inputs keyed by UI input id, values already resolved to primitives or file paths
    inputs: Dict[str, Any]
    # Optional: user-selected component choices (e.g., scheduler, transformer, etc.)
    selected_components: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    folder_uuid: Optional[str] = None


class WarmupEngineRequest(BaseModel):
    # Identify the manifest to warm. One of these must be provided.
    manifest_id: Optional[str] = None
    yaml_path: Optional[str] = None
    # Optional: user-selected component choices (e.g., scheduler, transformer, etc.)
    selected_components: Optional[Dict[str, Any]] = None
    # "disk" | "engine" | "both"
    mode: Optional[str] = "engine"
    job_id: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    type: Optional[str] = None
    error: Optional[str] = None


def _resolve_manifest_path(manifest_id: Optional[str], yaml_path: Optional[str]) -> str:
    """Resolve the manifest absolute path from id or explicit yaml_path."""
    if yaml_path:
        p = Path(yaml_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"YAML not found: {yaml_path}")
        return str(p.resolve())

    if manifest_id:
        manifest = get_manifest(manifest_id)
        if not manifest:
            raise HTTPException(
                status_code=404, detail=f"Manifest not found: {manifest_id}"
            )
        return str(MANIFEST_BASE_PATH / manifest["full_path"])

    raise HTTPException(
        status_code=400, detail="Provide either manifest_id or yaml_path"
    )


@router.post("/run", response_model=JobResponse)
def run_engine(request: RunEngineRequest):
    """Submit a job to execute a manifest with provided inputs."""
    try:
        manifest_path = _resolve_manifest_path(request.manifest_id, request.yaml_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Choose device resources for scheduling
    device_index, device_type = get_best_gpu()
    resources = get_ray_resources(device_index, device_type, load_profile="heavy")

    job_id = request.job_id or str(uuid.uuid4())
    bridge = get_ray_ws_bridge()

    try:
        from .ray_tasks import get_engine_runner_actor  # lazy import to avoid cycles
        from .ray_app import get_ray_app

        # Ensure Ray is initialized (warm actor lookup/creation requires it).
        get_ray_app()

        runner = get_engine_runner_actor(
            device_index=device_index, device_type=device_type, resources=resources
        )
        ref = submit_tracked_job(
            job_id=job_id,
            job_type="engine",
            meta={
                "manifest_path": manifest_path,
                "device_type": device_type,
                "device_index": device_index,
            },
            submit=lambda: runner.run_engine_from_manifest.remote(
                manifest_path,
                job_id,
                bridge,
                request.inputs,
                request.selected_components or {},
                request.folder_uuid,
            ),
        )
        logger.info(
            f"Engine run submitted job_id={job_id} manifest={manifest_path} resources={resources}"
        )
        return JobResponse(job_id=job_id, status="queued", message="Engine job created")
    except Exception as e:
        logger.error(f"Failed to submit engine run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit: {e}")


@router.post("/warmup", response_model=JobResponse)
def warmup_engine(request: WarmupEngineRequest):
    """
    Best-effort warmup for a manifest.

    - mode="disk": warm OS page cache for weight files (no inference).
    - mode="engine": instantiate engine into the per-worker warm pool.
    - mode="both": do both.
    """
    try:
        manifest_path = _resolve_manifest_path(request.manifest_id, request.yaml_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    mode = (request.mode or "engine").strip().lower()
    job_id = request.job_id or str(uuid.uuid4())

    # Choose scheduling resources.
    # - Disk-only warmup doesn't need a GPU.
    # - Engine warmup must target a GPU worker to be useful.
    if mode == "disk":
        resources = get_ray_resources(
            device_index=None, device_type="cpu", load_profile="light"
        )
    else:
        device_index, device_type = get_best_gpu()
        resources = get_ray_resources(device_index, device_type, load_profile="light")

    try:
        from .ray_app import get_ray_app
        from .ray_tasks import get_engine_runner_actor, warmup_engine_from_manifest  # lazy import

        # Ensure Ray is initialized (warm actor lookup/creation requires it).
        get_ray_app()

        if mode == "disk":
            # Disk warmup is CPU-only and doesn't need to go through the GPU actor.
            ref = submit_tracked_job(
                job_id=job_id,
                job_type="engine_warmup",
                meta={"manifest_path": manifest_path, "mode": mode},
                submit=lambda: warmup_engine_from_manifest.options(**resources).remote(
                    manifest_path, request.selected_components or {}, mode=mode
                ),
            )
        else:
            runner = get_engine_runner_actor(
                device_index=device_index, device_type=device_type, resources=resources
            )
            ref = submit_tracked_job(
                job_id=job_id,
                job_type="engine_warmup",
                meta={"manifest_path": manifest_path, "mode": mode},
                submit=lambda: runner.warmup_engine_from_manifest.remote(
                    manifest_path, request.selected_components or {}, mode=mode
                ),
            )
        return JobResponse(
            job_id=job_id, status="queued", message=f"Warmup queued (mode={mode})"
        )
    except Exception as e:
        logger.error(f"Failed to submit engine warmup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit: {e}")


@router.get("/status/{job_id}")
def engine_status(job_id: str):
    return job_store.status(job_id)


@router.get("/result/{job_id}", response_model=ResultResponse)
def engine_result(job_id: str):
    data = job_store.get(job_id)
    if not data:
        return ResultResponse(job_id=job_id, status="unknown", error="Job not found")
    ref = data.get("ref")
    try:
        ready, _ = ray.wait([ref], timeout=0)
        if not ready:
            return ResultResponse(job_id=job_id, status="running")
        result = ray.get(ready[0])
        return ResultResponse(
            job_id=job_id,
            status=result.get("status", "complete"),
            result_path=result.get("result_path"),
            type=result.get("type"),
            error=result.get("error"),
        )
    except Exception as e:
        return ResultResponse(job_id=job_id, status="error", error=str(e))


@router.post("/cancel/{job_id}", response_model=JobResponse)
def cancel_engine(job_id: str):
    result = job_store.cancel(job_id)
    status = result.get("status", "unknown")
    if status in ["cancelled", "canceled"]:
        return JobResponse(job_id=job_id, status=status, message=result.get("message"))
    raise HTTPException(status_code=404, detail=result.get("message", "Job not found"))
