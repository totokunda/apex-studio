"""
API endpoints for preprocessor operations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from typing import Set
from pathlib import Path
import os
import shutil
import json
import ray
from .ray_tasks import run_preprocessor
from .job_store import submit_tracked_job, job_store
from .preprocessor_registry import list_preprocessors, get_preprocessor_details
from .params import validate_and_convert_params
from .ray_resources import get_best_gpu, get_ray_resources
from src.utils.defaults import DEFAULT_CACHE_PATH
from loguru import logger
import uuid
import asyncio
from .ws_manager import websocket_manager, get_ray_ws_bridge
from .engine_resource_guard import maybe_release_warm_engine_for_non_engine_request

router = APIRouter(prefix="/preprocessor", tags=["preprocessor"])

# Legacy: kept for compatibility, but new registrations go through unified store
legacy_job_store: Dict[str, ray.ObjectRef] = {}


# Background task to poll updates from Ray bridge and send to websockets
async def poll_ray_updates():
    """Background task that polls the Ray bridge for updates and forwards to websockets"""
    # Ray may be started asynchronously (see `api.main`), so the websocket bridge might
    # not be available immediately. We retry until Ray is ready to avoid silently
    # disabling websocket-driven progress for all jobs.
    bridge = None
    logger.info("Starting polling Ray bridge for websocket updates")

    poll_interval_s = float(os.environ.get("RAY_WS_POLL_INTERVAL_S", "0.1"))
    max_updates_per_pull = int(os.environ.get("RAY_WS_MAX_UPDATES_PER_PULL", "200"))
    max_bridge_job_ids = int(os.environ.get("RAY_WS_MAX_BRIDGE_JOB_IDS", "5000"))

    while True:
        try:
            if bridge is None:
                try:
                    # Don't trigger Ray initialization from this polling loop.
                    # Ray is started asynchronously during app startup; until it's ready
                    # we simply sleep and retry.
                    from .ray_app import is_ray_ready

                    if not is_ray_ready():
                        await asyncio.sleep(0.5)
                        continue
                    bridge = get_ray_ws_bridge()
                    logger.info("Ray websocket bridge ready; polling enabled")
                except Exception:
                    # Ray not initialized yet (or bridge creation failed); retry shortly.
                    await asyncio.sleep(0.5)
                    continue

            # Get all job IDs that might have updates
            all_job_ids = set()
            try:
                all_job_ids.update(job_store.all_job_ids())
            except Exception:
                pass
            # include legacy jobs if any
            try:
                all_job_ids.update(legacy_job_store.keys())
            except Exception:
                pass

            # Also check the bridge for any job IDs it knows about
            try:
                bridge_job_ids = ray.get(
                    bridge.get_all_job_ids.remote(max_bridge_job_ids), timeout=0.05
                )
                all_job_ids.update(bridge_job_ids)
            except:
                pass

            # Check each job for updates
            for job_id in all_job_ids:
                try:
                    updates = ray.get(
                        bridge.get_updates.remote(job_id, max_updates_per_pull),
                        timeout=0.05,
                    )
                    if updates:
                        for update in updates:
                            await websocket_manager.send_update(job_id, update)
                except ray.exceptions.GetTimeoutError:
                    pass  # No updates available
                except Exception as e:
                    # If an oversized msgpack buffer caused this, the safest recovery is
                    # to clear the queued updates for this job so polling can resume.
                    try:
                        if "Unable to allocate internal buffer" in str(e):
                            ray.get(bridge.clear_updates.remote(job_id), timeout=0.05)
                    except Exception:
                        pass
                    logger.error(f"Error getting updates for job {job_id}: {e}")

            await asyncio.sleep(poll_interval_s)
        except Exception as e:
            logger.error(f"Error in poll_ray_updates: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # Bridge may have died (Ray restart); force re-create on next iteration.
            bridge = None
            await asyncio.sleep(1)


class DownloadRequest(BaseModel):
    preprocessor_name: str
    job_id: Optional[str] = None


class RunRequest(BaseModel):
    preprocessor_name: str
    input_path: str
    job_id: Optional[str] = None
    download_if_needed: bool = True
    params: Optional[Dict[str, Any]] = None
    start_frame: Optional[int] = None  # For video only, None means from beginning
    end_frame: Optional[int] = None  # For video only, None means to end


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    type: Optional[str] = None
    preprocessor: Optional[str] = None
    error: Optional[str] = None


class DeleteResponse(BaseModel):
    preprocessor_name: str
    status: str
    deleted_files: Optional[List[str]] = None
    message: Optional[str] = None


@router.get("/list")
def list_all_preprocessors(check_downloaded: bool = True):
    """
    List all available preprocessors with detailed parameter information.

    Args:
        check_downloaded: If True, include download status for each preprocessor

    Returns:
        List of preprocessor metadata including parameters, types, defaults, and download status
    """
    try:
        preprocessors = list_preprocessors(check_downloaded=check_downloaded)
        return {"count": len(preprocessors), "preprocessors": preprocessors}
    except Exception as e:
        logger.error(f"Failed to list preprocessors: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list preprocessors: {str(e)}"
        )


@router.get("/get/{preprocessor_name}")
def get_preprocessor(preprocessor_name: str):
    """
    Get detailed information about a specific preprocessor.

    Args:
        preprocessor_name: Name of the preprocessor

    Returns:
        Detailed preprocessor information including all parameters
    """
    try:
        details = get_preprocessor_details(preprocessor_name)
        return details
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get preprocessor details: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get preprocessor details: {str(e)}"
        )


@router.get("/ray-status")
def ray_status():
    """Check if Ray is running and get cluster info"""
    try:
        if not ray.is_initialized():
            return {
                "ray_connected": False,
                "message": "Ray is not initialized. Start the API to initialize Ray.",
            }

        resources = ray.available_resources()
        cluster_resources = ray.cluster_resources()
        nodes = ray.nodes()

        return {
            "ray_connected": True,
            "available_resources": resources,
            "cluster_resources": cluster_resources,
            "num_nodes": len(nodes),
            "nodes": [
                {
                    "NodeID": node["NodeID"],
                    "Alive": node["Alive"],
                    "Resources": node["Resources"],
                }
                for node in nodes
            ],
        }
    except Exception as e:
        return {
            "error": str(e),
            "ray_connected": False,
            "message": "Error connecting to Ray cluster",
        }


@router.post("/run", response_model=JobResponse)
def trigger_run(request: RunRequest):
    """
    Run a preprocessor on input media

    This creates a Ray job for processing. Use the job_id with the
    websocket endpoint /ws/job/{job_id} to get real-time updates.

    If download_if_needed is True and the model is not available,
    it will be downloaded first.
    """

    # Validate preprocessor exists via YAML registry
    try:
        details = get_preprocessor_details(request.preprocessor_name)
    except ValueError:
        available = [p["id"] for p in list_preprocessors(check_downloaded=False)]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preprocessor: {request.preprocessor_name}. Available: {available}",
        )

    # Validate input path exists
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Input file not found: {request.input_path}"
        )

    # If a warm engine is idling on the GPU, release it so this non-engine job can run.
    # If an engine job is active/queued, we do nothing (Ray will queue naturally).
    try:
        maybe_release_warm_engine_for_non_engine_request()
    except Exception:
        pass

    # Get best GPU for the task
    device_index, device_type = get_best_gpu()
    resources = get_ray_resources(device_index, device_type, load_profile="medium")

    logger.info(
        f"Submitting run task for preprocessor: {request.preprocessor_name}, input: {request.input_path}, resources: {resources}"
    )

    # Validate and convert parameters
    try:
        parameter_definitions = details.get("parameters", [])
        kwargs = validate_and_convert_params(
            request.params or {}, parameter_definitions
        )
        logger.info(f"Validated parameters: {kwargs}")
    except ValueError as e:
        logger.error(f"Parameter validation failed: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Parameter validation failed: {str(e)}"
        )

    # Submit Ray task
    try:
        job_id = request.job_id or str(uuid.uuid4())
        bridge = get_ray_ws_bridge()

        # Submit task with resource constraints
        task_ref = submit_tracked_job(
            job_id=job_id,
            job_type="preprocessor",
            meta={
                "preprocessor_name": request.preprocessor_name,
                "input_path": request.input_path,
            },
            # Force non-persistent Ray workers for preprocessors: we do not want any
            # warm/persistent model state after the job completes.
            submit=lambda: run_preprocessor.options(**resources).remote(
                request.preprocessor_name,
                request.input_path,
                job_id,
                bridge,
                request.start_frame,
                request.end_frame,
                **kwargs,
            ),
        )
        legacy_job_store[job_id] = task_ref

        logger.info(f"Run task submitted with job_id: {job_id}")

        return JobResponse(
            job_id=job_id,
            status="queued",
            message=f"Processing job created for {request.preprocessor_name}",
        )
    except Exception as e:
        logger.error(f"Failed to submit run task: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit Ray task: {str(e)}"
        )


@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Get the status of a job

    Note: For real-time updates, use the websocket endpoint /ws/job/{job_id}
    """
    return job_store.status(job_id)


@router.delete("/delete/{preprocessor_name}", response_model=DeleteResponse)
def delete_preprocessor(preprocessor_name: str):
    """
    Delete downloaded model files for a preprocessor and unmark it as downloaded.
    Removes files listed under the manifest's files section if present.
    """
    from src.api.preprocessor_registry import _load_preprocessor_yaml
    from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
    from src.preprocess.base_preprocessor import BasePreprocessor

    try:
        info = _load_preprocessor_yaml(preprocessor_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    files = info.get("files", [])
    base = Path(DEFAULT_PREPROCESSOR_SAVE_PATH)
    deleted: List[str] = []
    for f in files:
        rel = f.get("path") if isinstance(f, dict) else None
        if not rel:
            continue
        abs_path = base / rel
        try:
            if abs_path.exists():
                abs_path.unlink()
                deleted.append(str(abs_path))
        except Exception:
            # Best-effort: skip errors deleting individual files
            pass
    # Unmark downloaded
    BasePreprocessor._unmark_as_downloaded(preprocessor_name)

    # Best-effort: also purge Hugging Face caches commonly used by downloads
    try:
        hf_dirs: Set[str] = set()

        # Prefer environment variables when set
        hf_home = os.environ.get("HF_HOME")
        hf_hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get(
            "HUGGINGFACE_HUB_CACHE"
        )
        transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
        datasets_cache = os.environ.get("HF_DATASETS_CACHE")

        # Fall back to huggingface_hub defaults if available
        try:
            from huggingface_hub import constants as hf_constants

            if not hf_home:
                hf_home = getattr(hf_constants, "HF_HOME", None)
            if not hf_hub_cache:
                hf_hub_cache = getattr(hf_constants, "HF_HUB_CACHE", None)
        except Exception:
            pass

        # Build likely default paths when envs are not set
        user_home = os.path.expanduser("~")
        default_hf_root = os.path.join(user_home, ".cache", "huggingface")
        if not hf_home:
            hf_home = default_hf_root
        if not hf_hub_cache and hf_home:
            hf_hub_cache = os.path.join(hf_home, "hub")
        if not transformers_cache and hf_home:
            transformers_cache = os.path.join(hf_home, "transformers")
        if not datasets_cache and hf_home:
            datasets_cache = os.path.join(hf_home, "datasets")

        # Only remove well-known cache directories (not entire HF_HOME to avoid tokens/settings)
        for path in [hf_hub_cache, transformers_cache, datasets_cache]:
            if path and os.path.isdir(path):
                hf_dirs.add(os.path.abspath(path))

        # As a best-effort, also look for standard subdirs under HF_HOME
        if hf_home and os.path.isdir(hf_home):
            for sub in ("hub", "transformers", "datasets"):
                p = os.path.join(hf_home, sub)
                if os.path.isdir(p):
                    hf_dirs.add(os.path.abspath(p))

        # Delete collected directories
        for d in sorted(hf_dirs):
            try:
                shutil.rmtree(d, ignore_errors=True)
                deleted.append(d)
            except Exception:
                pass
    except Exception:
        # Never fail the request due to cache cleanup attempts
        pass

    return DeleteResponse(
        preprocessor_name=preprocessor_name,
        status="deleted",
        deleted_files=deleted,
        message=f"Removed {len(deleted)} file(s)/dir(s) including Hugging Face caches",
    )


@router.get("/result/{job_id}", response_model=ResultResponse)
def get_result(job_id: str):
    """
    Get the result file path for a completed job

    Returns the path to the cached result file.
    """
    # Check if job is in store
    if job_id in legacy_job_store:
        task_ref = legacy_job_store[job_id]

        # Check if task is ready
        ready_refs, remaining_refs = ray.wait([task_ref], timeout=0)

        if ready_refs:
            # Task is complete, get result
            try:
                result = ray.get(ready_refs[0])
                if result and result.get("status") == "complete":
                    return ResultResponse(
                        job_id=job_id,
                        status="complete",
                        result_path=result.get("result_path"),
                        type=result.get("type"),
                        preprocessor=result.get("preprocessor"),
                    )
                elif result and result.get("status") == "error":
                    return ResultResponse(
                        job_id=job_id, status="error", error=result.get("error")
                    )
            except Exception as e:
                return ResultResponse(job_id=job_id, status="error", error=str(e))
        else:
            # Still running
            return ResultResponse(
                job_id=job_id, status="running", error="Job is still processing"
            )

    # Try to load from metadata file
    cache_path = Path(DEFAULT_CACHE_PATH) / "preprocessor_results" / job_id
    metadata_path = cache_path / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return ResultResponse(
            job_id=job_id,
            status="complete",
            result_path=metadata.get("result_path"),
            type=metadata.get("type"),
            preprocessor=metadata.get("preprocessor"),
        )

    # Job not complete or not found
    return ResultResponse(job_id=job_id, status="unknown", error="Result not found")


@router.post("/cancel/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str):
    """
    Cancel a running job
    Stops the execution of a job and removes it from the active job list.
    """
    result = job_store.cancel(job_id)
    if result.get("status") in ["cancelled", "canceled"]:
        return JobResponse(
            job_id=job_id, status=result["status"], message=result.get("message")
        )
    raise HTTPException(status_code=404, detail=result.get("message", "Job not found"))
