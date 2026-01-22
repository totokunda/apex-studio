from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger


# Job types registered in `src/api/engine.py`.
_ENGINE_JOB_TYPES = {"engine", "engine_warmup"}


def _has_incomplete_engine_jobs() -> bool:
    """
    Return True when there is any engine(-related) Ray job not yet completed.

    We treat both "running" and "queued" as incomplete. This is intentionally conservative:
    if an engine job is queued behind a warm actor, we should not kill that actor.
    """
    try:
        import ray

        from .job_store import job_store

        # Best-effort: job_store is in-memory; it's fine to read its private map here.
        jobs = getattr(job_store, "_jobs", {}) or {}
        for _job_id, data in jobs.items():
            try:
                if (data or {}).get("type") not in _ENGINE_JOB_TYPES:
                    continue
                ref = (data or {}).get("ref")
                if ref is None:
                    continue
                ready, _ = ray.wait([ref], timeout=0)
                if not ready:
                    return True
            except Exception:
                # If we cannot reliably inspect a job, do not risk killing the actor.
                return True
        return False
    except Exception:
        # If Ray isn't initialized or anything goes wrong, do nothing.
        return True


def maybe_release_warm_engine_for_non_engine_request() -> Dict[str, Any]:
    """
    Best-effort: if the warm GPU engine actor exists and no engine jobs are active/queued,
    offload all warm-pooled engine modules and kill the actor so Ray gets the GPU back.

    This is intentionally a "simple" policy:
    - If any engine job is incomplete, we do nothing (let it run/queue).
    - Otherwise, we try to release VRAM/RAM and the Ray GPU token before scheduling
      non-engine work (preprocessor/postprocessor).
    """
    try:
        from .ray_app import get_ray_app
        from .ray_resources import get_best_gpu, get_ray_resources
        from .ray_tasks import kill_engine_runner_actor, _engine_runner_actor_name

        ray = get_ray_app()

        if _has_incomplete_engine_jobs():
            return {"released": False, "reason": "engine_job_incomplete"}

        device_index, device_type = get_best_gpu()
        if device_type != "cuda" or device_index is None:
            return {"released": False, "reason": "no_cuda_device"}

        actor_name = _engine_runner_actor_name(
            device_index=device_index, device_type=device_type
        )
        try:
            runner = ray.get_actor(actor_name)
        except Exception:
            return {"released": False, "reason": "engine_actor_not_running"}

        # Run the free op inside the warm actor process so it can see the per-process pool.
        # Use a light resource profile; this work should be quick and not hoard CPUs.
        resources = get_ray_resources(device_index, device_type, load_profile="light")
        _ = resources  # (kept for logging/debugging if needed)

        try:
            ref = runner.free_unused_modules_in_warm_pool.remote(active=[], target="disk")
            result: Optional[Dict[str, Any]] = ray.get(ref)
        except Exception as e:
            logger.warning(f"auto-release: failed to offload warm pool: {e}")
            return {"released": False, "reason": "offload_failed", "error": str(e)}

        skipped = []
        if isinstance(result, dict):
            skipped = result.get("skipped_in_use") or []

        # If anything is "in_use", do not kill the actor (avoid disrupting a run).
        if skipped:
            return {
                "released": False,
                "reason": "warm_pool_in_use",
                "skipped_in_use": skipped,
                "worker_result": result,
            }

        # Kill the actor process to release the CUDA context and return the Ray GPU token.
        killed = bool(
            kill_engine_runner_actor(device_index=device_index, device_type=device_type)
        )
        logger.info(
            f"auto-release: engine actor {actor_name} released (killed={killed})"
        )
        return {
            "released": True,
            "actor_killed": killed,
            "actor_name": actor_name,
            "worker_result": result,
        }
    except Exception as e:
        # Never fail the user request because cleanup couldn't run.
        logger.debug(f"auto-release: skipped due to error: {e}")
        return {"released": False, "reason": "exception", "error": str(e)}

