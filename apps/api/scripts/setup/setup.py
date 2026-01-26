from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path


def _ensure_src_importable() -> None:
    """
    Ensure `import src` works in both repo and bundled layouts.

    Repo layout:
    - this script: `apps/api/scripts/setup/setup.py`
    - package root: `apps/api/` (contains `src/`)

    Bundled layout:
    - installer runs this script from a bundle root that typically also contains `src/`
    """
    try:
        import src  # noqa: F401

        return
    except Exception:
        pass

    candidates: list[str] = []

    bundle_root = os.environ.get("APEX_BUNDLE_ROOT")
    if bundle_root:
        try:
            candidates.append(str(Path(bundle_root).expanduser().resolve()))
        except Exception:
            candidates.append(str(bundle_root))

    # `.../scripts/setup/setup.py` -> `.../apps/api`
    try:
        candidates.append(str(Path(__file__).resolve().parents[2]))
    except Exception:
        pass

    for c in candidates:
        if not c:
            continue
        if c not in sys.path:
            sys.path.insert(0, c)
        try:
            import src  # noqa: F401

            return
        except Exception:
            continue


_ensure_src_importable()






def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apex Engine setup helper: download optional models and toggle persisted flags."
    )
    parser.add_argument(
        "--apex_home_dir",
        type=str,
        required=True,
        help="Path to Apex home directory (persisted config location).",
    )
    parser.add_argument(
        "--mask_model_type",
        type=str,
        choices=["sam2_tiny", "sam2_small", "sam2_base_plus", "sam2_large", "sam3"],
        help="Download the selected mask model weights into the preprocessor directory.",
    )
    parser.add_argument(
        "--install_rife",
        action="store_true",
        help="Download the RIFE postprocessor weights into the postprocessor directory.",
    )
    parser.add_argument(
        "--enable_image_render_steps",
        action="store_true",
        help="Persist ENABLE_IMAGE_RENDER_STEP=true in config.",
    )
    
    parser.add_argument(
        "--enable_video_render_steps",
        action="store_true",
        help="Persist ENABLE_VIDEO_RENDER_STEP=true in config.",
    )

    parser.add_argument(
        "--skip_attention_verification",
        action="store_true",
        help="Skip attention backend verification (not recommended; installer normally runs this).",
    )

    # Optional job id for structured progress consumers (e.g. Electron installer)
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Job id included in emitted progress JSON (defaults to a random UUID).",
    )

    # Optional terminal progress logging (useful for test_setup.sh / CLI verification)
    parser.add_argument(
        "--log_progress_callbacks",
        action="store_true",
        help="Also log progress callback updates to the terminal (stdout).",
    )

    return parser.parse_args(argv)


@dataclass
class MultiFileProgress:
    """
    Aggregates per-file byte callbacks (current,total,label) into a best-effort overall fraction.
    """

    files: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    last_frac: float = 0.0

    def reset(self) -> None:
        self.files = {}
        self.last_frac = 0.0

    def update(
        self, current: int, total: Optional[int], label: Optional[str]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        name = (label or "unknown").strip() or "unknown"
        try:
            cur_i = int(current or 0)
        except Exception:
            cur_i = 0
        tot_i: Optional[int]
        try:
            tot_i = int(total) if total is not None else None
        except Exception:
            tot_i = None

        self.files[name] = {"current": cur_i, "total": tot_i}

        # Prefer accurate totals when available.
        downloaded_known = 0
        total_known = 0
        downloaded_any = 0
        for v in self.files.values():
            c = v.get("current") or 0
            t = v.get("total")
            if isinstance(c, int):
                downloaded_any += max(0, c)
            if isinstance(c, int) and isinstance(t, int) and t > 0:
                downloaded_known += max(0, min(c, t))
                total_known += t

        frac: Optional[float]
        if total_known > 0:
            frac = max(0.0, min(1.0, downloaded_known / total_known))
        else:
            # Fallback when we don't know file sizes (some downloaders omit totals):
            # provide a smooth, monotonic approximation based on bytes seen so far.
            if downloaded_any <= 0:
                frac = 0.0
            else:
                # 0..~0.95 over the first few hundred MiB, then the caller will mark complete at the end.
                scale = 100 * 1024 * 1024  # 100MiB
                frac = 1.0 - float(pow(2.718281828, -float(downloaded_any) / float(scale)))
                frac = min(0.99, max(0.0, frac))

        # Enforce monotonicity to avoid jitter when multiple files start/finish.
        try:
            frac = max(self.last_frac, float(frac))
        except Exception:
            frac = self.last_frac
        self.last_frac = float(frac)

        metadata = {
            "filename": name,
            "current_bytes": cur_i,
            "total_bytes": tot_i,
            # compatibility with various frontend extractors
            "downloaded": cur_i,
            "total": tot_i,
            "bytes_downloaded": cur_i,
            "bytes_total": tot_i,
            "files": self.files,
        }
        return frac, metadata


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Required environment for defaults/config paths
    os.environ["APEX_HOME_DIR"] = args.apex_home_dir

    from src.utils.defaults import (
        get_preprocessor_path,
        get_postprocessor_path,
        get_config_store_path,
    )
    from src.mixins import LoaderMixin

    preprocessor_path = get_preprocessor_path()
    postprocessor_path = get_postprocessor_path()
    config_store_path = get_config_store_path()
    job_id = (args.job_id or "").strip() or str(uuid.uuid4())

    # Determine which tasks will run for overall progress mapping.
    # Keep this in the same order we execute below so UIs can render phases sequentially.
    tasks: list[str] = []
    if args.mask_model_type:
        tasks.append("mask")
    if args.install_rife:
        tasks.append("rife")
    # Attention verification should run during install so users can see what backends work.
    # It is opt-out for dev/CI via --skip_attention_verification.
    if not args.skip_attention_verification:
        tasks.append("attention")
    # Config updates (render-step toggles) should also show progress in the UI.
    if args.enable_image_render_steps or args.enable_video_render_steps:
        tasks.append("config")
    weight = 1.0 / max(1, len(tasks))
    completed: Dict[str, float] = {t: 0.0 for t in tasks}
    trackers: Dict[str, MultiFileProgress] = {t: MultiFileProgress() for t in tasks}

    def _fmt_bytes(n: Optional[int]) -> str:
        if not isinstance(n, int) or n < 0:
            return "?"
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        v = float(n)
        i = 0
        while v >= 1024.0 and i < len(units) - 1:
            v /= 1024.0
            i += 1
        if i == 0:
            return f"{int(v)}{units[i]}"
        return f"{v:.1f}{units[i]}"

    # Throttle terminal logs to avoid spamming on very frequent byte callbacks.
    _terminal_state: Dict[str, Dict[str, Any]] = {}

    def send_update(
        task: str,
        progress: Optional[float],
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "processing",
    ) -> None:
        md = metadata or {}
        payload = {
            "progress": progress,
            "message": message,
            "status": status,
            "metadata": {**md, "task": task, "job_id": job_id},
        }
        key = task or "setup"
        now = time.monotonic()
        prev = _terminal_state.get(key, {})
        prev_t = float(prev.get("t", 0.0) or 0.0)
        prev_p = prev.get("p")
        prev_file = prev.get("filename")

        filename = md.get("filename")
        cur_b = (
            md.get("current_bytes")
            if isinstance(md.get("current_bytes"), int)
            else None
        )
        tot_b = (
            md.get("total_bytes") if isinstance(md.get("total_bytes"), int) else None
        )

        # Emit if:
        # - status changes (complete/error)
        # - we cross a meaningful progress delta
        # - filename changes
        # - or enough time has passed
        should = False
        if status in ("complete", "error"):
            should = True
        elif filename and filename != prev_file:
            should = True
        elif progress is None or prev_p is None:
            should = True
        else:
            try:
                if abs(float(progress) - float(prev_p)) >= 0.01:
                    should = True
            except Exception:
                should = True
        if not should and (now - prev_t) >= 0.25:
            should = True

        if should:
            if args.log_progress_callbacks:
                pct = "?"
                if isinstance(progress, (int, float)):
                    pct = f"{max(0.0, min(1.0, float(progress))) * 100.0:5.1f}%"
                suffix = ""
                if cur_b is not None or tot_b is not None or filename:
                    suffix = f" | {filename or 'unknown'} {_fmt_bytes(cur_b)}/{_fmt_bytes(tot_b)}"
                print(f"[setup][{task}] {pct} {message}{suffix}", flush=True)

            # Structured progress for machine consumers (Electron installer, CI, etc.)
            if os.environ.get("APEX_SETUP_PROGRESS_JSON") == "1":
                print(json.dumps(payload, ensure_ascii=False), flush=True)

            _terminal_state[key] = {"t": now, "p": progress, "filename": filename}

    def emit(
        task: str, p_task: Optional[float], message: str, metadata: Dict[str, Any]
    ):
        # Map per-task progress into an overall [0,1] best-effort progress
        if task in completed and isinstance(p_task, (int, float)):
            completed[task] = max(0.0, min(1.0, float(p_task)))

        overall: Optional[float] = None
        if tasks and all(isinstance(completed.get(t), (int, float)) for t in tasks):
            overall = 0.0
            for t in tasks:
                overall += weight * float(completed.get(t, 0.0))
            overall = max(0.0, min(1.0, overall))

        send_update(
            task=task,
            progress=overall,
            message=message,
            metadata={
                **metadata,
                # Prefer phase-specific progress in the UI. `progress` is overall across tasks.
                "task_progress": (
                    max(0.0, min(1.0, float(p_task)))
                    if isinstance(p_task, (int, float))
                    else None
                ),
            },
            status="processing",
        )

    def verify_attention_backends_with_progress() -> None:
        """
        Verify attention backends (best-effort) and stream per-backend progress.

        Important goals:
        - keep failures isolated (some backends can crash/segfault). This is handled by
          `src.attention.functions.verify_attention_backends()` which verifies in separate processes.
        - do NOT fail the entire installer if optional backends fail; we just report and continue
        - populate the attention cache via `verify_attention_backends(force_refresh=True)`
        """
        if "attention" not in completed:
            return

        # In the installer we run setup.py with cwd=bundleRoot. Keep that as bundle_root.
        bundle_root = Path(os.environ.get("APEX_BUNDLE_ROOT") or os.getcwd()).resolve()
        # Make bundle root visible to any subprocesses spawned by verification.
        os.environ["APEX_BUNDLE_ROOT"] = str(bundle_root)

        emit(
            "attention",
            0.0,
            "Verifying attention backends…",
            {"phase": "start", "bundle_root": str(bundle_root)},
        )

        # NOTE: We intentionally avoid `smoke_tests.*` here. Smoke tests are not bundled
        # into production releases, while `src.attention.functions` is.
        try:
            from src.attention.functions import (
                attention_register,
                verify_attention_backends_detailed,
            )
        except Exception as e:
            completed["attention"] = 1.0
            send_update(
                task="attention",
                progress=None,
                message=f"Attention verification skipped (unable to import verifier): {e}",
                metadata={"phase": "done", "status": "skipped", "task_progress": 1.0},
                status="processing",
            )
            return

        # 1) Enumerate candidate backends from the runtime registry.
        try:
            available = sorted([str(k) for k in attention_register.all_available().keys()])
        except Exception:
            available = []

        emit(
            "attention",
            0.05,
            f"Discovered {len(available) or 0} attention backend(s) to check…",
            {"phase": "list", "available_count": len(available)},
        )

        total = len(available)
        if total == 0:
            completed["attention"] = 1.0
            send_update(
                task="attention",
                progress=None,
                message="Attention verification skipped (no candidates discovered).",
                metadata={"phase": "done", "status": "skipped", "task_progress": 1.0},
                status="processing",
            )
            return

        emit(
            "attention",
            0.1,
            "Running backend verification…",
            {"phase": "check", "total": total},
        )

        ok: list[str] = []
        failed: list[dict[str, Any]] = []

        def _on_backend_event(ev: dict[str, Any]) -> None:
            try:
                if ev.get("phase") != "backend":
                    return
                backend = str(ev.get("backend") or "")
                if not backend:
                    return
                idx = int(ev.get("index") or 0)
                tot = int(ev.get("total") or total or 1)
                status = str(ev.get("status") or "")

                # Map per-backend progress into a smooth 0..1 range for the attention phase.
                # Reserve 0.10..0.98 for per-backend steps.
                base = 0.10
                span = 0.88
                # Start of a backend check bumps slightly; completion bumps more.
                step = 0.0
                if status == "start":
                    step = 0.15
                elif status in ("ok", "failed", "timeout"):
                    step = 0.95
                p = base + span * (min(max(idx + step, 0.0), float(tot)) / float(max(1, tot)))
                p = max(0.0, min(0.99, float(p)))

                msg = (
                    f"Checking {backend}…"
                    if status == "start"
                    else f"{backend}: {status}"
                )
                emit(
                    "attention",
                    p,
                    msg,
                    {
                        "phase": "backend",
                        "backend": backend,
                        "backend_status": status,
                        "index": idx,
                        "total": tot,
                    },
                )
            except Exception:
                return

        try:
            res = verify_attention_backends_detailed(
                force_refresh=True, progress_callback=_on_backend_event
            )
            ok = sorted([str(x) for x in (res.get("working") or [])])
            failed_names = [str(x) for x in (res.get("failed") or [])]
            failed = [{"backend": n, "reason": "failed verification"} for n in failed_names]
        except Exception as e:
            # Best-effort: do not fail installer; report and continue.
            ok = []
            failed = [{"backend": "*", "reason": f"verification failed: {e}"}]

        completed["attention"] = 1.0
        summary = (
            f"Attention verification complete ({len(ok)} ok, {len(failed)} failed)"
        )
        send_update(
            task="attention",
            progress=None,
            message=summary,
            metadata={
                "phase": "done",
                "ok": ok,
                "failed": failed,
                "task_progress": 1.0,
                "status": "complete",
            },
            status="complete",
        )

    def emit_config_step(
        p_task: float, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit progress for config toggles. This shows up as task='config' so the frontend can map it
        into the update_configs phase while still using the same overall progress mapping.
        """
        md: Dict[str, Any] = metadata or {}
        emit("config", max(0.0, min(1.0, float(p_task))), message, md)

    def _persist_config_updates_best_effort(updates: Dict[str, Any]) -> None:
        """
        Persist config updates so they survive backend restarts.

        Prefer the canonical `_update_persisted_config` implementation used by the API.
        If it can't be imported (e.g. packaging/path differences), fall back to a small
        local writer that uses the same persisted JSON config store.
        """
        # Filter None values to match the API behavior.
        safe_updates = {k: v for k, v in (updates or {}).items() if v is not None}
        if not safe_updates:
            return

        try:
            # Canonical implementation (may import FastAPI/Pydantic modules).
            from src.api.config import _update_persisted_config as _official_update

            _official_update(**safe_updates)
            return
        except Exception:
            pass

        # Fallback: best-effort update of the persisted config store JSON.
        try:
            from src.utils.config_store import (
                config_store_lock,
                read_json_dict,
                write_json_dict_atomic,
            )

            p = Path(str(config_store_path))
            with config_store_lock(p):
                data = read_json_dict(p)
                for k, v in safe_updates.items():
                    data[k] = v
                write_json_dict_atomic(p, data, indent=2)
        except Exception as e:
            # Keep parity with the API helper: do not fail setup on persistence errors.
            print(f"Warning: failed to persist config settings: {e}", flush=True)

    def mask_progress_cb(
        current: int, total: Optional[int], label: Optional[str] = None
    ):
        if "mask" not in trackers:
            return
        frac, md = trackers["mask"].update(current, total, label)
        filename = md.get("filename") or ""

        # Remove UUID (standard v4 pattern) or long hex hashes (SHA256) if present
        filename = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{64}",
            "",
            filename,
            flags=re.I,
        )
        # Clean up any leftover underscores/hyphens from the removal
        filename = re.sub(r"[-_]{2,}", "-", filename).strip("-_ ")

        if filename:
            md["filename"] = filename

        emit(
            "mask",
            frac,
            f"Downloading mask model… {filename}",
            md,
        )

    def rife_progress_cb(
        current: int, total: Optional[int], label: Optional[str] = None
    ):
        if "rife" not in trackers:
            return
        frac, md = trackers["rife"].update(current, total, label)
        filename = md.get("filename") or ""

        # Remove UUID (standard v4 pattern) or long hex hashes (SHA256) if present
        filename = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{64}",
            "",
            filename,
            flags=re.I,
        )
        # Clean up any leftover underscores/hyphens from the removal
        filename = re.sub(r"[-_]{2,}", "-", filename).strip("-_ ")

        if filename:
            md["filename"] = filename

        emit(
            "rife",
            frac,
            f"Downloading RIFE… {filename}",
            md,
        )

    def download_mask_model(model_type):
        # Lazy import: SAM mask stack is heavy and unnecessary for other setup tasks.
        from src.mask.mask import ModelType, MODEL_WEIGHTS

        model_type = ModelType(model_type)
        model_weight = MODEL_WEIGHTS[model_type]
        try:
            trackers["mask"].reset()
        except Exception:
            trackers["mask"] = MultiFileProgress()
        completed["mask"] = 0.0
        send_update(
            task="mask",
            progress=0.0 if tasks else None,
            message=f"Starting mask model download: {model_type.value}",
            metadata={"model_type": model_type.value},
            status="processing",
        )
        loader = LoaderMixin()
        loader._download(
            model_weight,
            save_path=preprocessor_path,
            progress_callback=mask_progress_cb,
        )
        completed["mask"] = 1.0
        send_update(
            task="mask",
            progress=1.0 if tasks == ["mask"] else None,
            message=f"Mask model download complete: {model_type.value}",
            metadata={"model_type": model_type.value, "status": "complete"},
            status="complete",
        )
        return loader

    def download_rife():
        # Use lightweight downloader to avoid importing torch/model code during setup.
        from src.postprocess.rife import download_rife_assets

        try:
            trackers["rife"].reset()
        except Exception:
            trackers["rife"] = MultiFileProgress()
        completed["rife"] = 0.0
        send_update(
            task="rife",
            progress=0.0 if tasks else None,
            message="Starting RIFE download",
            metadata={},
            status="processing",
        )
        download_rife_assets(
            save_path=postprocessor_path, progress_callback=rife_progress_cb
        )
        completed["rife"] = 1.0
        send_update(
            task="rife",
            progress=1.0 if tasks == ["rife"] else None,
            message="RIFE download complete",
            metadata={"status": "complete"},
            status="complete",
        )
        return None

    try:
        if args.mask_model_type:
            download_mask_model(args.mask_model_type)

        if args.install_rife:
            download_rife()

        if not args.skip_attention_verification:
            verify_attention_backends_with_progress()

        # Config toggles (render-step flags)
        if args.enable_image_render_steps or args.enable_video_render_steps:
            # Lazy import: config module import can be heavy depending on environment/setup.
            # Best-effort: show config progress even if we only toggle one setting.
            emit_config_step(0.0, "Updating config…", {"status": "processing"})
            if args.enable_image_render_steps:
                emit_config_step(
                    0.2,
                    "Enabling image render steps…",
                    {"key": "ENABLE_IMAGE_RENDER_STEP"},
                )
                _persist_config_updates_best_effort(
                    {"ENABLE_IMAGE_RENDER_STEP": "true"}
                )
            if args.enable_video_render_steps:
                emit_config_step(
                    0.6,
                    "Enabling video render steps…",
                    {"key": "ENABLE_VIDEO_RENDER_STEP"},
                )
                _persist_config_updates_best_effort(
                    {"ENABLE_VIDEO_RENDER_STEP": "true"}
                )
            completed["config"] = 1.0
            send_update(
                task="config",
                progress=1.0 if tasks == ["config"] else None,
                message="Config updated",
                metadata={"status": "complete"},
                status="complete",
            )

        send_update(
            task="setup",
            progress=1.0 if tasks else None,
            message="Setup complete",
            metadata={"status": "complete"},
            status="complete",
        )
        return 0
    except Exception as e:
        send_update(
            task="setup",
            progress=None,
            message=f"Setup failed: {e}",
            metadata={"status": "error"},
            status="error",
        )
        raise
    finally:
        # Give stdout consumers a brief moment to flush buffers before exiting.
        try:
            time.sleep(0.05)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
