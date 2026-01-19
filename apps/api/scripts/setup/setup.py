from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path


def _update_persisted_config(updates: dict, config_store_path: Path) -> None:
    import json

    """
    Persist config-related values (paths, hf_token, etc.) so they survive backend restarts.
    """
    try:
        data = {}
        if config_store_path.exists():
            try:
                with config_store_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        data = existing
            except Exception:
                data = {}

        for key, value in updates.items():
            if value is not None:
                data[key] = value

        config_store_path.parent.mkdir(parents=True, exist_ok=True)
        with config_store_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        # Silently ignore persistence errors; API behavior should not depend on disk writes.
        # For debugging, you may want to log this error.
        print(f"Warning: failed to persist config settings: {e}")


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

        total_downloaded = 0
        total_size = 0
        for v in self.files.values():
            c = v.get("current") or 0
            t = v.get("total")
            if isinstance(c, int):
                total_downloaded += c
            if isinstance(t, int) and t > 0:
                total_size += t

        frac: Optional[float] = None
        if total_size > 0:
            frac = max(0.0, min(1.0, total_downloaded / total_size))

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

    # Determine which tasks will run for overall progress mapping
    tasks: list[str] = []
    if args.mask_model_type:
        tasks.append("mask")
    if args.install_rife:
        tasks.append("rife")
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
            metadata=metadata,
            status="processing",
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

    def mask_progress_cb(
        current: int, total: Optional[int], label: Optional[str] = None
    ):
        if "mask" not in trackers:
            return
        frac, md = trackers["mask"].update(current, total, label)
        emit(
            "mask",
            frac,
            f"Downloading mask model… {md.get('filename')}",
            md,
        )

    def rife_progress_cb(
        current: int, total: Optional[int], label: Optional[str] = None
    ):
        if "rife" not in trackers:
            return
        frac, md = trackers["rife"].update(current, total, label)
        emit(
            "rife",
            frac,
            f"Downloading RIFE… {md.get('filename')}",
            md,
        )

    def download_mask_model(model_type):
        # Lazy import: SAM mask stack is heavy and unnecessary for other setup tasks.
        from src.mask.mask import ModelType, MODEL_WEIGHTS

        model_type = ModelType(model_type)
        model_weight = MODEL_WEIGHTS[model_type]
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
        if args.install_rife:
            download_rife()

        if args.mask_model_type:
            download_mask_model(args.mask_model_type)

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
                _update_persisted_config(
                    {"ENABLE_IMAGE_RENDER_STEP": "true"}, config_store_path
                )
            if args.enable_video_render_steps:
                emit_config_step(
                    0.6,
                    "Enabling video render steps…",
                    {"key": "ENABLE_VIDEO_RENDER_STEP"},
                )
                _update_persisted_config(
                    {"ENABLE_VIDEO_RENDER_STEP": "true"}, config_store_path
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
