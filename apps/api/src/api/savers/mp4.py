from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os

from loguru import logger as _default_logger


def _env_flag(name: str, default: bool = False) -> bool:
    """
    Parse a boolean-ish environment variable.

    Accepts: 1/0, true/false, yes/no, on/off (case-insensitive).
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def optimize_mp4_for_editor_in_place(
    video_path: str,
    *,
    fps: Optional[int] = None,
    gop_frames: Optional[int] = None,
    logger=_default_logger,
) -> bool:
    """
    Best-effort post-pass for MP4s intended for interactive playback in editors.

    Goals:
    - Make MP4 seekable/streamable via `-movflags +faststart`
    - Optionally enforce CFR + a shorter GOP for smoother scrubbing
    - Keep behavior best-effort: on failure, leave the original file intact

    Controlled by env:
    - APEX_VIDEO_EDITOR_OPTIMIZE: enable/disable (default: true)
    - APEX_VIDEO_EDITOR_ENGINE_GOP: engine-result GOP in frames (default: 1)
    - APEX_VIDEO_EDITOR_PREPROCESSOR_GOP: preprocessor-result GOP in frames (default: 4)
    - APEX_VIDEO_EDITOR_GOP_FRAMES: keyframe interval in frames (default: unset)
    - APEX_VIDEO_EDITOR_GOP_SECONDS: keyframe interval in seconds (default: 1.0)
    - APEX_VIDEO_EDITOR_CRF: x264 CRF (default: 18)
    - APEX_VIDEO_EDITOR_PRESET: x264 preset (default: veryfast)
    - APEX_VIDEO_EDITOR_NO_BFRAMES: disable B-frames (default: true)
    - APEX_VIDEO_EDITOR_FFMPEG_TIMEOUT_SECONDS: timeout for ffmpeg (default: 60)
    """
    try:
        from src.utils.ffmpeg import get_ffmpeg_path, run_ffmpeg

        if not _env_flag("APEX_VIDEO_EDITOR_OPTIMIZE", default=True):
            return False

        if not (isinstance(video_path, str) and os.path.isfile(video_path)):
            return False

        base = Path(video_path)
        if base.suffix.lower() != ".mp4":
            return False

        # Prefer explicit GOP frames if provided; then env override; then GOP seconds.
        gop_frames_int: Optional[int] = None
        if gop_frames is not None:
            try:
                gop_frames_int = int(gop_frames)
            except Exception:
                gop_frames_int = None
        if gop_frames_int is None:
            try:
                _raw = os.environ.get("APEX_VIDEO_EDITOR_GOP_FRAMES")
                if _raw is not None and str(_raw).strip() != "":
                    gop_frames_int = int(str(_raw).strip())
            except Exception:
                gop_frames_int = None
        if gop_frames_int is not None:
            gop_frames_int = max(1, min(1000, int(gop_frames_int)))

        try:
            gop_seconds = float(os.environ.get("APEX_VIDEO_EDITOR_GOP_SECONDS", "1.0") or "1.0")
        except Exception:
            gop_seconds = 1.0
        gop_seconds = max(0.25, min(10.0, gop_seconds))

        try:
            crf = int(os.environ.get("APEX_VIDEO_EDITOR_CRF", "18") or "18")
        except Exception:
            crf = 18
        crf = max(0, min(51, crf))

        preset = str(os.environ.get("APEX_VIDEO_EDITOR_PRESET", "veryfast") or "veryfast").strip() or "veryfast"
        no_bframes = _env_flag("APEX_VIDEO_EDITOR_NO_BFRAMES", default=True)

        fps_int: Optional[int] = None
        if fps is not None:
            try:
                fps_int = int(max(1, round(float(fps))))
            except Exception:
                fps_int = None

        temp_out_path = base.with_name(f"{base.stem}_editor{base.suffix}")

        try:
            timeout_s = float(os.environ.get("APEX_VIDEO_EDITOR_FFMPEG_TIMEOUT_SECONDS", "60") or "60")
        except Exception:
            timeout_s = 60.0
        timeout_s = max(5.0, min(60.0 * 10.0, float(timeout_s)))

        ffmpeg_path = get_ffmpeg_path()

        # GOP tuning for smoother seeking/scrubbing (requires re-encode).
        gop: Optional[int] = None
        if gop_frames_int is not None:
            gop = int(gop_frames_int)
        elif fps_int is not None:
            gop = int(max(1, round(float(fps_int) * float(gop_seconds))))

        # Common flags for daemon-safe invocations.
        common: List[str] = [
            ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(base),
            # Video only: audio is muxed elsewhere in the pipeline.
            "-an",
            "-map",
            "0:v:0",
        ]

        candidate_cmds: List[List[str]] = []

        # If we don't know FPS and no GOP override is requested, prefer fast stream copy
        # to relocate the moov atom (`+faststart`) without re-encoding.
        if fps_int is None and gop is None:
            candidate_cmds.append(
                common
                + [
                    "-c:v",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(temp_out_path),
                ]
            )

        # Re-encode path (needed for CFR/GOP/B-frames tuning).
        cmd: List[str] = (
            common
            + [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-sc_threshold",
                "0",
            ]
        )

        if gop is not None:
            cmd.extend(["-g", str(gop), "-keyint_min", str(gop)])
        if fps_int is not None:
            cmd.extend(["-fps_mode", "cfr", "-r", str(fps_int)])
        if no_bframes:
            cmd.extend(["-x264-params", "bframes=0"])

        cmd.extend(["-movflags", "+faststart", str(temp_out_path)])
        candidate_cmds.append(cmd)

        last_err: Optional[str] = None
        for attempt, cmd_i in enumerate(candidate_cmds, start=1):
            try:
                log_path = base.with_name(f"{base.stem}_editor_ffmpeg_{attempt}.log")
                rc, lp, _ = run_ffmpeg(cmd_i, timeout_s=timeout_s, log_path=log_path)
                if rc == 0 and temp_out_path.is_file():
                    try:
                        temp_out_path.replace(base)
                        return True
                    except Exception as move_err:
                        last_err = f"move_failed: {move_err}"
                        break
                last_err = f"ffmpeg_failed: code={rc} log={lp}"
            except Exception as e:
                last_err = str(e)

        if last_err:
            logger.warning(f"MP4 editor optimization failed for {video_path}: {last_err}")
        return False
    except Exception as e:
        logger.warning(f"MP4 editor optimization crashed for {video_path}: {e}")
        return False

