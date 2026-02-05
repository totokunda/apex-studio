from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import os
import shutil

from loguru import logger as _default_logger

from src.api.savers.ffmpeg_mux import mux_audio_from_source_video_to_output
from src.api.savers.mp4 import optimize_mp4_for_editor_in_place


def save_frames_with_source_audio(
    *,
    frames: List,
    input_video_path: str,
    output_dir: Path,
    fps: int,
    logger=_default_logger,
) -> Tuple[str, Optional[str]]:
    """
    Save `frames` as MP4 into `output_dir`, then mux audio from `input_video_path`.
    Returns (final_out_path, video_only_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_only_path = str(output_dir / "result_video.mp4")
    final_out_path = str(output_dir / "result.mp4")

    from diffusers.utils import export_to_video

    export_to_video(frames, video_only_path, fps=int(max(1, fps)), quality=8.0)

    # Optimize MP4 before muxing audio.
    try:
        try:
            engine_gop = int(os.environ.get("APEX_VIDEO_EDITOR_ENGINE_GOP", "1") or "1")
        except Exception:
            engine_gop = 1
        optimize_mp4_for_editor_in_place(
            video_only_path, fps=int(max(1, fps)), gop_frames=engine_gop, logger=logger
        )
    except Exception:
        pass

    ok = mux_audio_from_source_video_to_output(
        video_only_path=video_only_path,
        source_video_path=input_video_path,
        output_path=final_out_path,
        job_dir=output_dir,
        logger=logger,
        audio_codec="copy",
        video_codec="copy",
    )
    if not ok:
        # If muxing failed (e.g., no audio stream), just use the video-only output.
        try:
            shutil.move(video_only_path, final_out_path)
        except Exception:
            final_out_path = video_only_path

    return final_out_path, video_only_path

