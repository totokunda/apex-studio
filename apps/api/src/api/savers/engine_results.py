from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from loguru import logger as _default_logger
from PIL import Image

from src.api.savers.ffmpeg_mux import (
    mux_audio_files_into_video_in_place,
    mux_audio_from_source_video_in_place,
)
from src.api.savers.mp4 import optimize_mp4_for_editor_in_place


def save_engine_output(
    *,
    output_obj: Any,
    job_dir: Path,
    filename_prefix: str = "result",
    final: bool = False,
    fps: int = 16,
    audio_inputs: Optional[List[str]] = None,
    is_upscaler_engine: bool = False,
    input_video_for_audio_mux: Optional[str] = None,
    logger=_default_logger,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Persist an engine output into `job_dir`.

    Supports:
    - str: passthrough path
    - PIL.Image: save as png/jpg
    - list of frames: export MP4
    - numpy-like: try save as PNG

    When final=True and output is a video:
    - optimize MP4 for editor (best-effort)
    - mux provided audio_inputs (best-effort)
    - for upscalers, mux audio from the source input video (best-effort)
    """
    result_path: Optional[str] = None
    media_type: Optional[str] = None

    try:
        # String path passthrough
        if isinstance(output_obj, str):
            result_path = output_obj
            media_type = "path"

        # Single image
        elif isinstance(output_obj, Image.Image):
            ext = "png" if final else "jpg"
            result_path = str(job_dir / f"{filename_prefix}.{ext}")
            output_obj.save(result_path)
            media_type = "image"

        # Sequence of frames -> MP4
        elif isinstance(output_obj, list) and len(output_obj) > 0:
            result_path = str(job_dir / f"{filename_prefix}.mp4")
            from diffusers.utils import export_to_video

            export_to_video(
                output_obj,
                result_path,
                fps=int(max(1, fps)),
                quality=8.0 if final else 3.0,
            )
            media_type = "video"

            # Optimize MP4 (best-effort) before muxing any audio.
            if final and result_path:
                try:
                    import os

                    try:
                        engine_gop = int(os.environ.get("APEX_VIDEO_EDITOR_ENGINE_GOP", "1") or "1")
                    except Exception:
                        engine_gop = 1
                    optimize_mp4_for_editor_in_place(
                        result_path, fps=int(max(1, fps)), gop_frames=engine_gop, logger=logger
                    )
                except Exception:
                    pass

            # If final video and we have audio inputs to save, mux them in (best-effort).
            if final and media_type == "video" and result_path and audio_inputs:
                try:
                    muxed = mux_audio_files_into_video_in_place(
                        video_path=result_path,
                        audio_paths=audio_inputs,
                        job_dir=job_dir,
                        logger=logger,
                    )
                    if muxed:
                        result_path = muxed
                except Exception as mux_err:
                    logger.warning(
                        "Audio muxing failed; returning video-only output. "
                        f"Error: {mux_err}"
                    )

            # Upscalers: preserve input video audio if present.
            if (
                final
                and media_type == "video"
                and result_path
                and is_upscaler_engine
                and input_video_for_audio_mux
            ):
                try:
                    muxed = mux_audio_from_source_video_in_place(
                        video_path=result_path,
                        source_video_path=input_video_for_audio_mux,
                        job_dir=job_dir,
                        logger=logger,
                    )
                    if muxed:
                        result_path = muxed
                except Exception as mux_err:
                    logger.warning(
                        "Upscaler input audio muxing failed; returning video-only output. "
                        f"Error: {mux_err}"
                    )

        else:
            # Fallback best-effort serialization
            try:
                arr = np.asarray(output_obj)
                result_path = str(job_dir / f"{filename_prefix}.png")
                Image.fromarray(arr).save(result_path)
                media_type = "image"
            except Exception as e:
                logger.error(f"Failed to save output: {e}")
                result_path = str(job_dir / f"{filename_prefix}.txt")
                with open(result_path, "w") as f:
                    f.write(str(type(output_obj)))
                media_type = "unknown"
    except Exception as save_err:
        logger.error(f"Failed to save output: {save_err}")
        raise

    return result_path, media_type

