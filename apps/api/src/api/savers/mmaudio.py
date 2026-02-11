from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from loguru import logger as _default_logger

from src.api.savers.audio_video import _write_wav_wave
from src.api.savers.ffmpeg_mux import mux_audio_files_into_video_in_place
from src.utils.ffmpeg import get_ffmpeg_path, run_ffmpeg


def _normalize_audio_output(audio_output: Any) -> Any:
    """Normalize MMAudio output to a tensor/array accepted by `_write_wav_wave`."""
    audio_tensor = audio_output
    if isinstance(audio_tensor, (list, tuple)) and len(audio_tensor) > 0:
        audio_tensor = audio_tensor[0]

    # Convert [S, C] to [C, S] when needed.
    if isinstance(audio_tensor, torch.Tensor):
        if (
            audio_tensor.ndim == 2
            and audio_tensor.shape[0] > 2
            and audio_tensor.shape[1] <= 2
        ):
            audio_tensor = audio_tensor.T
    else:
        arr = np.asarray(audio_tensor)
        if arr.ndim == 2 and arr.shape[0] > 2 and arr.shape[1] <= 2:
            audio_tensor = arr.T

    return audio_tensor


def save_mmaudio_output(
    *,
    audio_output: Any,
    job_dir: Path,
    filename_prefix: str = "result",
    input_video_path: str | None = None,
    logger=_default_logger,
    wav_sample_rate: int = 44100,
    mp3_bitrate: str = "192k",
) -> Tuple[str, str]:
    """
    Save MMAudio output as:
    - MP4 with muxed generated audio when `input_video_path` is provided.
    - MP3 when `input_video_path` is absent.
    """
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_tensor = _normalize_audio_output(audio_output)
    audio_wav_path = str(job_dir / f"{filename_prefix}_audio.wav")
    _write_wav_wave(audio_tensor, audio_wav_path, sample_rate=wav_sample_rate)

    if input_video_path:
        if not os.path.isfile(input_video_path):
            raise RuntimeError(
                "MMAudio received a `video` input, but the resolved path does not exist."
            )

        output_video_path = str(job_dir / f"{filename_prefix}.mp4")
        shutil.copyfile(input_video_path, output_video_path)

        muxed = mux_audio_files_into_video_in_place(
            video_path=output_video_path,
            audio_paths=[audio_wav_path],
            job_dir=job_dir,
            logger=logger,
            audio_codec="aac",
            audio_bitrate="192k",
            audio_sample_rate=wav_sample_rate,
            audio_channels=2,
        )
        if muxed:
            return muxed, "video"

        # Fallback mux path if in-place helper fails.
        fallback_path = job_dir / f"{filename_prefix}_fallback.mp4"
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-i",
            input_video_path,
            "-i",
            audio_wav_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(int(wav_sample_rate)),
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(fallback_path),
        ]
        rc, lp, _ = run_ffmpeg(
            cmd,
            log_path=job_dir / f"{filename_prefix}_mmaudio_mux.log",
        )
        if rc == 0 and fallback_path.is_file():
            fallback_path.replace(Path(output_video_path))
            return output_video_path, "video"

        raise RuntimeError(f"Failed to mux MMAudio output onto input video (log={lp})")

    mp3_path = str(job_dir / f"{filename_prefix}.mp3")
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-i",
        audio_wav_path,
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        str(mp3_bitrate),
        "-ar",
        str(int(wav_sample_rate)),
        "-ac",
        "2",
        mp3_path,
    ]
    rc, lp, _ = run_ffmpeg(
        cmd,
        log_path=job_dir / f"{filename_prefix}_mmaudio_mp3.log",
    )
    if rc != 0 or not os.path.isfile(mp3_path):
        raise RuntimeError(f"Failed to save MMAudio MP3 output (log={lp})")

    return mp3_path, "audio"

