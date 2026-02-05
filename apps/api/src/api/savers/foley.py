from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger as _default_logger

from src.api.savers.ffmpeg_mux import mux_audio_files_into_video_in_place
from src.utils.save_audio_video import _write_wav_wave


def mux_foley_audio_onto_input_video(
    *,
    input_video_path: str,
    audio_output: Any,
    job_dir: Path,
    output_video_path: str,
    logger=_default_logger,
    wav_sample_rate: int = 48000,
) -> str:
    """
    HunyuanVideo-Foley returns an audio waveform. This helper:
    - writes it to WAV inside job_dir
    - copies the original input video into output_video_path
    - muxes the WAV into output_video_path (best-effort)

    Returns the final video path (output_video_path). If muxing fails, returns a
    video-only copy (still at output_video_path when possible).
    """
    import os
    import shutil
    import numpy as np
    import torch

    if not (isinstance(input_video_path, str) and os.path.isfile(input_video_path)):
        raise RuntimeError("input_video_path must be an existing file")

    job_dir.mkdir(parents=True, exist_ok=True)

    # Normalize engine output: `audio.unbind(0)` -> take first item.
    audio_tensor = audio_output
    try:
        if isinstance(audio_tensor, (list, tuple)) and len(audio_tensor) > 0:
            audio_tensor = audio_tensor[0]
    except Exception:
        pass

    # Normalize layout to what `_write_wav_wave` expects: [C, S] or [S].
    try:
        if isinstance(audio_tensor, torch.Tensor):
            if audio_tensor.ndim == 2 and audio_tensor.shape[0] > 2 and audio_tensor.shape[1] <= 2:
                audio_tensor = audio_tensor.T
        else:
            arr = np.asarray(audio_tensor)
            if arr.ndim == 2 and arr.shape[0] > 2 and arr.shape[1] <= 2:
                audio_tensor = arr.T
    except Exception:
        pass

    audio_wav_path = str(job_dir / "result_audio.wav")
    _write_wav_wave(audio_tensor, audio_wav_path, sample_rate=wav_sample_rate)

    # Copy input video to output location so we never mutate the original.
    try:
        shutil.copyfile(input_video_path, output_video_path)
    except Exception as copy_err:
        logger.warning(f"Failed to copy input video for foley mux: {copy_err}")
        # If we can't even copy, fall back to returning original (best-effort).
        return input_video_path

    # Now mux WAV into the copied video in-place.
    muxed: Optional[str] = mux_audio_files_into_video_in_place(
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
        return muxed

    logger.warning("Foley audio mux failed; returning video-only output.")
    return output_video_path

