from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os

from loguru import logger as _default_logger


def mux_audio_files_into_video_in_place(
    *,
    video_path: str,
    audio_paths: List[str],
    job_dir: Path,
    logger=_default_logger,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    audio_sample_rate: int = 48000,
    audio_channels: int = 2,
) -> Optional[str]:
    """
    Best-effort: mux one or more audio files into `video_path` using ffmpeg.
    On success overwrites the original file and returns `video_path`.
    """
    from src.utils.ffmpeg import get_ffmpeg_path, run_ffmpeg

    try:
        valid_audio_paths = [
            p for p in audio_paths if isinstance(p, str) and os.path.isfile(p)
        ]
        if not (isinstance(video_path, str) and os.path.isfile(video_path)):
            return None
        if not valid_audio_paths:
            return None

        base = Path(video_path)
        temp_out_path = base.with_name(f"{base.stem}_with_audio{base.suffix}")
        log_path = job_dir / f"{base.stem}_mux_audio.log"

        cmd: List[str] = [get_ffmpeg_path(), "-y", "-i", video_path]
        for ap in valid_audio_paths:
            cmd.extend(["-i", ap])

        if len(valid_audio_paths) == 1:
            cmd.extend(
                [
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    audio_codec,
                    "-b:a",
                    str(audio_bitrate),
                    "-ar",
                    str(int(audio_sample_rate)),
                    "-ac",
                    str(int(audio_channels)),
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(temp_out_path),
                ]
            )
        else:
            inputs_count = len(valid_audio_paths)
            filter_inputs = "".join(f"[{i+1}:a]" for i in range(inputs_count))
            filter_spec = (
                f"{filter_inputs}amix=inputs={inputs_count}:dropout_transition=0[aout]"
            )
            cmd.extend(
                [
                    "-filter_complex",
                    filter_spec,
                    "-map",
                    "0:v:0",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    audio_codec,
                    "-b:a",
                    str(audio_bitrate),
                    "-ar",
                    str(int(audio_sample_rate)),
                    "-ac",
                    str(int(audio_channels)),
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(temp_out_path),
                ]
            )

        rc, lp, _ = run_ffmpeg(cmd, log_path=log_path)
        if rc != 0 or not temp_out_path.is_file():
            logger.warning(f"ffmpeg audio mux failed (code={rc}) (log={lp})")
            return None

        try:
            temp_out_path.replace(base)
        except Exception as move_err:
            logger.warning(
                f"ffmpeg audio mux succeeded but failed to move into place: {move_err}"
            )
            return None
        return str(base)
    except Exception as e:
        logger.warning(f"Failed to mux audio into video: {e}")
        return None


def mux_audio_from_source_video_in_place(
    *,
    video_path: str,
    source_video_path: str,
    job_dir: Path,
    logger=_default_logger,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    audio_sample_rate: int = 48000,
    audio_channels: int = 2,
) -> Optional[str]:
    """
    Best-effort: mux the first audio track from `source_video_path` into `video_path`.
    On success overwrites the original file and returns `video_path`.
    """
    from src.utils.ffmpeg import get_ffmpeg_path, run_ffmpeg

    try:
        if not (
            isinstance(video_path, str)
            and isinstance(source_video_path, str)
            and os.path.isfile(video_path)
            and os.path.isfile(source_video_path)
        ):
            return None

        base = Path(video_path)
        temp_out_path = base.with_name(f"{base.stem}_with_audio{base.suffix}")
        log_path = job_dir / f"{base.stem}_mux_audio_from_source.log"

        cmd: List[str] = [
            get_ffmpeg_path(),
            "-y",
            "-i",
            video_path,
            "-i",
            source_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            "-b:a",
            str(audio_bitrate),
            "-ar",
            str(int(audio_sample_rate)),
            "-ac",
            str(int(audio_channels)),
            "-shortest",
            "-movflags",
            "+faststart",
            str(temp_out_path),
        ]
        rc, lp, _ = run_ffmpeg(cmd, log_path=log_path)
        if rc != 0 or not temp_out_path.is_file():
            return None
        try:
            temp_out_path.replace(base)
        except Exception as move_err:
            logger.warning(
                f"ffmpeg audio mux succeeded but failed to move into place: {move_err}"
            )
            return None
        return str(base)
    except Exception as e:
        logger.warning(f"Failed to mux audio from source video: {e}")
        return None


def mux_audio_from_source_video_to_output(
    *,
    video_only_path: str,
    source_video_path: str,
    output_path: str,
    job_dir: Path,
    logger=_default_logger,
    audio_codec: str = "copy",
    video_codec: str = "copy",
) -> bool:
    """
    Mux audio from `source_video_path` onto `video_only_path` and write to `output_path`.
    Returns True on success.
    """
    from src.utils.ffmpeg import get_ffmpeg_path, run_ffmpeg

    try:
        if not (
            isinstance(video_only_path, str)
            and isinstance(source_video_path, str)
            and os.path.isfile(video_only_path)
            and os.path.isfile(source_video_path)
        ):
            return False

        log_path = job_dir / "ffmpeg_mux_audio.log"
        cmd: List[str] = [
            get_ffmpeg_path(),
            "-y",
            "-i",
            video_only_path,
            "-i",
            source_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            video_codec,
            "-c:a",
            audio_codec,
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]
        rc, lp, _ = run_ffmpeg(cmd, log_path=log_path)
        if rc != 0 or not Path(output_path).is_file():
            logger.warning(f"ffmpeg mux audio failed (code={rc}) (log={lp})")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to mux audio: {e}")
        return False

