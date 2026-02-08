from __future__ import annotations

"""
Audio/video saving helpers used by API workers.

Why this exists:
- `src/api/ray_tasks.py` needs to save outputs for engines that return
  (video_frames, audio_waveform) tuples (e.g. OVI, LTX2 TI2V).
- The old implementation lived in `src/utils/save_audio_video.py` and used
  MoviePy + SoundFile for OVI, which is not always available in bundled/API
  environments. This module provides a consistent PyAV-based implementation.
"""

from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union, Any, List
from diffusers.utils.export_utils import export_to_video
import av
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from src.utils.ffmpeg import get_ffmpeg_path
import subprocess
import shutil
import tempfile
import os
import wave

JobDir = Union[str, Path]


def _as_path(job_dir: JobDir | None) -> Path | None:
    if job_dir is None:
        return None
    return job_dir if isinstance(job_dir, Path) else Path(str(job_dir))


def _to_uint8_rgb_frames_from_cf_hw(
    video_numpy: np.ndarray,
) -> np.ndarray:
    """
    Convert (C, F, H, W) -> (F, H, W, 3) uint8 RGB.
    Accepts values in [-1, 1] or [0, 255].
    """
    if not isinstance(video_numpy, np.ndarray):
        video_numpy = np.asarray(video_numpy)
    if video_numpy.ndim != 4:
        raise ValueError(f"Expected video with 4 dims (C,F,H,W); got shape={video_numpy.shape}")
    if video_numpy.shape[0] not in (1, 3):
        raise ValueError(f"Expected C in (1,3); got C={video_numpy.shape[0]}")

    # (C, F, H, W) -> (F, H, W, C)
    vf = np.transpose(video_numpy, (1, 2, 3, 0))

    # Normalize to uint8.
    vmax = float(np.max(vf)) if vf.size else 0.0
    if vmax <= 1.0:
        vf = np.clip(vf, -1.0, 1.0)
        vf = ((vf + 1.0) / 2.0 * 255.0).astype(np.uint8)
    else:
        vf = np.clip(vf, 0.0, 255.0).astype(np.uint8)

    # Expand grayscale to RGB if needed.
    if vf.shape[-1] == 1:
        vf = np.repeat(vf, 3, axis=-1)

    return vf


def _to_uint8_rgb_frames_any(video: Any) -> Iterator[np.ndarray]:
    """
    Yield frames as (H, W, 3) uint8 RGB.

    Supports:
    - np.ndarray shaped (F,H,W,C) or (C,F,H,W)
    - torch.Tensor shaped (F,H,W,C) or (C,F,H,W) or (1,F,H,W,C)
    - iterator yielding torch.Tensor / np.ndarray chunks shaped (F,H,W,C)
    """

    def _yield_from_fhwc(arr: np.ndarray) -> Iterator[np.ndarray]:
        if arr.ndim != 4:
            raise ValueError(f"Expected frames (F,H,W,C); got shape={arr.shape}")
        if arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected channel-last C in (1,3,4); got shape={arr.shape}")
        a = arr
        if a.dtype != np.uint8:
            vmax = float(np.max(a)) if a.size else 0.0
            if vmax <= 1.0:
                a = np.clip(a, -1.0, 1.0)
                a = ((a + 1.0) / 2.0 * 255.0).astype(np.uint8)
            else:
                a = np.clip(a, 0.0, 255.0).astype(np.uint8)
        if a.shape[-1] == 4:
            a = a[..., :3]
        if a.shape[-1] == 1:
            a = np.repeat(a, 3, axis=-1)
        for f in a:
            yield f

    def _coerce_chunk(x: Any) -> np.ndarray:
        if isinstance(x, tuple) and x:
            # Some pipelines may yield (tensor, index) style tuples.
            x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.detach().to("cpu")
            return x.numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    # Iterator/generator path
    if hasattr(video, "__iter__") and not isinstance(video, (np.ndarray, torch.Tensor, bytes, str)):
        it = iter(video)
        for chunk in it:
            arr = _coerce_chunk(chunk)
            # If chunk looks like (C,F,H,W), convert first.
            if arr.ndim == 4 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
                arr = _to_uint8_rgb_frames_from_cf_hw(arr)
            yield from _yield_from_fhwc(arr)
        return

    # Array/tensor path
    if isinstance(video, torch.Tensor):
        v = video.detach()
        if v.ndim == 5 and v.shape[0] == 1:
            v = v[0]
        video = v.to("cpu").numpy()

    if not isinstance(video, np.ndarray):
        video = np.asarray(video)

    if video.ndim != 4:
        raise ValueError(f"Unsupported video shape={video.shape}")

    # (C,F,H,W)
    if video.shape[0] in (1, 3) and video.shape[-1] not in (1, 3, 4):
        frames = _to_uint8_rgb_frames_from_cf_hw(video)
        yield from _yield_from_fhwc(frames)
        return

    # (F,H,W,C)
    yield from _yield_from_fhwc(video)


def _coerce_audio_to_stereo_float32(audio: Any) -> np.ndarray:
    """
    Return audio as float32 ndarray shaped (n_samples, 2) in [-1, 1].
    Accepts 1D mono, (n_samples, n_channels), or (n_channels, n_samples).
    """
    if audio is None:
        raise ValueError("audio is None")

    if isinstance(audio, torch.Tensor):
        a = audio.detach().to("cpu").numpy()
    else:
        a = np.asarray(audio)

    if a.ndim == 1:
        a = a[:, None]  # (n_samples, 1)
    elif a.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D; got shape={a.shape}")

    # If it's (channels, samples), transpose to (samples, channels)
    if a.shape[0] in (1, 2) and a.shape[1] > a.shape[0]:
        # ambiguous case: could already be (samples, channels) with samples=1/2, but that’s unlikely.
        # treat small first dimension as channels.
        a = a.T

    # Ensure 2 channels
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    elif a.shape[1] != 2:
        # Best-effort: keep first 2 channels
        a = a[:, :2]

    # Convert dtype/range
    if np.issubdtype(a.dtype, np.integer):
        # Assume int16-like PCM
        maxv = float(np.iinfo(a.dtype).max)
        a = (a.astype(np.float32) / maxv).clip(-1.0, 1.0)
    else:
        a = a.astype(np.float32, copy=False)
        a = np.clip(a, -1.0, 1.0)

    return a


def _prepare_audio_stream(
    container: av.container.Container, audio_sample_rate: int
) -> av.audio.AudioStream:
    audio_stream = container.add_stream("aac", rate=int(audio_sample_rate))
    audio_stream.codec_context.sample_rate = int(audio_sample_rate)
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, int(audio_sample_rate))
    return audio_stream


def _resample_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    frame_in: av.AudioFrame,
) -> None:
    cc = audio_stream.codec_context
    target_format = cc.format or "fltp"  # AAC typically uses fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        rframe.pts = audio_next_pts
        rframe.time_base = Fraction(1, int(target_rate))
        audio_next_pts += rframe.samples
        rframe.sample_rate = int(target_rate)
        for packet in audio_stream.encode(rframe):
            container.mux(packet)

    for packet in audio_stream.encode():
        container.mux(packet)


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    samples_stereo: np.ndarray,
    audio_sample_rate: int,
) -> None:
    """
    samples_stereo: float32 ndarray shaped (n_samples, 2) in [-1, 1]
    """
    if samples_stereo.ndim != 2 or samples_stereo.shape[1] != 2:
        raise ValueError(f"Expected (n_samples,2) audio; got shape={samples_stereo.shape}")

    # Convert to int16 PCM
    pcm = np.clip(samples_stereo, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16, copy=False)  # (n_samples, 2)

    # Planar s16p expects (n_channels, n_samples)
    planar = np.ascontiguousarray(pcm.T)
    frame_in = av.AudioFrame.from_ndarray(planar, format="s16p", layout="stereo")
    frame_in.sample_rate = int(audio_sample_rate)
    frame_in.time_base = Fraction(1, int(audio_sample_rate))

    _resample_audio(container, audio_stream, frame_in)


def save_video_ltx2(
    video: Union[torch.Tensor, np.ndarray, Iterator[Any]],
    audio: Union[torch.Tensor, np.ndarray, None],
    filename_prefix: str,
    sample_rate: int | None = 24000,
    fps: int = 25,
    video_chunks_number: int = 1,
    job_dir: JobDir | None = None,
) -> Tuple[str, str]:
    """
    Save an MP4 from an LTX2-style output, optionally muxing audio.

    Compatible with:
    - video as torch tensor / numpy array / iterator yielding chunks
    - audio as torch tensor / numpy array
    """
    out_dir = _as_path(job_dir)
    output_path = str((out_dir / f"{filename_prefix}.mp4") if out_dir else Path(f"{filename_prefix}.mp4"))

    frames_iter = _to_uint8_rgb_frames_any(video)
    # Consume first frame to get dimensions (and to validate early).
    try:
        first_frame = next(frames_iter)
    except StopIteration:
        raise ValueError("Video had zero frames")

    if first_frame.ndim != 3 or first_frame.shape[-1] != 3:
        raise ValueError(f"Expected frame (H,W,3); got shape={first_frame.shape}")

    height, width, _ = first_frame.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream(
        "libx264",
        rate=int(fps),
        options={"preset": "veryfast", "tune": "zerolatency"},
    )
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    try:
        stream.codec_context.max_b_frames = 0
    except Exception:
        pass

    audio_stream = None
    if audio is not None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is provided")
        audio_stream = _prepare_audio_stream(container, int(sample_rate))
        audio_stereo = _coerce_audio_to_stereo_float32(audio)
    else:
        audio_stereo = None

    def _all_frames() -> Iterator[np.ndarray]:
        yield first_frame
        yield from frames_iter

    for frame_array in tqdm(_all_frames(), total=None):
        vf = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    if audio_stream is not None and audio_stereo is not None:
        _write_audio(container, audio_stream, audio_stereo, int(sample_rate))

    container.close()
    return output_path, "video"


def save_video_ovi(
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    filename_prefix: str = "result",
    sample_rate: int = 16000,
    fps: int = 24,
    job_dir: JobDir | None = None,
) -> Tuple[str, str]:
    """
    Save an MP4 from an OVI-style output:
    - video_numpy: (C, F, H, W) in [-1,1] or [0,255]
    - audio_numpy: 1D/2D waveform in [-1,1] (mono or stereo supported)

    This is intentionally implemented with PyAV (no MoviePy/SoundFile dependency).
    """
    out_dir = _as_path(job_dir)
    output_path = str((out_dir / f"{filename_prefix}.mp4") if out_dir else Path(f"{filename_prefix}.mp4"))

    frames = _to_uint8_rgb_frames_from_cf_hw(video_numpy)  # (F,H,W,3) uint8
    if frames.shape[0] == 0:
        raise ValueError("OVI video had zero frames")
    height, width, _ = frames[0].shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream(
        "libx264",
        rate=int(fps),
        options={"preset": "veryfast", "tune": "zerolatency"},
    )
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    try:
        stream.codec_context.max_b_frames = 0
    except Exception:
        pass

    audio_stream = None
    audio_stereo = None
    if audio_numpy is not None:
        audio_stream = _prepare_audio_stream(container, int(sample_rate))
        audio_stereo = _coerce_audio_to_stereo_float32(audio_numpy)

    for frame_array in frames:
        vf = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    if audio_stream is not None and audio_stereo is not None:
        _write_audio(container, audio_stream, audio_stereo, int(sample_rate))

    container.close()
    return output_path, "video"




def _write_wav_wave(audio, wav_path, sample_rate=44100):
    """
    Write int16 PCM WAV using standard library wave.
    - audio: torch.Tensor or np.ndarray, shape [samples] / [channels, samples]
      - If float, assumed range is approximately [-1, 1], will be converted to int16 PCM
    """
    if isinstance(audio, torch.Tensor):
        a = audio.detach().cpu().numpy()
    else:
        a = np.asarray(audio)

    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"audio shape needs to be [S] / [C,S], current shape is {a.shape}")

    channels, samples = int(a.shape[0]), int(a.shape[1])
    if channels > 2:
        a = a[:2, :]
        channels = 2

    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, -1.0, 1.0)
        a = (a * 32767.0).astype(np.int16)
    elif a.dtype != np.int16:
        a = np.clip(a, -32768, 32767).astype(np.int16)

    if channels == 1:
        interleaved = a.reshape(-1)
    else:
        interleaved = a.T.reshape(-1)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(interleaved.tobytes(order="C"))


def save_video_mova(
    video: List[Image.Image] | List[np.ndarray], 
    audio: torch.Tensor | None, 
    filename_prefix: str, 
    sample_rate: int = 44100, 
    fps: int = 24, 
    quality: int = 9, 
    job_dir: Optional[str] = None
) -> None:
    """
    Save video with audio.
    - video: List[PIL.Image | np.ndarray]
    - audio: torch.Tensor or None
    - filename_prefix: Output mp4 file name prefix
    - sample_rate: Audio sample rate (default 44100)
    - fps: Video frame rate (default 24)
    - quality: Video quality (default 9)
    - job_dir: Output directory (default None)
    """

    ffmpeg_path = get_ffmpeg_path()
        
    if job_dir is not None:
        job_dir = Path(job_dir)
        save_path = str(job_dir / f"{filename_prefix}.mp4")
    else:
        save_path = f"{filename_prefix}.mp4"

    with tempfile.TemporaryDirectory(prefix='save_vwa_') as tmp_dir:
        tmp_video = os.path.join(tmp_dir, 'video.mp4')
        tmp_audio = os.path.join(tmp_dir, 'audio.wav')
        if isinstance(video[0], list):
            video = video[0]
        export_to_video(video, tmp_video, fps=fps, quality=quality)
        _write_wav_wave(audio, tmp_audio, sample_rate=sample_rate)

        cmd = [
            ffmpeg_path, '-y',
            '-i', tmp_video,
            '-i', tmp_audio,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            '-shortest',
            save_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # Fallback to save only video
            print(f"[save_video_with_audio] ffmpeg reuse failed, saving only video. Error: {e.stderr.decode(errors='ignore')[:500]}")
            # Copy temporary video to target
            shutil.copyfile(tmp_video, save_path)
            
    return save_path, "video"