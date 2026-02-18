from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch

try:
    import torchaudio.functional as _taF  # type: ignore
except Exception:  # pragma: no cover
    _taF = None


@dataclass(frozen=True)
class AudioPostprocessConfig:
    """
    Lightweight audio cleanup for model-generated waveforms.

    This intentionally does NOT implement true LUFS (gated K-weighted) measurement; instead we do
    pragmatic cleanup that avoids common artifacts:
    - DC removal
    - High-pass filtering (remove rumble / very low frequency bias)
    - RMS-based loudness normalization (dBFS target) with boost limits
    - Peak limiting (avoid clipping)
    - Optional soft clip for extra headroom
    """

    # Remove DC offset per-channel (mean over time).
    dc_remove: bool = True

    # High-pass filter cutoff in Hz (0 disables).
    highpass_hz: float = 30.0

    # RMS loudness target in dBFS (approximate). Common values: -20 .. -14.
    target_rms_dbfs: float = -16.0

    # Max amplification we allow during normalization (to avoid boosting noise).
    max_boost_db: float = 9.0

    # If signal is quieter than this, reduce allowed boost further.
    low_level_rms_dbfs: float = -45.0
    low_level_max_boost_db: float = 3.0

    # Peak limiting (linear). 0.98 ~ -0.17 dBFS.
    peak_limit: float = 0.98

    # Optional soft clip (tanh) to shave peaks gently.
    soft_clip: bool = True
    soft_clip_drive: float = 1.5


def _to_channels_first_2d(audio: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Normalize audio tensor to shape [C, T] for processing.

    Accepts shapes:
    - [T]
    - [C, T]
    - [B, C, T]
    """
    if audio.ndim == 1:
        return audio.unsqueeze(0), (1,)
    if audio.ndim == 2:
        return audio, (audio.shape[0],)
    if audio.ndim == 3:
        b, c, t = audio.shape
        return audio.reshape(b * c, t), (b, c)
    raise ValueError(
        f"Unsupported audio tensor shape {tuple(audio.shape)}; expected 1D/2D/3D."
    )


def _restore_shape(audio_2d: torch.Tensor, restore_shape: tuple[int, ...]) -> torch.Tensor:
    if restore_shape == (1,):
        return audio_2d.squeeze(0)
    if len(restore_shape) == 1:
        return audio_2d
    if len(restore_shape) == 2:
        b, c = restore_shape
        t = audio_2d.shape[-1]
        return audio_2d.reshape(b, c, t)
    raise ValueError(f"Invalid restore_shape={restore_shape}")


def _highpass_fallback_iir(
    audio_2d: torch.Tensor, *, sample_rate: int, cutoff_hz: float
) -> torch.Tensor:
    """
    Simple first-order high-pass as a fallback when torchaudio isn't available.

    y[n] = x[n] - x[n-1] + a * y[n-1], where a = exp(-2*pi*fc/sr)
    """
    if cutoff_hz <= 0:
        return audio_2d
    a = float(math.exp(-2.0 * math.pi * float(cutoff_hz) / float(sample_rate)))
    x = audio_2d
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0]
    for i in range(1, x.shape[1]):
        y[:, i] = x[:, i] - x[:, i - 1] + a * y[:, i - 1]
    return y


@torch.no_grad()
def postprocess_audio_waveform(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    config: Optional[AudioPostprocessConfig] = None,
) -> torch.Tensor:
    """
    Apply DC removal, high-pass, and loudness/peak normalization to a waveform tensor.

    Args:
        audio: Tensor shaped [T], [C, T], or [B, C, T], expected roughly in [-1, 1].
        sample_rate: Audio sample rate in Hz.
        config: Optional configuration.
    """
    cfg = config or AudioPostprocessConfig()
    if audio.numel() == 0:
        return audio

    orig_device = audio.device
    x = audio.to(dtype=torch.float32)
    x2d, restore_shape = _to_channels_first_2d(x)

    # 1) DC removal (per-channel)
    if cfg.dc_remove:
        x2d = x2d - x2d.mean(dim=-1, keepdim=True)

    # 2) High-pass filter
    if cfg.highpass_hz and cfg.highpass_hz > 0.0:
        if _taF is not None:
            try:
                x2d = _taF.highpass_biquad(
                    x2d, sample_rate=int(sample_rate), cutoff_freq=float(cfg.highpass_hz)
                )
            except Exception:
                x2d = _highpass_fallback_iir(
                    x2d, sample_rate=int(sample_rate), cutoff_hz=float(cfg.highpass_hz)
                )
        else:  # pragma: no cover
            x2d = _highpass_fallback_iir(
                x2d, sample_rate=int(sample_rate), cutoff_hz=float(cfg.highpass_hz)
            )

    # 3) RMS loudness normalization (approx. dBFS) with boost limits
    eps = 1e-8
    rms = torch.sqrt(torch.mean(x2d * x2d) + eps)
    rms_dbfs = 20.0 * torch.log10(rms + eps)

    max_boost_db = (
        float(cfg.low_level_max_boost_db)
        if float(rms_dbfs.item()) < float(cfg.low_level_rms_dbfs)
        else float(cfg.max_boost_db)
    )
    gain_db = float(cfg.target_rms_dbfs) - float(rms_dbfs.item())
    gain_db = max(-60.0, min(max_boost_db, gain_db))
    gain = 10.0 ** (gain_db / 20.0)
    x2d = x2d * float(gain)

    # 4) Peak limiting (hard)
    peak = torch.amax(torch.abs(x2d)).clamp_min(eps)
    if float(peak.item()) > float(cfg.peak_limit):
        x2d = x2d * (float(cfg.peak_limit) / float(peak.item()))

    # 5) Optional soft clip for extra headroom
    if cfg.soft_clip and cfg.soft_clip_drive and cfg.soft_clip_drive > 0.0:
        drive = float(cfg.soft_clip_drive)
        x2d = torch.tanh(x2d * drive) / float(math.tanh(drive))

    x2d = torch.clamp(x2d, -1.0, 1.0)
    out = _restore_shape(x2d, restore_shape)
    return out.to(device=orig_device, dtype=torch.float32)

