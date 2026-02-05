import torch

from src.utils.audio_postprocess import postprocess_audio_waveform


def test_postprocess_audio_waveform_dc_removal_and_limits():
    sr = 24000
    t = torch.arange(0, sr, dtype=torch.float32) / float(sr)

    # Stereo signal with DC offset and a low-frequency component.
    left = 0.25 * torch.sin(2.0 * torch.pi * 220.0 * t) + 0.2  # DC offset
    right = 0.25 * torch.sin(2.0 * torch.pi * 220.0 * t + 0.3) - 0.15  # DC offset
    audio = torch.stack([left, right], dim=0)

    out = postprocess_audio_waveform(audio, sample_rate=sr)

    assert out.shape == audio.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()

    # Means should be ~0 after DC removal (allow small residual due to filtering).
    means = out.mean(dim=-1)
    assert torch.all(torch.abs(means) < 5e-3)

    # Output should be bounded to [-1, 1].
    assert float(out.abs().max().item()) <= 1.0 + 1e-6

