import torchaudio
from src.types import InputAudio
import torch
from typing import TYPE_CHECKING
import torch
import torchaudio
from torch import nn

class AudioProcessor(nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling."""

    def __init__(
        self,
        sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_waveform(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate if needed."""
        if source_rate == target_rate:
            return waveform
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.to(device=waveform.device, dtype=waveform.dtype)

    def waveform_to_mel(
        self,
        waveform: torch.Tensor,
        waveform_sample_rate: int,
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram [batch, channels, time, n_mels]."""
        waveform = self.resample_waveform(waveform, waveform_sample_rate, self.sample_rate)

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


if TYPE_CHECKING:
    from src.engine.base_engine import BaseEngine
    base_class = BaseEngine
else:
    base_class = object

class LTX2AudioProcessingMixin(base_class):
    """Mixin for audio processing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_sampling_rate = self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        self.mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        self.mel_hop_length = self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        self.n_fft = 1024
        self.audio_processor = AudioProcessor(
            sample_rate=self.audio_sampling_rate,
            mel_bins=self.mel_bins,
            mel_hop_length=self.mel_hop_length,
            n_fft=self.n_fft,
        )

    def encode_audio_latents_grid_(
        self,
        audio: InputAudio,
        generator: torch.Generator = None,
        offload: bool = True,
    ) -> torch.Tensor:
        """
        Encode an audio input into *grid* latents of shape [1, C, L, M] (unpacked).

        This is intentionally separated from token packing so callers (e.g. conditioning injection)
        can slice/inject in grid space first, then pack.
        """
        audio_array = self._load_audio(audio, sample_rate=self.audio_sampling_rate)
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).unsqueeze(0)

        audio_config = self.load_config_by_name("audio_vae")
        target_channels = audio_config.in_channels if getattr(audio_config, "in_channels", None) is not None else 2
        if audio_tensor.shape[1] != target_channels:
            if audio_tensor.shape[1] == 1 and target_channels > 1:
                audio_tensor = audio_tensor.repeat(1, target_channels, 1)
            elif target_channels == 1:
                audio_tensor = audio_tensor.mean(dim=1, keepdim=True)
            else:
                audio_tensor = audio_tensor[:, :target_channels, :]
                if audio_tensor.shape[1] < target_channels:
                    pad_channels = target_channels - audio_tensor.shape[1]
                    pad = torch.zeros(
                        (audio_tensor.shape[0], pad_channels, audio_tensor.shape[2]),
                        dtype=audio_tensor.dtype,
                        device=audio_tensor.device,
                    )
                    audio_tensor = torch.cat([audio_tensor, pad], dim=1)

        mel = self.audio_processor.waveform_to_mel(audio_tensor, self.audio_sampling_rate)

        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
        self.to_device(self.audio_vae)
        mel = mel.to(dtype=self.audio_vae.dtype, device=self.audio_vae.device)

        posterior = self.audio_vae.encode(mel, return_dict=False)[0]
        audio_latents_grid = posterior.sample(generator=generator) if generator is not None else posterior.mode()

        if offload:
            self._offload("audio_vae")

        return audio_latents_grid

    def prepare_audio_latents_(
        self,
        audio: InputAudio,
        generator: torch.Generator = None,
        latent_length: int = 121,
        offload: bool = True,
    ) -> torch.Tensor:
        """Prepare packed audio latents (legacy helper)."""
        audio_latents = self.encode_audio_latents_grid_(audio=audio, generator=generator, offload=offload)

        if latent_length is not None:
            if audio_latents.shape[2] < latent_length:
                pad_length = latent_length - audio_latents.shape[2]
                pad = torch.zeros(
                    audio_latents.shape[0],
                    audio_latents.shape[1],
                    pad_length,
                    audio_latents.shape[3],
                    dtype=audio_latents.dtype,
                    device=audio_latents.device,
                )
                audio_latents = torch.cat([audio_latents, pad], dim=2)
            elif audio_latents.shape[2] > latent_length:
                audio_latents = audio_latents[:, :, :latent_length]

        audio_latents = self._pack_audio_latents(audio_latents)
        audio_latents = self.audio_vae.normalize_latents(audio_latents)
        return audio_latents