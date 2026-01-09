from typing import Optional, Tuple
import numpy as np

def save_audio_video_output(
        video_numpy: np.ndarray,
        audio_numpy: Optional[np.ndarray] = None,
        filename_prefix: str = "result",
        sample_rate: int = 16000,
        resample_audio_rate: int = 24000,
        fps: int = 24,
        job_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Combine a sequence of video frames with an optional audio track and save as an MP4.
        Args:
            output_path (str): Path to the output MP4 file.
            video_numpy (np.ndarray): Numpy array of frames. Shape (C, F, H, W).
                                      Values can be in range [-1, 1] or [0, 255].
            audio_numpy (Optional[np.ndarray]): 1D or 2D numpy array of audio samples, range [-1, 1].
            sample_rate (int): Sample rate of the audio in Hz. Defaults to 16000.
            resample_audio_rate (int): The *input* audio sample rate in Hz. If it differs from
                                       `sample_rate`, audio is resampled to `sample_rate` before
                                       muxing. Defaults to 24000.
            fps (int): Frames per second for the video. Defaults to 24.
        Returns:
            str: Path to the saved MP4 file.
        """
        
        from moviepy.editor import ImageSequenceClip, AudioFileClip
        import soundfile as wavfile
        import tempfile
        import math

        def _resample_audio_numpy(
            audio: np.ndarray, src_rate: int, dst_rate: int
        ) -> np.ndarray:
            """
            Resample float audio from src_rate -> dst_rate.
            Supports 1D (n,) or 2D (n, ch) / (ch, n) arrays.
            """
            if src_rate == dst_rate:
                return audio
            if src_rate <= 0 or dst_rate <= 0:
                raise ValueError(
                    f"Invalid sample rates: src_rate={src_rate}, dst_rate={dst_rate}"
                )

            x = np.asarray(audio)
            if x.ndim == 1:
                x2 = x[:, None]
                channel_first = False
            elif x.ndim == 2:
                # Heuristic: treat the smaller dimension as channels when ambiguous.
                if x.shape[0] <= x.shape[1]:
                    x2 = x.T  # (n, ch)
                    channel_first = True
                else:
                    x2 = x  # (n, ch)
                    channel_first = False
            else:
                raise ValueError(f"audio_numpy must be 1D or 2D, got shape={x.shape}")

            x2 = x2.astype(np.float32, copy=False)

            # Prefer high-quality, fast polyphase resampling if SciPy is available.
            try:
                from scipy.signal import resample_poly  # type: ignore

                g = math.gcd(src_rate, dst_rate)
                up = dst_rate // g
                down = src_rate // g
                y2 = resample_poly(x2, up=up, down=down, axis=0).astype(
                    np.float32, copy=False
                )
            except Exception:
                # Fallback: linear interpolation (good enough for previews, avoids hard dependency).
                n_in = x2.shape[0]
                n_out = int(round(n_in * (dst_rate / float(src_rate))))
                if n_out <= 0:
                    return x2[:0]
                t_in = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float32)
                t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
                y2 = np.stack(
                    [np.interp(t_out, t_in, x2[:, c]).astype(np.float32) for c in range(x2.shape[1])],
                    axis=1,
                )

            # Restore original orientation.
            if x.ndim == 1:
                y = y2[:, 0]
            else:
                y = y2.T if channel_first else y2
            return np.clip(y, -1.0, 1.0)

        # Validate inputs
        assert isinstance(
            video_numpy, np.ndarray
        ), "video_numpy must be a numpy array"
        assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
        assert video_numpy.shape[0] in {
            1,
            3,
        }, "video_numpy must have 1 or 3 channels"
        if audio_numpy is not None:
            assert isinstance(
                audio_numpy, np.ndarray
            ), "audio_numpy must be a numpy array"
            assert (
                np.abs(audio_numpy).max() <= 1.0
            ), "audio_numpy values must be in range [-1, 1]"
            # If the provided audio is at a different rate than the muxing rate,
            # resample it before writing the temporary WAV.
            if resample_audio_rate != sample_rate:
                audio_numpy = _resample_audio_numpy(
                    audio_numpy, src_rate=resample_audio_rate, dst_rate=sample_rate
                )
        # Reorder dimensions: (C, F, H, W) → (F, H, W, C)
        video_numpy = video_numpy.transpose(1, 2, 3, 0)
        # Normalize frames if values are in [-1, 1]
        if video_numpy.max() <= 1.0:
            video_numpy = np.clip(video_numpy, -1, 1)
            video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
        else:
            video_numpy = video_numpy.astype(np.uint8)
        # Convert numpy array to a list of frames
        frames = list(video_numpy)
        # Create video clip
        clip = ImageSequenceClip(frames, fps=fps)
        # Add audio if provided
        if audio_numpy is not None:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", mode="wb", delete=False
            ) as temp_audio_file:
                wavfile.write(
                    temp_audio_file.name,
                    (audio_numpy * 32767).astype(np.int16),
                    sample_rate,
                )
                audio_clip = AudioFileClip(temp_audio_file.name)
                final_clip = clip.set_audio(audio_clip)
        else:
            final_clip = clip
        # Write final video to disk
        output_path = str(job_dir / f"{filename_prefix}.mp4")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            verbose=False,
            logger=None,
        )
        final_clip.close()
        return output_path, "video"