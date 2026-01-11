from typing import Optional, Tuple
import numpy as np
import math
from collections.abc import Generator, Iterator
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch._prims_common import DeviceLikeType
from tqdm import tqdm
from loguru import logger

DEFAULT_IMAGE_CRF = 33

def save_video_ovi(
        video_numpy: np.ndarray,
        audio_numpy: Optional[np.ndarray] = None,
        filename_prefix: str = "result",
        sample_rate: int = 16000,
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
        import os


        def _audio_to_soundfile_frames(audio: np.ndarray) -> np.ndarray:
            """
            Normalize audio array to the shape expected by soundfile: (n_frames,) or (n_frames, n_channels).
            The engine may produce (channels, n_frames); we transpose that to (n_frames, channels).
            """
            x = np.asarray(audio)
            if x.ndim == 1:
                return np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)
            if x.ndim != 2:
                raise ValueError(f"audio_numpy must be 1D or 2D, got shape={x.shape}")

            # Heuristic: treat the smaller dimension as channels when ambiguous.
            if x.shape[0] <= x.shape[1]:
                x = x.T  # (n_frames, n_channels)
            return np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)

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
        audio_path: Optional[str] = None
        audio_clip = None
        # Add audio if provided
        if audio_numpy is not None:
            try:
                # Create a path, close it, then let soundfile write it (avoid writing to an already-open handle).
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                    audio_path = temp_audio_file.name

                audio_frames = _audio_to_soundfile_frames(audio_numpy)
                wavfile.write(
                    audio_path,
                    audio_frames,
                    sample_rate,
                    format="WAV",
                    subtype="PCM_16",
                )

                audio_clip = AudioFileClip(audio_path)
                final_clip = clip.set_audio(audio_clip)
            finally:
                # MoviePy will hold the file open until the clip is closed; we clean up at the end.
                # (See below where we close clips.)
                pass
        else:
            final_clip = clip
        # Write final video to disk
        if job_dir is not None:
            output_path = str(job_dir / f"{filename_prefix}.mp4")
        else:
            output_path = f"{filename_prefix}.mp4"
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            verbose=False,
            logger=None,
        )
        # Ensure resources are released (especially the temp audio file on some platforms).
        try:
            final_clip.close()
        finally:
            try:
                if audio_clip is not None:
                    audio_clip.close()
            finally:
                if audio_numpy is not None and audio_path is not None:
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass
        return output_path, "video"
    
    
    




def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image tensor with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Tensor with shape (F (optional), H, W, C) F = 1 if input is 3D, otherwise input shape[0]
    """
    height, width = image.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    # rearrange and remove batch dimension
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    # preserve input dimensions
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Resize tensor preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        latent: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D input or (1, C, F, height, width) for 4D input
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape

    scale = max(height / src_h, width / src_w)
    # Use ceil to avoid floating-point rounding causing new_h/new_w to be
    # slightly smaller than target, which would result in negative crop offsets.
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def normalize_latent(latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (latent / 127.5 - 1.0).to(device=device, dtype=dtype)


def load_image_conditioning(
    image_path: str, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    image = decode_image(image_path=image_path)
    image = preprocess(image=image)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    image = normalize_latent(image, device, dtype)
    return image


def load_video_conditioning(
    video_path: str, height: int, width: int, frame_cap: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Loads a video from a path and preprocesses it for conditioning.
    Note: The video is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    frames = decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device)
    result = None
    for f in frames:
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_latent(frame, device, dtype)
        result = frame if result is None else torch.cat([result, frame], dim=2)
    return result


def decode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    np_array = np.array(image)[..., :3]
    return np_array


def _write_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, samples: torch.Tensor, audio_sample_rate: int
) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def save_video_ltx2(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: torch.Tensor | None,
    filename_prefix: str,
    sample_rate: int | None = 24000,
    fps: int = 25,
    video_chunks_number: int = 1,
    job_dir: Optional[str] = None,
) -> None:
    if job_dir is not None:
        output_path = str(job_dir / f"{filename_prefix}.mp4")
    else:
        output_path = f"{filename_prefix}.mp4"
    if isinstance(video, torch.Tensor):
        video = iter([video])

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, sample_rate)

    def all_tiles(
        first_chunk: torch.Tensor, tiles_generator: Generator[tuple[torch.Tensor, int], None, None]
    ) -> Generator[tuple[torch.Tensor, int], None, None]:
        yield first_chunk
        yield from tiles_generator

    for video_chunk in tqdm(all_tiles(first_chunk, video), total=video_chunks_number):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, sample_rate)

    container.close()
    return output_path, "video"

def decode_audio_from_file(path: str, device: torch.device) -> torch.Tensor | None:
    container = av.open(path)
    try:
        audio = []
        audio_stream = next(s for s in container.streams if s.type == "audio")
        for frame in container.decode(audio_stream):
            audio.append(torch.tensor(frame.to_ndarray(), dtype=torch.float32, device=device).unsqueeze(0))
        container.close()
        audio = torch.cat(audio)
    except StopIteration:
        audio = None
    finally:
        container.close()

    return audio


def decode_video_from_file(path: str, frame_cap: int, device: DeviceLikeType) -> Generator[torch.Tensor]:
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            frame_cap = frame_cap - 1
            if frame_cap == 0:
                break
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: float = DEFAULT_IMAGE_CRF) -> np.array:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array