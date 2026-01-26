import torch
import math
from typing import List, Union, Optional, Tuple
from PIL import Image
from src.mixins import LoaderMixin
import torch.nn.functional as F
import io
import av
import numpy as np
import torchvision.transforms.functional as TVF


def _encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def _decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def compress(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (
        (image[: (image.shape[0] // 2) * 2, : (image.shape[1] // 2) * 2] * 255.0)
        .byte()
        .cpu()
        .numpy()
    )
    with io.BytesIO() as output_file:
        _encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = _decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor


class LTXVideoCondition(LoaderMixin):
    def __init__(
        self,
        image: Optional[Image.Image] = None,
        video: Optional[List[Image.Image]] = None,
        height: int = 704,
        width: int = 1216,
        frame_number: int = 0,
        target_frames: int | None = None,
        conditioning_strength: float = 1.0,
        vae_scale_factor_spatial: int = 32,
        media_x: Optional[int] = None,
        media_y: Optional[int] = None,
        padding: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 0),
    ):
        if image is not None:
            self._media_item = self._load_image(image)
        elif video is not None:
            self._media_item = self._load_video(video)
        else:
            raise ValueError("No media item provided")
        self.frame_number = frame_number
        self.conditioning_strength = conditioning_strength
        self.media_x = media_x
        self.media_y = media_y
        self.height = height
        self.width = width
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        num_frames = (
            1 if isinstance(self._media_item, Image.Image) else len(self._media_item)
        )

        num_frames = self.trim_conditioning_sequence(
            self.frame_number, num_frames, target_frames or num_frames
        )

        self.media_item = self.load_media_file(
            self._media_item,
            self.height,
            self.width,
            num_frames,
            padding,
            just_crop=True,
        )

    def load_media_file(
        self,
        media: List[Image.Image] | Image.Image,
        height: int,
        width: int,
        max_frames: int,
        padding: tuple[int, int, int, int],
        just_crop: bool = False,
    ) -> torch.Tensor:

        if isinstance(media, List):
            num_input_frames = min(len(media), max_frames)

            # Read and preprocess the relevant frames from the video file.
            frames = []
            for i in range(num_input_frames):
                frame = media[i]
                frame_tensor = self.load_image_to_tensor_with_resize_and_crop(
                    frame, height, width, just_crop=just_crop
                )
                frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
                frames.append(frame_tensor)
            # Stack frames along the temporal dimension
            media_tensor = torch.cat(frames, dim=2)
        else:  # Input image
            media_tensor = self.load_image_to_tensor_with_resize_and_crop(
                media, height, width, just_crop=just_crop
            )
            media_tensor = torch.nn.functional.pad(media_tensor, padding)
        return media_tensor

    def load_image_to_tensor_with_resize_and_crop(
        self,
        image_input: Union[str, Image.Image],
        target_height: int = 512,
        target_width: int = 768,
        just_crop: bool = False,
    ) -> torch.Tensor:
        """Load and process an image into a tensor.

        Args:
            image_input: Either a file path (str) or a PIL Image object
            target_height: Desired height of output tensor
            target_width: Desired width of output tensor
            just_crop: If True, only crop the image to the target size without resizing
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(
                "image_input must be either a file path or a PIL Image object"
            )

        input_width, input_height = image.size
        aspect_ratio_target = target_width / target_height
        aspect_ratio_frame = input_width / input_height

        if aspect_ratio_frame > aspect_ratio_target:
            new_width = int(input_height * aspect_ratio_target)
            new_height = input_height
            x_start = (input_width - new_width) // 2
            y_start = 0
        else:
            new_width = input_width
            new_height = int(input_width / aspect_ratio_target)
            x_start = 0
            y_start = (input_height - new_height) // 2

        image = image.crop(
            (x_start, y_start, x_start + new_width, y_start + new_height)
        )
        if not just_crop:
            image = image.resize((target_width, target_height))

        frame_tensor = TVF.to_tensor(image)  # PIL -> tensor (C, H, W), [0,1]
        frame_tensor = TVF.gaussian_blur(frame_tensor, kernel_size=3, sigma=1.0)
        frame_tensor_hwc = frame_tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        frame_tensor_hwc = compress(frame_tensor_hwc)
        frame_tensor = (
            frame_tensor_hwc.permute(2, 0, 1) * 255.0
        )  # (H, W, C) -> (C, H, W)
        frame_tensor = (frame_tensor / 127.5) - 1.0
        # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
        return frame_tensor.unsqueeze(0).unsqueeze(2)

    def trim_conditioning_sequence(
        self, start_frame: int, sequence_num_frames: int, target_num_frames: int
    ):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.

        Returns:
            int: updated sequence length
        """

        scale_factor = self.vae_scale_factor_spatial
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames
