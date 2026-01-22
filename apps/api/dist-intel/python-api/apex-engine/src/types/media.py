from typing import Union, List
from PIL import Image
import numpy as np
import torch

InputImage = Union[Image.Image, np.ndarray, str, torch.Tensor]
InputVideo = Union[str, List[str], np.ndarray, torch.Tensor, List[Image.Image]]
InputAudio = Union[str, List[str], np.ndarray, torch.Tensor]
OutputImage = Image.Image
OutputVideo = List[Image.Image]
InputMedia = Union[InputImage, InputVideo]
OutputMedia = Union[OutputImage, OutputVideo]
