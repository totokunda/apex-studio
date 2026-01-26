from PIL import Image
from typing import Union
from src.utils.defaults import (
    DEFAULT_CONFIG_SAVE_PATH,
    DEFAULT_COMPONENTS_PATH,
    get_components_path,
)
import numpy as np
import torch
from typing import Union, Dict, Any, List
from src.helpers.helpers import helpers
import torch.nn as nn
from src.helpers.base import BaseHelper
from transformers import CLIPImageProcessor
from src.utils.module import find_class_recursive
import importlib


@helpers("clip")
class CLIP(BaseHelper):

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        model_config_path: str = None,
        model_config: Dict[str, Any] | None = None,
        save_path: str = DEFAULT_COMPONENTS_PATH,
        config_save_path: str = DEFAULT_CONFIG_SAVE_PATH,
        processor_class: str = "AutoProcessor",
        model_class: str = "AutoModel",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path, save_path=save_path, config_path=model_config_path
        )
        self.config_save_path = config_save_path
        self.processor_class = processor_class
        self.model_class = model_class
        processor_class = find_class_recursive(
            importlib.import_module("transformers"), processor_class
        )

        preprocessor_path = self._download(preprocessor_path, get_components_path())

        self.processor = processor_class.from_pretrained(preprocessor_path)

        self.model = self._load_model(
            {
                "type": "clip",
                "base": model_class,
                "model_path": model_path,
                "config_path": model_config_path,
                "config": model_config,
            },
            module_name="transformers",
        )

    def find_key_with_type(self, config: Dict[str, Any]) -> str:
        for key, value in config.items():
            if "type" in key:
                return key
        return None

    @torch.no_grad()
    def __call__(
        self,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        hidden_states_layer: int = -1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        **kwargs,
    ):
        if isinstance(image, list):
            images = [self._load_image(img) for img in image]
        else:
            images = [self._load_image(image)]

        if device is None:
            device = self.model.device
        if dtype is None:
            dtype = self.model.dtype

        processed_images = self.processor(images, return_tensors="pt", **kwargs).to(
            device=device, dtype=dtype
        )

        image_embeds = self.model(**processed_images, output_hidden_states=True)

        image_embeds = image_embeds.hidden_states[hidden_states_layer]
        return image_embeds

    def __str__(self):
        return f"CLIPPreprocessor(model={self.model}, preprocessor={self.processor})"

    def __repr__(self):
        return self.__str__()
