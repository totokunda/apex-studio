from src.mixins import LoaderMixin, ToMixin
from torch import nn
from PIL import Image
import importlib
from typing import Dict, Any, List, Tuple
from transformers.image_processing_utils import ImageProcessingMixin
from src.utils.module import find_class_recursive
from transformers import AutoProcessor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from src.register import ClassRegister
from enum import Enum
from collections import OrderedDict
import numpy as np
import json
from loguru import logger

class BaseOutput(OrderedDict):
    """
    Lightweight output container that supports both attribute and dict-style access.

    - Initializes attributes from keyword arguments.
    - Keeps attributes and mapping entries in sync on assignment.
    - Preserves field order using subclass type annotations when available.
    - Provides tuple-style indexing (e.g., out[0]) over non-None fields in order.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        # Collect annotated fields from the MRO to preserve a meaningful order
        annotated_fields: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            annotated_fields.update(getattr(cls, "__annotations__", {}) or {})

        ordered_keys = (
            list(annotated_fields.keys()) if annotated_fields else list(kwargs.keys())
        )

        # Initialize annotated fields first (default to None if not provided)
        for key in ordered_keys:
            value = kwargs.get(key, None)
            if value is not None:
                super().__setitem__(key, value)
                super().__setattr__(key, value)
            else:
                # Ensure attribute exists even if not provided
                super().__setattr__(key, None)

        # Add any extra keys not present in annotations
        for key, value in kwargs.items():
            if key not in ordered_keys:
                super().__setitem__(key, value)
                super().__setattr__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep mapping and attributes in sync; add key when first set to non-None
        if not name.startswith("_") and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        # Mirror into attribute for convenience
        super().__setattr__(key, value)

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, str):
            # Dict-like access by key
            return dict(self.items())[k]
        # Tuple-like access by index or slice
        return self.to_tuple()[k]

    def to_tuple(self):
        # Convert to tuple of non-None values preserving key order
        return tuple(value for key, value in self.items() if value is not None)

    def _jsonify(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalars
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [self._jsonify(v) for v in obj]
        if isinstance(obj, (dict, OrderedDict)):
            return {k: self._jsonify(v) for k, v in obj.items()}
        return obj

    def to_dict(self) -> Dict[str, Any]:
        # Return a JSON-serializable plain dict view
        return {k: self._jsonify(v) for k, v in self.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return self.to_json()

    def __repr__(self) -> str:
        return self.to_json()


class PreprocessorType(Enum):
    IMAGE = "image"
    IMAGE_TEXT = "image_text"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    OTHER = "other"
    POSE = "pose"


class BaseHelper(LoaderMixin, ToMixin, nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.logger = logger
        self.kwargs = kwargs
