from abc import ABC, abstractmethod
import torch
from src.mixins import LoaderMixin
from src.register import ClassRegister
from enum import Enum


class PostprocessorCategory(Enum):
    UPSCALER = "upscaler"
    FRAME_INTERPOLATION = "frame_interpolation"
    SAFETY_CHECKER = "safety_checker"


class BasePostprocessor(torch.nn.Module, LoaderMixin):
    def __init__(self, engine=None, category: PostprocessorCategory = None, **kwargs):
        super().__init__()
        self.engine = engine
        self.device = engine.device if engine is not None else None
        self.component_conf = kwargs
        self.category = (
            category if category is not None else PostprocessorCategory.UPSCALER
        )
        self.engine_type = "pt"

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "BasePostprocessor::__call__ method must be implemented by child classes"
        )


postprocessor_registry = ClassRegister()
