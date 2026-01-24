from typing import MutableMapping, Type, Dict, Any
from abc import ABC, abstractmethod
from diffusers.models.modeling_utils import ModelMixin
from src.register import ClassRegister

TRANSFORMERS_REGISTRY = ClassRegister()


def get_transformer(name: str):
    return TRANSFORMERS_REGISTRY.get(name.lower())
