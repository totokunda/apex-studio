import gc
import torch
from typing import Dict, Union
from safetensors.torch import safe_open
import os


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
