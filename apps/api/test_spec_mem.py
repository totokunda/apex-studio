import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import torch
from src.engine.registry import UniversalEngine

engine = UniversalEngine(
    yaml_path="manifest/video/ltx2-19b-text-to-image-to-video-distilled-1.0.0.v1.yml",
    selected_components={
        "transformer": {"variant": "default"},
    },
    attention_type="sdpa",
).engine
engine.device = torch.device("cpu")

engine.load_component_by_type("transformer")
transformer = engine.transformer
print(transformer)
