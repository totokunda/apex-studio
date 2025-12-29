from dataclasses import dataclass
import mlx.core as mx


@dataclass
class Transformer2DModelOutput:
    sample: mx.array
