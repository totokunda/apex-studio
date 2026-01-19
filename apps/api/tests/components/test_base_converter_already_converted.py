import torch

from src.converters.base_converter import BaseConverter


class _NoOpConverter(BaseConverter):
    """
    Converter with no rename/special rules. Historically, `_already_converted` would
    return False because there are no target markers; with `model_keys` it should
    be able to detect already-converted state_dicts.
    """

    def __init__(self):
        super().__init__()


class _SimpleRenameConverter(BaseConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {"old.block": "new.block"}


def test_already_converted_uses_model_keys_when_provided():
    converter = _NoOpConverter()

    sd = {
        "layer.weight": torch.zeros(1),
        "layer.bias": torch.zeros(1),
        "block.0.proj.weight": torch.zeros(1),
        "block.0.proj.bias": torch.zeros(1),
        "norm.weight": torch.zeros(1),
        "norm.bias": torch.zeros(1),
        "attn.to_q.weight": torch.zeros(1),
        "attn.to_k.weight": torch.zeros(1),
        "attn.to_v.weight": torch.zeros(1),
        "attn.to_out.weight": torch.zeros(1),
    }

    model_keys = list(sd.keys()) + ["extra.unused.param"]
    assert converter._already_converted(sd, model_keys) is True


def test_model_keys_mismatch_does_not_false_positive():
    converter = _SimpleRenameConverter()

    sd = {
        "old.block.weight": torch.zeros(1),
        "old.block.bias": torch.zeros(1),
    }

    # Target model expects the post-conversion keys, so this ckpt is not yet converted.
    model_keys = ["new.block.weight", "new.block.bias"]
    assert converter._already_converted(sd, model_keys) is False
