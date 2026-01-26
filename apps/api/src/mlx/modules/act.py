from mlx import nn
import mlx.core as mx

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(
            f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}"
        )


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approx="none", bias: bool = True):
        super().__init__()
        self.approx = approx
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.gelu = nn.GELU(approx=approx)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states)
        return self.gelu(hidden_states)
