import mlx.core as mx
import torch
from typing import Dict, List
import mlx.nn as nn
from typing import Any
import numpy as np


def convert_dtype_to_torch(dtype: str | torch.dtype | mx.Dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, mx.Dtype):
        dtype_as_str = str(dtype).replace("mlx.core.", "")
        return getattr(torch, dtype_as_str)
    else:
        return torch.dtype(dtype)


def convert_dtype_to_mlx(dtype: str | torch.dtype | mx.Dtype) -> mx.Dtype:
    if isinstance(dtype, mx.Dtype):
        return dtype
    elif isinstance(dtype, torch.dtype):
        dtype_as_str = str(dtype).replace("torch.", "")
        return getattr(mx, dtype_as_str)
    else:
        return mx.Dtype(dtype)


def to_mlx(t: torch.Tensor) -> mx.array:
    torch_dtype = t.dtype
    mx_dtype = convert_dtype_to_mlx(torch_dtype)
    return mx.array(t.detach().to("cpu", copy=False).numpy()).astype(
        dtype=mx_dtype, stream=mx.default_device()
    )


def to_torch(a: mx.array) -> torch.Tensor:
    mx_dtype = a.dtype
    torch_dtype = convert_dtype_to_torch(mx_dtype)
    # get the device of the mlx array
    default_device = mx.default_device()
    if default_device.type.name == "gpu":
        return torch.from_numpy(np.array(a, copy=False)).to(torch_dtype).to("mps")
    else:
        return torch.from_numpy(np.array(a, copy=False)).to(torch_dtype)


def check_mlx_convolutional_weights(
    state_dict: Dict[str, mx.array], model: nn.Module
) -> bool:
    # Go through model and find all the conv3d and conv2d layers and check that weights are in the same shape, as mlx has them backwrads compatible
    for name, param in model.named_modules():
        if isinstance(param, nn.Conv3d):
            if name in state_dict:
                if state_dict[name].shape != param.weight.shape:
                    state_dict[name] = state_dict[name].transpose(0, 2, 3, 4, 1)
                    if state_dict[name].shape != param.weight.shape:
                        raise ValueError(
                            f"Weight {name} has shape {state_dict[name].shape} but expected {param.weight.shape}"
                        )
        elif isinstance(param, nn.Conv2d):
            if name in state_dict:
                if state_dict[name].shape != param.weight.shape:
                    state_dict[name] = state_dict[name].transpose(0, 2, 3, 1)
                    if state_dict[name].shape != param.weight.shape:
                        raise ValueError(
                            f"Weight {name} has shape {state_dict[name].shape} but expected {param.weight.shape}"
                        )
    return True


def torch_to_mlx(
    state_dict: Dict[str, Any] | List[torch.Tensor] | torch.Tensor,
) -> Dict[str, Any] | List[mx.array] | mx.array:
    if isinstance(state_dict, list):
        return [torch_to_mlx(item) for item in state_dict]
    elif isinstance(state_dict, dict):
        return {key: torch_to_mlx(value) for key, value in state_dict.items()}
    else:
        if isinstance(state_dict, torch.Tensor):
            return to_mlx(state_dict)
        else:
            return state_dict


def mlx_to_torch(
    state_dict: Dict[str, Any] | List[mx.array] | mx.array,
) -> Dict[str, Any] | List[torch.Tensor] | torch.Tensor:
    if isinstance(state_dict, list):
        return [mlx_to_torch(item) for item in state_dict]
    elif isinstance(state_dict, dict):
        return {key: mlx_to_torch(value) for key, value in state_dict.items()}
    else:
        if isinstance(state_dict, mx.array):
            return to_torch(state_dict)
        else:
            return state_dict
