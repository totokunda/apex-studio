from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

try:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
except Exception:  # pragma: no cover - MLX is not available on Windows/Linux
    mx = None  # type: ignore
    nn = None  # type: ignore


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError(
            "MLX is not available on this platform/environment. "
            "Install/use MLX only on macOS (Apple Silicon)."
        )


def convert_dtype_to_torch(dtype: str | torch.dtype | Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    elif mx is not None and isinstance(dtype, mx.Dtype):
        dtype_as_str = str(dtype).replace("mlx.core.", "")
        return getattr(torch, dtype_as_str)
    else:
        if isinstance(dtype, str):
            dtype_as_str = dtype.replace("torch.", "")
            return getattr(torch, dtype_as_str)
        raise TypeError(f"Unsupported dtype: {type(dtype)}")


def convert_dtype_to_mlx(dtype: str | torch.dtype | Any):
    _require_mlx()
    if isinstance(dtype, mx.Dtype):  # type: ignore[attr-defined]
        return dtype
    elif isinstance(dtype, torch.dtype):
        dtype_as_str = str(dtype).replace("torch.", "")
        return getattr(mx, dtype_as_str)  # type: ignore[union-attr]
    else:
        if isinstance(dtype, str):
            dtype_as_str = dtype.replace("mlx.core.", "")
            return getattr(mx, dtype_as_str)  # type: ignore[union-attr]
        raise TypeError(f"Unsupported dtype: {type(dtype)}")


def to_mlx(t: torch.Tensor):
    _require_mlx()
    torch_dtype = t.dtype
    mx_dtype = convert_dtype_to_mlx(torch_dtype)
    return mx.array(t.detach().to("cpu", copy=False).numpy()).astype(  # type: ignore[union-attr]
        dtype=mx_dtype, stream=mx.default_device()  # type: ignore[union-attr]
    )


def to_torch(a) -> torch.Tensor:
    _require_mlx()
    mx_dtype = a.dtype
    torch_dtype = convert_dtype_to_torch(mx_dtype)
    # get the device of the mlx array
    default_device = mx.default_device()  # type: ignore[union-attr]
    if default_device.type.name == "gpu":
        return torch.from_numpy(np.array(a, copy=False)).to(torch_dtype).to("mps")
    else:
        return torch.from_numpy(np.array(a, copy=False)).to(torch_dtype)


def check_mlx_convolutional_weights(
    state_dict: Dict[str, Any], model: Any
) -> bool:
    _require_mlx()
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
) -> Dict[str, Any] | List[Any] | Any:
    _require_mlx()
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
    state_dict: Dict[str, Any] | List[Any] | Any,
) -> Dict[str, Any] | List[torch.Tensor] | torch.Tensor:
    _require_mlx()
    if isinstance(state_dict, list):
        return [mlx_to_torch(item) for item in state_dict]
    elif isinstance(state_dict, dict):
        return {key: mlx_to_torch(value) for key, value in state_dict.items()}
    else:
        if isinstance(state_dict, mx.array):
            return to_torch(state_dict)
        else:
            return state_dict
