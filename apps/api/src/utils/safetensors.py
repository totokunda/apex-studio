from typing import Dict, Union
import os
import torch
import safetensors
import mlx.core as mx


def is_safetensors_file(file_path: str, framework: str = "pt"):
    try:
        with safetensors.safe_open(file_path, framework=framework, device="cpu") as f:
            f.keys()
        return True
    except Exception:
        return False


def is_floating_point_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]


def load_safetensors(
    filename: Union[str, os.PathLike],
    device: Union[str, int] = "cpu",
    dtype: torch.dtype | mx.Dtype = None,
    framework: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result: Dict[str, torch.Tensor] = {}

    # Cache local variables and flags outside the loop to minimize
    # repeated attribute lookups and isinstance checks in the hot path.
    framework_is_np = framework == "np"
    has_dtype = dtype is not None
    is_torch_dtype = isinstance(dtype, torch.dtype)
    is_mx_dtype = isinstance(dtype, mx.Dtype)
    to_mx_array = mx.array
    is_fp = is_floating_point_tensor

    with safetensors.safe_open(filename, framework=framework, device=device) as f:
        # Fast path: no framework conversion and no dtype casting.
        if not framework_is_np and not has_dtype:
            for k in f.offset_keys():
                result[k] = f.get_tensor(k)
            return result

        # General path with optional framework and/or dtype conversion.
        for k in f.offset_keys():
            tensor = f.get_tensor(k)

            if framework_is_np:
                tensor = to_mx_array(tensor)

            if has_dtype:
                if (
                    is_torch_dtype
                    and isinstance(tensor, torch.Tensor)
                    and is_fp(tensor)
                ):
                    if tensor.dtype != dtype:
                        tensor = tensor.to(dtype, copy=False, non_blocking=True)
                elif is_mx_dtype and isinstance(tensor, mx.array):
                    if tensor.dtype != dtype:
                        tensor = tensor.astype(dtype)

            result[k] = tensor
    return result
