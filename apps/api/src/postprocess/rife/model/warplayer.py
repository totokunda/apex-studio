import torch
import torch.nn as nn

try:
    from src.utils.defaults import get_torch_device
except ModuleNotFoundError:
    import os
    import sys

    # Try to add the project root (parent of 'src') to sys.path
    current_dir = os.path.dirname(__file__)
    for i in range(6):
        candidate = os.path.abspath(os.path.join(current_dir, *([".."] * i)))
        if os.path.isdir(os.path.join(candidate, "src")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            break
    from src.utils.defaults import get_torch_device

device = get_torch_device()
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    padding_mode = "zeros" if tenInput.device.type == "mps" else "border"
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
