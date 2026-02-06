import io
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from src.helpers.helpers import helpers

def unzip_to_array(
    data: bytes, key: Union[str, List[str]] = "array"
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    bytes_io = io.BytesIO(data)

    if isinstance(key, str):
        # Load the NPZ data from the BytesIO object
        with np.load(bytes_io) as data:
            return data[key]
    else:
        get = {}
        with np.load(bytes_io) as data:
            for k in key:
                get[k] = data[k]
        return get

def process_tracks(
    tracks_np: np.ndarray, frame_size: Tuple[int, int], quant_multi: int = 8, **kwargs
):
    # tracks: shape [t, h, w, 3] => samples align with 24 fps, model trained with 16 fps.
    # frame_size: tuple (W, H)

    tracks = torch.from_numpy(tracks_np).float() / quant_multi
    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))
    tracks, visibles = tracks[..., :2], tracks[..., 2:3]
    short_edge = min(*frame_size)

    tracks = tracks - torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = tracks / short_edge * 2

    visibles = visibles * 2 - 1

    trange = (
        torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)
    )

    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)
    out_0 = out_[:1]
    out_l = out_[1:]  # 121 => 120 | 1
    out_l = torch.repeat_interleave(out_l, 2, dim=0)[1::3]  # 120 => 240 => 80
    return torch.cat([out_0, out_l], dim=0)


@helpers("wan.ati")
class WanATI:
    def __call__(
        self,
        tracks: np.ndarray | str,
        width: int,
        height: int,
        quant_multi: int = 8,
        **kwargs,
    ):
        if isinstance(tracks, str):
            tracks = torch.load(tracks)

        tracks_np = unzip_to_array(tracks)
        tracks = process_tracks(
            tracks_np, (width, height), quant_multi=quant_multi, **kwargs
        )
        return tracks
