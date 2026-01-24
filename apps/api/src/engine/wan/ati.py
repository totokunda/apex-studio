import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from .shared import WanShared

# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union
import torch


from typing import List, Optional, Tuple, Union
import torch


# Refer to https://github.com/Angtian/VoGE/blob/main/VoGE/Utils.py
def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int = 1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert (
        len(ind.shape) > dim
    ), "Index must have the target dim, but get dim: %d, ind shape: %s" % (
        dim,
        str(ind.shape),
    )

    target = target.expand(
        *tuple(
            [ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)]
            + [
                -1,
            ]
            * (len(target.shape) - dim)
        )
    )

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1) : :])

    return torch.gather(target, dim=dim, index=ind_pad)


def merge_final(
    vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor
):
    """

    :param vert_attr: [n, d] or [b, n, d] color or feature of each vertex
    :param weight: [b(optional), w, h, M] weight of selected vertices
    :param vert_assign: [b(optional), w, h, M] selective index
    :return:
    """
    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        assert vert_attr.shape[0] > vert_assign.max()
        # [n, d] ind: [b(optional), w, h, M]-> [b(optional), w, h, M, d]
        sel_attr = ind_sel(
            vert_attr[(None,) * target_dim],
            vert_assign.type(torch.long),
            dim=target_dim,
        )
    else:
        assert vert_attr.shape[1] > vert_assign.max()
        sel_attr = ind_sel(
            vert_attr[(slice(None),) + (None,) * (target_dim - 1)],
            vert_assign.type(torch.long),
            dim=target_dim,
        )

    # [b(optional), w, h, M]
    final_attr = torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)
    return final_attr


def patch_motion(
    tracks: torch.FloatTensor,  # (B, T, N, 4)
    vid: torch.FloatTensor,  # (C, T, H, W)
    temperature: float = 220.0,
    training: bool = True,
    tail_dropout: float = 0.2,
    vae_divide: tuple = (4, 16),
    topk: int = 2,
):
    with torch.no_grad():
        _, T, H, W = vid.shape
        N = tracks.shape[2]
        _, tracks, visible = torch.split(
            tracks, [1, 2, 1], dim=-1
        )  # (B, T, N, 2) | (B, T, N, 1)
        tracks_n = tracks / torch.tensor(
            [W / min(H, W), H / min(H, W)], device=tracks.device
        )
        tracks_n = tracks_n.clamp(-1, 1)
        visible = visible.clamp(0, 1)

        if tail_dropout > 0 and training:
            TT = visible.shape[1]
            rrange = torch.arange(TT, device=visible.device, dtype=visible.dtype)[
                None, :, None, None
            ]
            rand_nn = torch.rand_like(visible[:, :1])
            rand_rr = torch.rand_like(visible[:, :1]) * (TT - 1)
            visible = visible * (
                (rand_nn > tail_dropout).type_as(visible)
                + (rrange < rand_rr).type_as(visible)
            ).clamp(0, 1)

        xx = torch.linspace(-W / min(H, W), W / min(H, W), W)
        yy = torch.linspace(-H / min(H, W), H / min(H, W), H)

        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(
            tracks.device
        )

        tracks_pad = tracks[:, 1:]
        visible_pad = visible[:, 1:]

        visible_align = visible_pad.view(T - 1, 4, *visible_pad.shape[2:]).sum(1)
        tracks_align = (tracks_pad * visible_pad).view(
            T - 1, 4, *tracks_pad.shape[2:]
        ).sum(1) / (visible_align + 1e-5)
        dist_ = (
            (tracks_align[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
        )  # T, H, W, N
        weight = torch.exp(-dist_ * temperature) * visible_align.clamp(0, 1).view(
            T - 1, 1, 1, N
        )
        vert_weight, vert_index = torch.topk(
            weight, k=min(topk, weight.shape[-1]), dim=-1
        )

    grid_mode = "bilinear"
    point_feature = torch.nn.functional.grid_sample(
        vid[vae_divide[0] :].permute(1, 0, 2, 3)[:1],
        tracks_n[:, :1].type(vid.dtype),
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=None,
    )
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0)  # N, C=16

    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(
        3, 0, 1, 2
    )  # T - 1, H, W, C => C, T - 1, H, W
    out_weight = vert_weight.sum(-1)  # T - 1, H, W

    # out feature -> already soft weighted
    mix_feature = out_feature + vid[vae_divide[0] :, 1:] * (1 - out_weight.clamp(0, 1))

    out_feature_full = torch.cat(
        [vid[vae_divide[0] :, :1], mix_feature], dim=1
    )  # C, T, H, W
    out_mask_full = torch.cat(
        [torch.ones_like(out_weight[:1]), out_weight], dim=0
    )  # T, H, W
    return torch.cat(
        [out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full], dim=0
    )


class WanATIEngine(WanShared):
    """WAN Audio-to-Video Engine Implementation"""

    def run(
        self,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        trajectory: np.ndarray | str = None,
        duration: int | str = 16,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
        enhance_kwargs: Dict[str, Any] = {},
    ):

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        batch_size = prompt_embeds.shape[0]

        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            negative_prompt_embeds = None

        if offload:
            self._offload("text_encoder")

        loaded_image = self._load_image(image)

        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image,
            max_area=height * width,
            mod_value=32 if expand_timesteps else 16,
        )

        if isinstance(trajectory, str):
            preprocessor = self.helpers["wan.ati"]
            tracks = preprocessor(trajectory, width, height)
            tracks = tracks.to(self.device)
        else:
            tracks = torch.from_numpy(trajectory).to(self.device)

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        if not self.transformer:
            self.load_component_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        self.to_device(self.transformer)

        if boundary_ratio is None and not expand_timesteps:
            image_embeds = self.helpers["clip"](
                loaded_image, hidden_states_layer=-2
            ).to(self.device, dtype=transformer_dtype)
        else:
            image_embeds = None

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload and boundary_ratio is None and not expand_timesteps:
            self._offload("clip")

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )

        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        num_frames = self._parse_num_frames(duration, fps)

        vae_config = self.load_config_by_type("vae")
        vae_scale_factor_spatial = getattr(
            vae_config, "scale_factor_spatial", self.vae_scale_factor_spatial
        )
        vae_scale_factor_temporal = getattr(
            vae_config, "scale_factor_temporal", self.vae_scale_factor_temporal
        )

        latents = self._get_latents(
            height,
            width,
            duration,
            num_channels_latents=getattr(vae_config, "z_dim", 16),
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        if preprocessed_image.ndim == 4:
            preprocessed_image = preprocessed_image.unsqueeze(2)

        video_condition = torch.cat(
            [
                preprocessed_image,
                preprocessed_image.new_zeros(
                    preprocessed_image.shape[0],
                    preprocessed_image.shape[1],
                    num_frames - 1,
                    height,
                    width,
                ),
            ],
            dim=2,
        )

        latent_condition = self.vae_encode(
            video_condition,
            offload=offload,
            dtype=latents.dtype,
            normalize_latents_dtype=latents.dtype,
        )

        batch_size, _, num_latent_frames, latent_height, latent_width = latents.shape

        if expand_timesteps:
            first_frame_mask = torch.ones(
                1,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=latents.dtype,
                device=latents.device,
            )
            first_frame_mask[:, :, 0] = 0
        else:
            mask_lat_size = torch.ones(
                batch_size,
                1,
                num_frames,
                latent_height,
                latent_width,
                device=self.device,
            )

            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
            )

            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(
                batch_size,
                -1,
                self.vae_scale_factor_temporal,
                latent_height,
                latent_width,
            )

            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latents.device)

            latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        if tracks is not None:
            tracks = (
                tracks.to(latent_condition).unsqueeze(0).repeat(num_videos, 1, 1, 1)
            )
            latent_condition = (
                patch_motion(tracks, latent_condition.squeeze(0), training=False)
                .unsqueeze(0)
                .repeat(num_videos, 1, 1, 1, 1)
            )

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * getattr(
                self.scheduler.config, "num_train_timesteps", 1000
            )
        else:
            boundary_timestep = None

        latents = self.denoise(
            boundary_timestep=boundary_timestep,
            timesteps=timesteps,
            latents=latents,
            latent_condition=latent_condition,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                enhance_kwargs=enhance_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    enhance_kwargs=enhance_kwargs,
                )
                if negative_prompt_embeds is not None
                else None
            ),
            transformer_dtype=transformer_dtype,
            use_cfg_guidance=use_cfg_guidance,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            expand_timesteps=expand_timesteps,
            first_frame_mask=first_frame_mask,
        )

        if offload:
            self._offload("transformer")

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
