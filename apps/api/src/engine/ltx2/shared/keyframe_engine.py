from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


class LTX2KeyframeConditioningMixin:
    """
    Mixin that implements *keyframe-token* conditioning similar to ltx-core's
    `VideoConditionByKeyframeIndex`:
    - Encode each conditioning (image or video) into VAE latent grid.
    - Pack into tokens and APPEND to the base latent sequence.
    - Create a per-token denoise_mask (1=strength 0=keep clean) and per-token clean_latent.
    - Create per-token coords for RoPE and time-shift them by pixel_frame_idx/fps.

    Assumes the host class provides:
    - self.vae_encode(...)
    - self._pack_latents(...)
    - self.transformer.rope.prepare_video_coords(...)
    - self.vae_temporal_compression_ratio, self.vae_spatial_compression_ratio
    - self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
    """

    def _base_video_token_count(
        self,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> int:
        pt = int(self.transformer_temporal_patch_size)
        ps = int(self.transformer_spatial_patch_size)
        return (latent_num_frames // pt) * (latent_height // ps) * (latent_width // ps)

    def _prepare_keyframe_conditioned_video_latents(
        self,
        *,
        # Generation target
        batch_size: int,
        num_channels_latents: int,
        pixel_num_frames: int,
        pixel_height: int,
        pixel_width: int,
        fps: float,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        # Base latents for the video sequence (packed tokens). If None, starts from noise.
        base_latents: Optional[torch.Tensor],
        # Conditioning items (each may be 1-frame image or multi-frame video latent input)
        cond_latent_inputs: List[torch.Tensor],
        cond_strengths: Optional[Union[float, List[float]]],
        cond_pixel_frame_indices: Optional[Union[int, List[int]]],
        offload: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
          - latents_tokens: [B, S_total, D]
          - denoise_mask_tokens: [B, S_total]  (scalar per token, 1=denoise/noise, 0=keep clean)
          - video_coords: [B, 3, S_total, 2]   (RoPE coords aligned to tokens)
          - clean_latents_tokens: [B, S_total, D] (used to re-impose hard conditioning if desired)
          - base_token_count: number of tokens that belong to the *base* video latent (before appended keyframe tokens)
        """
        latent_num_frames = (pixel_num_frames - 1) // int(self.vae_temporal_compression_ratio) + 1
        latent_height = pixel_height // int(self.vae_spatial_compression_ratio)
        latent_width = pixel_width // int(self.vae_spatial_compression_ratio)

        base_token_count = self._base_video_token_count(latent_num_frames, latent_height, latent_width)

        # --- Base latents (tokens) ---
        if base_latents is not None:
            latents_tokens = base_latents.to(device=device, dtype=dtype)
            if latents_tokens.ndim != 3:
                raise ValueError(f"`base_latents` must be a packed [B, S, D] tensor, got shape {latents_tokens.shape}.")
        else:
            shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
            noise_grid = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents_tokens = self._pack_latents(
                noise_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )

        if latents_tokens.shape[0] != batch_size:
            raise ValueError(f"Packed base latents batch {latents_tokens.shape[0]} != batch_size {batch_size}.")
        if latents_tokens.shape[1] != base_token_count:
            # If caller passed in latents with conditioning tokens, they should pass only the base tokens.
            raise ValueError(
                f"Packed base latents token count {latents_tokens.shape[1]} != expected base_token_count {base_token_count}. "
                "Pass only the base video tokens (without appended conditioning tokens)."
            )

        clean_latents_tokens = latents_tokens.clone()
        denoise_mask_tokens = torch.ones(
            (batch_size, base_token_count), device=device, dtype=dtype
        )
        video_coords = self.transformer.rope.prepare_video_coords(
            batch_size, latent_num_frames, latent_height, latent_width, device, fps=fps
        )

        # --- Normalize conditioning strengths and frame indices ---
        num_conds = len(cond_latent_inputs)
        if num_conds == 0:
            return latents_tokens, denoise_mask_tokens, video_coords, clean_latents_tokens, base_token_count

        if cond_strengths is None:
            strengths_t = torch.full((num_conds,), 1.0, device=device, dtype=torch.float32)
        elif isinstance(cond_strengths, (int, float)):
            strengths_t = torch.full((num_conds,), float(cond_strengths), device=device, dtype=torch.float32)
        else:
            if len(cond_strengths) != num_conds:
                raise ValueError(
                    f"`cond_strengths` length {len(cond_strengths)} must match number of conditionings {num_conds}."
                )
            strengths_t = torch.tensor([float(s) for s in cond_strengths], device=device, dtype=torch.float32)
        strengths_t = strengths_t.clamp(0.0, 1.0).to(dtype=dtype)

        if cond_pixel_frame_indices is None:
            pixel_idx_t = torch.zeros((num_conds,), device=device, dtype=torch.long)
        elif isinstance(cond_pixel_frame_indices, int):
            pixel_idx_t = torch.full((num_conds,), int(cond_pixel_frame_indices), device=device, dtype=torch.long)
        else:
            if len(cond_pixel_frame_indices) != num_conds:
                raise ValueError(
                    f"`cond_pixel_frame_indices` length {len(cond_pixel_frame_indices)} must match number of conditionings {num_conds}."
                )
            pixel_idx_t = torch.tensor([int(i) for i in cond_pixel_frame_indices], device=device, dtype=torch.long)

        if pixel_idx_t.min().item() < 0 or pixel_idx_t.max().item() >= pixel_num_frames:
            raise ValueError(
                f"All cond pixel frame indices must be in [0, {pixel_num_frames - 1}]. "
                f"Got min={pixel_idx_t.min().item()} max={pixel_idx_t.max().item()}."
            )

        # --- Append each conditioning as keyframe tokens ---
        encode_generator = generator[0] if isinstance(generator, list) and len(generator) > 0 else generator
        for k, latent_in in enumerate(cond_latent_inputs):
            # latent_in is expected to be pixel-space preprocessed tensor:
            # - image: [1, 3, H, W]
            # - video: [1, 3, F, H, W]
            if latent_in.ndim == 4:
                latent_in = latent_in.unsqueeze(2)  # [1, 3, 1, H, W]
            if latent_in.ndim != 5:
                raise ValueError(f"Conditioning input must be [1,3,H,W] or [1,3,F,H,W], got {latent_in.shape}")

            encoded = self.vae_encode(
                latent_in,
                encode_generator,
                offload=offload,
                sample_mode="mode",
            ).to(dtype)

            # Ensure spatial shape matches the generation latent grid.
            if encoded.shape[-2:] != (latent_height, latent_width):
                raise ValueError(
                    f"Conditioning latent spatial shape {tuple(encoded.shape[-2:])} != expected {(latent_height, latent_width)}. "
                    "Make sure conditioning inputs were resized to the same pixel height/width as the generation."
                )

            encoded = encoded.repeat(batch_size, 1, 1, 1, 1)  # broadcast across batch
            cond_tokens = self._pack_latents(
                encoded, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )

            # Build coords for the conditioning tokens and shift time by pixel_frame_idx/fps (keyframe index).
            cond_lat_frames = int(encoded.shape[2])
            cond_coords = self.transformer.rope.prepare_video_coords(
                batch_size, cond_lat_frames, latent_height, latent_width, device, fps=fps
            )
            time_shift_s = float(pixel_idx_t[k].item()) / float(fps)
            cond_coords[:, 0, :, :] = cond_coords[:, 0, :, :] + time_shift_s

            # Create denoise mask for appended tokens (scalar per token).
            # strength=1 => denoise_mask=0 (keep clean)
            cond_denoise_scalar = float(1.0 - strengths_t[k].item())
            cond_denoise_mask = torch.full(
                (batch_size, cond_tokens.shape[1]),
                fill_value=cond_denoise_scalar,
                device=device,
                dtype=dtype,
            )

            # Initialize appended tokens as GaussianNoiser would.
            cond_noise = randn_tensor(
                cond_tokens.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            cond_latents_init = cond_noise * cond_denoise_mask.unsqueeze(-1) + cond_tokens * (
                1.0 - cond_denoise_mask.unsqueeze(-1)
            )

            latents_tokens = torch.cat([latents_tokens, cond_latents_init], dim=1)
            clean_latents_tokens = torch.cat([clean_latents_tokens, cond_tokens], dim=1)
            denoise_mask_tokens = torch.cat([denoise_mask_tokens, cond_denoise_mask], dim=1)
            video_coords = torch.cat([video_coords, cond_coords], dim=2)

        return latents_tokens, denoise_mask_tokens, video_coords, clean_latents_tokens, base_token_count


