from src.engine.ltx2.shared import LTX2Shared
from typing import Union, List, Optional, Dict, Any, Callable
import torch
import numpy as np
import copy
from diffusers.utils.torch_utils import randn_tensor
from src.helpers.ltx2.upsampler import upsample_video
from src.types import InputImage, InputAudio    
from einops import rearrange   
from src.utils.cache import empty_cache
from src.utils.progress import safe_emit_progress, make_mapped_progress

class LTX2TI2VEngine(LTX2Shared):
    """LTX2 Text-to-Image-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        
    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        offload: bool = True,
    ) -> torch.Tensor:

        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        
        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        denoise_mask = torch.ones(shape, device=device, dtype=dtype) * noise_scale
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = noise * denoise_mask + latents * (1 - denoise_mask)
        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )
        return latents, None
    
    
    def prepare_latents_image_conditioning(
        self,
        image: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        noise_scale: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        strengths: Optional[Union[float, List[float]]] = None,
        pixel_frame_indices: Optional[Union[int, List[int]]] = None,
        offload: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: Official ltx-pipelines uses `VideoConditionByLatentIndex(latent_idx=frame_idx)`, i.e. frame indices are
        # in *latent frame* space (after VAE temporal compression). Historically our API called this
        # `pixel_frame_indices` and converted pixel->latent.
        #
        # To stay backward compatible:
        # - if max index < latent_num_frames => treat as latent indices (official semantics)
        # - else                           => treat as pixel indices and convert to latent indices
        pixel_num_frames = num_frames
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, latent_num_frames, height, width)
        mask_shape = (batch_size, 1, latent_num_frames, height, width)

        # Denoise mask semantics match ltx-core:
        # - denoise_mask = 1.0 => fully denoise/noise
        # - denoise_mask = 0.0 => keep clean (conditioning)

        # Determine number of conditioning items (multiple images injected into a *single* latent sequence).
        if image is not None:
            num_conds = int(image.shape[0])
        elif isinstance(strengths, list):
            num_conds = len(strengths)
        elif isinstance(pixel_frame_indices, list):
            num_conds = len(pixel_frame_indices)
        else:
            num_conds = 1

        # Normalize strengths to per-conditioning list/tensor.
        if strengths is None:
            strengths_t = torch.full((num_conds,), 1.0, device=device, dtype=torch.float32)
        elif isinstance(strengths, (int, float)):
            strengths_t = torch.full((num_conds,), float(strengths), device=device, dtype=torch.float32)
        else:
            if len(strengths) != num_conds:
                raise ValueError(
                    f"Provided `strengths` has length {len(strengths)}, but expected length {num_conds}."
                )
            strengths_t = torch.tensor([float(s) for s in strengths], device=device, dtype=torch.float32)
        strengths_t = strengths_t.clamp(0.0, 1.0).to(dtype=dtype)

        # Normalize frame indices to per-conditioning list/tensor (pixel frame indices).
        if pixel_frame_indices is None:
            pixel_frame_indices_t = torch.zeros((num_conds,), device=device, dtype=torch.long)
        elif isinstance(pixel_frame_indices, int):
            pixel_frame_indices_t = torch.full((num_conds,), int(pixel_frame_indices), device=device, dtype=torch.long)
        else:
            if len(pixel_frame_indices) != num_conds:
                raise ValueError(
                    f"Provided `pixel_frame_indices` has length {len(pixel_frame_indices)}, but expected length {num_conds}."
                )
            pixel_frame_indices_t = torch.tensor([int(i) for i in pixel_frame_indices], device=device, dtype=torch.long)

        if pixel_frame_indices_t.min().item() < 0:
            raise ValueError(
                f"All `pixel_frame_indices` must be >= 0. Got min={pixel_frame_indices_t.min().item()}."
            )

        # Decide whether indices are latent-frame indices (official) or pixel-frame indices (legacy).
        if pixel_frame_indices_t.max().item() < latent_num_frames:
            frame_indices_t = pixel_frame_indices_t
        else:
            if pixel_frame_indices_t.max().item() >= pixel_num_frames:
                raise ValueError(
                    f"All `pixel_frame_indices` must be in [0, {pixel_num_frames - 1}] (pixel/video frames). "
                    f"Got min={pixel_frame_indices_t.min().item()} max={pixel_frame_indices_t.max().item()}."
                )
            frame_indices_t = pixel_frame_indices_t // int(self.vae_temporal_compression_ratio)

        if frame_indices_t.max().item() >= latent_num_frames:
            raise ValueError(
                f"Latent frame indices must be in [0, {latent_num_frames - 1}] (latent frames after temporal compression). "
                f"Got min={frame_indices_t.min().item()} max={frame_indices_t.max().item()}."
            )

        # Build denoise_mask in latent grid space [B, 1, F, H, W]
        denoise_mask_grid = torch.ones(mask_shape, device=device, dtype=dtype)
        for k in range(num_conds):
            denoise_mask_grid[:, :, int(frame_indices_t[k].item())] = 1.0 - strengths_t[k]

        denoise_mask_tokens = self._pack_latents(
            denoise_mask_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        ).mean(dim=-1)

        # Start from initial latent grid if provided, otherwise an empty (zero) grid.
        if latents is not None:
            if latents.ndim == 5:
                if tuple(latents.shape) != shape:
                    raise ValueError(
                        f"Provided `latents` grid has shape {tuple(latents.shape)}, but expected {shape} "
                        f"for (batch_size={batch_size}, channels={num_channels_latents}, frames={num_frames}, "
                        f"height={height}, width={width})."
                    )
                base_latent_grid = latents.to(device=device, dtype=dtype)
            elif latents.ndim == 3:
                # Advanced usage: already packed tokens (B, tokens, C). In this case we can't safely replace
                # per-frame latents here, so we only return the provided latents with the computed denoise mask.
                if latents.shape[:2] != denoise_mask_tokens.shape:
                    raise ValueError(
                        f"Provided `latents` tokens have shape {tuple(latents.shape)}, but expected "
                        f"{tuple(denoise_mask_tokens.shape) + (num_channels_latents,)}."
                    )
                # No safe way to compute clean latents in token space here.
                return latents.to(device=device, dtype=dtype), denoise_mask_tokens, None
            else:
                raise ValueError(
                    f"Provided `latents` must be either packed tokens (ndim=3) or a latent grid (ndim=5). "
                    f"Got ndim={latents.ndim} with shape={tuple(latents.shape)}."
                )
        else:
            base_latent_grid = torch.zeros(shape, device=device, dtype=dtype)

        if image is None:
            # If latents were provided, allow running without re-injecting images (still returns the denoise mask).
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # Match ltx-core GaussianNoiser: scaled_mask = denoise_mask * noise_scale
            scaled_mask = denoise_mask_grid * float(noise_scale)
            latents_grid = noise * scaled_mask + base_latent_grid * (1.0 - scaled_mask)
            latents_tokens = self._pack_latents(
                latents_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
            clean_latents_tokens = self._pack_latents(
                base_latent_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )
            return latents_tokens, denoise_mask_tokens, clean_latents_tokens

        # Inject each conditioning into the base latent grid at its frame index (replace semantics, like ltx-pipelines).

        # If generator is a list (per-batch), pick the first for deterministic VAE-encode (sample_mode='mode' usually ignores it anyway).
        encode_generator = generator[0] if isinstance(generator, list) and len(generator) > 0 else generator

        for k in range(num_conds):
            img = image[k]
            encoded = self.vae_encode(
                img.unsqueeze(0).unsqueeze(2),
                sample_generator=encode_generator,
                offload=offload,
                sample_mode="mode",
            ).to(dtype)
            encoded = encoded.repeat(batch_size, 1, 1, 1, 1)
            f_idx = int(frame_indices_t[k].item())

            base_latent_grid[:, :, f_idx : f_idx + 1] = encoded

        clean_latents_tokens = self._pack_latents(
            base_latent_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # Match ltx-core GaussianNoiser: scaled_mask = denoise_mask * noise_scale
        scaled_mask = denoise_mask_grid * float(noise_scale)
        latents_grid = noise * scaled_mask + base_latent_grid * (1.0 - scaled_mask)

        latents_tokens = self._pack_latents(
            latents_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        return latents_tokens, denoise_mask_tokens, clean_latents_tokens

    def prepare_audio_latents(
        self,
        audio: InputAudio | List[InputAudio] = None,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        num_mel_bins: int = 64,
        num_frames: int = 121,
        frame_rate: float = 25.0,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        strengths: Optional[Union[float, List[float]]] = None,
        range_indices: Optional[Union[tuple[int, int], List[tuple[int, int]]]] = None,
        offload: bool = True,
    ) -> tuple[torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]]:
        duration_s = num_frames / frame_rate
        latents_per_second = (
            float(sampling_rate) / float(hop_length) / float(self.audio_vae_temporal_compression_ratio)
        )
        latent_length = round(duration_s * latents_per_second)
        # TODO: confirm whether this logic is correct
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        token_dim = int(num_channels_latents * latent_mel_bins)
        tokens_shape = (batch_size, latent_length, token_dim)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Base *clean* tokens: either from provided latents or empty (zeros).
        if latents is not None:
            if latents.ndim == 4:
                expected_grid = (batch_size, num_channels_latents, latent_length, latent_mel_bins)
                if tuple(latents.shape) != expected_grid:
                    raise ValueError(
                        f"Provided `latents` grid has shape {tuple(latents.shape)}, but expected {expected_grid}."
                    )
                clean_latents_tokens = self._pack_audio_latents(latents.to(device=device, dtype=dtype))
            elif latents.ndim == 3:
                if tuple(latents.shape) != tokens_shape:
                    raise ValueError(
                        f"Provided `latents` tokens have shape {tuple(latents.shape)}, but expected {tokens_shape}."
                    )
                clean_latents_tokens = latents.to(device=device, dtype=dtype)
            else:
                raise ValueError(
                    f"Provided `latents` must be either packed tokens (ndim=3) or a latent grid (ndim=4). "
                    f"Got ndim={latents.ndim} with shape={tuple(latents.shape)}."
                )
        else:
            clean_latents_tokens = torch.zeros(tokens_shape, device=device, dtype=dtype)

        # Start with a full-denoise mask (1.0) in token space: [B, L]
        denoise_mask_tokens = torch.ones((batch_size, latent_length), device=device, dtype=dtype)

        # If no audio provided, just return noisy version of current clean tokens (with mask applied).
        if audio is None:
            noise_tokens = randn_tensor(tokens_shape, generator=generator, device=device, dtype=dtype)
            scaled_mask_tokens = denoise_mask_tokens * float(noise_scale)
            latents_tokens = noise_tokens * scaled_mask_tokens.unsqueeze(-1) + clean_latents_tokens * (
                1.0 - scaled_mask_tokens.unsqueeze(-1)
            )
            return latents_tokens, latent_length, denoise_mask_tokens, clean_latents_tokens

        # Normalize to list of audio conditionings.
        audio_list = audio if isinstance(audio, list) else [audio]
        num_conds = len(audio_list)

        # Normalize strengths to per-conditioning list/tensor.
        if strengths is None:
            strengths_t = torch.full((num_conds,), 1.0, device=device, dtype=torch.float32)
        elif isinstance(strengths, (int, float)):
            strengths_t = torch.full((num_conds,), float(strengths), device=device, dtype=torch.float32)
        else:
            if len(strengths) != num_conds:
                raise ValueError(
                    f"Provided `strengths` has length {len(strengths)}, but expected length {num_conds}."
                )
            strengths_t = torch.tensor([float(s) for s in strengths], device=device, dtype=torch.float32)
        strengths_t = strengths_t.clamp(0.0, 1.0).to(dtype=dtype)

        # Normalize range indices: list of (start, end) in *audio latent index* space.
        # If not provided:
        # - single audio => [0, min(cond_len, latent_length)]
        # - multiple audios => place them back-to-back (each range length is its encoded latent length, clipped to remaining).
        if range_indices is None:
            ranges: List[tuple[int, int]] = []
            cursor = 0
            for k in range(num_conds):
                if cursor >= latent_length:
                    ranges.append((latent_length, latent_length))
                    continue
                # We'll decide range length after encoding (based on the audio's encoded latent length).
                ranges.append((cursor, latent_length))  # provisional end; fixed per-item below
                cursor = latent_length  # sentinel; corrected later
        elif isinstance(range_indices, tuple):
            if num_conds != 1:
                raise ValueError(
                    f"Provided a single `range_indices` tuple, but got {num_conds} audio conditionings. "
                    f"Provide a list of ranges (one per audio)."
                )
            ranges = [range_indices]
        else:
            if len(range_indices) != num_conds:
                raise ValueError(
                    f"Provided `range_indices` has length {len(range_indices)}, but expected length {num_conds}."
                )
            ranges = list(range_indices)

        # Keep audio_vae on device while encoding + normalizing all conditionings, then optionally offload once.
        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
        self.to_device(self.audio_vae)
        encode_generator = generator[0] if isinstance(generator, list) and len(generator) > 0 else generator

        for k in range(num_conds):
            aud_grid = self.encode_audio_latents_grid_(
                audio=audio_list[k], generator=encode_generator, offload=False
            ).to(device=device, dtype=dtype)

            # Resolve this conditioning's range.
            start, end = ranges[k]
            start = int(start)
            end = int(end)
            if range_indices is None:
                # Default: back-to-back by encoded latent length (clipped to remaining).
                # For k==0, start is 0; for later items, we'll recompute start as end of previous.
                if k == 0:
                    start = 0
                else:
                    start = ranges[k - 1][1]
                seg_len = int(min(int(aud_grid.shape[2]), max(0, latent_length - start)))
                end = start + seg_len
                ranges[k] = (start, end)
            else:
                if start < 0 or end < 0:
                    raise ValueError(f"All `range_indices` must be >= 0. Got ({start}, {end}).")
                if start > end:
                    raise ValueError(f"`range_indices` must be (start <= end). Got ({start}, {end}).")
                if start > latent_length or end > latent_length:
                    raise ValueError(
                        f"`range_indices` must be within [0, {latent_length}]. Got ({start}, {end})."
                    )

            if start == end:
                continue

            seg_len = end - start

            # Crop/pad time axis to segment length.
            if aud_grid.shape[2] < seg_len:
                pad_t = seg_len - aud_grid.shape[2]
                aud_grid = torch.cat(
                    [
                        aud_grid,
                        torch.zeros(
                            aud_grid.shape[0],
                            aud_grid.shape[1],
                            pad_t,
                            aud_grid.shape[3],
                            dtype=aud_grid.dtype,
                            device=aud_grid.device,
                        ),
                    ],
                    dim=2,
                )
            elif aud_grid.shape[2] > seg_len:
                aud_grid = aud_grid[:, :, :seg_len, :]

            # Patchify to tokens and normalize (matches `prepare_audio_latents_` behavior).
            cond_tokens = self._pack_audio_latents(aud_grid)  # [1, seg_len, token_dim]
            cond_tokens = self.audio_vae.normalize_latents(cond_tokens)
            cond_tokens = cond_tokens.repeat(batch_size, 1, 1)  # [B, seg_len, token_dim]
            


            # Replace clean latents + denoise mask over the selected range.
            clean_latents_tokens[:, start:end] = cond_tokens
            denoise_mask_tokens[:, start:end] = 1.0 - strengths_t[k]

        if offload:
            self._offload("audio_vae")

        noise_tokens = randn_tensor(tokens_shape, generator=generator, device=device, dtype=dtype)
        scaled_mask_tokens = denoise_mask_tokens * float(noise_scale)
        latents_tokens = noise_tokens * scaled_mask_tokens.unsqueeze(-1) + clean_latents_tokens * (
            1.0 - scaled_mask_tokens.unsqueeze(-1)
        )

        return latents_tokens, latent_length, denoise_mask_tokens, clean_latents_tokens

    def run(
        self,
        audio: InputAudio | List[InputAudio] = None,
        image: InputImage | List[InputImage] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        duration: str | int = 121,
        fps: float = 25.0,
        num_inference_steps: int = 40,
        timesteps: List[int] = None,
        use_distilled_stage_1: bool = False,
        use_distilled_stage_2: bool = False,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        audio_strengths: Optional[Union[float, List[float]]] = None,
        audio_range_indices: Optional[Union[tuple[int, int], List[tuple[int, int]]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 1024,
        offload: bool = True,
        return_latents: bool = False,
        upsample: bool = True,
        image_strengths: Optional[Union[float, List[float]]] = None,
        image_pixel_frame_indices: Optional[Union[int, List[int]]] = None,
        use_gradient_estimation: bool = False,
        ge_gamma: float = 2.0,
        empty_cache_after_step: bool = True,
        cfg_sequential: bool = False,
        chunking_profile: str = "none",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs,
    ):
        
        # Progress mapping:
        # - If `upsample=True` (stage-1 + stage-2 refinement), map stage-1 to [0.00, 0.90] and stage-2 to [0.90, 1.00]
        #   so progress doesn't "reset" mid-run.
        if upsample and not use_distilled_stage_2:
            stage1_progress_callback = make_mapped_progress(progress_callback, 0.00, 0.90)
            stage2_progress_callback = make_mapped_progress(progress_callback, 0.90, 1.00)
        else:
            stage1_progress_callback = progress_callback
            stage2_progress_callback = progress_callback

        safe_emit_progress(stage1_progress_callback, 0.0, "Starting text-to-image-to-video pipeline")
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        num_frames = self._parse_num_frames(duration, fps)

      
        
        # Image conditioning should be applied in BOTH stages (stage-2 starts from upscaled latents but still
        # replaces latent slices + denoise mask), matching ltx-pipelines distilled behavior.
        height = round(height / self.vae_spatial_compression_ratio) * self.vae_spatial_compression_ratio
        width = round(width / self.vae_spatial_compression_ratio) * self.vae_spatial_compression_ratio
        
        
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None
        target_height = height
        target_width = width
        
        if upsample:
            height = target_height // 2
            width = target_width // 2
            
        
        if image is not None:
            safe_emit_progress(stage1_progress_callback, 0.03, "Loading and preprocessing conditioning image(s)")
            # Multiple images are treated as multiple conditionings for a *single* latent sequence (not batch items).
            if not isinstance(image, list):
                image = [image]
            image = [self._load_image(img) for img in image]
            image = [self._aspect_ratio_resize(img, max_area=height * width, mod_value=self.vae_spatial_compression_ratio)[0] for img in image]
            if not use_distilled_stage_2:
                width, height = image[0].size
                
            condition_images = [self.video_processor.preprocess(img, height=height, width=width) for img in image]
            condition_images = torch.cat(condition_images, dim=0)
        else:
            condition_images = None
            

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        
        if self.preloaded_loras and "ltx-2-19b-distilled-lora-384" in self.preloaded_loras and not use_distilled_stage_2:
            self.logger.info("Disabling LTX2 19B Distilled LoRA 384 with scale 0.0")
            self._previous_lora_scale = self.preloaded_loras["ltx-2-19b-distilled-lora-384"].scale
            self.preloaded_loras["ltx-2-19b-distilled-lora-384"].scale = 0.0

        # 3. Prepare text embeddings
        safe_emit_progress(stage1_progress_callback, 0.08, "Encoding prompt")
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
            offload=offload,
        )
        
        if self.do_classifier_free_guidance:
            # We still build the combined [uncond; cond] embedding stack for connectors, but optionally
            # avoid duplicating the *latents* batch during denoising by running two forward passes.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            
        connectors = self.helpers["connectors"]
        self.to_device(connectors)

        additive_attention_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        safe_emit_progress(stage1_progress_callback, 0.12, "Running connector(s)")
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = connectors(
            prompt_embeds, additive_attention_mask, additive_mask=True
        )
        
        del connectors
        if offload:
            self._offload("connectors")
            


        # 4. Prepare latent variables
        safe_emit_progress(stage1_progress_callback, 0.18, "Preparing latents")
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width
        
        transformer_config = self.load_config_by_type("transformer")
        num_channels_latents = transformer_config.in_channels
        
        

        clean_latents = None
        freeze_latent_frame_indices: Optional[List[int]] = None
        
        # Determine if we should use image conditioning logic:
        # - condition_images is provided (normal case), OR
        # - latents are provided with conditioning info (stage 2 refinement case - use upsampled latents as conditioning)
        use_image_conditioning = (
            condition_images is not None
            or (latents is not None and (image_strengths is not None or image_pixel_frame_indices is not None))
        )
        
        if use_image_conditioning:
            latents, denoise_mask, clean_latents = self.prepare_latents_image_conditioning(
                image=condition_images,
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                noise_scale=noise_scale,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents,
                strengths=image_strengths,
                pixel_frame_indices=image_pixel_frame_indices,
                offload=offload,
            )
            
            denoise_mask_model = (
                torch.cat([denoise_mask, denoise_mask], dim=0) if self.do_classifier_free_guidance else denoise_mask
            )

            # Build a list of latent-frame indices to freeze (i.e. never denoise), matching the "diffusers slicing"
            # approach: only update the frames we want to denoise.
            #
            # Supports multiple conditioned images/frames. We freeze frames whose strength is (approximately) 1.0.
            try:
                # Determine num_conds from condition_images if available, otherwise from strengths/indices
                if condition_images is not None:
                    num_conds = int(condition_images.shape[0])
                elif isinstance(image_strengths, list):
                    num_conds = len(image_strengths)
                elif isinstance(image_pixel_frame_indices, list):
                    num_conds = len(image_pixel_frame_indices)
                else:
                    num_conds = 1
                if image_strengths is None:
                    strengths_list = [1.0] * num_conds
                elif isinstance(image_strengths, (int, float)):
                    strengths_list = [float(image_strengths)] * num_conds
                else:
                    strengths_list = [float(s) for s in image_strengths]
                    if len(strengths_list) != num_conds:
                        raise ValueError(
                            f"`image_strengths` length {len(strengths_list)} must match number of conditioned images {num_conds}."
                        )

                if image_pixel_frame_indices is None:
                    idx_list = [0] * num_conds
                elif isinstance(image_pixel_frame_indices, int):
                    idx_list = [int(image_pixel_frame_indices)] * num_conds
                else:
                    idx_list = [int(i) for i in image_pixel_frame_indices]
                    if len(idx_list) != num_conds:
                        raise ValueError(
                            f"`image_pixel_frame_indices` length {len(idx_list)} must match number of conditioned images {num_conds}."
                        )

                # Heuristic matches prepare_latents_image_conditioning:
                # - if max idx fits in latent frame range => treat as latent indices
                # - else treat as pixel indices and convert to latent indices
                if len(idx_list) > 0:
                    if max(idx_list) < latent_num_frames:
                        latent_idx_list = idx_list
                    else:
                        latent_idx_list = [i // int(self.vae_temporal_compression_ratio) for i in idx_list]
                else:
                    latent_idx_list = []

                freeze = set()
                for li, s in zip(latent_idx_list, strengths_list):
                    if s >= 0.999:  # treat ~1.0 as "fully frozen"
                        freeze.add(int(li))
                freeze_latent_frame_indices = sorted([i for i in freeze if 0 <= i < latent_num_frames])
            except Exception as e:
                # If anything goes wrong, fall back to mask-based re-imposition only.
                self.logger.warning(f"Failed to compute freeze_latent_frame_indices; falling back to mask-only. Error: {e}")
                freeze_latent_frame_indices = None
        else:
            latents, denoise_mask = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                noise_scale,
                torch.float32,
                device,
                generator,
                latents,
                offload,
            )
            denoise_mask_model = None

        num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        
        num_channels_latents_audio = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )
        
        audio_latents, audio_num_frames, audio_denoise_mask, audio_clean_latents = self.prepare_audio_latents(
            audio=audio,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            num_mel_bins=num_mel_bins,
            num_frames=num_frames,  # Video frames, audio frames will be calculated from this
            frame_rate=fps,
            sampling_rate=self.audio_sampling_rate,
            hop_length=self.audio_hop_length,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
            noise_scale=noise_scale,
            strengths=audio_strengths,
            range_indices=audio_range_indices,
        )

        audio_denoise_mask_model = (
            torch.cat([audio_denoise_mask, audio_denoise_mask], dim=0)
            if self.do_classifier_free_guidance
            else audio_denoise_mask
        )

        # 5. Prepare timesteps
        safe_emit_progress(stage1_progress_callback, 0.28, "Preparing timesteps")
        if use_distilled_stage_1:
            # Diffusers FlowMatchEulerDiscreteScheduler expects `sigmas` to be the per-step schedule
            # and will append a terminal 0 internally. Our config lists include that terminal 0 already
            # (matching ltx-core/ltx-pipelines), so we must drop it here to avoid an extra step.
            sigmas = self.distilled_stage_1_sigma_values
            if len(sigmas) > 0 and float(sigmas[-1]) == 0.0:
                sigmas = sigmas[:-1]
        elif use_distilled_stage_2:
            sigmas = self.distilled_stage_2_sigma_values
            if len(sigmas) > 0 and float(sigmas[-1]) == 0.0:
                sigmas = sigmas[:-1]
        else:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        
        if not use_distilled_stage_2 and not use_distilled_stage_1:
            mu = self.calculate_shift(
                video_sequence_length,
                self.scheduler.config.get("base_image_seq_len", 1024),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.95),
                self.scheduler.config.get("max_shift", 2.05),
            )
        else:
            mu = None
            self.scheduler.config.use_dynamic_shifting = False
        
        self.scheduler.set_shift(1.0)  # critical: prevents the extra static shift
        self.scheduler.config.shift_terminal = None
        self.scheduler.config.use_karras_sigmas = False
        self.scheduler.config.use_exponential_sigmas = False
        self.scheduler.config.use_beta_sigmas = False
        self.scheduler.config.invert_sigmas = False

        # For now, duplicate the scheduler for use with the audio latents in the default (scheduler.step) path.
        audio_scheduler = None
        if not use_gradient_estimation:
            audio_scheduler = copy.deepcopy(self.scheduler)
            _, _ = self._get_timesteps(
                audio_scheduler,
                num_inference_steps,
                timesteps,
                sigmas=sigmas,
                mu=mu,
            )

        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)


        if chunking_profile != "none":
            self.transformer.set_chunking_profile(chunking_profile)
        
        if self.preloaded_loras and "ltx-2-19b-distilled-lora-384" in self.preloaded_loras and use_distilled_stage_2:
            self.logger.info(f"Applying LTX2 19B Distilled LoRA 384 with scale {getattr(self, '_previous_lora_scale', 1.0)}")
            lora = self.preloaded_loras["ltx-2-19b-distilled-lora-384"]
            lora.scale = getattr(self, "_previous_lora_scale", 1.0)
            self.apply_loras([(lora.source, lora.scale)], adapter_names=[lora.name])

        # Pre-compute video and audio positional ids as they will be the same at each step of the denoising loop
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=fps
        )
        
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device, fps=fps
        )

        # 7. Denoising loop
        # NOTE: FlowMatchEulerDiscreteScheduler operates over `scheduler.sigmas` (includes a terminal 0).
        # For gradient estimation we implement the Euler update directly, matching ltx-pipelines:
        #   v_k = (x_k - x0_k) / sigma_k,  v_total = ge_gamma*(v_k - v_{k-1}) + v_{k-1}
        #   x_{k+1} = x_k + (sigma_{k+1} - sigma_k) * v_total
        self._latent_num_frames = latent_num_frames
        self._latent_height = latent_height
        self._latent_width = latent_width
        self._audio_num_frames = audio_num_frames
        self._latent_mel_bins = latent_mel_bins
        
        sigmas_t = getattr(self.scheduler, "sigmas", None)
        if use_gradient_estimation:
            if getattr(getattr(self.scheduler, "config", None), "stochastic_sampling", False):
                self.logger.warning(
                    "Gradient-estimation sampling requested, but scheduler.config.stochastic_sampling=True. "
                    "Falling back to default Euler scheduler.step."
                )
                use_gradient_estimation = False
            elif sigmas_t is None:
                raise ValueError("Gradient-estimation sampling requires scheduler.sigmas to be available.")

        if use_gradient_estimation:
            # Ensure sigmas are on device and float32 for stable scalar math.
            if not torch.is_tensor(sigmas_t):
                sigmas_t = torch.tensor(sigmas_t, device=device, dtype=torch.float32)
            else:
                sigmas_t = sigmas_t.to(device=device, dtype=torch.float32)

            # In FlowMatchEulerDiscreteScheduler, `sigmas` has length N+1, while `timesteps` has length N.
            num_steps = int(sigmas_t.shape[0]) - 1
            timesteps_loop = timesteps[:num_steps]

            prev_video_velocity = None
            prev_video_velocity_grid = None
            prev_audio_velocity = None

            denoise_progress_callback = make_mapped_progress(stage1_progress_callback, 0.50, 0.90)
            safe_emit_progress(stage1_progress_callback, 0.45, "Starting denoise (gradient-estimation)")
            with self._progress_bar(total=num_steps) as progress_bar:
                for i, t in enumerate(timesteps_loop):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    # Predict velocity (GE mode).
                    # If CFG is enabled, the pipeline expects a "virtual 2x batch" layout: [uncond; cond].
                    # With `cfg_sequential=True`, we compute those two halves sequentially (batch=B each),
                    # then concatenate back to 2B so indexing/chunking semantics remain intact.
                    if self.do_classifier_free_guidance and cfg_sequential:
                        bsz = latents.shape[0]
                        uncond_idx = slice(0, bsz)
                        cond_idx = slice(bsz, 2 * bsz)

                        latent_model_input = latents.to(prompt_embeds.dtype)
                        audio_latent_model_input = audio_latents.to(prompt_embeds.dtype)

                        timestep_2b = t.expand(2 * bsz)
                        if denoise_mask_model is not None:
                            video_timestep_2b = timestep_2b.unsqueeze(-1) * denoise_mask_model
                        else:
                            video_timestep_2b = timestep_2b

                        if audio_denoise_mask_model is not None:
                            audio_timestep_2b = timestep_2b.unsqueeze(-1) * audio_denoise_mask_model
                        else:
                            audio_timestep_2b = timestep_2b

                        with self.transformer.cache_context("cond_uncond"):
                            vel_video_uncond, vel_audio_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=connector_prompt_embeds[uncond_idx],
                                audio_encoder_hidden_states=connector_audio_prompt_embeds[uncond_idx],
                                timestep=video_timestep_2b[uncond_idx],
                                audio_timestep=audio_timestep_2b[uncond_idx],
                                encoder_attention_mask=connector_attention_mask[uncond_idx],
                                audio_encoder_attention_mask=connector_attention_mask[uncond_idx],
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=fps,
                                audio_num_frames=audio_num_frames,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )
                            vel_video_text, vel_audio_text = self.transformer(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=connector_prompt_embeds[cond_idx],
                                audio_encoder_hidden_states=connector_audio_prompt_embeds[cond_idx],
                                timestep=video_timestep_2b[cond_idx],
                                audio_timestep=audio_timestep_2b[cond_idx],
                                encoder_attention_mask=connector_attention_mask[cond_idx],
                                audio_encoder_attention_mask=connector_attention_mask[cond_idx],
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=fps,
                                audio_num_frames=audio_num_frames,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )

                        vel_pred_video = torch.cat([vel_video_uncond, vel_video_text], dim=0).float()
                        vel_pred_audio = torch.cat([vel_audio_uncond, vel_audio_text], dim=0).float()
                    else:
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                        audio_latent_model_input = (
                            torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                        )
                        audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

                        timestep_2b = t.expand(latent_model_input.shape[0])
                        if denoise_mask_model is not None:
                            video_timestep_2b = timestep_2b.unsqueeze(-1) * denoise_mask_model
                        else:
                            video_timestep_2b = timestep_2b

                        if audio_denoise_mask_model is not None:
                            audio_timestep_2b = timestep_2b.unsqueeze(-1) * audio_denoise_mask_model
                        else:
                            audio_timestep_2b = timestep_2b

                        with self.transformer.cache_context("cond_uncond"):
                            vel_pred_video, vel_pred_audio = self.transformer(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=connector_prompt_embeds,
                                audio_encoder_hidden_states=connector_audio_prompt_embeds,
                                timestep=video_timestep_2b,
                                audio_timestep=audio_timestep_2b,
                                encoder_attention_mask=connector_attention_mask,
                                audio_encoder_attention_mask=connector_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=fps,
                                audio_num_frames=audio_num_frames,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )
                        vel_pred_video = vel_pred_video.float()
                        vel_pred_audio = vel_pred_audio.float()

                    if self.do_classifier_free_guidance:
                        vel_video_uncond, vel_video_text = vel_pred_video.chunk(2)
                        vel_pred_video = vel_video_uncond + self.guidance_scale * (vel_video_text - vel_video_uncond)

                        vel_audio_uncond, vel_audio_text = vel_pred_audio.chunk(2)
                        vel_pred_audio = vel_audio_uncond + self.guidance_scale * (vel_audio_text - vel_audio_uncond)

                        if self.guidance_rescale > 0:
                            vel_pred_video = self.rescale_noise_cfg(
                                vel_pred_video, vel_video_text, guidance_rescale=self.guidance_rescale
                            )
                            vel_pred_audio = self.rescale_noise_cfg(
                                vel_pred_audio, vel_audio_text, guidance_rescale=self.guidance_rescale
                            )

                    sigma = sigmas_t[i]
                    sigma_next = sigmas_t[i + 1]
                    dt = sigma_next - sigma

                    # Build denoised (x0) from current sample and predicted velocity.
                    # FlowMatchEulerDiscreteScheduler uses: x0 = sample - sigma * velocity
                    denoised_video = latents - sigma * vel_pred_video
                    denoised_audio = audio_latents - sigma * vel_pred_audio

                    # Post-process denoised with hard-conditioning (ltx-core semantics):
                    # denoise_mask=1 => denoise/noise; denoise_mask=0 => keep clean latent.
                    if denoise_mask is not None and clean_latents is not None:
                        m = denoise_mask.unsqueeze(-1)
                        denoised_video = (denoised_video * m + clean_latents.float() * (1.0 - m)).to(denoised_video.dtype)

                    # Convert to velocities (v = (x - x0) / sigma) and apply gradient estimation.
                    if float(sigma.item()) == 0.0:
                        raise ValueError("Sigma can't be 0.0 during gradient-estimation update.")

                    # If requested, freeze fully-conditioned latent frames (diffusers slicing approach) in GE mode too.
                    # We do the video GE update in unpacked grid space so we can freeze frame indices precisely.
                    if freeze_latent_frame_indices is not None and len(freeze_latent_frame_indices) > 0:
                        latents_grid = self._unpack_latents(
                            latents,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )
                        denoised_video_grid = self._unpack_latents(
                            denoised_video,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )

                        freeze_set = set(int(i) for i in freeze_latent_frame_indices)
                        freeze_idx = [fi for fi in range(latent_num_frames) if fi in freeze_set]

                        # Ensure frozen frames never change by forcing x0 == x for those frames (=> velocity 0).
                        if len(freeze_idx) > 0:
                            denoised_video_grid[:, :, freeze_idx] = latents_grid[:, :, freeze_idx]

                        cur_video_velocity_grid = (latents_grid - denoised_video_grid) / sigma

                        if prev_video_velocity_grid is not None:
                            total_video_velocity_grid = (
                                ge_gamma * (cur_video_velocity_grid - prev_video_velocity_grid) + prev_video_velocity_grid
                            )
                        else:
                            total_video_velocity_grid = cur_video_velocity_grid

                        denoised_video_grid = latents_grid - sigma * total_video_velocity_grid

                        # If this is the final sigma -> 0 transition, return x0 directly (matches ltx-pipelines behavior).
                        if float(sigma_next.item()) == 0.0:
                            latents = self._pack_latents(
                                denoised_video_grid.to(latents_grid.dtype),
                                self.transformer_spatial_patch_size,
                                self.transformer_temporal_patch_size,
                            ).to(latents.dtype)
                        else:
                            latents = self._pack_latents(
                                (latents_grid + dt * total_video_velocity_grid).to(latents_grid.dtype),
                                self.transformer_spatial_patch_size,
                                self.transformer_temporal_patch_size,
                            ).to(latents.dtype)

                        prev_video_velocity_grid = cur_video_velocity_grid
                        # Keep token-space path variables consistent (not used when freezing is active).
                        cur_video_velocity = None
                        total_video_velocity = None
                    else:
                        cur_video_velocity = (latents - denoised_video) / sigma
                    cur_audio_velocity = (audio_latents - denoised_audio) / sigma

                    if freeze_latent_frame_indices is None or len(freeze_latent_frame_indices) == 0:
                        if prev_video_velocity is not None:
                            total_video_velocity = ge_gamma * (cur_video_velocity - prev_video_velocity) + prev_video_velocity
                            denoised_video = latents - sigma * total_video_velocity
                        else:
                            total_video_velocity = cur_video_velocity

                    if prev_audio_velocity is not None:
                        total_audio_velocity = ge_gamma * (cur_audio_velocity - prev_audio_velocity) + prev_audio_velocity
                        denoised_audio = audio_latents - sigma * total_audio_velocity
                    else:
                        total_audio_velocity = cur_audio_velocity

                    # If this is the final sigma -> 0 transition, return x0 directly (matches ltx-pipelines behavior).
                    if float(sigma_next.item()) == 0.0:
                        if freeze_latent_frame_indices is None or len(freeze_latent_frame_indices) == 0:
                            latents = denoised_video.to(latents.dtype)
                        audio_latents = denoised_audio.to(audio_latents.dtype)
                        progress_bar.update()
                        break

                    # Euler step in sigma-space.
                    if freeze_latent_frame_indices is None or len(freeze_latent_frame_indices) == 0:
                        latents = (latents + dt * total_video_velocity).to(latents.dtype)
                    audio_latents = (audio_latents + dt * total_audio_velocity).to(audio_latents.dtype)

                    # Track previous velocities (use the *current* velocity, not the corrected one).
                    if freeze_latent_frame_indices is None or len(freeze_latent_frame_indices) == 0:
                        prev_video_velocity = cur_video_velocity
                    prev_audio_velocity = cur_audio_velocity

                    # call the callback, if provided
                    if i == len(timesteps_loop) - 1 or (
                        (i + 1) % max(int(getattr(self.scheduler, "order", 1)), 1) == 0
                    ):
                        progress_bar.update()
                        denoise_progress_callback(
                            float(i + 1) / float(max(num_steps, 1)),
                            f"Denoising step {i + 1}/{num_steps}",
                        )
                
                    if empty_cache_after_step:
                        empty_cache()
        else:
            # Default (traditional Euler) scheduler stepping.
            # For now, duplicate the scheduler for use with the audio latents
            if audio_scheduler is None:
                audio_scheduler = copy.deepcopy(self.scheduler)
                _, _ = self._get_timesteps(
                    audio_scheduler,
                    num_inference_steps,
                    timesteps,
                    sigmas=sigmas,
                    mu=mu,
                )

            denoise_progress_callback = make_mapped_progress(stage1_progress_callback, 0.50, 0.90)
            safe_emit_progress(stage1_progress_callback, 0.45, "Starting denoise")
            with self._progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    # Predict noise (default mode).
                    # If CFG is enabled, the pipeline expects a "virtual 2x batch" layout: [uncond; cond].
                    # With `cfg_sequential=True`, we compute those two halves sequentially (batch=B each),
                    # then concatenate back to 2B so indexing/chunking semantics remain intact.
                    if self.do_classifier_free_guidance and cfg_sequential:
                        bsz = latents.shape[0]
                        uncond_idx = slice(0, bsz)
                        cond_idx = slice(bsz, 2 * bsz)

                        latent_model_input = latents.to(prompt_embeds.dtype)
                        audio_latent_model_input = audio_latents.to(prompt_embeds.dtype)

                        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        timestep_2b = t.expand(2 * bsz)
                        if denoise_mask_model is not None:
                            video_timestep_2b = timestep_2b.unsqueeze(-1) * denoise_mask_model
                        else:
                            video_timestep_2b = timestep_2b

                        if audio_denoise_mask_model is not None:
                            audio_timestep_2b = timestep_2b.unsqueeze(-1) * audio_denoise_mask_model
                        else:
                            audio_timestep_2b = timestep_2b
                            
                       
                        with self.transformer.cache_context("cond_uncond"):
                            noise_pred_video_uncond, noise_pred_audio_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=connector_prompt_embeds[uncond_idx],
                                audio_encoder_hidden_states=connector_audio_prompt_embeds[uncond_idx],
                                timestep=video_timestep_2b[uncond_idx],
                                audio_timestep=audio_timestep_2b[uncond_idx],
                                encoder_attention_mask=connector_attention_mask[uncond_idx],
                                audio_encoder_attention_mask=connector_attention_mask[uncond_idx],
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=fps,
                                audio_num_frames=audio_num_frames,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )
                            noise_pred_video_text, noise_pred_audio_text = self.transformer(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=connector_prompt_embeds[cond_idx],
                                audio_encoder_hidden_states=connector_audio_prompt_embeds[cond_idx],
                                timestep=video_timestep_2b[cond_idx],
                                audio_timestep=audio_timestep_2b[cond_idx],
                                encoder_attention_mask=connector_attention_mask[cond_idx],
                                audio_encoder_attention_mask=connector_attention_mask[cond_idx],
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=fps,
                                audio_num_frames=audio_num_frames,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )

                        noise_pred_video = torch.cat([noise_pred_video_uncond, noise_pred_video_text], dim=0).float()
                        noise_pred_audio = torch.cat([noise_pred_audio_uncond, noise_pred_audio_text], dim=0).float()
                    else:
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                        audio_latent_model_input = (
                            torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                        )
                        audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

                        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        timestep_2b = t.expand(latent_model_input.shape[0])
                        if denoise_mask_model is not None:
                            video_timestep_2b = timestep_2b.unsqueeze(-1) * denoise_mask_model
                        else:
                            video_timestep_2b = timestep_2b

                        if audio_denoise_mask_model is not None:
                            audio_timestep_2b = timestep_2b.unsqueeze(-1) * audio_denoise_mask_model
                        else:
                            audio_timestep_2b = timestep_2b
                        
                        
                 
                        with self.transformer.cache_context("cond_uncond"):
                            noise_pred_video, noise_pred_audio = self.transformer(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=connector_prompt_embeds,
                                audio_encoder_hidden_states=connector_audio_prompt_embeds,
                                timestep=video_timestep_2b,
                                audio_timestep=audio_timestep_2b,
                                encoder_attention_mask=connector_attention_mask,
                                audio_encoder_attention_mask=connector_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=fps,
                                audio_num_frames=audio_num_frames,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )
                        noise_pred_video = noise_pred_video.float()
                        noise_pred_audio = noise_pred_audio.float()

                    if self.do_classifier_free_guidance:
                        noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
                        noise_pred_video = noise_pred_video_uncond + self.guidance_scale * (
                            noise_pred_video_text - noise_pred_video_uncond
                        )

                        noise_pred_audio_uncond, noise_pred_audio_text = noise_pred_audio.chunk(2)
                        noise_pred_audio = noise_pred_audio_uncond + self.guidance_scale * (
                            noise_pred_audio_text - noise_pred_audio_uncond
                        )

                        if self.guidance_rescale > 0:
                            # Based on 3.4. in https://huggingface.co/papers/2305.08891
                            noise_pred_video = self.rescale_noise_cfg(
                                noise_pred_video, noise_pred_video_text, guidance_rescale=self.guidance_rescale
                            )
                            noise_pred_audio = self.rescale_noise_cfg(
                                noise_pred_audio, noise_pred_audio_text, guidance_rescale=self.guidance_rescale
                            )

                    # compute the previous noisy sample x_t -> x_t-1
                    if freeze_latent_frame_indices is not None and len(freeze_latent_frame_indices) > 0:
                        # Unpack -> step only unfrozen frames -> stitch -> repack.
                        noise_pred_video_grid = self._unpack_latents(
                            noise_pred_video,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )
                        latents_grid = self._unpack_latents(
                            latents,
                            latent_num_frames,
                            latent_height,
                            latent_width,
                            self.transformer_spatial_patch_size,
                            self.transformer_temporal_patch_size,
                        )

                        freeze_set = set(int(i) for i in freeze_latent_frame_indices)
                        denoise_idx = [fi for fi in range(latent_num_frames) if fi not in freeze_set]
                        if len(denoise_idx) > 0:
                            noise_pred_slice = noise_pred_video_grid[:, :, denoise_idx]
                            latents_slice = latents_grid[:, :, denoise_idx]
                            pred_slice = self.scheduler.step(noise_pred_slice, t, latents_slice, return_dict=False)[0]
                            latents_grid[:, :, denoise_idx] = pred_slice

                        latents = self._pack_latents(
                            latents_grid, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                        )
                    else:
                        latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                    # NOTE: for now duplicate scheduler for audio latents in case self.scheduler sets internal state in
                    # the step method (such as _step_index)
                    audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

                    # Re-impose hard conditioning each step (matches ltx-pipelines `post_process_latent` semantics).
                    # Without this, the conditioned frame(s) can drift immediately after the first step.
                    if denoise_mask is not None and clean_latents is not None:
                        m = denoise_mask.unsqueeze(-1)
                        latents = (latents * m + clean_latents.float() * (1.0 - m)).to(latents.dtype)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)
                    ):
                        progress_bar.update()
                        denoise_progress_callback(
                            float(i + 1) / float(max(num_inference_steps, 1)),
                            f"Denoising step {i + 1}/{num_inference_steps}",
                        )
                    
                    if empty_cache_after_step:
                        empty_cache()
        
        if offload:
            self._offload("transformer")

        if upsample:
            safe_emit_progress(stage1_progress_callback, 0.92, "Upsampling latents (stage-2 prep)")
            upsampler = self.helpers["latent_upsampler"]
            self.to_device(upsampler)
            vae_dtype = self.component_dtypes["vae"]
            upsampler.to(dtype=vae_dtype)
            if not getattr(self, "video_vae", None):
                self.load_component_by_name("video_vae")
            self.to_device(self.video_vae)
            latents = self._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)
            latents = upsample_video(latents, self.video_vae, upsampler)
            del upsampler
            if offload:
                self._offload("latent_upsampler")
            if offload:
                self._offload("video_vae")
            
            
            # call run function again with the upsampled latents
            noise_scale = self.distilled_stage_2_sigma_values[0]
            target_height = latents.shape[3] * self.vae_spatial_compression_ratio
            target_width = latents.shape[4] * self.vae_spatial_compression_ratio

            # Stage 2 refinement: Do NOT pass image to avoid re-encoding at wrong resolution.
            # The upsampled latents already contain the conditioning from stage 1.
            # Pass image_strengths and image_pixel_frame_indices to maintain the denoise mask and freeze mechanism.
            safe_emit_progress(stage1_progress_callback, 0.98, "Starting stage-2 refinement")
            return self.run(
                image=image, 
                audio=audio,
                audio_strengths=audio_strengths,
                audio_range_indices=audio_range_indices,
                prompt=prompt,
                height=target_height,
                width=target_width,
                duration=duration,
                fps=fps,
                num_inference_steps=num_inference_steps,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                offload=offload,
                audio_latents=audio_latents if audio is not None else None,
                return_latents=return_latents,
                upsample=False,
                seed=seed,
                image_strengths=image_strengths,
                image_pixel_frame_indices=image_pixel_frame_indices,
                guidance_scale=1.0,
                guidance_rescale=0.0,
                use_distilled_stage_2=True,
                noise_scale=noise_scale,
                use_gradient_estimation=False,
                ge_gamma=0.0,
                progress_callback=stage2_progress_callback,
            )

        if return_latents:
            safe_emit_progress(stage1_progress_callback, 1.0, "Returning latents")
            return (latents, audio_latents)
        
        latents = self._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        # enable tiling
        self.logger.info("Enabling tiling for video VAE")
        self.video_vae.enable_tiling()
        
        latents = self.video_vae.denormalize_latents(
            latents
        )
        latents = latents.to(prompt_embeds.dtype)
        if not self.video_vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size
            timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                :, None, None, None, None
            ]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise
        latents = latents.to(self.video_vae.dtype)
        safe_emit_progress(stage1_progress_callback, 0.94, "Decoding video latents")
        video = self.video_vae.decode(latents, timestep, return_dict=False)[0]

        if offload:
            self._offload("video_vae")
            
        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
            
        self.to_device(self.audio_vae)
        audio_latents = audio_latents.to(self.audio_vae.dtype)
        audio_latents = self.audio_vae.denormalize_latents(
            audio_latents
        )

        audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)
        # enable tiling
        
        safe_emit_progress(stage1_progress_callback, 0.96, "Decoding audio latents")
        generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        
        if offload:
            self._offload("audio_vae")
            
        # load vocoder
        vocoder = self.helpers["vocoder"]
        self.to_device(vocoder)
        
        safe_emit_progress(stage1_progress_callback, 0.98, "Vocoder synthesis")
        audio = vocoder(generated_mel_spectrograms)
        
        if offload:
            self._offload("vocoder")

        video = self._convert_to_uint8(video).cpu()
        audio = audio.squeeze(0).cpu().float()
        
        safe_emit_progress(stage1_progress_callback, 1.0, "Completed text-to-image-to-video pipeline")
        
        return video, audio
    
