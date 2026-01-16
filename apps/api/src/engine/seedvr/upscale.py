"""
SeedVR Video Super-Resolution Engine Implementation.

This engine implements the SeedVR video upscaling inference flow,
adapted to work with the Apex engine architecture.
"""

import os
import gc
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from src.engine.base_engine import BaseEngine
from src.types import InputVideo, InputImage
from src.utils.cache import empty_cache
from src.utils.progress import safe_emit_progress, make_mapped_progress
from diffusers.video_processor import VideoProcessor
from src.engine.seedvr.shared.colorfix import wavelet_reconstruction
from src.utils.assets import get_asset_path

# Paths to pre-computed prompt embeddings (works in dev checkout and installed app bundle)
# Don't hard-fail at import time; fail when the engine is actually used.
POS_EMB_PATH = get_asset_path("seedvr", "pos_emb.pt", must_exist=False)
NEG_EMB_PATH = get_asset_path("seedvr", "neg_emb.pt", must_exist=False)


@dataclass
class SamplerModelArgs:
    """Arguments passed to the model function during sampling."""

    x_t: Tensor
    t: Tensor
    i: int


class LinearInterpolationSchedule:
    """
    Linear interpolation schedule (lerp) as used by flow matching and rectified flow.
    x_t = (1 - t/T) * x_0 + (t/T) * x_T
    """

    def __init__(self, T: float = 1000.0):
        self._T = T

    @property
    def T(self) -> float:
        return self._T

    def A(self, t: Tensor) -> Tensor:
        """Coefficient for x_0."""
        return 1 - (t / self.T)

    def B(self, t: Tensor) -> Tensor:
        """Coefficient for x_T (noise)."""
        return t / self.T

    def forward(self, x_0: Tensor, x_T: Tensor, t: Tensor) -> Tensor:
        """Diffusion forward: interpolate between x_0 and x_T."""
        t = self._expand_dims(t, x_0.ndim)
        return self.A(t) * x_0 + self.B(t) * x_T

    def convert_from_pred(
        self, pred: Tensor, pred_type: str, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Convert model prediction to x_0 and x_T."""
        t = self._expand_dims(t, x_t.ndim)
        A_t = self.A(t)
        B_t = self.B(t)

        if pred_type == "v_lerp":
            # v_lerp prediction type: pred = x_T - x_0
            pred_x_0 = (x_t - B_t * pred) / (A_t + B_t)
            pred_x_T = (x_t + A_t * pred) / (A_t + B_t)
        elif pred_type == "x_T":
            pred_x_T = pred
            pred_x_0 = (x_t - B_t * pred_x_T) / A_t
        elif pred_type == "x_0":
            pred_x_0 = pred
            pred_x_T = (x_t - A_t * pred_x_0) / B_t
        else:
            raise NotImplementedError(f"Unknown prediction type: {pred_type}")

        return pred_x_0, pred_x_T

    @staticmethod
    def _expand_dims(t: Tensor, ndim: int) -> Tensor:
        """Expand tensor dimensions for broadcasting."""
        while t.ndim < ndim:
            t = t.unsqueeze(-1)
        return t


class SeedVRUpscaleEngine(BaseEngine):
    """SeedVR Video Super-Resolution Engine."""

    # Default settings matching SeedVR INFERENCE script (not training config!)
    # See inference_seedvr_3b.py lines 93 and 138
    # Note: training config has different values (cfg=7.5, noise_scale=0.25)
    default_cfg_scale: float = 6.5  # Original inference uses 6.5
    default_cfg_rescale: float = 0.0
    default_sample_steps: int = 50
    default_cond_noise_scale: float = 0.1  # Original inference uses 0.1

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)

        # Initialize video processor
        # From vae config: spatial_downsample_factor: 8, temporal_downsample_factor: 4
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # VAE config from main.yaml:
        # vae.scaling_factor: 0.9152
        # vae.dtype: bfloat16
        self._vae_scaling_factor = 0.9152
        self._vae_shifting_factor = 0.0
        self._vae_dtype = torch.bfloat16

        # Diffusion config from main.yaml:
        # diffusion.schedule.type: lerp, T: 1000.0
        # diffusion.sampler.prediction_type: v_lerp
        self._schedule = LinearInterpolationSchedule(T=1000.0)
        self._prediction_type = "v_lerp"

        # CFG partial application (config default is 1.0 = always apply CFG)
        # diffusion.cfg.partial: 1 (not specified, defaults to 1)
        self._cfg_partial = 1.0

        # Pre-loaded embeddings cache
        self._pos_embeds: Optional[Tensor] = None
        self._neg_embeds: Optional[Tensor] = None

    # -------------------------------------------------------------------------
    # Prompt Embeddings
    # -------------------------------------------------------------------------

    def _load_prompt_embeddings(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Load pre-computed positive and negative prompt embeddings."""
        if self._pos_embeds is None:
            self._pos_embeds = torch.load(POS_EMB_PATH, map_location="cpu")
        if self._neg_embeds is None:
            self._neg_embeds = torch.load(NEG_EMB_PATH, map_location="cpu")

        return (
            self._pos_embeds.to(device),
            self._neg_embeds.to(device),
        )

    # -------------------------------------------------------------------------
    # Video Preprocessing
    # -------------------------------------------------------------------------

    def _is_image_file(self, filename: str) -> bool:
        """Check if filename is an image."""
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif"}
        return os.path.splitext(filename.lower())[1] in image_exts

    def _prepare_video_tensor(
        self,
        video_input: Union[InputVideo, InputImage],
        target_height: int,
        target_width: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        fps: float = 24.0,
    ) -> Tuple[Tensor, float]:
        """
        Prepare video tensor from input, applying necessary transforms.

        Returns:
            Tuple of (video_tensor [C, T, H, W], fps)
        """
        # Load video or image
        if isinstance(video_input, Image.Image):
            # Single image input
            frames = [video_input]
        elif isinstance(video_input, str) and self._is_image_file(video_input):
            # Image file path
            frames = [self._load_image(video_input)]
        else:
            # Video input
            frames = self._load_video(video_input, fps=fps)

        # Resize frames to target resolution while preserving aspect ratio
        processed_frames = []
        for frame in frames:
            # Resize using area interpolation (like NaResize)
            frame = self._aspect_ratio_resize(
                frame, max_area=target_height * target_width, mod_value=16
            )[0]
            processed_frames.append(frame)

        # Convert to tensor [T, C, H, W] normalized to [-1, 1]
        frame_tensors = []
        for frame in tqdm(processed_frames, desc="Converting frames to tensors"):
            tensor = torch.from_numpy(__import__("numpy").array(frame)).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            frame_tensors.append(tensor)

        video_tensor = torch.stack(frame_tensors, dim=0)  # [T, C, H, W]
        video_tensor = rearrange(video_tensor, "t c h w -> c t h w")
        video_tensor = video_tensor.to(device=device, dtype=dtype)

        return video_tensor

    def _resize_frame(
        self,
        frame: Image.Image,
        target_height: int,
        target_width: int,
    ) -> Image.Image:
        """Resize frame to target dimensions using area interpolation."""
        return frame.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def _divisible_crop(
        self, tensor: Tensor, divisor: Tuple[int, int] = (16, 16)
    ) -> Tensor:
        """Crop tensor to be divisible by the given divisor."""
        _, t, h, w = tensor.shape
        new_h = (h // divisor[0]) * divisor[0]
        new_w = (w // divisor[1]) * divisor[1]

        # Center crop
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2

        return tensor[:, :, start_h : start_h + new_h, start_w : start_w + new_w]

    def _pad_video_frames(self, video: Tensor, sp_size: int = 1) -> Tuple[Tensor, int]:
        """
        Pad video frames to match required frame count for sequence parallel.

        Args:
            video: Input tensor [C, T, H, W]
            sp_size: Sequence parallel size

        Returns:
            Tuple of (padded_video, original_length)
        """
        t = video.size(1)
        original_length = t

        if t == 1:
            return video, original_length

        if t <= 4 * sp_size:
            # Pad to minimum length
            padding_needed = 4 * sp_size - t + 1
            padding = video[:, -1:].repeat(1, padding_needed, 1, 1)
            video = torch.cat([video, padding], dim=1)
        elif (t - 1) % (4 * sp_size) != 0:
            # Pad to make (t-1) divisible by (4 * sp_size)
            padding_needed = 4 * sp_size - ((t - 1) % (4 * sp_size))
            padding = video[:, -1:].repeat(1, padding_needed, 1, 1)
            video = torch.cat([video, padding], dim=1)

        return video, original_length

    # -------------------------------------------------------------------------
    # Video Chunking
    # -------------------------------------------------------------------------

    def _split_video_into_chunks(
        self,
        video: Tensor,
        chunk_frames: int,
        overlap_frames: int,
    ) -> Tuple[List[Tensor], List[Tuple[int, int]]]:
        """
        Split video tensor into overlapping chunks.

        Args:
            video: Input tensor [C, T, H, W]
            chunk_frames: Number of frames per chunk
            overlap_frames: Number of overlapping frames between chunks

        Returns:
            Tuple of (list of chunk tensors, list of (start, end) indices)
        """
        c, t, h, w = video.shape

        if t <= chunk_frames:
            # No chunking needed
            return [video], [(0, t)]

        chunks = []
        indices = []
        stride = chunk_frames - overlap_frames

        start = 0
        while start < t:
            end = min(start + chunk_frames, t)
            chunk = video[:, start:end]
            chunks.append(chunk)
            indices.append((start, end))

            if end >= t:
                break
            start += stride

        return chunks, indices

    def _blend_chunks(
        self,
        chunks: List[Tensor],
        indices: List[Tuple[int, int]],
        total_frames: int,
        overlap_frames: int,
    ) -> Tensor:
        """
        Blend processed chunks back into a single video tensor.

        Uses cosine blending in overlap regions for smooth transitions.

        Args:
            chunks: List of processed chunk tensors [T, C, H, W]
            indices: List of (start, end) frame indices for each chunk
            total_frames: Total number of output frames
            overlap_frames: Number of overlapping frames between chunks

        Returns:
            Blended video tensor [T, C, H, W]
        """
        if len(chunks) == 1:
            return chunks[0]

        # Get dimensions from first chunk
        _, c, h, w = chunks[0].shape
        device = chunks[0].device
        dtype = chunks[0].dtype

        # Initialize output tensor and weight accumulator
        output = torch.zeros(total_frames, c, h, w, device=device, dtype=dtype)
        weights = torch.zeros(total_frames, 1, 1, 1, device=device, dtype=dtype)

        for chunk, (start, end) in zip(chunks, indices):
            chunk_len = end - start

            # Create blending weights for this chunk
            chunk_weights = torch.ones(chunk_len, 1, 1, 1, device=device, dtype=dtype)

            # Apply fade-in at the start (except for first chunk)
            if start > 0:
                fade_len = min(overlap_frames, chunk_len)
                # Cosine fade-in: 0 -> 1
                t = torch.linspace(
                    0, torch.pi / 2, fade_len, device=device, dtype=dtype
                )
                fade_in = torch.sin(t).view(-1, 1, 1, 1)
                chunk_weights[:fade_len] = fade_in

            # Apply fade-out at the end (except for last chunk)
            if end < total_frames:
                fade_len = min(overlap_frames, chunk_len)
                # Cosine fade-out: 1 -> 0
                t = torch.linspace(
                    0, torch.pi / 2, fade_len, device=device, dtype=dtype
                )
                fade_out = torch.cos(t).view(-1, 1, 1, 1)
                chunk_weights[-fade_len:] = chunk_weights[-fade_len:] * fade_out

            # Accumulate weighted chunk
            output[start:end] += chunk[:chunk_len] * chunk_weights
            weights[start:end] += chunk_weights

        # Normalize by accumulated weights
        output = output / weights.clamp(min=1e-8)

        return output

    # -------------------------------------------------------------------------
    # VAE Operations
    # -------------------------------------------------------------------------

    def _seedvr_vae_encode(
        self,
        samples: List[Tensor],
        offload: bool = False,
        conv_max_mem: float = 0.5,
        norm_max_mem: float = 0.5,
        split_size: int = 4,
        memory_device: str = "same",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tensor]:
        """
        Encode samples using SeedVR VAE with proper scaling.

        Args:
            samples: List of tensors [C, T, H, W]
            offload: Whether to offload VAE after encoding

        Returns:
            List of latent tensors
        """
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        if hasattr(self.vae, "set_memory_limit"):
            self.vae.set_memory_limit(
                conv_max_mem=conv_max_mem, norm_max_mem=norm_max_mem
            )
        if hasattr(self.vae, "set_causal_slicing"):
            self.vae.set_causal_slicing(
                split_size=split_size, memory_device=memory_device
            )

        latents = []
        dtype = self._vae_dtype
        scale = self._vae_scaling_factor
        shift = self._vae_shifting_factor

        total = len(samples)
        safe_emit_progress(progress_callback, 0.0, "VAE encode: starting")

        for idx, sample in enumerate(tqdm(samples, desc="Encoding samples")):
            # Add batch dimension and move to device
            sample = sample.unsqueeze(0).to(device=self.device, dtype=dtype)

            if hasattr(self.vae, "preprocess"):
                sample = self.vae.preprocess(sample)
            latent = self.vae.encode(sample).latent

            # Ensure proper shape: [B, C, T, H, W] -> [B, T, H, W, C]
            latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
            latent = rearrange(latent, "b c t h w -> b t h w c")

            # Apply scaling
            latent = (latent - shift) * scale

            # Remove batch dimension
            latents.append(latent.squeeze(0))

            if total > 0:
                p = (idx + 1) / total
                safe_emit_progress(
                    progress_callback,
                    p,
                    f"VAE encode: {idx + 1}/{total} ({int(p * 100)}%)",
                )

        safe_emit_progress(progress_callback, 1.0, "VAE encode: done")

        if offload:
            self._offload("vae")

        # Synchronize before clearing cache for consistent memory behavior
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        empty_cache()

        return latents

    def _seedvr_vae_decode(
        self,
        latents: List[Tensor],
        offload: bool = False,
        conv_max_mem: float = 0.5,
        norm_max_mem: float = 0.5,
        split_size: int = 4,
        memory_device: str = "same",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tensor]:
        """
        Decode latents using SeedVR VAE with proper scaling.

        Args:
            latents: List of latent tensors
            offload: Whether to offload VAE after decoding

        Returns:
            List of sample tensors
        """
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        if hasattr(self.vae, "set_memory_limit"):
            self.vae.set_memory_limit(
                conv_max_mem=conv_max_mem, norm_max_mem=norm_max_mem
            )
        if hasattr(self.vae, "set_causal_slicing"):
            self.vae.set_causal_slicing(
                split_size=split_size, memory_device=memory_device
            )
        samples = []
        dtype = self._vae_dtype
        scale = self._vae_scaling_factor
        shift = self._vae_shifting_factor

        total = len(latents)
        safe_emit_progress(progress_callback, 0.0, "VAE decode: starting")

        for idx, latent in enumerate(tqdm(latents, desc="Decoding latents")):
            # Add batch dimension
            latent = latent.unsqueeze(0).to(device=self.device, dtype=dtype)

            # Reverse scaling
            latent = latent / scale + shift

            # Rearrange: [B, T, H, W, C] -> [B, C, T, H, W]
            latent = rearrange(latent, "b t h w c -> b c t h w")
            latent = latent.squeeze(2)  # Remove temporal dim if single frame

            sample = self.vae.decode(latent).sample
            if hasattr(self.vae, "postprocess"):
                sample = self.vae.postprocess(sample)

            samples.append(sample.squeeze(0).cpu())

            if total > 0:
                p = (idx + 1) / total
                safe_emit_progress(
                    progress_callback,
                    p,
                    f"VAE decode: {idx + 1}/{total} ({int(p * 100)}%)",
                )

        safe_emit_progress(progress_callback, 1.0, "VAE decode: done")

        if offload:
            self._offload("vae")

        # Synchronize before clearing cache for consistent memory behavior
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        empty_cache()

        return samples

    # -------------------------------------------------------------------------
    # Diffusion Sampling
    # -------------------------------------------------------------------------

    def _get_sampling_timesteps(
        self,
        num_steps: int,
        device: torch.device,
    ) -> Tensor:
        """Generate uniform trailing sampling timesteps."""
        T = self._schedule.T
        # Uniform trailing timesteps from T to 0
        timesteps = torch.linspace(T, 0, num_steps + 1, device=device)
        return timesteps

    def _timestep_transform(self, timesteps: Tensor, latent_shape: Tensor) -> Tensor:
        """
        Apply timestep shifting based on resolution (as in SeedVR).

        This shifts timesteps based on the spatial/temporal resolution
        to account for different noise levels needed at different resolutions.
        """
        device = timesteps.device

        # Get resolution info
        vt = self.vae_scale_factor_temporal
        vs = self.vae_scale_factor_spatial

        frames = (latent_shape[0] - 1) * vt + 1
        heights = latent_shape[1] * vs
        widths = latent_shape[2] * vs

        # Compute shift factor using linear functions
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(256 * 256, 1.0, 1024 * 1024, 3.2)
        vid_shift_fn = get_lin_function(256 * 256 * 37, 1.0, 1280 * 720 * 145, 5.0)

        if frames > 1:
            shift = vid_shift_fn(heights * widths * frames)
        else:
            shift = img_shift_fn(heights * widths)

        shift = torch.tensor(shift, device=device)

        # Apply shift transformation
        timesteps = timesteps / self._schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self._schedule.T

        return timesteps

    def _get_condition(
        self,
        noise: Tensor,
        latent_blur: Tensor,
    ) -> Tensor:
        """
        Create conditioning tensor for super-resolution.

        Args:
            noise: Noise tensor [T, H, W, C]
            latent_blur: Blurred/low-res latent [T, H, W, C]

        Returns:
            Condition tensor [T, H, W, C+1]
        """
        t, h, w, c = noise.shape
        cond = torch.zeros([t, h, w, c + 1], device=noise.device, dtype=noise.dtype)

        # SR task: use blurred latent as condition with mask=1
        cond[..., :-1] = latent_blur
        cond[..., -1:] = 1.0

        return cond

    def _add_cond_noise(
        self,
        x: Tensor,
        aug_noise: Tensor,
        cond_noise_scale: float = 0.1,
    ) -> Tensor:
        """Add noise to condition latent."""
        device = x.device
        t = torch.tensor([1000.0 * cond_noise_scale], device=device)

        # Get shape for timestep transform
        shape = torch.tensor(x.shape[:-1], device=device)

        # Transform timestep
        t = self._timestep_transform(t, shape)

        # Apply forward diffusion
        x = self._schedule.forward(x, aug_noise, t)
        return x

    def _classifier_free_guidance(
        self,
        pos_pred: Tensor,
        neg_pred: Tensor,
        cfg_scale: float,
        cfg_rescale: float = 0.0,
    ) -> Tensor:
        """Apply classifier-free guidance."""
        # Standard CFG
        pred = neg_pred + cfg_scale * (pos_pred - neg_pred)

        # Optional rescaling (phi in some papers)
        if cfg_rescale > 0:
            std_pos = pos_pred.std(dim=list(range(1, pos_pred.ndim)), keepdim=True)
            std_cfg = pred.std(dim=list(range(1, pred.ndim)), keepdim=True)
            pred = pred * (std_pos / std_cfg) * cfg_rescale + pred * (1 - cfg_rescale)

        return pred

    def _euler_step(
        self,
        pred: Tensor,
        x_t: Tensor,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """
        Euler step from timestep t to timestep s.

        Args:
            pred: Model prediction
            x_t: Current sample
            t: Current timestep
            s: Target timestep

        Returns:
            Sample at timestep s
        """
        T = self._schedule.T

        # Expand dimensions
        t = self._schedule._expand_dims(t, x_t.ndim)
        s = self._schedule._expand_dims(s, x_t.ndim)

        # Convert prediction to x_0 and x_T
        pred_x_0, pred_x_T = self._schedule.convert_from_pred(
            pred, self._prediction_type, x_t, t
        )

        # Step to s
        s_clamped = s.clamp(0, T)
        pred_x_s = self._schedule.forward(pred_x_0, pred_x_T, s_clamped)

        # Handle edge cases
        pred_x_s = torch.where(s >= 0, pred_x_s, pred_x_0)
        pred_x_s = torch.where(s <= T, pred_x_s, pred_x_T)

        return pred_x_s

    # -------------------------------------------------------------------------
    # Main Inference
    # -------------------------------------------------------------------------

    def _flatten_latents(self, latents: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Flatten list of latents for batched processing."""
        shapes = torch.stack(
            [torch.tensor(x.shape[:-1], device=latents[0].device) for x in latents]
        )
        flat = torch.cat([x.flatten(0, -2) for x in latents])
        return flat, shapes

    def _unflatten_latents(self, flat: Tensor, shapes: Tensor) -> List[Tensor]:
        """Unflatten batched latents back to list."""
        lengths = shapes.prod(-1)
        splits = flat.split(lengths.tolist())
        return [x.unflatten(0, s.tolist()) for x, s in zip(splits, shapes)]

    @torch.no_grad()
    def _inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        text_pos_embeds: Tensor,
        text_neg_embeds: Tensor,
        cfg_scale: float,
        cfg_rescale: float,
        num_steps: int,
        dit_offload: bool = True,
        decode: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tensor]:
        """
        Run diffusion inference loop.

        Args:
            noises: List of noise tensors
            conditions: List of condition tensors
            text_pos_embeds: Positive text embeddings
            text_neg_embeds: Negative text embeddings
            cfg_scale: Classifier-free guidance scale
            cfg_rescale: CFG rescale factor
            num_steps: Number of sampling steps
            dit_offload: Whether to offload transformer after inference
            decode: Whether to VAE decode at the end (True) or return output latents (False)
            progress_callback: Optional progress callback

        Returns:
            If decode=True: list of decoded samples [C, T, H, W] (or [C, H, W] for single frame).
            If decode=False: list of output latents [T, H, W, C].
        """
        batch_size = len(noises)

        if batch_size == 0:
            return []

        safe_emit_progress(progress_callback, 0.0, "Preparing SeedVR sampling")

        # Ensure transformer is loaded
        if self.transformer is None:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        # Flatten for batched processing
        latents, latent_shapes = self._flatten_latents(noises)
        latents_cond, _ = self._flatten_latents(conditions)

        # Get timesteps
        timesteps = self._get_sampling_timesteps(num_steps, self.device)

        safe_emit_progress(progress_callback, 0.01, "Starting SeedVR sampling")

        # Sampling loop
        sampling_end = 0.92 if decode else 1.0
        emit_sampling_progress = make_mapped_progress(
            progress_callback, 0.0, sampling_end
        )
        emit_decode_progress = (
            make_mapped_progress(progress_callback, sampling_end, 1.0) if decode else None
        )

        with self._progress_bar(total=num_steps, desc="SeedVR Sampling") as pbar:
            for i, (t, s) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                # Prepare model input
                model_input = torch.cat([latents, latents_cond], dim=-1)

                # Positive prediction
                with torch.autocast(self.device.type, torch.bfloat16, enabled=True):
                    pos_pred = self.transformer(
                        vid=model_input,
                        txt=text_pos_embeds,
                        vid_shape=latent_shapes,
                        txt_shape=torch.tensor(
                            [[text_pos_embeds.shape[0]]], device=self.device
                        ),
                        timestep=t.expand(batch_size),
                    ).vid_sample

                    # Negative prediction for CFG
                    neg_pred = self.transformer(
                        vid=model_input,
                        txt=text_neg_embeds,
                        vid_shape=latent_shapes,
                        txt_shape=torch.tensor(
                            [[text_neg_embeds.shape[0]]], device=self.device
                        ),
                        timestep=t.expand(batch_size),
                    ).vid_sample

                # Apply CFG with partial application
                # From original: scale = cfg_scale if (i+1)/len(timesteps) <= partial else 1.0
                # diffusion.cfg.partial defaults to 1.0 (always apply full CFG)
                progress_ratio = (i + 1) / num_steps
                effective_cfg_scale = (
                    cfg_scale if progress_ratio <= self._cfg_partial else 1.0
                )

                pred = self._classifier_free_guidance(
                    pos_pred, neg_pred, effective_cfg_scale, cfg_rescale
                )

                # Euler step
                latents = self._euler_step(pred, latents, t, s)

                pbar.update(1)

                lp = (i + 1) / num_steps
                emit_sampling_progress(lp, f"SeedVR Sampling {int(lp * 100)}%")

        if dit_offload:
            self._offload("transformer")

        # Unflatten
        latents = self._unflatten_latents(latents, latent_shapes)

        if not decode:
            # Final synchronization and cache clear
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            empty_cache()
            safe_emit_progress(progress_callback, 1.0, "SeedVR sampling complete")
            return latents

        # Synchronize and clear cache before VAE decode
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        empty_cache()

        # VAE decode
        self.to_device(self.vae)
        samples = self._seedvr_vae_decode(
            latents,
            offload=dit_offload,
            progress_callback=emit_decode_progress,
        )

        # Free latents after decoding
        del latents

        if dit_offload:
            self._offload("vae")

        # Final synchronization and cache clear
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        empty_cache()

        safe_emit_progress(progress_callback, 1.0, "SeedVR sampling complete")

        return samples

    def run(
        self,
        video: Optional[InputVideo] = None,
        image: Optional[InputImage] = None,
        height: int = 720,
        width: int = 1280,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        num_inference_steps: int = 50,
        cond_noise_scale: float = 0.1,
        seed: int = 666,
        sp_size: int = 1,
        output_fps: float = 24.0,
        color_fix: bool = True,
        offload: bool = True,
        progress_callback: Optional[Callable] = None,
        vae_conv_max_mem: float = 0.5,
        vae_norm_max_mem: float = 0.5,
        vae_split_size: int = 4,
        vae_memory_device: str = "same",
        chunk_frames: Optional[int] = None,
        chunk_overlap: int = 8,
        use_chunking: bool = False,
        **kwargs,
    ) -> Union[List[Image.Image], Image.Image]:
        """
        Run SeedVR video super-resolution.

        Default values match SeedVR INFERENCE script (inference_seedvr_3b.py):
            - diffusion.schedule.T: 1000.0
            - diffusion.sampler.prediction_type: v_lerp
            - cfg_scale: 6.5 (inference default, config has 7.5)
            - cfg_rescale: 0
            - num_inference_steps: 50
            - cond_noise_scale: 0.1 (inference default, config has 0.25)
            - vae.scaling_factor: 0.9152
            - vae.dtype: bfloat16

        Args:
            video: Input video path, URL, or tensor
            image: Input image (alternative to video)
            height: Target output height (default 720 for 720p)
            width: Target output width (default 1280 for 720p)
            cfg_scale: Classifier-free guidance scale (inference: 6.5)
            cfg_rescale: CFG rescale factor (default: 0)
            num_inference_steps: Number of diffusion steps (default: 50)
            cond_noise_scale: Noise scale for conditioning (inference: 0.1)
            seed: Random seed (default: 666)
            sp_size: Sequence parallel size (should be 1 for images)
            output_fps: Output FPS (uses input FPS if None)
            color_fix: Whether to apply wavelet color correction
            offload: Whether to offload models after use
            progress_callback: Optional progress callback
            chunk_frames: Number of frames per chunk for long videos (None = no chunking).
                          Recommended: 33-65 frames depending on VRAM.
            chunk_overlap: Number of overlapping frames between chunks for blending (default: 8).
                          Higher values give smoother transitions but increase compute.

        Returns:
            List of PIL Images (for video) or single Image (for image input)
        """
        assert video is not None or image is not None, "video or image is required"

        safe_emit_progress(progress_callback, 0.0, "Starting SeedVR upscale")
        vae_split_size = max(vae_split_size, 4)

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Determine input type
        is_image_input = image is not None

        # Prepare input
        if is_image_input:
            if sp_size > 1:
                raise ValueError("sp_size should be 1 for image inputs")
            input_data = image
        else:
            input_data = video

        # Load and preprocess video/image
        safe_emit_progress(progress_callback, 0.03, "Preparing input frames")
        video_tensor = self._prepare_video_tensor(
            input_data,
            target_height=height,
            target_width=width,
            device=self.device,
            dtype=self._vae_dtype,
            fps=output_fps,
        )
        # Apply divisible crop
        safe_emit_progress(progress_callback, 0.07, "Cropping to divisible size")
        video_tensor = self._divisible_crop(video_tensor, (16, 16))

        total_frames = video_tensor.shape[1]

        # Store original video for color fix (before any padding)
        input_video_for_colorfix = video_tensor.clone()

        # Load prompt embeddings once
        safe_emit_progress(progress_callback, 0.10, "Loading prompt embeddings")
        text_pos_embeds, text_neg_embeds = self._load_prompt_embeddings(self.device)

        # Determine if chunking is needed
        use_chunking = (
            chunk_frames is not None
            and total_frames > chunk_frames
            and not is_image_input
            and use_chunking
        )

        if use_chunking:
            self.logger.info(
                f"Chunking enabled: {total_frames} frames -> chunks of {chunk_frames} with {chunk_overlap} overlap"
            )
            safe_emit_progress(progress_callback, 0.12, "Splitting video into chunks")

            # Split video into chunks - extract chunk data as CPU tensors to avoid holding GPU memory
            chunks_cpu, chunk_indices = self._split_video_into_chunks(
                video_tensor, chunk_frames, chunk_overlap
            )
            # Move chunks to CPU immediately to free GPU memory
            chunks_cpu = [c.cpu() for c in chunks_cpu]

            # Move input video to CPU - we only need it for color fix at the end
            input_video_for_colorfix = input_video_for_colorfix.cpu()

            # Free the original GPU video tensor
            del video_tensor
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            empty_cache()

            num_chunks = len(chunks_cpu)

            # -----------------------------------------------------------------
            # Phase 1: pad all chunks (CPU) then VAE-encode all chunks (single call)
            # -----------------------------------------------------------------
            safe_emit_progress(progress_callback, 0.16, "Padding chunks")
            padded_chunks_cpu: List[Tensor] = []
            chunk_original_lens: List[int] = []
            for chunk in chunks_cpu:
                chunk_padded, chunk_original_len = self._pad_video_frames(
                    chunk, sp_size
                )
                padded_chunks_cpu.append(chunk_padded)
                chunk_original_lens.append(chunk_original_len)

            # We no longer need the raw chunks list
            del chunks_cpu
            gc.collect()

            safe_emit_progress(progress_callback, 0.20, "Encoding all chunk latents")
            cond_latents_all = self._seedvr_vae_encode(
                padded_chunks_cpu,
                offload=True,
                conv_max_mem=vae_conv_max_mem,
                norm_max_mem=vae_norm_max_mem,
                split_size=vae_split_size,
                memory_device=vae_memory_device,
                progress_callback=make_mapped_progress(progress_callback, 0.20, 0.28),
            )

            # Free padded inputs and move condition latents to CPU to reduce VRAM
            del padded_chunks_cpu
            cond_latents_all = [lat.cpu() for lat in cond_latents_all]
            gc.collect()
            empty_cache()

            # -----------------------------------------------------------------
            # Phase 2: process each chunk output (sampling in latent space)
            # -----------------------------------------------------------------
            self.logger.info("Sampling chunks (latents only)...")
            output_latents_cpu: List[Tensor] = []

            # Reserve [0.28, 0.80] for per-chunk sampling
            chunks_overall_start = 0.28
            chunks_overall_end = 0.80

            for chunk_idx, (start, end) in enumerate(chunk_indices):
                self.logger.info(
                    f"Sampling chunk {chunk_idx + 1}/{num_chunks} (frames {start}-{end})"
                )

                chunk_start_p = chunks_overall_start + (
                    chunk_idx / max(1, num_chunks)
                ) * (chunks_overall_end - chunks_overall_start)
                chunk_end_p = chunks_overall_start + (
                    (chunk_idx + 1) / max(1, num_chunks)
                ) * (chunks_overall_end - chunks_overall_start)
                chunk_progress = make_mapped_progress(
                    progress_callback, chunk_start_p, chunk_end_p
                )
                chunk_progress(0.0, f"Chunk {chunk_idx + 1}/{num_chunks}: preparing")

                # Move this chunk's condition latent to GPU only when needed
                cond_latent = cond_latents_all[chunk_idx].to(device=self.device)

                # Generate noise (use seed offset for reproducibility across chunks)
                chunk_seed = seed + chunk_idx if seed is not None else None
                if chunk_seed is not None:
                    torch.manual_seed(chunk_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(chunk_seed)

                chunk_progress(
                    0.12, f"Chunk {chunk_idx + 1}/{num_chunks}: initializing noise"
                )
                noises = [torch.randn_like(cond_latent)]
                aug_noises = [torch.randn_like(cond_latent)]

                noised_cond_latents = [
                    self._add_cond_noise(latent, aug_noise, cond_noise_scale)
                    for latent, aug_noise in zip([cond_latent], aug_noises)
                ]
                del aug_noises

                conditions = [
                    self._get_condition(noise, cond_latent_)
                    for noise, cond_latent_ in zip(noises, noised_cond_latents)
                ]
                del noised_cond_latents, cond_latent
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                empty_cache()

                # Reserve a sub-span inside the chunk for sampling
                chunk_sampling_progress = make_mapped_progress(
                    chunk_progress, 0.18, 0.98
                )

                # Run inference for this chunk (latents only; decode later)
                out_latents = self._inference(
                    noises=noises,
                    conditions=conditions,
                    text_pos_embeds=text_pos_embeds,
                    text_neg_embeds=text_neg_embeds,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    num_steps=num_inference_steps,
                    dit_offload=False,
                    decode=False,
                    progress_callback=chunk_sampling_progress,
                )

                # Free per-chunk inputs after inference
                del noises, conditions

                # Move output latents to CPU immediately to free VRAM
                for latent in out_latents:
                    output_latents_cpu.append(latent.cpu())
                del out_latents

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect()
                empty_cache()

            # Free condition latents after sampling all chunks
            del cond_latents_all
            gc.collect()
            empty_cache()
            
            # offload dit
            if offload:
                self._offload("transformer")

            # -----------------------------------------------------------------
            # Phase 3: decode chunk outputs, then blend on CPU
            # -----------------------------------------------------------------
            safe_emit_progress(progress_callback, 0.82, "Decoding chunk outputs")
            decoded_chunks = self._seedvr_vae_decode(
                output_latents_cpu,
                offload=True,
                conv_max_mem=vae_conv_max_mem,
                norm_max_mem=vae_norm_max_mem,
                split_size=vae_split_size,
                memory_device=vae_memory_device,
                progress_callback=make_mapped_progress(progress_callback, 0.82, 0.90),
            )
            del output_latents_cpu
            gc.collect()
            empty_cache()

            processed_chunks: List[Tensor] = []
            for sample, chunk_original_len in zip(decoded_chunks, chunk_original_lens):
                if sample.ndim == 3:
                    sample = sample.unsqueeze(1)
                sample = rearrange(sample, "c t h w -> t c h w")
                if sample.shape[0] > chunk_original_len:
                    sample = sample[:chunk_original_len]
                processed_chunks.append(sample.cpu())

            del decoded_chunks, chunk_original_lens
            gc.collect()

            # Blend chunks together (all on CPU now)
            self.logger.info("Blending chunks...")
            safe_emit_progress(progress_callback, 0.90, "Blending chunks")
            blended_sample = self._blend_chunks(
                processed_chunks, chunk_indices, total_frames, chunk_overlap
            )

            # Free processed chunks after blending
            del processed_chunks
            gc.collect()

            # Apply color fix if requested (both tensors already on CPU)
            if color_fix:
                safe_emit_progress(progress_callback, 0.92, "Applying color correction")
                blended_sample = wavelet_reconstruction(
                    blended_sample,
                    input_video_for_colorfix.permute(1, 0, 2, 3),
                )

            # Free input video after color fix
            del input_video_for_colorfix
            gc.collect()

            # Convert to output format
            safe_emit_progress(progress_callback, 0.95, "Converting output frames")
            blended_sample = rearrange(blended_sample, "t c h w -> t h w c")
            blended_sample = (
                blended_sample.clamp(-1, 1).mul(0.5).add(0.5).mul(255).round()
            )
            blended_sample = blended_sample.to(torch.uint8).cpu().numpy()

            # Convert to PIL images
            output_frames = []
            for frame in blended_sample:
                output_frames.append(Image.fromarray(frame))

        else:
            # Original non-chunked processing
            # Pad frames if needed
            safe_emit_progress(progress_callback, 0.12, "Padding frames (if needed)")
            video_tensor, original_length = self._pad_video_frames(
                video_tensor, sp_size
            )

            # VAE encode the conditioned latents
            self.logger.info(f"Encoding video: {video_tensor.shape}")
            safe_emit_progress(progress_callback, 0.18, "Encoding latents")

            cond_latents = self._seedvr_vae_encode(
                [video_tensor],
                offload=True,
                conv_max_mem=vae_conv_max_mem,
                norm_max_mem=vae_norm_max_mem,
                split_size=vae_split_size,
                memory_device=vae_memory_device,
                progress_callback=make_mapped_progress(progress_callback, 0.18, 0.24),
            )

            # Generate noise and augmentation noise
            safe_emit_progress(progress_callback, 0.24, "Initializing noise")
            noises = [torch.randn_like(latent) for latent in cond_latents]
            aug_noises = [torch.randn_like(latent) for latent in cond_latents]

            # Add noise to condition latents
            noised_cond_latents = [
                self._add_cond_noise(latent, aug_noise, cond_noise_scale)
                for latent, aug_noise in zip(cond_latents, aug_noises)
            ]

            # Build conditions
            conditions = [
                self._get_condition(noise, cond_latent)
                for noise, cond_latent in zip(noises, noised_cond_latents)
            ]

            # Run inference
            self.logger.info(f"Starting inference with {num_inference_steps} steps")
            sampling_progress = make_mapped_progress(progress_callback, 0.30, 0.86)
            samples = self._inference(
                noises=noises,
                conditions=conditions,
                text_pos_embeds=text_pos_embeds,
                text_neg_embeds=text_neg_embeds,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                num_steps=num_inference_steps,
                dit_offload=offload,
                progress_callback=sampling_progress,
            )

            # Process outputs
            output_frames = []
            for i, sample in enumerate(samples):
                # Trim to original length
                if sample.ndim == 3:
                    # Single frame: [C, H, W]
                    sample = sample.unsqueeze(1)  # [C, 1, H, W]

                sample = rearrange(sample, "c t h w -> t c h w")

                if sample.shape[0] > original_length:
                    sample = sample[:original_length]

                # Apply color fix if requested
                if color_fix:
                    safe_emit_progress(
                        progress_callback, 0.90, "Applying color correction"
                    )
                    sample = wavelet_reconstruction(
                        sample.cpu(),
                        input_video_for_colorfix[:, :original_length]
                        .cpu()
                        .permute(1, 0, 2, 3),
                    )

                # Convert to output format
                safe_emit_progress(progress_callback, 0.95, "Converting output frames")
                sample = rearrange(sample, "t c h w -> t h w c")
                sample = sample.clamp(-1, 1).mul(0.5).add(0.5).mul(255).round()
                sample = sample.to(torch.uint8).cpu().numpy()

                # Convert to PIL images
                for frame in sample:
                    output_frames.append(Image.fromarray(frame))

        gc.collect()
        empty_cache()

        safe_emit_progress(progress_callback, 1.0, "SeedVR upscale complete")

        if is_image_input:
            return [output_frames[0] if output_frames else None]
        else:
            return [output_frames]

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable, timestep: Optional[torch.Tensor] = None, image: Optional[bool] = False):
        self.logger.warning("Rendering step not supported for SeedVR upscale")