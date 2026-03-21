from src.engine.ltx22.shared.engine import LTX2Shared
from typing import Union, List, Optional, Tuple, Callable
import torch
from src.helpers.ltx2.upsampler import upsample_video
from src.types import InputImage, InputAudio, InputVideo
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.engine.ltx22.shared.diffusion_steps import SchedulerDiffusionStep, configure_scheduler_sigmas
from src.engine.ltx22.shared.noisers import GaussianNoiser
from src.engine.ltx22.shared.types import LatentState, VideoPixelShape
from src.engine.ltx22.shared.protocols import DiffusionStepProtocol
from src.engine.ltx22.shared.helpers import combined_image_conditionings, denoise_audio_video, euler_denoising_loop, multi_modal_guider_denoising_func, simple_denoising_func
from src.engine.ltx22.shared.utils import PipelineComponents
from src.engine.ltx22.shared.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from src.engine.ltx22.shared.tiling import _build_tiling_config
from src.vae.ltx2.model import decode_video 
from src.vae.ltx2audio.model import decode_audio
from src.engine.ltx22.shared.conditioning import VideoConditionByReferenceLatent, ConditioningItem, ConditioningItemAttentionStrengthWrapper
from src.engine.ltx22.shared.media_io import load_video_conditioning
import numpy as np
from src.lora.disable import disable_adapters_by_filter
from src.engine.ltx22.shared.types import VideoLatentShape
from einops import rearrange

class LTX2ICLoRAEngine(LTX2Shared):
    """LTX2 Text-to-Image-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.reference_downscale_factor = 1
        
    
    @staticmethod
    def _downsample_mask_to_latent(
        mask: torch.Tensor,
        target_latent_shape: VideoLatentShape,
    ) -> torch.Tensor:
        """
        Downsample a pixel-space mask to latent space using VAE scale factors.
        Handles causal temporal downsampling: the first frame is kept separately
        (temporal scale factor = 1 for the first frame), while the remaining
        frames are downsampled by the VAE's temporal scale factor.
        Args:
            mask: Pixel-space mask of shape (B, 1, F_pixel, H_pixel, W_pixel).
                Values in [0, 1].
            target_latent_shape: Expected latent shape after VAE encoding.
                Used to determine the target (F_latent, H_latent, W_latent).
        Returns:
            Flattened latent-space mask of shape (B, F_lat * H_lat * W_lat),
            matching the patchifier's token ordering (f, h, w).
        """
        b = mask.shape[0]
        f_lat = target_latent_shape.frames
        h_lat = target_latent_shape.height
        w_lat = target_latent_shape.width

        # Step 1: Spatial downsampling (area interpolation per frame)
        f_pix = mask.shape[2]
        spatial_down = torch.nn.functional.interpolate(
            rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
            size=(h_lat, w_lat),
            mode="area",
        )
        spatial_down = rearrange(spatial_down, "(b f) 1 h w -> b 1 f h w", b=b)

        # Step 2: Causal temporal downsampling
        # First frame: kept as-is (causal VAE encodes first frame independently)
        first_frame = spatial_down[:, :, :1, :, :]  # (B, 1, 1, H_lat, W_lat)

        if f_pix > 1 and f_lat > 1:
            # Remaining frames: downsample by temporal factor via group-mean
            t = (f_pix - 1) // (f_lat - 1)  # temporal downscale factor
            assert (f_pix - 1) % (f_lat - 1) == 0, (
                f"Pixel frames ({f_pix}) not compatible with latent frames ({f_lat}): "
                f"(f_pix - 1) must be divisible by (f_lat - 1)"
            )
            rest = rearrange(spatial_down[:, :, 1:, :, :], "b 1 (f t) h w -> b 1 f t h w", t=t)
            rest = rest.mean(dim=3)  # (B, 1, F_lat-1, H_lat, W_lat)
            latent_mask = torch.cat([first_frame, rest], dim=2)  # (B, 1, F_lat, H_lat, W_lat)
        else:
            latent_mask = first_frame

        # Flatten to (B, F_lat * H_lat * W_lat) matching patchifier token order (f, h, w)
        return rearrange(latent_mask, "b 1 f h w -> b (f h w)")
        
    def _create_conditionings(
        self,
        images: list[tuple[str, int, float]],
        video_conditioning: list[tuple[InputVideo, float]],
        height: int,
        width: int,
        num_frames: int,
        offload: bool = True,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
    ) -> list[ConditioningItem]:
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        
        dtype = self.component_dtypes["transformer"]
        
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=self.video_vae.encoder,
            dtype=dtype,
            device=self.device,
        )

        for video_path, strength in video_conditioning:
            # Load video at scaled-down resolution (if scale > 1)
            video = self._load_video(video_path, num_frames=num_frames)
            video = [self._aspect_ratio_resize(frame, max_area=int(height * width), mod_value=128)[0] for frame in video]
            # convert to tensor
            video = [torch.from_numpy(np.array(frame)).to(self.device).unsqueeze(0) for frame in video]
            video = load_video_conditioning(
                video=video,
                height=height,
                width=width,
                frame_cap=num_frames,
                dtype=dtype,
                device=self.device,
            )
            encoded_video = self.video_vae.encoder(video)
            reference_video_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)

            # Build attention_mask for ConditioningItemAttentionStrengthWrapper
            if conditioning_attention_mask is not None:
                # Downsample pixel-space mask to latent space, then scale by strength
                latent_mask = self._downsample_mask_to_latent(
                    mask=conditioning_attention_mask,
                    target_latent_shape=reference_video_shape,
                )
                attn_mask = latent_mask * conditioning_attention_strength
            elif conditioning_attention_strength < 1.0:
                # Use scalar strength only
                attn_mask = conditioning_attention_strength
            else:
                attn_mask = None

            cond = VideoConditionByReferenceLatent(
                latent=encoded_video,
                downscale_factor=self.reference_downscale_factor,
                strength=strength,
            )
            if attn_mask is not None:
                cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=attn_mask)
            conditionings.append(cond)
            
        
        if offload:
            self._offload("video_vae")

        return conditionings
    

    def _load_mask_video(
        self,
        mask_video: InputVideo,
        height: int,
        width: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).
        The mask video is loaded, resized to (height, width), converted to
        grayscale, and normalised to [0, 1].
        Args:
            mask_path: Path to the mask video file.
            height: Target height in pixels.
            width: Target width in pixels.
            num_frames: Maximum number of frames to load.
        Returns:
            Tensor of shape ``(1, 1, F, H, W)`` with values in ``[0, 1]``.
        """
        video = self._load_video(mask_video, num_frames=num_frames)
        video = [self._aspect_ratio_resize(frame, max_area=int(height * width), mod_value=128)[0] for frame in video]
            # convert to tensor
        video = [torch.from_numpy(np.array(frame)).to(self.device).unsqueeze(0) for frame in video]
        mask_video = load_video_conditioning(
            video=video,       
            height=height,
            width=width,
            dtype=self.component_dtypes["transformer"],
            device=self.device,
        )
        # mask_video shape: (1, C, F, H, W) — take mean over channels for grayscale
        mask = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W)
        # Normalise to [0, 1] — load_video_conditioning applies normalize_latent,
        # so undo that: values are in [-1, 1], remap to [0, 1]
        mask = (mask + 1.0) / 2.0
        return mask.clamp(0.0, 1.0)

    def run(
        self, 
        prompt: Union[str, List[str]] = None,
        video: InputVideo = None,
        seed: int | None = None,
        height: int = 512,
        width: int = 768,
        duration: str | int = 121,
        fps: float = 25,
        num_inference_steps: int = 8,
        tile_size: int = 512,
        image: Optional[InputImage] = None,
        last_image: Optional[InputImage] = None,
        audio: Optional[InputAudio] = None,
        images: List[Tuple[InputImage, int, float]] | None = None,
        audio_strength: float = 1.0,
        offload: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        condition_mask: InputVideo | None = None,
        condition_mask_strength: float = 1.0,
        **kwargs,
        ):
        safe_emit_progress(progress_callback, 0.0, "Starting image-conditioned LoRA pipeline")
        height = round(height / (self.vae_spatial_compression_ratio * 2) * (self.vae_spatial_compression_ratio * 2))
        width = round(width / (self.vae_spatial_compression_ratio * 2) * (self.vae_spatial_compression_ratio * 2))
        
        num_frames = self._parse_num_frames(duration, fps)

        tiling_config = _build_tiling_config(tile_size=tile_size, fps=fps)
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)
        
        noiser = GaussianNoiser(generator=generator)
        
        
        if not getattr(self, "scheduler", None):
            self.load_component_by_type("scheduler")
        safe_emit_progress(progress_callback, 0.05, "Encoding text")
        if images is None:
            images = []
        
        new_height = height
        new_width = width
        
        if last_image is not None:
            last_image = self._load_image(last_image)
            
            last_image, new_height, new_width = self._aspect_ratio_resize(last_image, max_area=int(height * width), mod_value=128)
            latent_idx = num_frames // 8
            images.append((last_image, latent_idx, 1.0))
            
            
        if image is not None:
            image = self._load_image(image)
            image, new_height, new_width = self._aspect_ratio_resize(image, max_area=int(height * width), mod_value=128)
            images.append((image, 0, 1.0))
        
        if new_height != height or new_width != width:
            height = new_height
            width = new_width
            
        
        condition_attention_mask = None
        if condition_mask is not None:
            condition_attention_mask = self._load_mask_video(condition_mask, height=height // 2, width=width // 2, num_frames=num_frames)

            
        text_encoder_results = self._encode_text([prompt], offload=offload)
        safe_emit_progress(progress_callback, 0.12, "Text encoded")
        context_p = text_encoder_results
        v_context_p, a_context_p, _ = context_p

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        configure_scheduler_sigmas(self.scheduler, stage_1_sigmas)
        stepper = SchedulerDiffusionStep(self.scheduler)
        
        dtype = self.component_dtypes["transformer"]
        pipeline_components = PipelineComponents(
            dtype=dtype,
            device=self.device,
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            denoise_progress_callback=None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=self.transformer,  # noqa: F821
                ),
                denoise_progress_callback=denoise_progress_callback,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=fps,
        )
        
        safe_emit_progress(progress_callback, 0.15, "Preparing stage 1")
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        if audio is not None:
            safe_emit_progress(progress_callback, 0.18, "Encoding audio")
            audio_conditionings = self._encode_audio(audio, device=self.device, strength=audio_strength, fps=fps, num_frames=num_frames, offload=offload)
        else:
            audio_conditionings = []
        video_conditioning = []
        if video is not None:
            video_conditioning = [(video, 1.0)]
        safe_emit_progress(progress_callback, 0.20, "Building conditionings")
        stage_1_conditionings = self._create_conditionings(
           images=images,
           video_conditioning=video_conditioning,
           height=stage_1_output_shape.height,
           width=stage_1_output_shape.width,
           num_frames=num_frames,
           offload=offload,
           conditioning_attention_mask=condition_attention_mask,
           conditioning_attention_strength=condition_mask_strength,
        )

        if offload:
            self._offload("video_vae")
            
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        dtype = self.component_dtypes["transformer"]
        stage_1_denoise_progress = make_mapped_progress(progress_callback, 0.25, 0.42)
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=pipeline_components,
            dtype=dtype,
            device=self.device,
            denoise_progress_callback=stage_1_denoise_progress,
        )
        safe_emit_progress(progress_callback, 0.42, "Stage 1 complete, upsampling latents")
        latent_upsampler = self.helpers["latent_upsampler"]
        self.to_device(latent_upsampler)
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        
        upscaled_video_latent = upsample_video(
            video_state.latent[:1],
            self.video_vae,
            upsampler=latent_upsampler,
        )
        
        if offload:
            del latent_upsampler
            self._offload("latent_upsampler")
            self._offload("video_vae")
        safe_emit_progress(progress_callback, 0.48, "Preparing stage 2")
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        configure_scheduler_sigmas(self.scheduler, distilled_sigmas)
        stepper = SchedulerDiffusionStep(self.scheduler)
        
        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            denoise_progress_callback=None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=self.transformer,  # noqa: F821
                ),
                denoise_progress_callback=denoise_progress_callback,
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=self.video_vae.encoder,
            dtype=dtype,
            device=self.device,
        )
        
        if offload:
            self._offload("video_vae")
            
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        dtype = self.component_dtypes["transformer"]
        
        # disable all loras that start with "ic_lora_"
        disable_adapters_by_filter(self.transformer, contains="ic_lora_")
        
        stage_2_denoise_progress = make_mapped_progress(progress_callback, 0.52, 0.88)
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            audio_conditionings=audio_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            denoise_progress_callback=stage_2_denoise_progress,
        )
        if offload:
            self._offload("transformer")
        safe_emit_progress(progress_callback, 0.88, "Decoding video")
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)

        decoded_video = decode_video(
            video_state.latent, self.video_vae.decoder, tiling_config, generator
        ).cpu()
        
        if offload:
            self._offload("video_vae")
        safe_emit_progress(progress_callback, 0.94, "Decoding audio")
        vocoder = self.helpers["vocoder"]
        self.to_device(vocoder)
        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
        self.to_device(self.audio_vae)

        decoded_audio = decode_audio(
            audio_state.latent, self.audio_vae.decoder, vocoder
        ).cpu()
        
        if offload:
            self._offload("audio_vae")
            del vocoder
            self._offload("vocoder")
        safe_emit_progress(progress_callback, 1.0, "Completed image-conditioned LoRA pipeline")
        return decoded_video, decoded_audio