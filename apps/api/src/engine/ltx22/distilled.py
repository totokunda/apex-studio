from src.engine.ltx22.shared.engine import LTX2Shared
from typing import Union, List, Optional, Tuple
import torch
from src.helpers.ltx2.upsampler import upsample_video
from src.types import InputImage, InputAudio
from src.utils.cache import empty_cache
from src.utils.progress import safe_emit_progress, make_mapped_progress
from src.engine.ltx2.multimodal_guidance import MultiModalGuider, MultiModalGuiderParams
from src.engine.ltx22.shared.diffusion_steps import SchedulerDiffusionStep, configure_scheduler_sigmas
from src.engine.ltx22.shared.guiders import MultiModalGuider, MultiModalGuiderParams
from src.engine.ltx22.shared.noisers import GaussianNoiser
from src.engine.ltx22.shared.types import LatentState, VideoPixelShape
from src.engine.ltx22.shared.protocols import DiffusionStepProtocol
from src.engine.ltx22.shared.helpers import image_conditionings_by_replacing_latent, denoise_audio_video, euler_denoising_loop, multi_modal_guider_denoising_func, simple_denoising_func
from src.engine.ltx22.shared.utils import PipelineComponents
from src.engine.ltx22.shared.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from src.engine.ltx22.shared.tiling import _build_tiling_config
from src.vae.ltx2.model import decode_video 
from src.vae.ltx2audio.model import decode_audio

class LTX2TI2VEngine(LTX2Shared):
    """LTX2 Text-to-Image-to-Video Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)

    def run(
        self, 
        prompt: Union[str, List[str]] = None,
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
        **kwargs,
        ):
        

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
            
        
        text_encoder_results = self._encode_text([prompt], offload=offload)
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
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
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
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=fps,
        )
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        
        
        if audio is not None:
            audio_conditionings = self._encode_audio(audio, device=self.device, strength=audio_strength, fps=fps, num_frames=num_frames, offload=offload)
        else:
            audio_conditionings = []
        
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
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
        )
        
        latent_upsampler = self.helpers["latent_upsampler"]
        self.to_device(latent_upsampler)
        
        upscaled_video_latent = upsample_video(
            video_state.latent[:1],
            self.video_vae,
            upsampler=latent_upsampler,
        )
        
        if offload:
            del latent_upsampler
            self._offload("latent_upsampler")
            
        
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        configure_scheduler_sigmas(self.scheduler, distilled_sigmas)
        stepper = SchedulerDiffusionStep(self.scheduler)
        
        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
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
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)
        
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=self.video_vae.encoder,
            dtype=dtype,
            device=self.device,
        )
        
        if offload:
            self._offload("video_vae")

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
        )
        
        if offload:
            self._offload("transformer")
        
        if not getattr(self, "video_vae", None):
            self.load_component_by_name("video_vae")
        self.to_device(self.video_vae)

        decoded_video = decode_video(
            video_state.latent, self.video_vae.decoder, tiling_config, generator
        )
        
        
        if offload:
            self._offload("video_vae")
        
        vocoder = self.helpers["vocoder"]
        self.to_device(vocoder)
        
        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
        self.to_device(self.audio_vae)

        decoded_audio = decode_audio(
            audio_state.latent, self.audio_vae.decoder, vocoder
        )
        
        if offload:
            self._offload("audio_vae")
            del vocoder
            self._offload("vocoder")
        
        return decoded_video, decoded_audio