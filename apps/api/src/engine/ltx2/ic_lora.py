from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from src.engine.ltx2.shared.keyframe_video_base import LTX2KeyframeVideoBaseEngine
from src.types import InputImage, InputVideo


class LTX2ICLoraEngine(LTX2KeyframeVideoBaseEngine):
    """
    LTX2 engine modeled after the ltx-pipelines IC-LoRA pipeline structure:
    - Supports *keyframe-token append* conditioning from images
    - Supports an additional conditioning *video* (control video) as keyframe tokens
    - Supports two-stage upsampling via the shared base implementation
    """

    @torch.inference_mode()
    def run(  # noqa: PLR0913
        self,
        image: Optional[Union[InputImage, List[InputImage]]] = None,
        image_strengths: Optional[Union[float, List[float]]] = None,
        image_pixel_frame_indices: Optional[Union[int, List[int]]] = None,
        conditioning_video: Optional[InputVideo] = None,
        conditioning_video_strength: float = 1.0,
        conditioning_video_pixel_frame_index: int = 0,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        duration: Union[str, int] = 121,
        fps: float = 25.0,
        num_inference_steps: int = 40,
        timesteps: List[int] = None,
        use_distilled_stage_1: bool = False,
        use_distilled_stage_2: bool = False,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
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
    ):
        return super().run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            images=image,
            image_strengths=image_strengths,
            image_pixel_frame_indices=image_pixel_frame_indices,
            conditioning_video=conditioning_video,
            conditioning_video_strength=conditioning_video_strength,
            conditioning_video_pixel_frame_index=conditioning_video_pixel_frame_index,
            height=height,
            width=width,
            duration=duration,
            fps=fps,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            use_distilled_stage_1=use_distilled_stage_1,
            use_distilled_stage_2=use_distilled_stage_2,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
            offload=offload,
            return_latents=return_latents,
            upsample=upsample,
        )
