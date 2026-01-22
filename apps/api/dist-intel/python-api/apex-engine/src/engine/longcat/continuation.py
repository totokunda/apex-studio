from typing import List, Optional, Union
import torch
from PIL import Image
from .shared import LongCatShared
from .vc import LongCatVCEngine
from .refine import LongCatRefineEngine
from src.types import InputVideo


class LongCatContinuationEngine(LongCatShared):
    """
    LongCat video‑continuation engine.

    This engine mirrors `run_demo_video_continuation.py` by:

    1. Extending an input video using the VC engine.
    2. Optionally running a distilled VC pass (if a CFG‑step LoRA is provided).
    3. Refining the distilled output to higher resolution with the refine engine.
    """

    def __init__(self, yaml_path: str, **kwargs):
        self._yaml_path = yaml_path
        super().__init__(yaml_path, **kwargs)

        self._vc_engine: Optional[LongCatVCEngine] = None
        self._refine_engine: Optional[LongCatRefineEngine] = None

    def _get_vc_engine(self) -> LongCatVCEngine:
        if self._vc_engine is None:
            self._vc_engine = self.sub_engines["vc"]
        return self._vc_engine

    def _get_refine_engine(self) -> LongCatRefineEngine:
        if self._refine_engine is None:
            self._refine_engine = self.sub_engines["refine"]
        return self._refine_engine

    def run(
        self,
        video: InputVideo,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        duration: str | int = 93,
        fps: int = 15,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        resolution: str = "480p",
        # Distillation stage
        enable_distill: bool = True,
        distill_num_inference_steps: int = 16,
        distill_guidance_scale: float = 1.0,
        # Refinement stage
        enable_refine: bool = True,
        refine_num_inference_steps: int = 50,
        spatial_refine_only: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        return_intermediates: bool = False,
        **kwargs,
    ):
        """
        Continue an input video with LongCat VC and optional refinement.

        Args:
            video: Input video (path, URL, tensor, ndarray, or list of frames).
            prompt: Prompt string (or list of one string) for continuation.
            negative_prompt: Optional negative prompt for the base VC pass.
            num_frames: Number of frames to generate for the continuation.
            num_cond_frames: Number of conditioning frames from the input.
            num_inference_steps: Steps for the base VC pass.
            guidance_scale: CFG scale for the base VC pass.
            resolution: Resolution bucket for VC ("480p" or "720p").
            enable_distill: If True, run an additional distilled VC pass.
            distill_num_inference_steps: Steps for the distilled VC pass.
            distill_guidance_scale: Guidance scale for the distilled VC pass.
            cfg_step_lora: Optional LoRA path to apply for the distilled VC.
            enable_refine: If True, run refinement over the distilled output.
            refine_num_inference_steps: Steps for the refinement pass.
            refinement_lora: Optional LoRA path for the refinement stage.
            spatial_refine_only: Passed through to refine engine.
            generator: Optional torch.Generator.
            seed: Optional seed when generator is not provided.
            return_intermediates: If True, return dict of stage videos.

        Returns:
            List[List[Image.Image]] or dict if `return_intermediates=True`.
        """
        num_frames = self._parse_num_frames(duration, fps)
        if isinstance(prompt, list):
            if not prompt:
                raise ValueError("`prompt` list must not be empty.")
            prompt = prompt[0]

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        vc_engine = self._get_vc_engine()
        refine_engine = self._get_refine_engine()
        # Refinement in the original pipeline does not use CFG; ensure
        # do_classifier_free_guidance is False on the refine engine.
        setattr(refine_engine, "_guidance_scale", 1.0)
        video, video_fps = self._load_video(video, return_fps=True)
        stride = max(1, round(video_fps / fps))

        # ---- Stage 1: Base VC continuation at 480p -------------------------
        self.logger.info("[LongCatContinuation] Stage 1: base VC continuation")

        vc_output = vc_engine.run(
            video=video[::stride],
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            duration=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
            **kwargs,
        )
        base_vc_video: List[Image.Image] = vc_output[0]

        if not base_vc_video:
            raise RuntimeError("VC engine returned an empty video.")

        # ---- Stage 2: Optional distilled VC pass ---------------------------
        self.logger.info("[LongCatContinuation] Stage 2: distilled VC pass")

        if enable_distill:
            distill_output = vc_engine.run(
                video=video[::stride],
                prompt=prompt,
                negative_prompt=None,
                resolution=resolution,
                duration=duration,
                num_cond_frames=num_cond_frames,
                num_inference_steps=distill_num_inference_steps,
                use_distill=True,
                guidance_scale=distill_guidance_scale,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=False,
                **kwargs,
            )
            distilled_video: List[Image.Image] = distill_output[0]
        else:
            distilled_video = base_vc_video

        # ---- Stage 3: Optional refinement to higher resolution -------------
        self.logger.info("[LongCatContinuation] Stage 3: refinement")

        if not enable_refine:
            final_video = distilled_video
            if return_intermediates:
                return {
                    "stage1": [base_vc_video],
                    "stage2": [distilled_video],
                    "stage3": [final_video],
                }
            return [final_video]

        # Use a reasonable target fps for conditioning video as in the demo:
        # 15 fps when only spatial refinement, 30 fps when also temporal.
        target_fps = 15 if spatial_refine_only else 30
        cond_video_frames, _fps = self._load_video(
            video, fps=target_fps, return_fps=True
        )

        cur_num_cond_frames = (
            num_cond_frames if spatial_refine_only else num_cond_frames * 2
        )

        refine_output = refine_engine.run(
            image=None,
            video=cond_video_frames,
            prompt=prompt,
            stage1_video=distilled_video,
            num_cond_frames=cur_num_cond_frames,
            num_inference_steps=refine_num_inference_steps,
            generator=generator,
            spatial_refine_only=spatial_refine_only,
            **kwargs,
        )

        refined_video: List[Image.Image] = refine_output[0]

        final_video = refined_video

        if return_intermediates:
            return {
                "stage1": [base_vc_video],
                "stage2": [distilled_video],
                "stage3": [final_video],
            }

        return [final_video]
