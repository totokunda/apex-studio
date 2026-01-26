from typing import List, Optional, Union

import torch
from PIL import Image

from .shared import LongCatShared
from .t2v import LongCatT2VEngine
from .vc import LongCatVCEngine
from .refine import LongCatRefineEngine


class LongCatLongVideoEngine(LongCatShared):
    """
    LongCat long‑video engine.

    This engine mirrors `run_demo_long_video.py` by chaining:

    1. Text‑to‑video (T2V) to generate the first segment.
    2. Repeated video continuation (VC) with the same prompt.
    3. Optional 720p refinement over the full sequence.
    """

    def __init__(self, yaml_path: str, **kwargs):
        self._yaml_path = yaml_path
        super().__init__(yaml_path, **kwargs)

        self._t2v_engine: Optional[LongCatT2VEngine] = None
        self._vc_engine: Optional[LongCatVCEngine] = None
        self._refine_engine: Optional[LongCatRefineEngine] = None

    def _get_t2v_engine(self) -> LongCatT2VEngine:
        if self._t2v_engine is None:
            self._t2v_engine = self.sub_engines["t2v"]
        return self._t2v_engine

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
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_segments: int = 11,
        duration: str | int = 93,
        fps: int = 15,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        resolution: str = "480p",
        spatial_refine_only: bool = False,
        enable_refine: bool = True,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        return_intermediates: bool = False,
        **kwargs,
    ):
        """
        Generate a long video from a single prompt.

        Args:
            prompt: Prompt string (or list of one string) used for all segments.
            negative_prompt: Optional negative prompt for T2V/VC.
            num_segments: Number of continuation segments (excluding the first T2V).
            duration: Duration of the video in seconds or frames.
            fps: Frames per second.
            num_cond_frames: Conditioning frames for VC.
            num_inference_steps: Diffusion steps for T2V/VC (and refine if enabled).
            guidance_scale: CFG scale for T2V/VC.
            resolution: Resolution bucket for VC ("480p" or "720p").
            spatial_refine_only: Passed through to refinement engine.
            enable_refine: If False, skip the 720p refinement stage.
            generator: Optional torch.Generator.
            seed: Optional seed when generator is not provided.
            return_intermediates: If True, return dict with stage videos.

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

        t2v_engine = self._get_t2v_engine()
        vc_engine = self._get_vc_engine()
        refine_engine = self._get_refine_engine()
        # Refinement in the original pipeline does not use CFG; ensure
        # do_classifier_free_guidance is False on the refine engine.
        setattr(refine_engine, "_guidance_scale", 1.0)

        # ---- Stage 1: base T2V segment ------------------------------------
        self.logger.info("[LongCatLongVideo] Stage 1: T2V base segment")

        t2v_output = t2v_engine.run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )
        base_video: List[Image.Image] = t2v_output[0]

        if not base_video:
            raise RuntimeError("T2V engine returned an empty video.")

        target_size = base_video[0].size
        all_generated_frames: List[Image.Image] = list(base_video)
        current_video: List[Image.Image] = base_video

        # ---- Stage 2: VC continuation for `num_segments` segments ----------
        self.logger.info(
            f"[LongCatLongVideo] Stage 2: VC continuation over {num_segments} segments"
        )

        stage2_full_video: List[Image.Image] = list(base_video)

        for segment_idx in range(num_segments):
            self.logger.info(
                f"[LongCatLongVideo] VC segment {segment_idx + 1}/{num_segments}"
            )

            vc_output = vc_engine.run(
                video=current_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                duration=num_frames,
                fps=fps,
                num_cond_frames=num_cond_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=False,
                enhance_hf=True,
                **kwargs,
            )
            new_video: List[Image.Image] = vc_output[0]
            all_generated_frames.extend(new_video[num_cond_frames:])
            stage2_full_video.extend(new_video[num_cond_frames:])

            current_video = new_video

        if not enable_refine:
            final_video = stage2_full_video
            if return_intermediates:
                return {
                    "stage1": [base_video],
                    "stage2": [stage2_full_video],
                    "stage3": [final_video],
                }
            return [final_video]

        # ---- Stage 3: 720p refinement across the full generated sequence ---
        self.logger.info("[LongCatLongVideo] Stage 3: 720p refinement")

        all_refine_frames: List[Image.Image] = []
        cur_condition_video: Optional[List[Image.Image]] = None
        cur_num_cond_frames = 0
        start_id = 0

        total_segments = num_segments + 1
        for segment_idx in range(total_segments):
            self.logger.info(
                f"[LongCatLongVideo] Refining segment {segment_idx + 1}/{total_segments}"
            )

            segment_stage1 = all_generated_frames[start_id : start_id + num_frames]
            if not segment_stage1:
                break

            refine_output = refine_engine.run(
                image=None,
                video=cur_condition_video,
                prompt="",
                stage1_video=segment_stage1,
                num_cond_frames=cur_num_cond_frames,
                num_inference_steps=num_inference_steps,
                generator=generator,
                spatial_refine_only=spatial_refine_only,
                **kwargs,
            )

            refined_segment: List[Image.Image] = refine_output[0]

            all_refine_frames.extend(refined_segment[cur_num_cond_frames:])
            cur_condition_video = refined_segment

            cur_num_cond_frames = (
                num_cond_frames if spatial_refine_only else num_cond_frames * 2
            )
            start_id = start_id + num_frames - num_cond_frames

        final_video = all_refine_frames or stage2_full_video

        if return_intermediates:
            return {
                "stage1": [base_video],
                "stage2": [stage2_full_video],
                "stage3": [final_video],
            }

        return [final_video]
