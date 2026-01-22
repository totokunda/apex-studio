from typing import List, Optional, Union
import torch
from PIL import Image
from .shared import LongCatShared
from .t2v import LongCatT2VEngine
from .vc import LongCatVCEngine
from .refine import LongCatRefineEngine


class LongCatInteractiveEngine(LongCatShared):
    """
    LongCat interactive long-video engine.

    This engine mirrors the behaviour of
    `run_demo_interactive_video.py` by composing the existing
    `LongCatT2VEngine`, `LongCatVCEngine`, and `LongCatRefineEngine`
    engines in three stages:

    1. Text → short video (T2V)
    2. Iterative video continuation (VC) across multiple segments
    3. High‑resolution refinement over all segments
    """

    def __init__(self, yaml_path: str, **kwargs):
        # Keep track of the manifest path so we can spin up child engines
        self._yaml_path = yaml_path
        super().__init__(yaml_path, **kwargs)

        self._t2v_engine: Optional[LongCatT2VEngine] = None
        self._vc_engine: Optional[LongCatVCEngine] = None
        self._refine_engine: Optional[LongCatRefineEngine] = None

    # Lazily create and cache sub‑engines so we share config and device
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
        prompts: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        duration: str | int = 93,
        fps: int = 15,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        resolution: str = "480p",
        spatial_refine_only: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        return_intermediates: bool = False,
        **kwargs,
    ):
        """
        Generate an interactive long video from a list of prompts.

        Args:
            prompts: List of prompts. The first prompt is used for the
                initial T2V segment, and each subsequent prompt is used
                for one VC continuation segment.
            negative_prompt: Optional negative prompt used for T2V/VC stages.
            num_frames: Number of frames per segment for the base 480p run.
            num_cond_frames: Number of conditioning frames for each VC segment.
            num_inference_steps: Diffusion sampling steps for all stages.
            guidance_scale: Classifier‑free guidance scale for T2V/VC.
            resolution: Resolution bucket for VC stage ("480p" or "720p").
            spatial_refine_only: If True, refinement only increases spatial
                resolution; otherwise it also doubles temporal resolution.
            generator: Optional torch.Generator to control randomness.
            seed: Optional seed used when `generator` is not provided.
            return_intermediates: If True, also return intermediate stage
                videos.

        Returns:
            List[List[Image.Image]]: Outer list is batch (size 1),
            inner list is the final refined video frames.
            If `return_intermediates=True`, a dict is returned with:
                {
                    "stage1": [frames],
                    "stage2": [frames],
                    "stage3": [frames],
                }
        """
        num_frames = self._parse_num_frames(duration, fps)
        if prompt is not None and not prompts:
            prompts = self.split_into_sentences(prompt)
            prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not prompts:
            raise ValueError("`prompts` must contain at least one prompt.")

        if len(prompts) == 1:
            # Degenerate case: treat as a single‑segment long video
            prompts = [prompts[0], prompts[0]]

        num_segments = len(prompts) - 1

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        t2v_engine = self._get_t2v_engine()
        vc_engine = self._get_vc_engine()
        refine_engine = self._get_refine_engine()
        # Refinement in the original pipeline does not use CFG; ensure
        # do_classifier_free_guidance is False on the refine engine.
        setattr(refine_engine, "_guidance_scale", 1.0)

        # ---- Stage 1: T2V base segment ------------------------------------
        self.logger.info("[LongCatInteractive] Stage 1: T2V base segment")

        t2v_output = t2v_engine.run(
            prompt=prompts[0],
            negative_prompt=negative_prompt,
            duration=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )
        self.offload_engine(t2v_engine)
        # Engines return [OutputVideo]; we operate on the single sample
        base_video: List[Image.Image] = t2v_output[0]

        if not base_video:
            raise RuntimeError("T2V engine returned an empty video.")

        target_size = base_video[0].size
        all_generated_frames: List[Image.Image] = list(base_video)
        current_video: List[Image.Image] = base_video

        # ---- Stage 2: VC long video over multiple segments -----------------
        self.logger.info(
            f"[LongCatInteractive] Stage 2: VC continuation over {num_segments} segments"
        )

        stage2_full_video: List[Image.Image] = list(base_video)
        with self._progress_bar(total=num_segments, desc="VC continuation") as pbar:
            for segment_idx in range(num_segments):
                self.logger.info(
                    f"[LongCatInteractive] VC segment {segment_idx + 1}/{num_segments}"
                )

                vc_output = vc_engine.run(
                    video=current_video,
                    prompt=prompts[segment_idx + 1],
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

                # Append only the newly generated continuation frames
                all_generated_frames.extend(new_video[num_cond_frames:])
                stage2_full_video.extend(new_video[num_cond_frames:])

                current_video = new_video
                pbar.update()

        # ---- Stage 3: 720p refinement across all segments ------------------
        self.logger.info(
            "[LongCatInteractive] Stage 3: 720p refinement over all segments"
        )

        all_refine_frames: List[Image.Image] = []
        cur_condition_video: Optional[List[Image.Image]] = None
        cur_num_cond_frames = 0
        start_id = 0
        self.offload_engine(vc_engine)

        with self._progress_bar(total=num_segments + 1, desc="Refining") as pbar:
            for segment_idx in range(num_segments + 1):
                self.logger.info(
                    f"[LongCatInteractive] Refining segment {segment_idx + 1}/{num_segments + 1}"
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

                # Append only newly refined frames beyond the conditioning window
                all_refine_frames.extend(refined_segment[cur_num_cond_frames:])
                cur_condition_video = refined_segment

                cur_num_cond_frames = (
                    num_cond_frames if spatial_refine_only else num_cond_frames * 2
                )
                start_id = start_id + num_frames - num_cond_frames
                pbar.update()

        self.offload_engine(refine_engine)
        final_video = all_refine_frames or stage2_full_video

        if return_intermediates:
            return {
                "stage1": [base_video],
                "stage2": [stage2_full_video],
                "stage3": [final_video],
            }

        # Match other engines: outer list is batch dimension
        return [final_video]
