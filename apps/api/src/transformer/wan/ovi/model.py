import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from .wan_base import (
    WanModel,
    WanLayerNorm,
    gradient_checkpointing,
)
from src.transformer.efficiency.mod import InplaceRMSNorm
from typing import Dict, Any
from loguru import logger


class OviFusionBlock(nn.Module):
    """
    A single fused audio+video block wrapper.

    Key property: this wrapper is called exactly once per layer in `OviModel.forward`.
    When group offloading targets `OviModel.fusion_blocks`, hooks attach to these wrappers
    (not to the underlying audio/video blocks), avoiding repeated hook execution within
    a single forward pass.
    """

    def __init__(self, vid_block: nn.Module, audio_block: nn.Module):
        super().__init__()
        self.vid_block = vid_block
        self.audio_block = audio_block

    def forward(
        self,
        *,
        vid,
        audio,
        vid_e,
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
        vid_context,
        vid_context_lens,
        audio_e,
        audio_seq_lens,
        audio_grid_sizes,
        audio_freqs,
        audio_context,
        audio_context_lens,
    ):
        # Audio: modulation + self-attention
        audio, audio_e_chunked = self.audio_block(
            x=audio,
            e=audio_e,
            seq_lens=audio_seq_lens,
            grid_sizes=audio_grid_sizes,
            freqs=audio_freqs,
            context=audio_context,
            context_lens=audio_context_lens,
            mode="modulation_self_attn",
        )

        # Video: modulation + self-attention
        vid, vid_e_chunked = self.vid_block(
            x=vid,
            e=vid_e,
            seq_lens=vid_seq_lens,
            grid_sizes=vid_grid_sizes,
            freqs=vid_freqs,
            context=vid_context,
            context_lens=vid_context_lens,
            mode="modulation_self_attn",
        )

        og_audio = audio

        # Audio: fusion cross-attention + FFN (attends to video)
        audio = self.audio_block(
            x=audio,
            e=audio_e_chunked,
            seq_lens=audio_seq_lens,
            grid_sizes=audio_grid_sizes,
            freqs=audio_freqs,
            context=audio_context,
            context_lens=audio_context_lens,
            mode="fusion_cross_attn_ffn",
            target_seq=vid,
            target_seq_lens=vid_seq_lens,
            target_grid_sizes=vid_grid_sizes,
            target_freqs=vid_freqs,
        )

        # Video: fusion cross-attention + FFN (attends to og_audio)
        vid = self.vid_block(
            x=vid,
            e=vid_e_chunked,
            seq_lens=vid_seq_lens,
            grid_sizes=vid_grid_sizes,
            freqs=vid_freqs,
            context=vid_context,
            context_lens=vid_context_lens,
            mode="fusion_cross_attn_ffn",
            target_seq=og_audio,
            target_seq_lens=audio_seq_lens,
            target_grid_sizes=audio_grid_sizes,
            target_freqs=audio_freqs,
        )

        return vid, audio


class OviModel(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self, video: Dict[str, Any] | None = None, audio: Dict[str, Any] | None = None
    ):
        super().__init__()
        has_video = True
        has_audio = True
        if video is not None:
            self.video_model = WanModel(**video)
        else:
            has_video = False
            self.video_model = None
            logger.warning("Warning: No video model is provided!")

        if audio is not None:
            self.audio_model = WanModel(**audio)
        else:
            has_audio = False
            self.audio_model = None
            logger.warning("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.inject_cross_attention_kv_projections()

            vid_blocks = list(self.video_model.blocks)
            audio_blocks = list(self.audio_model.blocks)

            # IMPORTANT:
            # - We want group offloading to target `fusion_blocks`, so each offloaded "block"
            #   is invoked once per iteration.
            # - We therefore move the underlying blocks out of `video_model.blocks` /
            #   `audio_model.blocks` and into the wrappers below.
            #
            # This prevents block-level offloading hooks from attaching to the raw blocks
            # (which are invoked multiple times per iteration in the fused schedule).
            self.video_model.blocks = nn.ModuleList()
            self.audio_model.blocks = nn.ModuleList()

            self.fusion_blocks = nn.ModuleList(
                [
                    OviFusionBlock(vid_block=v, audio_block=a)
                    for v, a in zip(vid_blocks, audio_blocks)
                ]
            )
            self.num_blocks = len(self.fusion_blocks)

        self.gradient_checkpointing = False
        self.init_weights()

    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(
                vid_block.dim, elementwise_affine=True
            )
            vid_block.cross_attn.norm_k_fusion = (
                InplaceRMSNorm(vid_block.dim, eps=1e-6, elementwise_affine=True)
                if vid_block.qk_norm
                else nn.Identity()
            )

        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(
                audio_block.dim, audio_block.dim
            )
            audio_block.cross_attn.v_fusion = nn.Linear(
                audio_block.dim, audio_block.dim
            )
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(
                audio_block.dim, elementwise_affine=True
            )
            audio_block.cross_attn.norm_k_fusion = (
                InplaceRMSNorm(audio_block.dim, eps=1e-6, elementwise_affine=True)
                if audio_block.qk_norm
                else nn.Identity()
            )

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def single_fusion_block_forward(
        self,
        vid_block,
        audio_block,
        vid,
        audio,
        vid_e,
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
        vid_context,
        vid_context_lens,
        audio_e,
        audio_seq_lens,
        audio_grid_sizes,
        audio_freqs,
        audio_context,
        audio_context_lens,
    ):
        """
        Fusion block forward that calls block.forward() with mode parameter.
        All block operations happen inside forward() for offloading compatibility.
        """
        # Debug: check input tensors
        if torch.isnan(vid).any():
            print(f"NaN in vid input!")
        if torch.isnan(audio).any():
            print(f"NaN in audio input!")
        if torch.isnan(vid_e).any():
            print(f"NaN in vid_e!")
        if torch.isnan(audio_e).any():
            print(f"NaN in audio_e!")

        # Audio: modulation + self-attention (via forward with mode)
        audio, audio_e_chunked = audio_block(
            x=audio,
            e=audio_e,
            seq_lens=audio_seq_lens,
            grid_sizes=audio_grid_sizes,
            freqs=audio_freqs,
            context=audio_context,
            context_lens=audio_context_lens,
            mode="modulation_self_attn",
        )
        if torch.isnan(audio).any():
            print(f"NaN after audio modulation_self_attn!")

        # Video: modulation + self-attention (via forward with mode)
        vid, vid_e_chunked = vid_block(
            x=vid,
            e=vid_e,
            seq_lens=vid_seq_lens,
            grid_sizes=vid_grid_sizes,
            freqs=vid_freqs,
            context=vid_context,
            context_lens=vid_context_lens,
            mode="modulation_self_attn",
        )
        if torch.isnan(vid).any():
            print(f"NaN after vid modulation_self_attn!")

        og_audio = audio

        # Audio: fusion cross-attention + FFN (attends to video, via forward with mode)
        audio = audio_block(
            x=audio,
            e=audio_e_chunked,
            seq_lens=audio_seq_lens,
            grid_sizes=audio_grid_sizes,
            freqs=audio_freqs,
            context=audio_context,
            context_lens=audio_context_lens,
            mode="fusion_cross_attn_ffn",
            target_seq=vid,
            target_seq_lens=vid_seq_lens,
            target_grid_sizes=vid_grid_sizes,
            target_freqs=vid_freqs,
        )
        if torch.isnan(audio).any():
            print(f"NaN after audio fusion_cross_attn_ffn!")

        # Video: fusion cross-attention + FFN (attends to og_audio, via forward with mode)
        vid = vid_block(
            x=vid,
            e=vid_e_chunked,
            seq_lens=vid_seq_lens,
            grid_sizes=vid_grid_sizes,
            freqs=vid_freqs,
            context=vid_context,
            context_lens=vid_context_lens,
            mode="fusion_cross_attn_ffn",
            target_seq=og_audio,
            target_seq_lens=audio_seq_lens,
            target_grid_sizes=audio_grid_sizes,
            target_freqs=audio_freqs,
        )
        if torch.isnan(vid).any():
            print(f"NaN after vid fusion_cross_attn_ffn!")

        return vid, audio

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        # Route to easycache_forward if fusion cache is enabled
        if getattr(self, "fusion_cache_enabled", False):
            return self.easycache_forward(
                vid=vid,
                audio=audio,
                t=t,
                vid_context=vid_context,
                audio_context=audio_context,
                vid_seq_len=vid_seq_len,
                audio_seq_len=audio_seq_len,
                clip_fea=clip_fea,
                clip_fea_audio=clip_fea_audio,
                y=y,
                first_frame_is_clean=first_frame_is_clean,
                slg_layer=slg_layer,
            )

        assert clip_fea is None
        assert y is None

        if vid is None or all([x is None for x in vid]):
            raise ValueError(
                "OviModel requires both `vid` and `audio` inputs for fused inference."
            )

        if audio is None or all([x is None for x in audio]):
            raise ValueError(
                "OviModel requires both `vid` and `audio` inputs for fused inference."
            )

        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid,
            t=t,
            context=vid_context,
            seq_len=vid_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
        )

        # NOTE: avoid per-step prints in production; this can severely slow generation.
        # Use logging at DEBUG level if needed.

        audio, audio_e, audio_kwargs = (
            self.audio_model.prepare_transformer_block_kwargs(
                x=audio,
                t=t,
                context=audio_context,
                seq_len=audio_seq_len,
                clip_fea=clip_fea_audio,
                y=None,
                first_frame_is_clean=False,
            )
        )

        # NOTE: avoid per-step prints in production; this can severely slow generation.
        # Use logging at DEBUG level if needed.

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i, block in enumerate(self.fusion_blocks):
            if slg_layer > 0 and i == slg_layer:
                continue
            vid, audio = gradient_checkpointing(
                enabled=(self.training and self.gradient_checkpointing),
                module=block,
                vid=vid,
                audio=audio,
                **kwargs,
            )

        vid = self.video_model.post_transformer_block_out(
            vid, vid_kwargs["grid_sizes"], vid_e
        )
        audio = self.audio_model.post_transformer_block_out(
            audio, audio_kwargs["grid_sizes"], audio_e
        )

        return vid, audio

    def enable_fusion_easy_cache(
        self,
        num_steps: int,
        thresh: float,
        ret_steps: int = 10 * 2,
        cutoff_steps: int | None = None,
        *,
        cache_on_cpu: bool = False,
        cache_cpu_dtype: torch.dtype = torch.bfloat16,
        cache_pin_memory: bool = False,
        rope_on_cpu: bool = False,
    ):
        """Enable easy cache for the fused OVI model."""
        # Allow environment-driven defaults so this can be enabled without changing call sites.
        # Params still take precedence when explicitly passed.
        if not cache_on_cpu and os.getenv("APEX_FUSION_CACHE_ON_CPU", "0") == "1":
            cache_on_cpu = True
        if not rope_on_cpu and os.getenv("APEX_ROPE_ON_CPU", "0") == "1":
            rope_on_cpu = True
        
 
        self.fusion_cache_enabled = True
        self.num_steps = num_steps
        self.thresh = thresh
        self.ret_steps = ret_steps
        self.cutoff_steps = cutoff_steps
        self.cnt = 0
        self.k_vid = None
        self.k_audio = None
        self.accumulated_error_even = 0
        self.should_calc_current_pair = True
        # Cache options
        self.fusion_cache_on_cpu = bool(cache_on_cpu)
        self.fusion_cache_cpu_dtype = cache_cpu_dtype
        self.fusion_cache_pin_memory = bool(cache_pin_memory)
        self.fusion_cache_rope_on_cpu = bool(rope_on_cpu)
        # Also propagate to the underlying WAN submodels (so non-cache forwards can reuse it).
        if getattr(self, "video_model", None) is not None:
            setattr(self.video_model, "rope_on_cpu", bool(rope_on_cpu))
        if getattr(self, "audio_model", None) is not None:
            setattr(self.audio_model, "rope_on_cpu", bool(rope_on_cpu))
        # Clear any existing cache state
        self.previous_raw_vid_input_even = None
        self.previous_raw_audio_input_even = None
        self.previous_raw_vid_output_even = None
        self.previous_raw_audio_output_even = None
        self.previous_raw_vid_output_odd = None
        self.previous_raw_audio_output_odd = None
        self.cache_vid_even = None
        self.cache_audio_even = None
        self.cache_vid_odd = None
        self.cache_audio_odd = None
        self.prev_prev_raw_vid_input_even = None
        self.prev_prev_raw_audio_input_even = None
        self.prev_vid_out_norm_even = None
        self.prev_audio_out_norm_even = None

    def reset_fusion_cache(self):
        """Reset cache state for a new generation."""
        self.cnt = 0
        self.k_vid = None
        self.k_audio = None
        self.accumulated_error_even = 0
        self.should_calc_current_pair = True
        self.previous_raw_vid_input_even = None
        self.previous_raw_audio_input_even = None
        self.previous_raw_vid_output_even = None
        self.previous_raw_audio_output_even = None
        self.previous_raw_vid_output_odd = None
        self.previous_raw_audio_output_odd = None
        self.cache_vid_even = None
        self.cache_audio_even = None
        self.cache_vid_odd = None
        self.cache_audio_odd = None
        self.prev_prev_raw_vid_input_even = None
        self.prev_prev_raw_audio_input_even = None
        self.prev_vid_out_norm_even = None
        self.prev_audio_out_norm_even = None

    def easycache_forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        """
        Forward with easy caching for the fused OVI model.
        Uses input change prediction to skip computation when possible.
        """
        assert clip_fea is None
        assert y is None

        # Handle single-modality cases (fall back to normal forward)
        if vid is None or all([x is None for x in vid]):
            raise ValueError(
                "OviModel requires both `vid` and `audio` inputs for fused inference."
            )

        if audio is None or all([x is None for x in audio]):
            raise ValueError(
                "OviModel requires both `vid` and `audio` inputs for fused inference."
            )
   
        # NOTE: we intentionally avoid cloning raw inputs here to keep VRAM low.
        # The diffusion loop should treat inputs as immutable per-step.
        raw_vid_input = vid
        raw_audio_input = audio

        cache_on_cpu = bool(getattr(self, "fusion_cache_on_cpu", False))
        cache_cpu_dtype = getattr(self, "fusion_cache_cpu_dtype", torch.bfloat16)
        cache_pin_memory = bool(getattr(self, "fusion_cache_pin_memory", False))
        rope_on_cpu = bool(getattr(self, "fusion_cache_rope_on_cpu", False))

        def _maybe_pin(x: torch.Tensor) -> torch.Tensor:
            if cache_pin_memory and x.device.type == "cpu" and not x.is_pinned():
                return x.pin_memory()
            return x

        def _to_cpu_list(xs):
            # Detach to avoid holding graphs; optionally pin for faster H2D/D2H.
            return [_maybe_pin(u.detach().to("cpu", dtype=cache_cpu_dtype)) for u in xs]

        def _l1_mean_lists(a_list, b_list):
            # Global mean L1 across a list of tensors of potentially different shapes.
            total = None
            total_elems = 0
            for a, b in zip(a_list, b_list):
                n = int(a.numel())
                if n == 0:
                    continue
                m = F.l1_loss(a, b, reduction="mean")
                total = (m * n) if total is None else (total + m * n)
                total_elems += n
            if total is None:
                # Shouldn't happen for non-empty inputs, but keep it safe.
                return torch.tensor(0.0, device=a_list[0].device)
            return total / max(total_elems, 1)

        def _mean_abs_lists(x_list):
            # Mean absolute value without materializing abs tensors.
            total = None
            total_elems = 0
            for x in x_list:
                n = int(x.numel())
                if n == 0:
                    continue
                s = torch.linalg.vector_norm(x, ord=1)
                total = s if total is None else (total + s)
                total_elems += n
            if total is None:
                return torch.tensor(0.0, device=x_list[0].device)
            return total / max(total_elems, 1)

        # Track which type of step (even=condition, odd=uncondition)
        self.is_even = self.cnt % 2 == 0

        # Only make decision on even (condition) steps
        if self.is_even:
            # Keep references to the previous inputs for change estimation.
            # IMPORTANT: do NOT overwrite these until we are about to return, otherwise
            # input_change collapses to ~0 and k-factor estimation breaks.
            prev_vid_input_even = self.previous_raw_vid_input_even
            prev_audio_input_even = self.previous_raw_audio_input_even

            # Always compute first ret_steps and last steps
            if self.cnt < self.ret_steps or self.cnt >= (
                (
                    (
                        getattr(self, "low_start_step", None) is not None
                        and getattr(self, "is_high_noise", False)
                    )
                    and (self.low_start_step - 1) * 2 - 2
                )
                or (
                    (
                        getattr(self, "low_start_step", None) is not None
                        and not getattr(self, "is_high_noise", False)
                    )
                    and (self.num_steps - self.low_start_step) * 2 - 2
                )
                or (self.num_steps * 2 - 2)
            ):
                self.should_calc_current_pair = True
                self.accumulated_error_even = 0
            else:
                # Check if we have previous step data for comparison
                if (
                    prev_vid_input_even is not None
                    and prev_audio_input_even is not None
                    and self.cache_vid_even is not None
                    and self.cache_audio_even is not None
                    and self.prev_vid_out_norm_even is not None
                    and self.prev_audio_out_norm_even is not None
                    and self.k_vid is not None
                    and self.k_audio is not None
                ):
                    if cache_on_cpu:
                        vid_now = _to_cpu_list(raw_vid_input)
                        audio_now = _to_cpu_list(raw_audio_input)
                        vid_input_change = _l1_mean_lists(
                            vid_now, prev_vid_input_even
                        )
                        audio_input_change = _l1_mean_lists(
                            audio_now, prev_audio_input_even
                        )
                    else:
                        vid_input_change = _l1_mean_lists(
                            raw_vid_input, prev_vid_input_even
                        )
                        audio_input_change = _l1_mean_lists(
                            raw_audio_input, prev_audio_input_even
                        )

                    # Predicted relative change: k * (input_change / prev_output_norm)
                    vid_pred_change = self.k_vid * (
                        float(vid_input_change) / (float(self.prev_vid_out_norm_even) + 1e-12)
                    )
                    audio_pred_change = self.k_audio * (
                        float(audio_input_change)
                        / (float(self.prev_audio_out_norm_even) + 1e-12)
                    )

                    combined_pred_change = (
                        vid_pred_change
                        if vid_pred_change >= audio_pred_change
                        else audio_pred_change
                    )
                    self.accumulated_error_even = float(self.accumulated_error_even) + float(
                        combined_pred_change
                    )

                    if float(self.accumulated_error_even) < float(self.thresh):
                        self.should_calc_current_pair = False
                    else:
                        self.should_calc_current_pair = True
                        self.accumulated_error_even = 0
                else:
                    # No previous data yet, must calculate
                    self.should_calc_current_pair = True

        # Check if we can use cached output and return early
        if (
            self.is_even
            and not self.should_calc_current_pair
            and self.cache_vid_even is not None
            and self.cache_audio_even is not None
        ):
            # Use cached output directly
            self.cnt += 1
            # Advance the "previous input" reference for the next decision.
            if cache_on_cpu:
                self.previous_raw_vid_input_even = _to_cpu_list(raw_vid_input)
                self.previous_raw_audio_input_even = _to_cpu_list(raw_audio_input)
            else:
                self.previous_raw_vid_input_even = [u.detach() for u in raw_vid_input]
                self.previous_raw_audio_input_even = [
                    u.detach() for u in raw_audio_input
                ]
            if cache_on_cpu:
                vid_out = []
                for u, v_cpu in zip(raw_vid_input, self.cache_vid_even):
                    v = v_cpu.to(device=u.device, dtype=u.dtype, non_blocking=True)
                    vid_out.append((u + v).float())
                audio_out = []
                for u, v_cpu in zip(raw_audio_input, self.cache_audio_even):
                    v = v_cpu.to(device=u.device, dtype=u.dtype, non_blocking=True)
                    audio_out.append((u + v).float())
                return vid_out, audio_out
            return (
                [(u + v).float() for u, v in zip(raw_vid_input, self.cache_vid_even)],
                [
                    (u + v).float()
                    for u, v in zip(raw_audio_input, self.cache_audio_even)
                ],
            )

        elif (
            not self.is_even
            and not self.should_calc_current_pair
            and self.cache_vid_odd is not None
            and self.cache_audio_odd is not None
        ):
            # Use cached output directly
            self.cnt += 1
            if cache_on_cpu:
                vid_out = []
                for u, v_cpu in zip(raw_vid_input, self.cache_vid_odd):
                    v = v_cpu.to(device=u.device, dtype=u.dtype, non_blocking=True)
                    vid_out.append((u + v).float())
                audio_out = []
                for u, v_cpu in zip(raw_audio_input, self.cache_audio_odd):
                    v = v_cpu.to(device=u.device, dtype=u.dtype, non_blocking=True)
                    audio_out.append((u + v).float())
                return vid_out, audio_out
            return (
                [(u + v).float() for u, v in zip(raw_vid_input, self.cache_vid_odd)],
                [
                    (u + v).float()
                    for u, v in zip(raw_audio_input, self.cache_audio_odd)
                ],
            )

        # Continue with normal processing since we need to calculate
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid,
            t=t,
            context=vid_context,
            seq_len=vid_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
            rope_on_cpu=rope_on_cpu,
        )

        audio, audio_e, audio_kwargs = (
            self.audio_model.prepare_transformer_block_kwargs(
                x=audio,
                t=t,
                context=audio_context,
                seq_len=audio_seq_len,
                clip_fea=clip_fea_audio,
                y=None,
                first_frame_is_clean=False,
                rope_on_cpu=rope_on_cpu,
            )
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i, block in enumerate(self.fusion_blocks):
            if slg_layer > 0 and i == slg_layer:
                continue
            vid, audio = gradient_checkpointing(
                enabled=(self.training and self.gradient_checkpointing),
                module=block,
                vid=vid,
                audio=audio,
                **kwargs,
            )
            

        vid_output = self.video_model.post_transformer_block_out(
            vid, vid_kwargs["grid_sizes"], vid_e
        )
        audio_output = self.audio_model.post_transformer_block_out(
            audio, audio_kwargs["grid_sizes"], audio_e
        )
        

        # Update cache and calculate change rates if needed
        if self.is_even:  # Condition path
            # Update k factors and cache delta for future skip reconstruction.
            if cache_on_cpu:
                vid_out_cpu = _to_cpu_list(vid_output)
                audio_out_cpu = _to_cpu_list(audio_output)
                vid_in_cpu = _to_cpu_list(raw_vid_input)
                audio_in_cpu = _to_cpu_list(raw_audio_input)
            
                # Compute k from true output_change / input_change (both as global means).
                if (
                    self.previous_raw_vid_output_even is not None
                    and self.previous_raw_audio_output_even is not None
                    and self.prev_prev_raw_vid_input_even is not None
                    and self.prev_prev_raw_audio_input_even is not None
                ):
                    vid_output_change = _l1_mean_lists(
                        vid_out_cpu, self.previous_raw_vid_output_even
                    )
                    audio_output_change = _l1_mean_lists(
                        audio_out_cpu, self.previous_raw_audio_output_even
                    )
                    vid_input_change = _l1_mean_lists(
                        vid_in_cpu, self.prev_prev_raw_vid_input_even
                    )
                    audio_input_change = _l1_mean_lists(
                        audio_in_cpu, self.prev_prev_raw_audio_input_even
                    )

                    self.k_vid = (
                        (vid_output_change / (vid_input_change + 1e-12)).item()
                        if float(vid_input_change) > 0.0
                        else 0.0
                    )
                    self.k_audio = (
                        (audio_output_change / (audio_input_change + 1e-12)).item()
                        if float(audio_input_change) > 0.0
                        else 0.0
                    )
                # Cache delta on CPU.
                self.cache_vid_even = [
                    _maybe_pin((u.detach() - v.detach()).to("cpu", dtype=cache_cpu_dtype))
                    for u, v in zip(vid_output, raw_vid_input)
                ]
                self.cache_audio_even = [
                    _maybe_pin((u.detach() - v.detach()).to("cpu", dtype=cache_cpu_dtype))
                    for u, v in zip(audio_output, raw_audio_input)
                ]
                self.previous_raw_vid_output_even = vid_out_cpu
                self.previous_raw_audio_output_even = audio_out_cpu
                self.prev_vid_out_norm_even = float(_mean_abs_lists(vid_out_cpu))
                self.prev_audio_out_norm_even = float(_mean_abs_lists(audio_out_cpu))
                # Track inputs:
                # - `previous_raw_*_input_even` follows the last even-step input (even if we skipped),
                #   used for incremental change detection.
                # - `prev_prev_raw_*_input_even` follows the last *computed* even-step input, used
                #   to align k estimation with `previous_raw_*_output_even`.
                self.previous_raw_vid_input_even = vid_in_cpu
                self.previous_raw_audio_input_even = audio_in_cpu
                self.prev_prev_raw_vid_input_even = vid_in_cpu
                self.prev_prev_raw_audio_input_even = audio_in_cpu
            else:
                # Approximate k using delta_change + input_change to avoid storing full previous outputs.
                vid_input_change = (
                    _l1_mean_lists(raw_vid_input, prev_vid_input_even)
                    if prev_vid_input_even is not None
                    else None
                )
                audio_input_change = (
                    _l1_mean_lists(raw_audio_input, prev_audio_input_even)
                    if prev_audio_input_even is not None
                    else None
                )

                # Cache delta on GPU.
                cache_vid = [u - v for u, v in zip(vid_output, raw_vid_input)]
                cache_audio = [u - v for u, v in zip(audio_output, raw_audio_input)]

                if (
                    vid_input_change is not None
                    and self.cache_vid_even is not None
                    and float(vid_input_change) > 0.0
                ):
                    vid_delta_change = _l1_mean_lists(cache_vid, self.cache_vid_even)
                    self.k_vid = float(
                        (vid_input_change + vid_delta_change)
                        / (vid_input_change + 1e-12)
                    )
                if (
                    audio_input_change is not None
                    and self.cache_audio_even is not None
                    and float(audio_input_change) > 0.0
                ):
                    audio_delta_change = _l1_mean_lists(cache_audio, self.cache_audio_even)
                    self.k_audio = float(
                        (audio_input_change + audio_delta_change)
                        / (audio_input_change + 1e-12)
                    )

                self.cache_vid_even = cache_vid
                self.cache_audio_even = cache_audio
                self.prev_vid_out_norm_even = float(_mean_abs_lists(vid_output))
                self.prev_audio_out_norm_even = float(_mean_abs_lists(audio_output))
                # Advance the "previous input" reference for the next decision.
                self.previous_raw_vid_input_even = [u.detach() for u in raw_vid_input]
                self.previous_raw_audio_input_even = [
                    u.detach() for u in raw_audio_input
                ]

        else:  # Uncondition path
            # Store cache delta for unconditional path.
            if cache_on_cpu:
                self.cache_vid_odd = [
                    _maybe_pin((u.detach() - v.detach()).to("cpu", dtype=cache_cpu_dtype))
                    for u, v in zip(vid_output, raw_vid_input)
                ]
                self.cache_audio_odd = [
                    _maybe_pin((u.detach() - v.detach()).to("cpu", dtype=cache_cpu_dtype))
                    for u, v in zip(audio_output, raw_audio_input)
                ]
            else:
                self.cache_vid_odd = [u - v for u, v in zip(vid_output, raw_vid_input)]
                self.cache_audio_odd = [
                    u - v for u, v in zip(audio_output, raw_audio_input)
                ]

        # Update counter
        self.cnt += 1
        return [u.float() for u in vid_output], [u.float() for u in audio_output]

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()

    def set_chunking_profile(self, profile_name: str) -> None:
        """
        Apply a predefined chunking profile to both video and audio models.
        """
        if self.video_model is not None:
            self.video_model.set_chunking_profile(profile_name)
        if self.audio_model is not None:
            self.audio_model.set_chunking_profile(profile_name)

    def list_chunking_profiles(self):
        """Return available chunking profile names."""
        if self.video_model is not None:
            return self.video_model.list_chunking_profiles()
        if self.audio_model is not None:
            return self.audio_model.list_chunking_profiles()
        return ()
