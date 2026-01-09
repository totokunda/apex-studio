import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from .wan_base import (
    WanModel,
    WanLayerNorm,
    WanRMSNorm,
    gradient_checkpointing,
)
from typing import Dict, Any
from loguru import logger


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
            self.num_blocks = len(self.video_model.blocks)
            self.inject_cross_attention_kv_projections()

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
                WanRMSNorm(vid_block.dim, eps=1e-6)
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
                WanRMSNorm(audio_block.dim, eps=1e-6)
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
            assert self.audio_model is not None

            return None, self.audio_model(
                x=audio,
                t=t,
                context=audio_context,
                seq_len=audio_seq_len,
                clip_fea=clip_fea_audio,
                y=None,
            )

        if audio is None or all([x is None for x in audio]):
            assert self.video_model is not None

            return (
                self.video_model(
                    x=vid,
                    t=t,
                    context=vid_context,
                    seq_len=vid_seq_len,
                    clip_fea=clip_fea,
                    y=y,
                    first_frame_is_clean=first_frame_is_clean,
                ),
                None,
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
        
        print("vid", vid[0].mean(), "vid_e", vid_e[0].mean())

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
        
        print("audio", audio[0].mean(), "audio_e", audio_e[0].mean())

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            vid, audio = gradient_checkpointing(
                enabled=(self.training and self.gradient_checkpointing),
                module=self.single_fusion_block_forward,
                vid_block=vid_block,
                audio_block=audio_block,
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
    ):
        """Enable easy cache for the fused OVI model."""
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
            assert self.audio_model is not None
            return None, self.audio_model(
                x=audio,
                t=t,
                context=audio_context,
                seq_len=audio_seq_len,
                clip_fea=clip_fea_audio,
                y=None,
            )

        if audio is None or all([x is None for x in audio]):
            assert self.video_model is not None
            return (
                self.video_model(
                    x=vid,
                    t=t,
                    context=vid_context,
                    seq_len=vid_seq_len,
                    clip_fea=clip_fea,
                    y=y,
                    first_frame_is_clean=first_frame_is_clean,
                ),
                None,
            )

        # Store original raw inputs for caching
        raw_vid_input = [u.clone() for u in vid]
        raw_audio_input = [u.clone() for u in audio]

        # Track which type of step (even=condition, odd=uncondition)
        self.is_even = self.cnt % 2 == 0

        # Only make decision on even (condition) steps
        if self.is_even:
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
                    self.previous_raw_vid_input_even is not None
                    and self.previous_raw_vid_output_even is not None
                    and self.previous_raw_audio_input_even is not None
                    and self.previous_raw_audio_output_even is not None
                ):

                    # Calculate input changes for video
                    vid_input_change = (
                        torch.cat(
                            [
                                (u - v).flatten()
                                for u, v in zip(
                                    raw_vid_input, self.previous_raw_vid_input_even
                                )
                            ]
                        )
                        .abs()
                        .mean()
                    )

                    # Calculate input changes for audio
                    audio_input_change = (
                        torch.cat(
                            [
                                (u - v).flatten()
                                for u, v in zip(
                                    raw_audio_input, self.previous_raw_audio_input_even
                                )
                            ]
                        )
                        .abs()
                        .mean()
                    )

                    # Compute predicted change if we have k factors
                    if self.k_vid is not None and self.k_audio is not None:
                        # Calculate output norms for relative comparison
                        vid_output_norm = (
                            torch.cat(
                                [u.flatten() for u in self.previous_raw_vid_output_even]
                            )
                            .abs()
                            .mean()
                        )
                        audio_output_norm = (
                            torch.cat(
                                [
                                    u.flatten()
                                    for u in self.previous_raw_audio_output_even
                                ]
                            )
                            .abs()
                            .mean()
                        )

                        vid_pred_change = self.k_vid * (
                            vid_input_change / vid_output_norm
                        )
                        audio_pred_change = self.k_audio * (
                            audio_input_change / audio_output_norm
                        )

                        # Use max of predicted changes
                        combined_pred_change = max(vid_pred_change, audio_pred_change)

                        # Accumulate predicted error
                        self.accumulated_error_even += combined_pred_change

                        # Decide if we need full calculation
                        if self.accumulated_error_even < self.thresh:
                            self.should_calc_current_pair = False
                        else:
                            self.should_calc_current_pair = True
                            self.accumulated_error_even = 0
                    else:
                        # First time after ret_steps or missing k factors, need to calculate
                        self.should_calc_current_pair = True
                else:
                    # No previous data yet, must calculate
                    self.should_calc_current_pair = True

            # Store current input state
            self.previous_raw_vid_input_even = [u.clone() for u in raw_vid_input]
            self.previous_raw_audio_input_even = [u.clone() for u in raw_audio_input]

        # Check if we can use cached output and return early
        if (
            self.is_even
            and not self.should_calc_current_pair
            and self.previous_raw_vid_output_even is not None
            and self.previous_raw_audio_output_even is not None
        ):
            # Use cached output directly
            self.cnt += 1
            vid_out = [
                (u + v).float() for u, v in zip(raw_vid_input, self.cache_vid_even)
            ]
            audio_out = [
                (u + v).float() for u, v in zip(raw_audio_input, self.cache_audio_even)
            ]
            return vid_out, audio_out

        elif (
            not self.is_even
            and not self.should_calc_current_pair
            and self.previous_raw_vid_output_odd is not None
            and self.previous_raw_audio_output_odd is not None
        ):
            # Use cached output directly
            self.cnt += 1
            vid_out = [
                (u + v).float() for u, v in zip(raw_vid_input, self.cache_vid_odd)
            ]
            audio_out = [
                (u + v).float() for u, v in zip(raw_audio_input, self.cache_audio_odd)
            ]
            return vid_out, audio_out

        # Continue with normal processing since we need to calculate
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid,
            t=t,
            context=vid_context,
            seq_len=vid_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
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
            )
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            vid, audio = gradient_checkpointing(
                enabled=(self.training and self.gradient_checkpointing),
                module=self.single_fusion_block_forward,
                vid_block=vid_block,
                audio_block=audio_block,
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
            # If we have previous output, calculate k factors for future predictions
            if (
                self.previous_raw_vid_output_even is not None
                and self.previous_raw_audio_output_even is not None
            ):
                # Calculate output change for video
                vid_output_change = (
                    torch.cat(
                        [
                            (u - v).flatten()
                            for u, v in zip(
                                vid_output, self.previous_raw_vid_output_even
                            )
                        ]
                    )
                    .abs()
                    .mean()
                )

                # Calculate output change for audio
                audio_output_change = (
                    torch.cat(
                        [
                            (u - v).flatten()
                            for u, v in zip(
                                audio_output, self.previous_raw_audio_output_even
                            )
                        ]
                    )
                    .abs()
                    .mean()
                )

                # Check if we have previous input state for comparison
                if (
                    self.prev_prev_raw_vid_input_even is not None
                    and self.prev_prev_raw_audio_input_even is not None
                ):
                    # Calculate input change for video
                    vid_input_change = (
                        torch.cat(
                            [
                                (u - v).flatten()
                                for u, v in zip(
                                    self.previous_raw_vid_input_even,
                                    self.prev_prev_raw_vid_input_even,
                                )
                            ]
                        )
                        .abs()
                        .mean()
                    )

                    # Calculate input change for audio
                    audio_input_change = (
                        torch.cat(
                            [
                                (u - v).flatten()
                                for u, v in zip(
                                    self.previous_raw_audio_input_even,
                                    self.prev_prev_raw_audio_input_even,
                                )
                            ]
                        )
                        .abs()
                        .mean()
                    )

                    self.k_vid = (
                        vid_output_change / vid_input_change
                        if vid_input_change > 0
                        else 0
                    )
                    self.k_audio = (
                        audio_output_change / audio_input_change
                        if audio_input_change > 0
                        else 0
                    )

            # Update history
            self.prev_prev_raw_vid_input_even = getattr(
                self, "previous_raw_vid_input_even", None
            )
            self.prev_prev_raw_audio_input_even = getattr(
                self, "previous_raw_audio_input_even", None
            )
            self.previous_raw_vid_output_even = [u.clone() for u in vid_output]
            self.previous_raw_audio_output_even = [u.clone() for u in audio_output]
            self.cache_vid_even = [u - v for u, v in zip(vid_output, raw_vid_input)]
            self.cache_audio_even = [
                u - v for u, v in zip(audio_output, raw_audio_input)
            ]

        else:  # Uncondition path
            # Store output for unconditional path
            self.previous_raw_vid_output_odd = [u.clone() for u in vid_output]
            self.previous_raw_audio_output_odd = [u.clone() for u in audio_output]
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
