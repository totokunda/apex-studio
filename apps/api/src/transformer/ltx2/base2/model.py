from enum import Enum
import torch
from src.engine.ltx22.shared.guidance.perturbations import BatchedPerturbationConfig
from src.transformer.ltx2.base2.adaln import AdaLayerNormSingle

from src.transformer.ltx2.base2.modality import Modality
from src.transformer.ltx2.base2.rope import LTXRopeType
from src.transformer.ltx2.base2.text_projection import PixArtAlphaTextProjection
from src.transformer.ltx2.base2.transformer import BasicAVTransformerBlock, TransformerConfig
from src.transformer.ltx2.base2.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from src.engine.ltx22.shared.utils import to_denoised
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_utils import ModelMixin
from typing import Any, Mapping, Optional, Tuple
from src.utils.cache import empty_cache

def avtransformer3d_config_from_base2_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert the Base2 `LTXModel` config/kwargs into an `AVTransformer3DModel`-style
    diffusers config dict (matching the keys/values expected by the legacy loader).

    Intended usage:
      - `av_cfg = avtransformer3d_config_from_base2_config(model.config)`
      - `av_cfg = avtransformer3d_config_from_base2_config(kwargs_dict)`
    """

    def _get(key: str, default: Any) -> Any:
        return cfg.get(key, default)

    # Base2 → legacy normalization naming
    qk_norm_in = _get("qk_norm", "rms_norm_across_heads")
    qk_norm_out = "rms_norm" if qk_norm_in == "rms_norm_across_heads" else qk_norm_in

    rope_type_in = _get("rope_type", "interleaved")
    # Base2 uses "interleaved"; legacy expects "split" for the same layout.
    rope_type_out = "split" if rope_type_in == "interleaved" else rope_type_in

    rope_double_precision = bool(_get("rope_double_precision", True))
    frequencies_precision = "float64" if rope_double_precision else "float32"

    pos_embed_max_pos = int(_get("pos_embed_max_pos", 20))
    base_height = int(_get("base_height", 2048))
    base_width = int(_get("base_width", 2048))

    audio_pos_embed_max_pos = int(_get("audio_pos_embed_max_pos", 20))

    timestep_scale_multiplier = int(_get("timestep_scale_multiplier", 1000))
    cross_attn_timestep_scale_multiplier = int(_get("cross_attn_timestep_scale_multiplier", 1000))

    return {
        "_class_name": "AVTransformer3DModel",
        "_diffusers_version": "0.25.1",
        "activation_fn": _get("activation_fn", "gelu-approximate"),
        "attention_bias": bool(_get("attention_bias", True)),
        "attention_head_dim": int(_get("attention_head_dim", 128)),
        "attention_type": "default",
        "caption_channels": int(_get("caption_channels", 3840)),
        "cross_attention_dim": int(_get("cross_attention_dim", 4096)),
        "double_self_attention": False,
        "dropout": 0,
        "in_channels": int(_get("in_channels", 128)),
        "norm_elementwise_affine": bool(_get("norm_elementwise_affine", False)),
        "norm_eps": float(_get("norm_eps", 1e-6)),
        "norm_num_groups": 32,
        "num_attention_heads": int(_get("num_attention_heads", 32)),
        "num_embeds_ada_norm": timestep_scale_multiplier,
        "num_layers": int(_get("num_layers", 48)),
        "num_vector_embeds": None,
        "only_cross_attention": False,
        "cross_attention_norm": True,
        "out_channels": int(_get("out_channels", 128)),
        "upcast_attention": False,
        "use_linear_projection": False,
        "qk_norm": qk_norm_out,
        "standardization_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "positional_embedding_theta": float(_get("rope_theta", 10000.0)),
        "positional_embedding_max_pos": [pos_embed_max_pos, base_height, base_width],
        "timestep_scale_multiplier": timestep_scale_multiplier,
        "av_ca_timestep_scale_multiplier": cross_attn_timestep_scale_multiplier,
        "causal_temporal_positioning": True,
        "audio_num_attention_heads": int(_get("audio_num_attention_heads", 32)),
        "audio_attention_head_dim": int(_get("audio_attention_head_dim", 64)),
        "use_audio_video_cross_attention": True,
        "share_ff": False,
        "audio_out_channels": int(_get("audio_out_channels", 128)),
        "audio_cross_attention_dim": int(_get("audio_cross_attention_dim", 2048)),
        "audio_positional_embedding_max_pos": [audio_pos_embed_max_pos],
        "av_cross_ada_norm": True,
        "use_embeddings_connector": True,
        "connector_attention_head_dim": 128,
        "connector_num_attention_heads": 30,
        "connector_num_layers": 2,
        "connector_positional_embedding_max_pos": [4096],
        "connector_num_learnable_registers": 128,
        "connector_norm_output": True,
        "use_middle_indices_grid": True,
        "rope_type": rope_type_out,
        "frequencies_precision": frequencies_precision,
    }


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)
    
    
class LTX2VideoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    @register_to_config
    def __init__(
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        in_channels: int = 128,  # Video Arguments
        out_channels: Optional[int] = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: Optional[int] = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str | LTXRopeType = LTXRopeType.INTERLEAVED,
        use_middle_indices_grid: bool = True,
        attention_type: None = None,
        chunking_profile: str = "none",
        ffn_chunk_size: Optional[int] = None,
        ffn_chunk_dim: int = 1,
    ) -> None:
        super().__init__()

        self._enable_gradient_checkpointing = False

        self.model_type = model_type
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = LTXRopeType(rope_type) if isinstance(rope_type, str) else rope_type
        self.double_precision_rope = rope_double_precision
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = rope_theta
        self.av_ca_timestep_scale_multiplier = cross_attn_timestep_scale_multiplier

        cross_pe_max_pos = None
        if self.model_type.is_video_enabled():
            self.positional_embedding_max_pos = [pos_embed_max_pos, base_height, base_width]
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if self.model_type.is_audio_enabled():
            self.audio_positional_embedding_max_pos = [audio_pos_embed_max_pos]
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if self.model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if self.model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
        )

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)

        self.adaln_single = AdaLayerNormSingle(self.inner_dim)

        # Video caption projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
        )

        # Audio caption projection
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: None,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )
    

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x
    
    
    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.forward_velocity(video, audio, perturbations)
        if vx is None and ax is None:
            return None, None
        if isinstance(vx, list) or isinstance(ax, list):
            video_list = video if isinstance(video, (list, tuple)) else [None] * len(vx)
            audio_list = audio if isinstance(audio, (list, tuple)) else [None] * len(ax)
            denoised_video = []
            denoised_audio = []
            for v, v_pred in zip(video_list, vx):
                if v is None or v_pred is None:
                    denoised_video.append(None)
                    continue
                v_timesteps = v.timesteps
                if v.frame_indices is not None:
                    v_timesteps = v_timesteps.gather(1, v.frame_indices)
                if v_timesteps is not None and v_timesteps.ndim == 2:
                    v_timesteps = v_timesteps.unsqueeze(-1)
                denoised_video.append(to_denoised(v.latent, v_pred, v_timesteps))
            for a, a_pred in zip(audio_list, ax):
                if a is None or a_pred is None:
                    denoised_audio.append(None)
                    continue
                a_timesteps = a.timesteps
                if a_timesteps is not None and a_timesteps.ndim == 2:
                    a_timesteps = a_timesteps.unsqueeze(-1)
                denoised_audio.append(to_denoised(a.latent, a_pred, a_timesteps))
            return denoised_video, denoised_audio

        if video is not None and video.frame_indices is not None:
            video_timesteps = video.timesteps.gather(1, video.frame_indices)
        else:
            video_timesteps = video.timesteps if video is not None else None
        if video_timesteps is not None and video_timesteps.ndim == 2:
            video_timesteps = video_timesteps.unsqueeze(-1)
        audio_timesteps = audio.timesteps if audio is not None else None
        if audio_timesteps is not None and audio_timesteps.ndim == 2:
            audio_timesteps = audio_timesteps.unsqueeze(-1)

        denoised_video = to_denoised(video.latent, vx, video_timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio_timesteps) if ax is not None else None
        return denoised_video, denoised_audio

    def forward_velocity(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")
        
        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        empty_cache()
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None
        empty_cache()

        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )
        
        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax
