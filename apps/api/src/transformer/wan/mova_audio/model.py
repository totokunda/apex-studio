"""
Code adapted from DiffSynth-Studio's Wan DiT implementation:
https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/wan_video_dit.py
"""

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from diffusers.loaders import PeftAdapterMixin
from src.transformer.wan.mova.model import DiTBlock
from src.transformer.efficiency.ops import apply_scale_shift_inplace


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    # x is fp32 after layer norm
    # print(f"{shift.dtype = }")
    return (x * (1 + scale) + shift).to(shift.dtype)


def _chunked_modulated_norm(
    norm_layer: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """
    Modulated norm with optional chunking along the sequence dimension to reduce peak memory.

    Expects `hidden_states` to be [B, S, D]. `scale`/`shift` can be broadcastable or per-token [B, S, D].
    """
    if hidden_states.ndim != 3:
        out = norm_layer(hidden_states)
        out = out.to(hidden_states.dtype) if out.dtype != hidden_states.dtype else out
        apply_scale_shift_inplace(out, scale, shift)
        return out

    _, S, _ = hidden_states.shape
    in_dtype = hidden_states.dtype

    if chunk_size is None or S <= chunk_size:
        out = norm_layer(hidden_states)
        out = out.to(in_dtype) if out.dtype != in_dtype else out
        apply_scale_shift_inplace(out, scale, shift)
        return out

    out = torch.empty_like(hidden_states)
    scale_per_token = scale.dim() == 3 and scale.shape[1] == S

    for i in range(0, S, chunk_size):
        end = min(i + chunk_size, S)
        hs_chunk = hidden_states[:, i:end, :]
        if scale_per_token:
            scale_chunk = scale[:, i:end, :]
            shift_chunk = shift[:, i:end, :]
        else:
            scale_chunk = scale
            shift_chunk = shift

        out[:, i:end, :].copy_(norm_layer(hs_chunk))
        apply_scale_shift_inplace(out[:, i:end, :], scale_chunk, shift_chunk)
    return out


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis

def legacy_precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0, base_tps=4.0, target_tps=44100/2048):
    s = float(base_tps) / float(target_tps)
    # 1d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, s)
    # Do not apply positional encoding to the remaining dimensions.
    no_freqs_cis = precompute_freqs_cis(dim // 3, end, theta, s)
    no_freqs_cis = torch.ones_like(no_freqs_cis)
    return f_freqs_cis, no_freqs_cis, no_freqs_cis


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, s: float = 1.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        *,
        modulated_norm_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # NOTE: `t_mod` used to be [B, C]. When B=1 broadcasting works, but for B>1
            # it does not align with [1, 2, C]. We therefore unsqueeze at dim=1 here.
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(1)
            ).chunk(2, dim=1)

        # Align modulation tensors to x to avoid fp32 intermediates.
        shift = shift.to(dtype=x.dtype, device=x.device)
        scale = scale.to(dtype=x.dtype, device=x.device)

        x = _chunked_modulated_norm(
            self.norm,
            x,
            scale,
            shift,
            chunk_size=modulated_norm_chunk_size,
        )
        del scale, shift
        return self.head(x)


class MOVAWanAudioModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _repeated_blocks = ("DiTBlock",)
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.vae_type = vae_type
        # self.patch_embedding = nn.Conv3d(
        #     in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding = nn.Conv1d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        if vae_type == "oobleck":
            self.freqs = legacy_precompute_freqs_cis_1d(head_dim, base_tps=4.0, target_tps=44100/2048)
        elif vae_type == "dac":
            self.freqs = precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        # Default: no chunking unless explicitly enabled.
        self.set_chunking_profile("none")

    # ----------------------------
    # Chunking profile presets
    # ----------------------------

    _CHUNKING_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
        "none": {
            "ffn_chunk_size": None,
            "modulated_norm_chunk_size": None,
            "norm_chunk_size": None,
            "out_modulated_norm_chunk_size": None,
            "rotary_emb_chunk_size": None,
        },
        "light": {
            # NOTE: FFN chunking changes matmul shapes and can change numerics enough
            # to diverge diffusion outputs. Keep it off by default; use
            # `set_chunk_feed_forward(...)` explicitly if you accept small differences.
            "ffn_chunk_size": 4096,
            "modulated_norm_chunk_size": 16384,
            "norm_chunk_size": 8192,
            "out_modulated_norm_chunk_size": 16384,
            "rotary_emb_chunk_size": None,
        },
        "balanced": {
            # See note above.
            "ffn_chunk_size": 2048,
            "modulated_norm_chunk_size": 8192,
            "norm_chunk_size": 4096,
            "out_modulated_norm_chunk_size": 8192,
            "rotary_emb_chunk_size": 1024,
        },
        "aggressive": {
            # See note above.
            "ffn_chunk_size": 1024,
            "modulated_norm_chunk_size": 4096,
            "norm_chunk_size": 2048,
            "out_modulated_norm_chunk_size": 4096,
            "rotary_emb_chunk_size": 256,
        },
    }

    def list_chunking_profiles(self) -> Tuple[str, ...]:
        return tuple(self._CHUNKING_PROFILES.keys())

    def set_chunk_feed_forward(
        self, chunk_size: Optional[int], dim: int = 1, deterministic: bool = False
    ) -> None:
        """
        Enable/disable chunked feed-forward on all transformer blocks.

        Args:
            chunk_size: Tokens per chunk. None to disable chunking.
            dim: Dimension to chunk along (typically 1 for sequence).
            deterministic: If True, use deterministic CUDA settings during chunking.
                This can reduce (but may not fully eliminate) numerical differences
                compared to non-chunked execution.
        """
        for block in self.blocks:
            block.set_chunk_feed_forward(chunk_size, dim=dim, deterministic=deterministic)

    def set_chunking_profile(self, profile_name: str) -> None:
        if profile_name not in self._CHUNKING_PROFILES:
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. "
                f"Available: {sorted(self._CHUNKING_PROFILES.keys())}"
            )
        p = self._CHUNKING_PROFILES[profile_name]
        self._chunking_profile_name = profile_name

        self._rotary_emb_chunk_size_default = p.get("rotary_emb_chunk_size", None)
        self._out_modulated_norm_chunk_size = p.get(
            "out_modulated_norm_chunk_size", None
        )

        self.set_chunk_feed_forward(p.get("ffn_chunk_size", None), dim=1)
        for block in self.blocks:
            block.set_chunk_norms(
                modulated_norm_chunk_size=p.get("modulated_norm_chunk_size", None),
                norm_chunk_size=p.get("norm_chunk_size", None),
            )

    # ----------------------------
    # Memory-efficient inference setup
    # ----------------------------

    def enable_memory_efficient_inference(
        self,
        chunking_profile: str = "balanced",
        rope_on_cpu: bool = True,
        ffn_chunk_size: Optional[int] = None,
        ffn_deterministic: bool = True,
    ) -> None:
        """
        Enable memory optimizations for inference.

        Args:
            chunking_profile: One of "none", "light", "balanced", "aggressive".
                Controls norm chunking and RoPE chunking. Default "balanced".
            rope_on_cpu: If True, cache RoPE frequencies on CPU and transfer
                per-chunk during attention. Saves VRAM. Default True.
            ffn_chunk_size: If set, chunk the FFN to reduce peak memory from the
                4× intermediate tensor. WARNING: This may cause small numerical
                differences that can compound over diffusion steps. Use
                ffn_deterministic=True to minimize (but not eliminate) this.
                Set to None (default) for identical results.
            ffn_deterministic: If True and ffn_chunk_size is set, use deterministic
                CUDA settings during FFN chunking to reduce numerical variation.

        Note:
            The FFN intermediate tensor (4× model dim) is a major memory consumer.
            Without chunking, this cannot be reduced. If you need to save FFN memory
            and can accept small output differences, set ffn_chunk_size (e.g., 2048).

        Example:
            # Maximum memory savings with identical results (no FFN chunking)
            model.enable_memory_efficient_inference()

            # Maximum memory savings, accepting potential small output differences
            model.enable_memory_efficient_inference(ffn_chunk_size=2048)
        """
        self.set_chunking_profile(chunking_profile)

        if ffn_chunk_size is not None:
            self.set_chunk_feed_forward(
                ffn_chunk_size, dim=1, deterministic=ffn_deterministic
            )

        # Store rope_on_cpu preference as a default for forward()
        if not hasattr(self, "_apex_forward_kwargs_defaults"):
            self._apex_forward_kwargs_defaults = {}
        self._apex_forward_kwargs_defaults["rope_on_cpu"] = rope_on_cpu

    # ----------------------------
    # RoPE caching (CPU) to save VRAM
    # ----------------------------

    def _get_rope_cpu_cache(self) -> Dict[tuple, torch.Tensor]:
        if not hasattr(self, "_rope_cpu_cache"):
            self._rope_cpu_cache = {}
        return self._rope_cpu_cache

    def _rope_cache_key(self, *, f: int) -> tuple:
        return (int(f),)

    def _build_rope_1d(self, *, f: int, device: torch.device) -> torch.Tensor:
        # Build WAN-style complex freqs for [T, D/2], where T = f.
        return torch.cat(
            [self.freqs[0][:f], self.freqs[1][:f], self.freqs[2][:f]], dim=-1
        ).to(device)

    def _build_rope_cached(
        self,
        *,
        f: int,
        device: torch.device,
        rope_on_cpu: bool = False,
    ) -> torch.Tensor:
        if not rope_on_cpu:
            return self._build_rope_1d(f=f, device=device)

        cache = self._get_rope_cpu_cache()
        key = self._rope_cache_key(f=f)
        if key not in cache:
            cache[key] = self._build_rope_1d(f=f, device=torch.device("cpu"))
        return cache[key]

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f -> b f c').contiguous()
        return x, grid_size  # x, grid_size: (f)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b f (p c) -> b c (f p)',
            f=grid_size[0],
            p=self.patch_size[0]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                rope_on_cpu: Optional[bool] = None,
                return_prepared: bool = False,
                **kwargs,
                ):
        if rope_on_cpu is None:
            rope_on_cpu = (
                getattr(self, "_apex_forward_kwargs_defaults", {}) or {}
            ).get("rope_on_cpu", False)

        rotary_emb_chunk_size = kwargs.pop("rotary_emb_chunk_size", None)
        if rotary_emb_chunk_size is None:
            rotary_emb_chunk_size = getattr(self, "_rotary_emb_chunk_size_default", None)

        model_dtype = self.dtype
        with torch.autocast(x.device.type, dtype=torch.float32):
            t = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep)
            )
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        t = t.to(model_dtype)
        t_mod = t_mod.to(model_dtype)

        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
            del clip_embdding
        
        if x.dtype != model_dtype:
            x = x.to(model_dtype)

        x, (f, ) = self.patchify(x)

        freqs = self._build_rope_cached(
            f=f, device=x.device, rope_on_cpu=bool(rope_on_cpu)
        )

        if return_prepared:
            return {
                "x": x,
                "context": context,
                "t": t,
                "t_mod": t_mod,
                "freqs": freqs,
                "grid_size": (f,),
                "rotary_emb_chunk_size": rotary_emb_chunk_size,
            }
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, rotary_emb_chunk_size=rotary_emb_chunk_size)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    rotary_emb_chunk_size=rotary_emb_chunk_size,
                )

        x = self.head(
            x,
            t,
            modulated_norm_chunk_size=getattr(
                self, "_out_modulated_norm_chunk_size", None
            ),
        )
        del t_mod, t, context, freqs
        x = self.unpatchify(x, (f, ))
        return x
