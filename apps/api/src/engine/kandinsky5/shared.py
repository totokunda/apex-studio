import html
import re
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers.utils import is_ftfy_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from src.engine.base_engine import BaseEngine

try:
    import ftfy
except ImportError:
    ftfy = None


def basic_clean(text: str) -> str:
    if is_ftfy_available() and ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))


class Kandinsky5Shared(BaseEngine):
    """Shared functionality for Kandinsky 5 engines (text/image-to-video)."""

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_qwen",
        "prompt_embeds_clip",
        "negative_prompt_embeds_qwen",
        "negative_prompt_embeds_clip",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_template = "\n".join(
            [
                "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ]
        )
        self.prompt_template_encode_start_idx = 129
        self._guidance_scale = 0.0
        self._interrupt = False
        self._num_timesteps = 0
        self._refresh_vae_factors()
        self._refresh_transformer_meta()

    def _refresh_vae_factors(self) -> None:
        vae = getattr(self, "vae", None)
        self.vae_scale_factor_temporal = (
            getattr(vae.config, "temporal_compression_ratio", 4)
            if vae is not None
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(vae.config, "spatial_compression_ratio", 8)
            if vae is not None
            else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def _refresh_transformer_meta(self) -> None:
        transformer = getattr(self, "transformer", None)
        if transformer is None:
            self.num_channels_latents = getattr(self, "num_channels_latents", 16)
            return
        self.num_channels_latents = getattr(transformer.config, "in_visual_dim", 16)

    def _ensure_models(self) -> None:
        if getattr(self, "text_encoder", None) is None:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)

        if getattr(self, "text_encoder_2", None) is None:
            try:
                self.load_component_by_name("text_encoder_2")
            except Exception:
                pass
        if getattr(self, "text_encoder_2", None) is None:
            raise ValueError(
                "`text_encoder_2` is required for Kandinsky5 text encoding but was not loaded."
            )
        self.to_device(self.text_encoder_2)

        if getattr(self, "transformer", None) is None:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        if getattr(self, "vae", None) is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        self._refresh_vae_factors()
        self._refresh_transformer_meta()

        if getattr(self, "scheduler", None) is None:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @staticmethod
    def fast_sta_nabla(
        T: int,
        H: int,
        W: int,
        wT: int = 3,
        wH: int = 3,
        wW: int = 3,
        device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        l = torch.Tensor([T, H, W]).amax()
        r = torch.arange(0, l, 1, dtype=torch.int16, device=device)
        mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
        sta_t, sta_h, sta_w = (
            mat[:T, :T].flatten(),
            mat[:H, :H].flatten(),
            mat[:W, :W].flatten(),
        )
        sta_t = sta_t <= wT // 2
        sta_h = sta_h <= wH // 2
        sta_w = sta_w <= wW // 2
        sta_hw = (
            (sta_h.unsqueeze(1) * sta_w.unsqueeze(0))
            .reshape(H, H, W, W)
            .transpose(1, 2)
            .flatten()
        )
        sta = (
            (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0))
            .reshape(T, T, H * W, H * W)
            .transpose(1, 2)
        )
        return sta.reshape(T * H * W, T * H * W)

    def get_sparse_params(
        self, sample: torch.Tensor, device: torch.device
    ) -> Optional[Dict[str, Any]]:
        assert self.transformer.config.patch_size[0] == 1
        _, T, H, W, _ = sample.shape
        T, H, W = (
            T // self.transformer.config.patch_size[0],
            H // self.transformer.config.patch_size[1],
            W // self.transformer.config.patch_size[2],
        )
        if self.transformer.config.attention_type != "nabla":
            return None

        sta_mask = self.fast_sta_nabla(
            T,
            H // 8,
            W // 8,
            self.transformer.config.attention_wT,
            self.transformer.config.attention_wH,
            self.transformer.config.attention_wW,
            device=device,
        )

        return {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": self.transformer.config.attention_type,
            "to_fractal": True,
            "P": self.transformer.config.attention_P,
            "wT": self.transformer.config.attention_wT,
            "wW": self.transformer.config.attention_wW,
            "wH": self.transformer.config.attention_wH,
            "add_sta": self.transformer.config.attention_add_sta,
            "visual_shape": (T, H, W),
            "method": self.transformer.config.attention_method,
        }

    def _encode_prompt_qwen(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
        num_videos_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.device
        dtype = dtype or getattr(self.text_encoder, "dtype", None) or torch.float32

        full_texts = [self.prompt_template.format(p) for p in prompt]

        inputs = self.text_encoder.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_sequence_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # cache the inputs
        if self.text_encoder.enable_cache:
            hash = self.text_encoder.hash(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_length": max_sequence_length
                    + self.prompt_template_encode_start_idx,
                }
            )
            cached = self.text_encoder.load_cached(hash)
            if cached is not None:
                return cached[0].to(dtype).to(device), cached[1].to(device)

        # encode the prompt
        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model()

        embeds = self.text_encoder.model(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )["hidden_states"][-1][:, self.prompt_template_encode_start_idx :]

        attention_mask = inputs["attention_mask"][
            :, self.prompt_template_encode_start_idx :
        ]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32)

        # cache the embeds and cu_seqlens
        if self.text_encoder.enable_cache:
            self.text_encoder.cache(hash, embeds, cu_seqlens)

        return embeds.to(dtype), cu_seqlens

    def _encode_prompt_clip(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_videos_per_prompt: int = 1,
    ) -> torch.Tensor:
        device = device or self.device
        dtype = dtype or getattr(self.text_encoder_2, "dtype", None) or torch.float32

        pooled_embed = self.text_encoder_2.encode(
            prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=dtype,
            pad_to_max_length=True,
            output_type="pooler_output",
            add_special_tokens=True,
            return_attention_mask=False,
            pad_with_zero=False,
            max_sequence_length=77,
            clean_text=False,
        )
        return pooled_embed.to(dtype)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device or self.device
        dtype = dtype or getattr(self.text_encoder, "dtype", None) or torch.float32

        if not isinstance(prompt, list):
            prompt = [prompt]

        prompt = [prompt_clean(p) for p in prompt]

        prompt_embeds_qwen, prompt_cu_seqlens = self._encode_prompt_qwen(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        prompt_embeds_clip = self._encode_prompt_clip(
            prompt=prompt,
            device=device,
            dtype=dtype,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        return prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
            num_channels_latents,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if self.transformer.visual_cond:
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                [
                    batch_size,
                    num_latent_frames,
                    int(height) // self.vae_scale_factor_spatial,
                    int(width) // self.vae_scale_factor_spatial,
                    1,
                ],
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        return latents

    def _get_scale_factor(self, height: int, width: int) -> tuple[float, float, float]:
        """Default scale factor used by Kandinsky5 transformer."""
        return (1, 2, 2)
