from src.engine.base_engine import BaseEngine
from diffusers.image_processor import VaeImageProcessor
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING, Callable
import torch
from diffusers.utils.torch_utils import randn_tensor
from src.utils.progress import safe_emit_progress


class ZImageShared(BaseEngine):
    """Base class for ZImage engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, *args, **kwargs):
        super().__init__(yaml_path, *args, **kwargs)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        self.num_channels_latents = (
            self.transformer.in_channels if self.transformer is not None else 16
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        safe_emit_progress(progress_callback, 0.05, "Preparing prompt encoding")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            progress_callback=progress_callback,
        )
        safe_emit_progress(progress_callback, 0.70, "Prompt encoded")

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = (
                    [negative_prompt]
                    if isinstance(negative_prompt, str)
                    else negative_prompt
                )
            assert len(prompt) == len(negative_prompt)
            safe_emit_progress(
                progress_callback, 0.75, "Preparing negative prompt encoding"
            )
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                progress_callback=progress_callback,
            )
            safe_emit_progress(progress_callback, 0.95, "Negative prompt encoded")
        else:
            negative_prompt_embeds = []
        safe_emit_progress(progress_callback, 1.0, "Prompt encoding complete")
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[torch.FloatTensor]:
        device = device or self.device
        dtype = self.component_dtypes["text_encoder"]

        if not self.text_encoder:
            safe_emit_progress(progress_callback, 0.10, "Loading text encoder")
            self.load_component_by_type("text_encoder")
            safe_emit_progress(progress_callback, 0.20, "Text encoder loaded")
            safe_emit_progress(progress_callback, 0.25, "Moving text encoder to device")
            self.to_device(self.text_encoder)
            safe_emit_progress(progress_callback, 0.30, "Text encoder on device")

        if prompt_embeds is not None:
            safe_emit_progress(
                progress_callback, 0.35, "Using provided prompt embeddings"
            )
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        safe_emit_progress(progress_callback, 0.40, "Tokenizing prompt(s)")
        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.text_encoder.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        safe_emit_progress(progress_callback, 0.65, "Encoding prompt embeddings")
        prompt_embeds, prompt_masks = self.text_encoder.encode(
            prompt,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=1,
            add_special_tokens=False,
            return_attention_mask=True,
            use_attention_mask=True,
            pad_with_zero=False,
            clean_text=False,
            output_type="hidden_states_all",
        )

        prompt_embeds = prompt_embeds[-2].to(device=device, dtype=dtype)
        prompt_masks = prompt_masks.bool().to(device=device)

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = torch.randn(
                shape,
                generator=torch.Generator(device="cpu").manual_seed(
                    generator.initial_seed()
                ),
                device="cpu",
                dtype=dtype,
            )
            latents = latents.to(device)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)
        return latents
