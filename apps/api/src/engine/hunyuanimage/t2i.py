from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING, Callable
import torch
import numpy as np
from diffusers.guiders import AdaptiveProjectedMixGuidance
from src.utils.progress import safe_emit_progress, make_mapped_progress
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING, Callable
from diffusers.image_processor import VaeImageProcessor
from src.engine.base_engine import BaseEngine
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
import re
from loguru import logger


class HunyuanImageT2IEngine(BaseEngine):

    def __init__(self, yaml_path: str, **kwargs):

        super().__init__(yaml_path, **kwargs)

        self.vae_scale_factor = (
            self.vae.config.spatial_compression_ratio
            if getattr(self, "vae", None)
            else 32
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 128
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 64

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_max_length: int = 1000,
        template: str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>",
        drop_idx: int = 34,
        hidden_state_skip_layer: int = 2,
    ):
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        prompt = [prompt] if isinstance(prompt, str) else prompt

        txt = [template.format(e) for e in prompt]

        if self.text_encoder is None:
            self.load_component_by_name("text_encoder")

        encoder_hidden_states, attention_mask = self.text_encoder.encode(
            txt,
            max_sequence_length=tokenizer_max_length + drop_idx,
            pad_to_max_length=True,
            use_attention_mask=True,
            return_attention_mask=True,
            clean_text=False,
            output_type="raw",
        )

        prompt_embeds = encoder_hidden_states.hidden_states[
            -(hidden_state_skip_layer + 1)
        ]
        prompt_embeds = prompt_embeds[:, drop_idx:]
        encoder_attention_mask = attention_mask[:, drop_idx:]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = encoder_attention_mask.to(device=device)

        return prompt_embeds, encoder_attention_mask

    def _get_byt5_prompt_embeds(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_max_length: int = 128,
    ):
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        if isinstance(prompt, list):
            raise ValueError("byt5 prompt should be a string")
        elif prompt is None:
            raise ValueError("byt5 prompt should not be None")

        if not hasattr(self, "text_encoder_2") or not self.text_encoder_2:
            self.load_component_by_name("text_encoder_2")

        prompt_embeds, attention_mask = self.text_encoder_2.encode(
            prompt,
            max_sequence_length=tokenizer_max_length,
            pad_to_max_length=True,
            use_attention_mask=True,
            return_attention_mask=True,
            clean_text=False,
            output_type="raw",
        )

        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        encoder_attention_mask = attention_mask.to(device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str], None],
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        offload: bool = False,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            batch_size (`int`):
                batch size of prompts, defaults to 1
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input
                argument.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
            prompt_embeds_mask_2 (`torch.Tensor`, *optional*):
                Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input
                argument using self.tokenizer_2 and self.text_encoder_2.
        """

        if prompt is None:
            prompt = [""] * batch_size

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt=prompt,
                tokenizer_max_length=self.tokenizer_max_length,
                template=self.prompt_template_encode,
                drop_idx=self.prompt_template_encode_start_idx,
            )

        if offload:
            del self.text_encoder


        if prompt_embeds_2 is None:
            prompt_embeds_2_list = []
            prompt_embeds_mask_2_list = []

            glyph_texts = [self.extract_glyph_text(p) for p in prompt]
            text_encoder_config = self.load_config_by_name("text_encoder_2")
            for glyph_text in glyph_texts:
                if glyph_text is None:
                    glyph_text_embeds = torch.zeros(
                        (
                            1,
                            self.tokenizer_2_max_length,
                            text_encoder_config["d_model"],
                        ),
                        device=self.device,
                    )
                    glyph_text_embeds_mask = torch.zeros(
                        (1, self.tokenizer_2_max_length),
                        device=self.device,
                        dtype=torch.int64,
                    )
                else:
                    glyph_text_embeds, glyph_text_embeds_mask = (
                        self._get_byt5_prompt_embeds(
                            prompt=glyph_text,
                            device=self.device,
                            tokenizer_max_length=self.tokenizer_2_max_length,
                        )
                    )

                prompt_embeds_2_list.append(glyph_text_embeds)
                prompt_embeds_mask_2_list.append(glyph_text_embeds_mask)

            if offload and hasattr(self, "text_encoder_2"):
                del self.text_encoder_2


            prompt_embeds_2 = torch.cat(prompt_embeds_2_list, dim=0)
            prompt_embeds_mask_2 = torch.cat(prompt_embeds_mask_2_list, dim=0)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(
            batch_size * num_images_per_prompt, seq_len
        )

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(
            batch_size * num_images_per_prompt, seq_len_2, -1
        )
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(
            batch_size * num_images_per_prompt, seq_len_2
        )

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    @staticmethod
    def extract_glyph_text(prompt: str):
        """
        Extract text enclosed in quotes for glyph rendering.

        Finds text in single quotes, double quotes, and Chinese quotes, then formats it for byT5 processing.

        Args:
            prompt: Input text prompt

        Returns:
            Formatted glyph text string or None if no quoted text found
        """
        text_prompt_texts = []
        pattern_quote_single = r"\'(.*?)\'"
        pattern_quote_double = r"\"(.*?)\""
        pattern_quote_chinese_single = r"‘(.*?)’"
        pattern_quote_chinese_double = r"“(.*?)”"

        matches_quote_single = re.findall(pattern_quote_single, prompt)
        matches_quote_double = re.findall(pattern_quote_double, prompt)
        matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
        matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

        text_prompt_texts.extend(matches_quote_single)
        text_prompt_texts.extend(matches_quote_double)
        text_prompt_texts.extend(matches_quote_chinese_single)
        text_prompt_texts.extend(matches_quote_chinese_double)

        if text_prompt_texts:
            glyph_text_formatted = (
                ". ".join([f'Text "{text}"' for text in text_prompt_texts]) + ". "
            )
        else:
            glyph_text_formatted = None

        return glyph_text_formatted

    def get_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        seed,
        generator,
        latents=None,
    ):
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        """Decode latents and render a preview image during denoising."""
        image = self.vae_decode(latents, offload=True)
        image = self._tensor_to_frame(image)
        render_on_step_callback(image[0])

    def run(
        self,
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str, None] = None,
        height: int = 768,
        width: int = 1344,
        num_inference_steps: int = 50,
        prompt_embeds: torch.Tensor = None,
        prompt_embeds_mask: torch.Tensor = None,
        prompt_embeds_2: torch.Tensor = None,
        prompt_embeds_mask_2: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds_mask: torch.Tensor = None,
        negative_prompt_embeds_2: torch.Tensor = None,
        negative_prompt_embeds_mask_2: torch.Tensor = None,
        seed: int = None,
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        sigmas: List[float] = None,
        timesteps: List[int] = None,
        timesteps_as_indices: bool = True,
        max_sequence_length: int = 1024,
        attention_kwargs: Dict[str, Any] = {},
        num_images: int = 1,
        distilled_guidance_scale: float = 3.25,
        return_latents: bool = False,
        offload: bool = True,
        render_on_step: bool = False,
        render_on_step_callback: Callable = None,
        render_on_step_interval: int = 3,
        progress_callback: Callable = None,
        **kwargs,
    ):

        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = (
            self.encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                batch_size=batch_size,
                num_images_per_prompt=num_images,
                prompt_embeds_2=prompt_embeds_2,
                prompt_embeds_mask_2=prompt_embeds_mask_2,
                offload=offload,
            )
        )

        safe_emit_progress(progress_callback, 0.15, "Encoded prompt")

        if self.transformer is None:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        safe_emit_progress(progress_callback, 0.25, "Transformer ready")

        dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(dtype)
        prompt_embeds_2 = prompt_embeds_2.to(dtype)
        # select guider
        if (
            not torch.all(prompt_embeds_2 == 0)
            and self.helpers["ocr_guider"] is not None
        ):
            # prompt contains ocr and pipeline has a guider for ocr
            guider = self.helpers["ocr_guider"]
        elif self.helpers["guider"] is not None:
            guider = self.helpers["guider"]
        # distilled model does not use guidance method, use default guider with enabled=False
        else:
            guider = AdaptiveProjectedMixGuidance(enabled=False)

        if guider._enabled and guider.num_conditions > 1:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                batch_size=batch_size,
                num_images_per_prompt=num_images,
                prompt_embeds_2=negative_prompt_embeds_2,
                prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
                offload=offload,
            )

            negative_prompt_embeds = negative_prompt_embeds.to(dtype)
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(dtype)
            safe_emit_progress(progress_callback, 0.28, "Guidance embeddings prepared")
        else:
            safe_emit_progress(
                progress_callback, 0.18, "Skipped negative prompt embeds"
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.get_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=self.transformer.dtype,
            device=self.device,
            seed=seed,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.32, "Initialized latent noise")

        # 5. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        safe_emit_progress(progress_callback, 0.36, "Scheduler ready")

        sigmas = (
            np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            if sigmas is None
            else sigmas
        )
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler, num_inference_steps, sigmas=sigmas
        )
        safe_emit_progress(
            progress_callback, 0.40, "Timesteps computed; starting denoise"
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance (for guidance-distilled model)
        if self.transformer.config.guidance_embeds and distilled_guidance_scale is None:
            raise ValueError(
                "`distilled_guidance_scale` is required for guidance-distilled model."
            )

        if self.transformer.config.guidance_embeds:
            guidance = (
                torch.tensor(
                    [distilled_guidance_scale] * latents.shape[0],
                    dtype=self.transformer.dtype,
                    device=self.device,
                )
                * 1000.0
            )

        else:
            guidance = None

        denoise_progress_callback = make_mapped_progress(progress_callback, 0.40, 0.92)

        self.scheduler.set_begin_index(0)
        total_steps = len(timesteps) if timesteps is not None else 0
        if denoise_progress_callback is not None:
            try:
                denoise_progress_callback(0.0, "Starting denoise")
            except Exception:
                pass

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if self.transformer.config.use_meanflow:
                    if i == len(timesteps) - 1:
                        timestep_r = torch.tensor([0.0], device=self.device)
                    else:
                        timestep_r = timesteps[i + 1]
                    timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype)
                else:
                    timestep_r = None

                # Step 1: Collect model inputs needed for the guidance method
                # conditional inputs should always be first element in the tuple
                guider_inputs = {
                    "encoder_hidden_states": (prompt_embeds, negative_prompt_embeds),
                    "encoder_attention_mask": (
                        prompt_embeds_mask,
                        negative_prompt_embeds_mask,
                    ),
                    "encoder_hidden_states_2": (
                        prompt_embeds_2,
                        negative_prompt_embeds_2,
                    ),
                    "encoder_attention_mask_2": (
                        prompt_embeds_mask_2,
                        negative_prompt_embeds_mask_2,
                    ),
                }

                # Step 2: Update guider's internal state for this denoising step
                guider.set_state(
                    step=i, num_inference_steps=num_inference_steps, timestep=t
                )

                # Step 3: Prepare batched model inputs based on the guidance method
                # The guider splits model inputs into separate batches for conditional/unconditional predictions.
                # For CFG with guider_inputs = {"encoder_hidden_states": (prompt_embeds, negative_prompt_embeds)}:
                # you will get a guider_state with two batches:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "__guidance_identifier__": "pred_cond"},      # conditional batch
                #       {"encoder_hidden_states": negative_prompt_embeds, "__guidance_identifier__": "pred_uncond"},  # unconditional batch
                #   ]
                # Other guidance methods may return 1 batch (no guidance) or 3+ batches (e.g., PAG, APG).
                guider_state = guider.prepare_inputs(guider_inputs)
                # Step 4: Run the denoiser for each batch
                # Each batch in guider_state represents a different conditioning (conditional, unconditional, etc.).
                # We run the model once per batch and store the noise prediction in guider_state_batch.noise_pred.
                for guider_state_batch in guider_state:
                    guider.prepare_models(self.transformer)

                    # Extract conditioning kwargs for this batch (e.g., encoder_hidden_states)
                    cond_kwargs = {
                        input_name: getattr(guider_state_batch, input_name)
                        for input_name in guider_inputs.keys()
                    }

                    # e.g. "pred_cond"/"pred_uncond"
                    context_name = getattr(guider_state_batch, guider._identifier_key)
                    with self.transformer.cache_context(context_name):
                        # Run denoiser and store noise prediction in this batch
                        guider_state_batch.noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep,
                            timestep_r=timestep_r,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            **cond_kwargs,
                        )[0]

                    # Cleanup model (e.g., remove hooks)
                    guider.cleanup_models(self.transformer)

                # Step 5: Combine predictions using the guidance method
                # The guider takes all noise predictions from guider_state and combines them according to the guidance algorithm.
                # Continuing the CFG example, the guider receives:
                #   guider_state = [
                #       {"encoder_hidden_states": prompt_embeds, "noise_pred": noise_pred_cond, "__guidance_identifier__": "pred_cond"},      # batch 0
                #       {"encoder_hidden_states": negative_prompt_embeds, "noise_pred": noise_pred_uncond, "__guidance_identifier__": "pred_uncond"},  # batch 1
                #   ]
                # And extracts predictions using the __guidance_identifier__:
                #   pred_cond = guider_state[0]["noise_pred"]      # extracts noise_pred_cond
                #   pred_uncond = guider_state[1]["noise_pred"]    # extracts noise_pred_uncond
                # Then applies CFG formula:
                #   noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                # Returns GuiderOutput(pred=noise_pred, pred_cond=pred_cond, pred_uncond=pred_uncond)
                noise_pred = guider(guider_state)[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != len(timesteps) - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if denoise_progress_callback is not None and total_steps > 0:
                    try:
                        denoise_progress_callback(
                            min((i + 1) / total_steps, 1.0),
                            f"Denoising step {i + 1}/{total_steps}",
                        )
                    except Exception:
                        pass

        safe_emit_progress(progress_callback, 0.92, "Denoising complete")

        if offload:
            self._offload("transformer")
            safe_emit_progress(progress_callback, 0.94, "Transformer offloaded")

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents
        else:
            image = self.vae_decode(latents, offload=offload)
            image = self._tensor_to_frame(image)
            safe_emit_progress(
                progress_callback, 1.0, "Completed text-to-image pipeline"
            )
            return image
