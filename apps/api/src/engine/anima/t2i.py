from src.engine.base_engine import BaseEngine
from typing import Union, List, Optional, Callable, Dict, Any
import numpy as np
import torch
from src.utils.progress import safe_emit_progress
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor


def _parse_parentheses(text: str) -> List[str]:
    result: List[str] = []
    current_item = ""
    nesting_level = 0
    for char in text:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def _token_weights(text: str, current_weight: float) -> List[tuple[str, float]]:
    # Ported from ComfyUI `comfy/sd1_clip.py` to match prompt weighting behavior.
    parts = _parse_parentheses(text)
    out: List[tuple[str, float]] = []
    for part in parts:
        weight = current_weight
        if len(part) >= 2 and part[-1] == ")" and part[0] == "(":
            inner = part[1:-1]
            colon_idx = inner.rfind(":")
            weight *= 1.1
            if colon_idx > 0:
                try:
                    weight = float(inner[colon_idx + 1 :])
                    inner = inner[:colon_idx]
                except Exception:
                    pass
            out += _token_weights(inner, weight)
        else:
            out.append((part, current_weight))
    return out


def _escape_important(text: str) -> str:
    # Matches ComfyUI escape behavior for literal parentheses.
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def _unescape_important(text: str) -> str:
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def _get_per_token_weights(
    prompt: str,
    tokenizer,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    ComfyUI-style per-token weight computation.
    
    Parses the prompt for weight annotations like (word:1.2) or ((emphasized)),
    tokenizes each weighted segment, and returns a weight tensor aligned to the
    token sequence.
    
    Args:
        prompt: The raw prompt string with optional weight syntax.
        tokenizer: The tokenizer (e.g., from llm_adapter).
        max_sequence_length: Maximum token sequence length.
        device: Target device for the weight tensor.
        dtype: Target dtype for the weight tensor.
        
    Returns:
        A tensor of shape (max_sequence_length,) with per-token weights.
    """
    escaped = _escape_important(prompt)
    segments = _token_weights(escaped, 1.0)
    
    # Build per-token weights by tokenizing each segment
    token_weights: List[float] = []
    
    for segment_text, weight in segments:
        cleaned_text = _unescape_important(segment_text)
        if not cleaned_text:
            continue
            
        # Tokenize this segment without special tokens to get raw token count
        segment_tokens = tokenizer(
            cleaned_text,
            add_special_tokens=False,
            return_tensors=None,
        )
        num_tokens = len(segment_tokens["input_ids"])
        token_weights.extend([weight] * num_tokens)
    
    # Pad or truncate to max_sequence_length
    if len(token_weights) < max_sequence_length:
        # Pad with 1.0 (neutral weight) for padding tokens
        token_weights.extend([1.0] * (max_sequence_length - len(token_weights)))
    else:
        token_weights = token_weights[:max_sequence_length]
    
    return torch.tensor(token_weights, device=device, dtype=dtype)


def _apply_token_weights_to_embeddings(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-token weights to embeddings.
    
    ComfyUI applies weights by scaling the embedding vectors directly.
    This modifies the magnitude of each token's contribution.
    
    Args:
        embeddings: Tensor of shape (batch, seq_len, hidden_dim).
        weights: Tensor of shape (batch, seq_len) or (seq_len,).
        
    Returns:
        Weighted embeddings of the same shape.
    """
    if weights.dim() == 1:
        # Expand for batch dimension
        weights = weights.unsqueeze(0).expand(embeddings.shape[0], -1)
    
    # Expand weights to match embedding dimensions: (batch, seq_len) -> (batch, seq_len, 1)
    weights = weights.unsqueeze(-1)
    
    return embeddings * weights


class AnimaT2IEngine(BaseEngine):
    """Anima Text-to-Image Engine Implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None) is not None
            and getattr(getattr(self.vae, "config", None), "block_out_channels", None) is not None
            else 8
        )

        self.num_channels_latents = (
            getattr(getattr(self.transformer, "config", None), "in_channels", 16)
            if self.transformer
            else 16
        )
        
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs
    
        
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        offload: bool = True,
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
            negative_prompt_embeds = None
        safe_emit_progress(progress_callback, 1.0, "Prompt encoding complete")
        
        if offload:
            self._offload("text_encoder")
            self._offload("anima.llm_adapter")
            
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
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

            
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        cleaned_prompt_list: List[str] = []
        for p in prompt_list:
            escaped = _escape_important(p)
            segments = _token_weights(escaped, 1.0)
            cleaned = "".join(_unescape_important(seg) for seg, _ in segments)
            cleaned_prompt_list.append(cleaned)

        safe_emit_progress(progress_callback, 0.40, "Tokenizing prompt(s)")

        prompt_embeds, prompt_masks = self.text_encoder.encode(
            cleaned_prompt_list,
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

        prompt_embeds = prompt_embeds[-1].to(device=device, dtype=dtype)
        prompt_masks = prompt_masks.bool().to(device=device)
        
        llm_adapter = self.helpers["anima.llm_adapter"]
        self.to_device(llm_adapter)
        # --- ComfyUI-style prompt weighting for the T5 token stream ---
        # ComfyUI's Anima tokenizer parses weights from the (t5xxl) prompt tokens and applies them *after*
        # llm_adapter preprocessing (see `ComfyUI/comfy/model_base.py::Anima.extra_conds`).
        tokenizer = getattr(llm_adapter, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("anima.llm_adapter is missing a tokenizer; cannot apply prompt weights.")

        target_info = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_sequence_length).to(device=device)
  
        target_input_ids = target_info.input_ids
        target_attention_mask = target_info.attention_mask
        
        adapted_prompt_embeds = llm_adapter(
            prompt_embeds,
            target_input_ids,
            target_attention_mask=target_attention_mask,
            source_attention_mask=prompt_masks,
        )
        

        if adapted_prompt_embeds.shape[1] < max_sequence_length:
            adapted_prompt_embeds = torch.nn.functional.pad(
                adapted_prompt_embeds,
                (0, 0, 0, max_sequence_length - adapted_prompt_embeds.shape[1]),
            )

        # --- Apply ComfyUI-style per-token weights ---
        # Compute per-token weights for each prompt in the batch and apply them
        # to the adapted embeddings. This scales each token's embedding by its
        # parsed weight (e.g., (word:1.3) -> 1.3x scaling).
        safe_emit_progress(progress_callback, 0.55, "Applying prompt weights")
        
        batch_weights = []
        for p in prompt_list:
            token_weights = _get_per_token_weights(
                prompt=p,
                tokenizer=tokenizer,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=adapted_prompt_embeds.dtype,
            )
            batch_weights.append(token_weights)
        
        # Stack weights into (batch, seq_len) tensor
        weights_tensor = torch.stack(batch_weights, dim=0)
        
        # Apply weights to embeddings
        adapted_prompt_embeds = _apply_token_weights_to_embeddings(
            adapted_prompt_embeds, weights_tensor
        )
        
        safe_emit_progress(progress_callback, 0.60, "Prompt weights applied")

        return adapted_prompt_embeds
    
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 16,
        height: int = 768,
        width: int = 1360,
        num_frames: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor    + 1
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # For FlowMatch schedulers, sigmas live in [0, 1] and the max sigma is usually 1.0.
        # If timesteps were already set on the scheduler, scale initial noise by the first sigma.
        sigma0 = getattr(self.scheduler, "sigmas", None)
        if sigma0 is not None and len(sigma0) > 0:
            return latents * sigma0[0].to(device=device, dtype=dtype)
        return latents

    def run(self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            sigmas: Optional[List[float]] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            cfg_normalization: bool = False,
            cfg_truncation: float = 1.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            seed: Optional[int] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
            return_latents: bool = False,
            offload: bool = True,
            render_on_step: bool = False,
            timesteps: Optional[List[torch.FloatTensor]] = None,
            render_on_step_callback: Optional[Callable] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_sequence_length: int = 512,
            render_on_step_interval: int = 3,
            progress_callback: Callable = None,
            **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting text-to-image pipeline")

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        self._progress_callback = progress_callback
        

        height = height or 1024
        width = width or 1024

        # Ensure dimensions are compatible with VAE compression and transformer patchification.
        multiple_of = self.vae_scale_factor * 2
        height = max(multiple_of, (int(height) // multiple_of) * multiple_of)
        width = max(multiple_of, (int(width) // multiple_of) * multiple_of)
        
        device = self.device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
            offload=offload,
        )

        if num_images_per_prompt and num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
        
        # 4. Prepare timesteps
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # Custom `timesteps` takes precedence over `sigmas`.
        custom_timesteps = timesteps

        # Diffusers FlowMatchEulerDiscreteScheduler expects per-step `sigmas` (without the terminal 0),
        # and will append a terminal 0 internally.
        if custom_timesteps is None:
            if sigmas is None:
                sigmas = np.linspace(
                    1.0,
                    1.0 / float(num_inference_steps),
                    num_inference_steps,
                    dtype=np.float32,
                )
            else:
                # If the caller passes a schedule that already includes the terminal 0, drop it to
                # avoid an extra step and sigma=0 division issues.
                sigmas = np.asarray(list(sigmas), dtype=np.float32)
                if len(sigmas) > 0 and float(sigmas[-1]) == 0.0:
                    sigmas = sigmas[:-1]

            mu = None
            if getattr(self.scheduler, "config", None) is not None and getattr(
                self.scheduler.config, "use_dynamic_shifting", False
            ):
                transformer_config = self.load_config_by_type("transformer")
                if isinstance(transformer_config, dict):
                    patch_spatial = int(transformer_config.get("patch_spatial", 2))
                    patch_temporal = int(transformer_config.get("patch_temporal", 1))
                else:
                    patch_spatial = int(getattr(transformer_config, "patch_spatial", 2))
                    patch_temporal = int(
                        getattr(transformer_config, "patch_temporal", 1)
                    )

                latent_height = height // self.vae_scale_factor
                latent_width = width // self.vae_scale_factor
                # Match the token sequence length after patchification.
                token_h = max(1, latent_height // max(patch_spatial, 1))
                token_w = max(1, latent_width // max(patch_spatial, 1))
                token_t = max(1, 1 // max(patch_temporal, 1))
                image_seq_len = int(token_h * token_w * token_t)

                mu = self.calculate_shift(
                    image_seq_len=image_seq_len,
                    base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                    max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                    base_shift=self.scheduler.config.get("base_shift", 0.5),
                    max_shift=self.scheduler.config.get("max_shift", 1.15),
                )

            timesteps, num_inference_steps = self._get_timesteps(
                self.scheduler,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                mu=mu,
            )
        else:
            timesteps, num_inference_steps = self._get_timesteps(
                self.scheduler,
                num_inference_steps=num_inference_steps,
                timesteps=custom_timesteps,
            )

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
            
        
        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.num_channels_latents,
            height,
            width,
            1,
            torch.float32,
            device,
            generator,
            latents,
        )

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device, dtype=transformer_dtype
            )
        
        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)

        with self._progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
  
                self._current_timestep = t

                # Diffusers FlowMatch Euler uses `t` in [0, num_train_timesteps]; models take normalized timesteps.
                num_train_timesteps = float(
                    getattr(getattr(self.scheduler, "config", None), "num_train_timesteps", 1000)
                )
                timestep = t.expand(latents.shape[0]).to(device=device, dtype=torch.float32)
                model_timestep = (timestep / num_train_timesteps).to(transformer_dtype)

                latent_model_input = latents.to(device=device, dtype=transformer_dtype)

                noise_pred_cond = self.transformer(
                    x=latent_model_input,
                    timestep=model_timestep,
                    context=prompt_embeds,
                    padding_mask=None,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        x=latent_model_input,
                        timestep=model_timestep,
                        context=negative_prompt_embeds,
                        padding_mask=None,
                        return_dict=False,
                    )[0]
                    # Standard CFG: uncond + scale * (cond - uncond)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond
                    
                

                # Keep the solver state in float32 for numerical stability.
                noise_pred = noise_pred.float()
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        tensor_image = self.vae_decode(latents, offload=offload)[:, :, 0]
        image = self._tensor_to_frame(tensor_image)
        safe_emit_progress(progress_callback, 1.0, "Completed text-to-image pipeline")
        return image
