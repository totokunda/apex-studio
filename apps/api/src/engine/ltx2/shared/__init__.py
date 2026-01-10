from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor
from typing import Optional, Union, List
from PIL import Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from src.engine.ltx2.shared.audio_processing import LTX2AudioProcessingMixin

class LTX2Shared(LTX2AudioProcessingMixin, BaseEngine):
    """LTX2 Shared Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        
        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        # TODO: check whether the MEL compression ratio logic here is corrct
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer") is not None else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 1024
        )
        
    
    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: Union[str, torch.device],
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Packs and normalizes text encoder hidden states, respecting padding. Normalization is performed per-batch and
        per-layer in a masked fashion (only over non-padded positions).

        Args:
            text_hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
                Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
            sequence_lengths (`torch.Tensor of shape `(batch_size,)`):
                The number of valid (non-padded) tokens for each batch instance.
            device: (`str` or `torch.device`, *optional*):
                torch device to place the resulting embeddings on
            padding_side: (`str`, *optional*, defaults to `"left"`):
                Whether the text tokenizer performs padding on the `"left"` or `"right"`.
            scale_factor (`int`, *optional*, defaults to `8`):
                Scaling factor to multiply the normalized hidden states by.
            eps (`float`, *optional*, defaults to `1e-6`):
                A small positive value for numerical stability when performing normalization.

        Returns:
            `torch.Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
                Normed and flattened text encoder hidden states.
        """
        batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
        original_dtype = text_hidden_states.dtype
        # Create padding mask
        token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        if padding_side == "right":
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask = token_indices < sequence_lengths[:, None]  # [batch_size, seq_len]
        elif padding_side == "left":
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices = seq_len - sequence_lengths[:, None]  # [batch_size, 1]
            mask = token_indices >= start_indices  # [B, T]
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        mask = mask[:, :, None, None]  # [batch_size, seq_len] --> [batch_size, seq_len, 1, 1]

        # Compute masked mean over non-padding positions of shape (batch_size, 1, 1, seq_len)
        masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
        num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

        # Compute min/max over non-padding positions of shape (batch_size, 1, 1 seq_len)
        x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        # Normalization
        normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized_hidden_states = normalized_hidden_states * scale_factor

        # Pack the hidden states to a 3D tensor (batch_size, seq_len, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
        normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
        return normalized_hidden_states

    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`str` or `torch.device`):
                torch device to place the resulting embeddings on
            dtype: (`torch.dtype`):
                torch dtype to cast the prompt embeds to
            max_sequence_length (`int`, defaults to 1024): Maximum sequence length to use for the prompt.
        """
        device = device or self.device
        dtype = dtype or self.component_dtypes["text_encoder"]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        
        self.to_device(self.text_encoder)

        if getattr(self.text_encoder, "tokenizer", None) is not None:
            # Gemma expects left padding for chat-style prompts
            self.text_encoder.tokenizer.padding_side = "left"
            if self.text_encoder.tokenizer.pad_token is None:
                self.text_encoder.tokenizer.pad_token = self.text_encoder.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        
        
        
        text_encoder_hidden_states, prompt_attention_mask = self.text_encoder.encode(
            prompt,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
            add_special_tokens=True,
            pad_with_zero=False,
            use_attention_mask=True,
            output_type="hidden_states_all",
            hidden_states_all_stack_dim=-1,
            return_attention_mask=True,
        )
        
        text_encoder_hidden_states = text_encoder_hidden_states.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)
        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=device,
            padding_side=self.text_encoder.tokenizer.padding_side,
            scale_factor=scale_factor,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        offload: bool = True,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )
            
        if offload:
            self._offload("text_encoder")

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
    
    
    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents
    
    @staticmethod
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: Optional[int] = None, patch_size_t: Optional[int] = None
    ) -> torch.Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if patch_size is not None and patch_size_t is not None:
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length / patch_size_t
            post_patch_mel_bins = latent_mel_bins / patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents = latents.transpose(1, 2).flatten(2, 3)  # [B, C, L, M] --> [B, L, C * M]
        return latents

    @staticmethod
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
    ) -> torch.Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        r"""
        Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
        Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
        Flawed](https://huggingface.co/papers/2305.08891).

        Args:
            noise_cfg (`torch.Tensor`):
                The predicted noise tensor for the guided diffusion process.
            noise_pred_text (`torch.Tensor`):
                The predicted noise tensor for the text-guided diffusion process.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                A rescale factor applied to the noise predictions.

        Returns:
            noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt
    
    @property
    def distilled_stage_1_sigma_values(self):
        return self.config.get("distilled_sigma_values", [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0])
    
    @property
    def distilled_stage_2_sigma_values(self):
        return self.config.get("distilled_stage_2_sigma_values", [0.909375, 0.725, 0.421875, 0.0])