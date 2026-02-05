from src.engine.base_engine import BaseEngine
from diffusers.video_processor import VideoProcessor
from typing import Optional, Union, List, Callable, Tuple
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from src.engine.ltx2.shared.audio_processing import LTX2AudioProcessingMixin
from einops import rearrange
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from src.engine.ltx22.shared.conditioning.types.latent_cond import AudioConditionByLatent
from src.engine.ltx22.shared.text_encoder_helpers import _norm_and_concat_padded_batch
from src.engine.ltx22.shared.types import AudioLatentShape, VideoPixelShape
from src.mixins.cache_mixin import sanitize_path_for_filename
from src.utils.defaults import get_cache_path
import os
from src.types.media import InputAudio
from src.vae.ltx2audio.ops import AudioProcessor



class LTX2Shared(LTX2AudioProcessingMixin, BaseEngine):
    """LTX2 Shared Engine Implementation"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio
            if getattr(self, "vae", None) is not None
            else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio
            if getattr(self, "vae", None) is not None
            else 8
        )
        # TODO: check whether the MEL compression ratio logic here is corrct
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio
            if getattr(self, "audio_vae", None) is not None
            else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio
            if getattr(self, "audio_vae", None) is not None
            else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size
            if getattr(self, "transformer", None) is not None
            else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t
            if getattr(self, "transformer") is not None
            else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate
            if getattr(self, "audio_vae", None) is not None
            else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length
            if getattr(self, "audio_vae", None) is not None
            else 160
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if getattr(self, "tokenizer", None) is not None
            else 1024
        )
        
        self.cache_file = os.path.join(
                get_cache_path(),
                f"text_encoder_{sanitize_path_for_filename(os.path.basename(yaml_path))}.safetensors",
            )

    def _parse_num_frames(self, duration: str | int, fps: float) -> int:
        if isinstance(duration, int):
            return duration
        elif isinstance(duration, str):
            if duration.endswith("s"):
                return int(duration[:-1]) * fps
            elif duration.endswith("f"):
                return int(duration[:-1])
            else:
                return int(duration)
        else:
            raise ValueError(f"Invalid duration: {duration}")
    
    def _run_connectors(self, connectors: torch.nn.Module, normed_concated_encoded_text_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = connectors(
            normed_concated_encoded_text_features, attention_mask
        )
        return connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask
    
    def _encode_audio(self, audio: InputAudio, device: Optional[torch.device] = None, strength: float = 1.0, fps: float = 25, num_frames: int = 121, offload: bool = True) -> torch.Tensor:
        device = device or self.device
        
        if not getattr(self, "audio_vae", None):
            self.load_component_by_name("audio_vae")
        self.to_device(self.audio_vae)
        dtype = self.component_dtypes["vae"]
        
        audio_array = self._load_audio(audio, sample_rate=self.audio_sampling_rate)
        audio_tensor = torch.from_numpy(audio_array)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        target_channels = int(getattr(self.audio_vae.encoder, "in_channels", audio_tensor.shape[1]))
        if target_channels <= 0:
            target_channels = audio_tensor.shape[1]
        if audio_tensor.shape[1] != target_channels:
            if audio_tensor.shape[1] == 1 and target_channels > 1:
                audio_tensor = audio_tensor.repeat(1, target_channels, 1)
            elif target_channels == 1:
                audio_tensor = audio_tensor.mean(dim=1, keepdim=True)
            else:
                audio_tensor = audio_tensor[:, :target_channels, :]
                if audio_tensor.shape[1] < target_channels:
                    pad_channels = target_channels - audio_tensor.shape[1]
                    pad = torch.zeros(
                        (audio_tensor.shape[0], pad_channels, audio_tensor.shape[2]),
                        dtype=audio_tensor.dtype,
                        device=audio_tensor.device,
                    )
                    audio_tensor = torch.cat([audio_tensor, pad], dim=1)
        
        audio_processor = AudioProcessor(
            sample_rate=self.audio_vae.encoder.sample_rate,
            mel_bins=self.audio_vae.encoder.mel_bins,
            mel_hop_length=self.audio_vae.encoder.mel_hop_length,
            n_fft=self.audio_vae.encoder.n_fft,
        )

        audio_processor = audio_processor.to(audio_tensor.device)
        mel = audio_processor.waveform_to_mel(audio_tensor, self.audio_sampling_rate)
        
        mel = mel.to(dtype=dtype, device=self.audio_vae.device)
        
        audio_latent = self.audio_vae.encoder(mel)
        audio_downsample = getattr(
        getattr(self.audio_vae.encoder, "patchifier", None),
        "audio_latent_downsample_factor",
                    4,
                )
        target_shape = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(
                batch=audio_latent.shape[0],
                frames=int(num_frames),
                width=1,
                height=1,
                fps=float(fps),
            ),
            channels=audio_latent.shape[1],
            mel_bins=audio_latent.shape[3],
            sample_rate=self.audio_vae.encoder.sample_rate,
            hop_length=self.audio_vae.encoder.mel_hop_length,
            audio_latent_downsample_factor=audio_downsample,
        )
        
        target_frames = target_shape.frames
        if audio_latent.shape[2] < target_frames:
            pad_frames = target_frames - audio_latent.shape[2]
            pad = torch.zeros(
                (audio_latent.shape[0], audio_latent.shape[1], pad_frames, audio_latent.shape[3]),
                device=audio_latent.device,
                dtype=audio_latent.dtype,
            )
            audio_latent = torch.cat([audio_latent, pad], dim=2)
        elif audio_latent.shape[2] > target_frames:
            audio_latent = audio_latent[:, :, :target_frames, :]
        audio_latent = audio_latent.to(device=self.device, dtype=dtype)
        audio_conditionings = [AudioConditionByLatent(audio_latent, strength)]
        
        return audio_conditionings
        
    def _encode_text(self, prompts=List[str], device: Optional[torch.device] = None, offload: bool = True) -> torch.Tensor:
        device = device or self.device
        normed_concated_encoded_text_features_list = []
        
       # hash each prompt individually
        prompt_hashes = [self.hash({"prompt": prompt}) for prompt in prompts]

        cached_prompts = []
        
        for prompt_hash in prompt_hashes:
            cached = self.load_cached(prompt_hash)
            if cached is not None:
                cached_prompts.append([cached[0].to(device), cached[1].to(device), cached[2].to(device)])
        
        if len(cached_prompts) == len(prompts):
            return cached_prompts
        
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)
        
    
        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model(no_weights=False)
        self.to_device(self.text_encoder.model)
        
        attention_mask_list = []
        dtype = self.component_dtypes["text_encoder"]
        for prompt in prompts:
            normed_concated_encoded_text_features, attention_mask = self._preprocess_text(prompt)
            normed_concated_encoded_text_features_list.append(normed_concated_encoded_text_features)
            attention_mask_list.append(attention_mask)
        
        
        if offload:
            self._offload("text_encoder")

        connectors = self.helpers["connectors"]
        self.to_device(connectors, device=device)
        
        result = []
        connectors = connectors.to(dtype)

        for normed_concated_encoded_text_features, attention_mask in zip(normed_concated_encoded_text_features_list, attention_mask_list):
            connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self._run_connectors(connectors, normed_concated_encoded_text_features, attention_mask)
            result.append((connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask))
        
        if offload:
            del connectors
            self._offload("connectors")
        
        if self.enable_cache:
            for prompt_hash, result_tuple in zip(prompt_hashes, result):
                self.cache(prompt_hash, result_tuple[0], result_tuple[1], result_tuple[2])
        
        return result
        
    
    def _preprocess_text(self, text: str, device: Optional[torch.device] = None, padding_side: str = "left") -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Encode a given string into feature tensors suitable for downstream tasks.
        Args:
            text (str): Input string to encode.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: Encoded features and a dictionary with attention mask.
        """
        device = device or self.device
        
        tokenizer = self.text_encoder.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.max_length = self.tokenizer_max_length
        
        token_pairs = self.tokenize_with_weights(tokenizer, text)["gemma"]

        
        model = self.text_encoder.model
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        normed_concated_encoded_text_features = self._run_norm_and_concat_padded_batch(
            hidden_states=outputs.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        
        return normed_concated_encoded_text_features, attention_mask
    
    def tokenize_with_weights(self, tokenizer: AutoTokenizer, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
    
        text = text.strip()
        encoded = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out

    
    def _run_norm_and_concat_padded_batch(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
    ) -> torch.Tensor:
        dtype = self.component_dtypes["text_encoder"]
        encoded_text_features = torch.stack(hidden_states, dim=-1)

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )
        
        return normed_concated_encoded_text_features.to(dtype)
       
