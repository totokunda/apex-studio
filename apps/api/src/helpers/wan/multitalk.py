from typing import Dict, Any, Union, List, Optional
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
from src.helpers.helpers import helpers
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
import torch
import librosa
import numpy as np
from einops import rearrange
import soundfile as sf
import os
import tempfile
import torch.nn.functional as F
import torch.nn as nn

from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=seq_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)


class Wav2Vec2ModelMultitalk(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = False

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@helpers("wan.multitalk")
class WanMultiTalk(nn.Module, LoaderMixin, OffloadMixin):
    def __init__(
        self,
        model_path: str,
        save_path: str = DEFAULT_COMPONENTS_PATH,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.save_path = save_path
        self.device = device
        config_path = kwargs.get("config_path", None)
        if config_path is None:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.isfile(config_path):
                config_path = None
        if config_path is None:
            config_path = "https://huggingface.co/totoku/apex-models/resolve/main/MeiGen-MultiTalk/audio_encoder/config.json"
        
        # Initialize Wav2Vec2 components if available
        try:
            # Try to import and initialize the wav2vec model for audio feature extraction
            self.wav2vec_model = self._load_model({
                "base": "helpers.wan.multitalk.Wav2Vec2ModelMultitalk",
                "model_path": model_path,
                "config_path": config_path,
            }, module_name="src")
            
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_dict({
                "do_normalize": False,
                "feature_size": 1,
                "padding_side": "right",
                "padding_value": 0.0,
                "return_attention_mask": False,
                "sampling_rate": 16000,
            })
        except Exception as e:
            print(f"Warning: Could not load Wav2Vec2 model from {model_path}: {e}")
            self.wav2vec_model = None
            self.wav2vec_feature_extractor = None

    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        audio: Optional[Union[str, List[str]]] = None,
        audio_paths: Optional[Dict[str, str]] = None,
        audio_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        audio_type: str = "para",  # "para" for parallel, "add" for sequential
        num_frames: int = 81,
        vae_scale: int = 4,
        dtype: torch.dtype = torch.float32,
        bbox: Optional[Dict[str, List[float]]] = None,
        face_scale: float = 0.05,
        **kwargs,
    ):
        """
        Preprocess inputs for MultiTalk model.

        Args:
            image: Input conditioning image
            audio: Single audio file path or list of audio file paths
            audio_paths: Dictionary mapping person names to audio file paths
            audio_embeddings: Pre-computed audio embeddings
            audio_type: Type of audio combination ("para" or "add")
            num_frames: Number of video frames to generate
            vae_scale: VAE temporal downsampling scale
            dtype: Target data type
            bbox: Bounding boxes for multiple people
            face_scale: Scale factor for face regions
        """

        # Load and process image
        loaded_image = self._load_image(image)

        # Determine number of people from inputs
        if audio_paths is not None:
            human_num = len(audio_paths)
            audio_files = list(audio_paths.values())
        elif audio_embeddings is not None:
            human_num = len(audio_embeddings)
            audio_files = None
        elif isinstance(audio, list):
            human_num = len(audio)
            audio_files = audio
        elif audio is not None:
            human_num = 1
            audio_files = [audio]
        else:
            raise ValueError(
                "Must provide either audio paths, embeddings, or audio files"
            )

        # Process audio inputs
        if audio_embeddings is not None:
            # Use pre-computed embeddings
            processed_audio = self._process_audio_embeddings(
                audio_embeddings, num_frames
            )
        else:
            # Extract features from audio files
            processed_audio = self._process_audio_files(
                audio_files, num_frames, audio_type
            )

        # Generate human masks for spatial attention
        human_masks = self._generate_human_masks(
            loaded_image, human_num, bbox, face_scale
        )

        # Prepare outputs
        result = {
            "image": loaded_image,
            "audio_embeddings": processed_audio,
            "human_masks": human_masks,
            "human_num": human_num,
            "num_frames": num_frames,
        }

        return result

    def _process_audio_files(
        self,
        audio_files: List[str],
        num_frames: int,
        audio_type: str = "para",
    ) -> torch.Tensor:
        """Process audio files into embeddings."""
        if self.wav2vec_model is None:
            raise ValueError(
                "Wav2Vec2 model not available. Please provide pre-computed embeddings."
            )

        # Load and prepare audio
        audio_arrays = []
        for audio_file in audio_files:
            if audio_file is None or audio_file == "None":
                # Create silent audio placeholder
                # Use duration based on typical audio length for video
                duration_seconds = num_frames / 25.0  # Assume 25 fps
                audio_array = np.zeros(int(duration_seconds * 16000))
            else:
                audio_array = self._load_audio(
                    audio_file, sample_rate=16000, normalize=True
                )
            audio_arrays.append(audio_array)

        # Combine audio based on type
        if audio_type == "para":
            # Parallel - keep separate
            combined_audio = audio_arrays
        elif audio_type == "add":
            # Sequential - concatenate
            max_len = max(len(arr) for arr in audio_arrays)
            padded_arrays = []
            for arr in audio_arrays:
                padded = np.concatenate([arr, np.zeros(max_len - len(arr))])
                padded_arrays.append(padded)

            # Create sequential version
            combined_audio = []
            for i, arr in enumerate(padded_arrays):
                if i == 0:
                    seq_audio = np.concatenate([arr, np.zeros(max_len)])
                else:
                    seq_audio = np.concatenate([np.zeros(max_len), arr])
                combined_audio.append(seq_audio)
        else:
            raise ValueError(f"Unknown audio_type: {audio_type}")

        # Extract features using Wav2Vec2
        audio_embeddings = []
        for audio_array in combined_audio:
            embedding = self._extract_audio_features(audio_array)
            audio_embeddings.append(embedding)

        return audio_embeddings

    def _process_audio_embeddings(
        self, audio_embeddings: Dict[str, torch.Tensor], num_frames: int
    ) -> torch.Tensor:
        """Process pre-computed audio embeddings."""
        embeddings_list = []
        for person_key in sorted(audio_embeddings.keys()):
            embedding = audio_embeddings[person_key]
            embeddings_list.append(embedding)
        return embeddings_list

    def _extract_audio_features(
        self, audio_array: np.ndarray, sr: int = 16000
    ) -> torch.Tensor:
        """Extract audio features using Wav2Vec2."""

        if self.wav2vec_model is None or self.wav2vec_feature_extractor is None:
            raise ValueError("Wav2Vec2 components not available")

        audio_duration = len(audio_array) / sr

        video_length = int(audio_duration * 25)

        device = self.wav2vec_model.device
        # Process audio with feature extractor
        audio_features = np.squeeze(
            self.wav2vec_feature_extractor(audio_array, sampling_rate=sr).input_values
        )

        audio_feature = torch.from_numpy(audio_features).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        # Extract embeddings using Wav2Vec2 model
        with torch.no_grad():
            embeddings = self.wav2vec_model(
                audio_feature, seq_len=video_length, output_hidden_states=True
            )

        # Stack hidden states from different layers
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        return audio_emb

    def _generate_human_masks(
        self,
        image: Image.Image,
        human_num: int,
        bbox: Optional[Dict[str, List[float]]] = None,
        face_scale: float = 0.05,
    ) -> torch.Tensor:
        """Generate spatial masks for humans in the image."""
        height, width = image.height, image.width

        if human_num == 1:
            # Single person - use full image
            human_mask = torch.ones([height, width])
            background_mask = torch.ones([height, width])
            masks = [human_mask, torch.ones_like(human_mask), background_mask]

        elif human_num == 2:
            if bbox is not None:
                # Use provided bounding boxes
                background_mask = torch.zeros([height, width])
                human_masks = []

                for person_key in sorted(bbox.keys()):
                    x_min, y_min, x_max, y_max = bbox[person_key]
                    human_mask = torch.zeros([height, width])
                    human_mask[int(x_min) : int(x_max), int(y_min) : int(y_max)] = 1
                    background_mask += human_mask
                    human_masks.append(human_mask)

                # Background is where no humans are
                background_mask = torch.where(
                    background_mask > 0, torch.tensor(0), torch.tensor(1)
                )
                human_masks.append(background_mask)
                masks = human_masks
            else:
                # Default: split image in half vertically
                x_min, x_max = int(height * face_scale), int(height * (1 - face_scale))

                # Left person
                human_mask1 = torch.zeros([height, width])
                lefty_min, lefty_max = int((width // 2) * face_scale), int(
                    (width // 2) * (1 - face_scale)
                )
                human_mask1[x_min:x_max, lefty_min:lefty_max] = 1

                # Right person
                human_mask2 = torch.zeros([height, width])
                righty_min = int((width // 2) * face_scale + (width // 2))
                righty_max = int((width // 2) * (1 - face_scale) + (width // 2))
                human_mask2[x_min:x_max, righty_min:righty_max] = 1

                # Background
                background_mask = torch.where(
                    (human_mask1 + human_mask2) > 0, torch.tensor(0), torch.tensor(1)
                )

                masks = [human_mask1, human_mask2, background_mask]
        else:
            raise ValueError(f"Unsupported number of humans: {human_num}")

        return torch.stack(masks, dim=0).float()
