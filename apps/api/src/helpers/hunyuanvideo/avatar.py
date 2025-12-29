from typing import Dict, Any
from PIL import Image
from transformers import WhisperModel, AutoFeatureExtractor
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
from torchvision import transforms
from transformers import CLIPImageProcessor
import torch
import librosa
import numpy as np
from typing import Union, List
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
from einops import rearrange
from src.helpers.hunyuanvideo.align import get_facemask, AlignImage
from torchvision.transforms import ToPILImage
import torch.nn as nn
from src.helpers.helpers import helpers


@helpers("hunyuanvideo.avatar")
class HunyuanAvatar(nn.Module, LoaderMixin, OffloadMixin):
    def __init__(
        self,
        model_path: str,
        save_path: str = DEFAULT_COMPONENTS_PATH,
        align_pt_path: str = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(model_path, save_path)
        self.model_path = model_path
        self.save_path = save_path
        self.wav2vec_model = WhisperModel.from_pretrained(
            model_path, cache_dir=save_path
        )
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path, cache_dir=save_path
        )
        self.align_image = AlignImage(
            save_path=save_path, pt_path=align_pt_path, device=device
        )
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(
                    (336, 336), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.4082107),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()

    def __call__(
        self,
        image: Image.Image,
        audio: str,
        fps: int = 25,
        num_frames: int = 129,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        ref_image = np.array(image)
        ref_image = torch.from_numpy(ref_image).to(device=device, dtype=dtype)

        audio_input, audio_len = self._extract_audio_features(audio)
        audio_prompts = audio_input[0].unsqueeze(0)

        motion_bucket_id_heads = np.array([25] * 4)
        motion_bucket_id_exps = np.array([30] * 4)
        motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads).unsqueeze(0)
        motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps).unsqueeze(0)
        fps = torch.from_numpy(np.array(fps)).unsqueeze(0).to(dtype=dtype)

        pixel_value_ref = rearrange(
            ref_image.clone().unsqueeze(0), "b h w c -> b c h w"
        )

        audio_prompts = [
            self._encode_audio(
                audio_feat.to(dtype=self.wav2vec_model.dtype),
                fps.item(),
                num_frames=num_frames,
            )
            for audio_feat in audio_prompts
        ]
        audio_prompts = torch.cat(audio_prompts, dim=0).to(device=device, dtype=dtype)

        self._offload("wav2vec_model")
        self._offload("align_image")

        uncond_audio_prompts = torch.zeros_like(audio_prompts)
        motion_exp = motion_bucket_id_exps.to(device=device, dtype=dtype)
        motion_pose = motion_bucket_id_heads.to(device=device, dtype=dtype)
        self.align_image.to(device=device)

        face_masks = get_facemask(
            pixel_value_ref.clone().unsqueeze(0), self.align_image, area=3.0
        )

        return {
            "fps": fps,
            "face_masks": face_masks,
            "motion_exp": motion_exp,
            "motion_pose": motion_pose,
            "uncond_audio_prompts": uncond_audio_prompts,
            "audio_prompts": audio_prompts,
        }

    def _extract_audio_features(self, audio: str):
        audio_input, sampling_rate = librosa.load(audio, sr=16000)
        assert sampling_rate == 16000

        audio_features = []
        window = 750 * 640
        for i in range(0, len(audio_input), window):
            audio_feature = self.audio_feature_extractor(
                audio_input[i : i + window],
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ).input_features
        audio_features.append(audio_feature)

        audio_features = torch.cat(audio_features, dim=-1)
        return audio_features, len(audio_input) // 640

    def _encode_audio(self, audio_feats, fps, num_frames=129):
        if fps == 25:
            start_ts = [0]
            step_ts = [1]
        elif fps == 12.5:
            start_ts = [0]
            step_ts = [2]
        num_frames = min(num_frames, 400)
        audio_feats = self.wav2vec_model.encoder(
            audio_feats.unsqueeze(0)[:, :, :3000].to(self.wav2vec_model.device),
            output_hidden_states=True,
        ).hidden_states
        audio_feats = torch.stack(audio_feats, dim=2)
        audio_feats = torch.cat([torch.zeros_like(audio_feats[:, :4]), audio_feats], 1)

        audio_prompts = []
        for bb in range(1):
            audio_feats_list = []
            for f in range(num_frames):
                cur_t = (start_ts[bb] + f * step_ts[bb]) * 2
                audio_clip = audio_feats[bb : bb + 1, cur_t : cur_t + 10]
                audio_feats_list.append(audio_clip)
            audio_feats_list = torch.stack(audio_feats_list, 1)
            audio_prompts.append(audio_feats_list)
        audio_prompts = torch.cat(audio_prompts)

        return audio_prompts
