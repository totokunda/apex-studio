from src.engine.base_engine import BaseEngine
import torch
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms.v2 as v2
import av
from src.types.media import InputVideo
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, List
import numpy as np
from einops import rearrange
from torchvision.transforms import Normalize
from diffusers.utils.torch_utils import randn_tensor
from typing import Callable
import math

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: Optional[list[np.ndarray]]

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]


class MMAudioTV2AEngine(BaseEngine):
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                             std=[0.26862954, 0.26130258, 0.27577711])

        # Lazily initialized HuggingFace tokenizer for the CLIP model (only used
        # if the `clip_model` component doesn't already expose one).
        self._sampling_rate = 44100
        self._spectrogram_frame_rate = 512
        self._latent_downsample_rate = 2
        self._sync_num_frames_per_segment = 16
        self._sync_step_size = 8
        self._sync_downsample_rate = 2
        self._clip_seq_len = getattr(self.transformer.config, "clip_seq_len", 64) if self.transformer is not None else 64
        self._sync_seq_len = getattr(self.transformer.config, "sync_seq_len", 192) if self.transformer is not None else 192
        self._text_seq_len = getattr(self.transformer.config, "text_seq_len", 77) if self.transformer is not None else 77
        self._clip_dim = getattr(self.transformer.config, "clip_dim", 1024) if self.transformer is not None else 1024
        self._sync_dim = getattr(self.transformer.config, "sync_dim", 768) if self.transformer is not None else 768
        self._text_dim = getattr(self.transformer.config, "text_dim", 1024) if self.transformer is not None else 1024
        self._latent_seq_len = getattr(self.transformer.config, "latent_seq_len", 345) if self.transformer is not None else 345
        self._latent_dim = getattr(self.transformer.config, "latent_dim", 40) if self.transformer is not None else 40
    
    def _get_hf_clip_model_and_tokenizer(self):
        """
        Ensure the HF transformers CLIP model is loaded once and return:
        - clip_model: `transformers.CLIPModel`-like module
        - tokenizer: HF tokenizer compatible with CLIP text encoder

        This is the shared loading path for both `encode_video_with_clip` and
        `encode_text`, since they use the same underlying model.
        """
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)
        
        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model()
            self.text_encoder.model_loaded = True
        
        tokenizer = self.text_encoder.tokenizer
        clip_model = self.text_encoder.model

        return clip_model, tokenizer
    
    def prepare_latents(self, bs: int, latent_seq_len: int, latent_dim: int, generator: torch.Generator) -> torch.Tensor:
        transformer_dtype = self.component_dtypes["transformer"]
        shape = (bs, latent_seq_len, latent_dim)
        latents = randn_tensor(shape, device=self.device, dtype=transformer_dtype, generator=generator)
        return latents
 
    def _read_frames(self, video_path: Path, list_of_fps: list[float], start_sec: float | None, end_sec: float | None,
                    need_all_frames: bool) -> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
        output_frames = [[] for _ in list_of_fps]
        next_frame_time_for_each_fps = [0.0 for _ in list_of_fps]
        time_delta_for_each_fps = [1 / fps for fps in list_of_fps]
        all_frames = []

        # container = av.open(video_path)
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = stream.guessed_rate
            stream.thread_type = 'AUTO'
            for packet in container.demux(stream):
                for frame in packet.decode():
                    frame_time = frame.time
                    if start_sec is not None and frame_time < start_sec:
                        continue
                    if end_sec is not None and frame_time > end_sec:
                        break

                    frame_np = None
                    if need_all_frames:
                        frame_np = frame.to_ndarray(format='rgb24')
                        all_frames.append(frame_np)

                    for i, _ in enumerate(list_of_fps):
                        this_time = frame_time
                        while this_time >= next_frame_time_for_each_fps[i]:
                            if frame_np is None:
                                frame_np = frame.to_ndarray(format='rgb24')

                            output_frames[i].append(frame_np)
                            next_frame_time_for_each_fps[i] += time_delta_for_each_fps[i]

        output_frames = [np.stack(frames) for frames in output_frames]
        return output_frames, all_frames, fps


    def _load_video_mmaudio(self, video_path: Path, duration_sec: float | None, load_all_frames: bool = True) -> VideoInfo:

        clip_transform = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        if not isinstance(video_path, Path) and not isinstance(video_path, str):
            # for gradio>6.0, video_path is a namespace
            video_path = Path(video_path.value)
        output_frames, all_frames, orig_fps = self._read_frames(video_path,
                                                          list_of_fps=[_CLIP_FPS, _SYNC_FPS],
                                                          start_sec=0,
                                                          end_sec=duration_sec,
                                                          need_all_frames=load_all_frames)

        clip_chunk, sync_chunk = output_frames
        clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
        sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

        clip_frames = clip_transform(clip_chunk)
        sync_frames = sync_transform(sync_chunk)

        clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
        sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

        if duration_sec is not None and clip_length_sec < duration_sec:
            self.logger.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
            self.logger.warning(f'Truncating to {clip_length_sec:.2f} sec')
            duration_sec = clip_length_sec

        if duration_sec is not None and sync_length_sec < duration_sec:
            self.logger.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
            self.logger.warning(f'Truncating to {sync_length_sec:.2f} sec')
            duration_sec = sync_length_sec
            
        if duration_sec is None:
            # Keep both streams consistent when loading the full video.
            duration_sec = min(clip_length_sec, sync_length_sec)

        # Ensure sync frame count is divisible by 14 (model constraint).
        # We prefer truncation; if too short, we pad by repeating last frame(s).
        sync_multiple = 14
        if sync_frames.shape[0] <= 0:
            raise ValueError("No sync frames decoded from the input video.")

        # Start from the requested duration (already clamped to available lengths above),
        # convert to frames, then snap to a multiple of `sync_multiple`.
        requested_sync_frames = min(int(_SYNC_FPS * duration_sec), int(sync_frames.shape[0]))
        target_sync_frames = (requested_sync_frames // sync_multiple) * sync_multiple
        if target_sync_frames == 0:
            target_sync_frames = sync_multiple

        if target_sync_frames != requested_sync_frames:
            self.logger.info(
                f"Adjusting sync frames from {requested_sync_frames} to {target_sync_frames} "
                f"to keep length divisible by {sync_multiple}."
            )

        # Apply target length to sync frames.
        if target_sync_frames <= sync_frames.shape[0]:
            sync_frames = sync_frames[:target_sync_frames]
        else:
            pad = target_sync_frames - sync_frames.shape[0]
            sync_frames = torch.cat([sync_frames, sync_frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

        # Update duration to match the sync stream we will actually use.
        duration_sec = target_sync_frames / _SYNC_FPS

        # Keep clip frames consistent with the new duration. Clip stream is allowed to be
        # truncated or padded (repeat last frame) if needed.
        requested_clip_frames = int(_CLIP_FPS * duration_sec)
        if clip_frames.shape[0] <= 0:
            raise ValueError("No clip frames decoded from the input video.")
        if requested_clip_frames <= clip_frames.shape[0]:
            clip_frames = clip_frames[:requested_clip_frames]
        else:
            pad = requested_clip_frames - clip_frames.shape[0]
            clip_frames = torch.cat([clip_frames, clip_frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

        video_info = VideoInfo(
            duration_sec=duration_sec,
            fps=orig_fps,
            clip_frames=clip_frames,
            sync_frames=sync_frames,
            all_frames=all_frames if load_all_frames else None,
        )
        
        return video_info


    def encode_video_with_clip(
        self, x: torch.Tensor, batch_size: int = -1, offload: bool = True
    ) -> torch.Tensor:
        """
        HuggingFace transformers CLIP image features.

        Return shape is identical to `FeaturesUtils.encode_video_with_clip`:
        (B, T, D) where D is the CLIP projection dim, and features are L2-normalized.
        """
        clip_model, _ = self._get_hf_clip_model_and_tokenizer()

        # x: (B, T, C, H, W) with H/W = 384 (DFN5B CLIP uses 384px inputs)
        b, t, c, h, w = x.shape
        assert c == 3 and h == _CLIP_SIZE and w == _CLIP_SIZE

        pixel_values = self.clip_preprocess(x)
        pixel_values = rearrange(pixel_values, "b t c h w -> (b t) c h w")
        pixel_values = pixel_values.to(self.device)

        outputs = []
        if batch_size < 0:
            batch_size = b * t
            
        for i in range(0, b * t, batch_size):
            feats = clip_model.get_image_features(pixel_values=pixel_values[i : i + batch_size], interpolate_pos_encoding=True)
            feats = F.normalize(feats, dim=-1)
            outputs.append(feats)

        res = torch.cat(outputs, dim=0)
        res = rearrange(res, "(b t) d -> b t d", b=b)
        
        return res
    

    @torch.inference_mode()
    def encode_text(self, text: list[str], offload: bool = True, normalize: bool = False) -> torch.Tensor:
        """
        HuggingFace transformers CLIP text features as *last hidden state*.

        This matches the intent/shape of the OpenCLIP `patch_clip()` hack used by
        `FeaturesUtils.encode_text`: we return the final layer last hidden state
        (B, L, H) and L2-normalize over the hidden dimension.
        """
        clip_model, tokenizer = self._get_hf_clip_model_and_tokenizer()

        # OpenCLIP defaults to context length 77; preserve that shape for downstream.
        text_inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Use the text tower directly to obtain last hidden states.
        # `clip_model.get_text_features(...)` returns pooled/projection features,
        # which is *not* what the original OpenCLIP patch returned.
        text_out = clip_model.text_model(
            input_ids=text_inputs.input_ids,
            attention_mask=getattr(text_inputs, "attention_mask", None)
        )
        last_hidden_state = text_out.last_hidden_state
        if normalize:
            last_hidden_state = F.normalize(last_hidden_state, dim=-1)
        
        return last_hidden_state

    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1, offload: bool = True) -> torch.Tensor:
        synchformer = self.helpers["synchformer"]
        self.to_device(synchformer)
        # x: (B, T, C, H, W) H/W: 384

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
 
        if offload:
            del synchformer
            self._offload("synchformer")
        
        return x
    
    
    def get_latent_seq_len(self, duration: float) -> int:
        return int(
            math.ceil(duration * self._sampling_rate / self._spectrogram_frame_rate /
                      self._latent_downsample_rate))

    def get_clip_seq_len(self, duration: float) -> int:
        return int(duration * _CLIP_FPS)

    def get_sync_seq_len(self, duration: float) -> int:
        num_frames = duration * _SYNC_FPS
        num_segments = (num_frames - self._sync_num_frames_per_segment) // self._sync_step_size + 1
        return int(num_segments * self._sync_num_frames_per_segment / self._sync_downsample_rate)

    def run(self,
            video: InputVideo = None,
            prompt: List[str] | str = None,
            negative_prompt: str = "",
            duration: str | float = 8.0,
            guidance_scale: float = 4.5,
            num_inference_steps: int = 50,
            seed: int = None,
            generator: torch.Generator = None,
            latents: torch.Tensor = None,
            offload: bool = True,
            progress_callback: Callable = None,
            use_video_duration: bool = False,
            **kwargs,
            ):
        
            if type(duration) == str:
                duration = duration.replace('s', '')
                duration = float(duration)
            
            use_cfg_guidance = guidance_scale > 1.0 and negative_prompt is not None
            
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            if video is not None:
                video_info = self._load_video_mmaudio(Path(video), duration if not use_video_duration else None)
                clip_frames = video_info.clip_frames.unsqueeze(0)
                sync_frames = video_info.sync_frames.unsqueeze(0)
                duration = video_info.duration_sec
            else:
                clip_frames = sync_frames = None
            
            

            latent_seq_len = self.get_latent_seq_len(duration)
            clip_seq_len = self.get_clip_seq_len(duration)
            sync_seq_len = self.get_sync_seq_len(duration)
            

            
            if clip_frames is not None:
                clip_features = self.encode_video_with_clip(clip_frames)
            if sync_frames is not None:
                sync_features = self.encode_video_with_sync(sync_frames)
            if prompt is not None:
                text_features = self.encode_text([prompt])
            if negative_prompt is not None:
                negative_text_features = self.encode_text([negative_prompt])
            
            if offload:
                self._offload("text_encoder")
            

            if not self.transformer:
                self.load_component_by_type("transformer")
            self.to_device(self.transformer)
            

            self.transformer.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)
            
            if clip_frames is None:
                clip_features = self.transformer.get_empty_clip_sequence(1)
            if sync_frames is None:
                sync_features = self.transformer.get_empty_sync_sequence(1)
            if prompt is None:
                text_features = self.transformer.get_empty_string_sequence(1)
            if negative_prompt is None:
                negative_text_features = self.transformer.get_empty_string_sequence(1)
            bs = text_features.shape[0]
            latents = self.prepare_latents(bs, latent_seq_len, self._latent_dim, generator)
            
            if not self.scheduler:
                self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)
            
            scheduler = self.scheduler
            timesteps, num_inference_steps = self._get_timesteps(
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
            )
            
            clip_features = clip_features.to(self.device, dtype=self.component_dtypes["transformer"])
            sync_features = sync_features.to(self.device, dtype=self.component_dtypes["transformer"])
            text_features = text_features.to(self.device, dtype=self.component_dtypes["transformer"])
            negative_text_features = negative_text_features.to(self.device, dtype=self.component_dtypes["transformer"])
            
 
            preprocessed_conditions = self.transformer.preprocess_conditions(clip_features, sync_features, text_features)
            empty_conditions = self.transformer.get_empty_conditions(
                bs, negative_text_features=negative_text_features if negative_prompt is not None else None
            )
            
            with self._progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Convert scheduler sigma to MMAudio's time convention.
                    # MMAudio: t in [0,1] where t=0 is noise, t=1 is clean data.
                    # Scheduler: sigma in [1,0] where sigma=1 is noise, sigma=0 is data.
                    # Relationship: mmaudio_t = 1 - sigma
                    sigma = scheduler.sigmas[i]
                    model_t = (1.0 - sigma).expand(latents.shape[0]).to(latents.dtype)

                    flow = self.transformer(
                        latent=latents,
                        t=model_t,
                        conditions=preprocessed_conditions,
                        return_dict=False,
                    )[0]
                    
                    if use_cfg_guidance:
                        uncond_flow = self.transformer(
                            latent=latents,
                            t=model_t,
                            conditions=empty_conditions,
                            return_dict=False,
                        )[0]
                        flow = guidance_scale * flow + (1 - guidance_scale) * uncond_flow
                    
                    # MMAudio predicts flow v = (data - noise), but the scheduler
                    # expects v = (noise - data). Negate to match.
                    latents = scheduler.step(-flow, t, latents, return_dict=False)[0]
                    
                    if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                        progress_bar.update()
            
            latents = self.transformer.unnormalize(latents)
            
            if offload:
                self._offload("transformer")
            
            
            if not self.vae:
                self.load_component_by_type("vae")
            self.to_device(self.vae)
            
            self.vae.tod.remove_weight_norm()
            latents = latents.transpose(1, 2)
            audio = self.vae.decode(latents)[0]
            
            if offload:
                self._offload("vae")
            
            audio = audio.float().cpu()
            return audio.unbind(dim=0)