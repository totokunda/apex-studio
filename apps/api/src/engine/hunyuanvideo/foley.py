from __future__ import annotations
from typing import List
import numpy as np
import torch
from src.engine.base_engine import BaseEngine
from src.types import InputVideo
from torchvision import transforms
from torchvision.transforms import v2
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
import av
from PIL import Image
from typing import Tuple


class HunyuanFoleyEngine(BaseEngine):
    """HunyuanVideo-Foley (video+text → audio waveform)."""
    
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.siglip_fps = 8
        self.syncformer_fps = 25
        self.siglip2_preprocess = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        
        self.syncformer_preprocess = v2.Compose(
            [
                v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        
        self.audio_vae_latent_dim = getattr(self.transformer.config, "audio_vae_latent_dim", 128) if self.transformer is not None else 128
        self.audio_frame_rate = getattr(self.transformer.config, "audio_frame_rate", 50) if self.transformer is not None else 50
        
    
    def _get_frames_av(
        self,
        video_path: str,
        fps: float,
        max_length: float = None,
    ) -> Tuple[np.ndarray, float]:
        end_sec = max_length if max_length is not None else 15
        next_frame_time_for_each_fps = 0.0
        time_delta_for_each_fps = 1 / fps

        all_frames = []
        output_frames = []

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            ori_fps = stream.guessed_rate
            stream.thread_type = "AUTO"
            for packet in container.demux(stream):
                for frame in packet.decode():
                    frame_time = frame.time
                    if frame_time < 0:
                        continue
                    if frame_time > end_sec:
                        break

                    frame_np = None

                    this_time = frame_time
                    while this_time >= next_frame_time_for_each_fps:
                        if frame_np is None:
                            frame_np = frame.to_ndarray(format="rgb24")

                        output_frames.append(frame_np)
                        next_frame_time_for_each_fps += time_delta_for_each_fps

        output_frames = np.stack(output_frames)

        vid_len_in_s = len(output_frames) / fps
        if max_length is not None and len(output_frames) > int(max_length * fps):
            output_frames = output_frames[: int(max_length * fps)]
            vid_len_in_s = max_length

        return output_frames, vid_len_in_s

    def extract_siglip_features(self, video: InputVideo):
        siglip_video, _ = self._get_frames_av(video, fps=self.siglip_fps)
        siglip_model = self.helpers["siglip_model"]
        siglip_video = [Image.fromarray(frame).convert('RGB') for frame in siglip_video]
        images = [self.siglip2_preprocess(image) for image in siglip_video]  # [T, C, H, W]
        clip_frames = torch.stack(images).to(self.device).unsqueeze(0)

        siglip_features = self.encode_video_with_siglip2(clip_frames, siglip_model)

        return siglip_features
    
    def extract_syncformer_features(self, video: InputVideo):
        syncformer_video, _ = self._get_frames_av(video, fps=self.syncformer_fps)
        images = torch.from_numpy(syncformer_video).permute(0, 3, 1, 2)
        sync_frames = self.syncformer_preprocess(images).unsqueeze(0)
        syncformer_model = self.helpers["syncformer_model"]
        syncformer_features = self.encode_video_with_syncformer(sync_frames, syncformer_model)
        vid_len_in_s = sync_frames.shape[1] / self.syncformer_fps
        return syncformer_features, vid_len_in_s
    

    def encode_video_with_syncformer(self, x: torch.Tensor, syncformer_model, batch_size: int = -1):
        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
 
        for i in range(num_segments):
            segments.append(x[:, i * step_size : i * step_size + segment_size])
        x = torch.stack(segments, dim=1).to(self.device)  # (B, num_segments, segment_size, 3, 224, 224)

        outputs = []
        if batch_size < 0:
            batch_size = b * num_segments
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
        for i in range(0, b * num_segments, batch_size):
            outputs.append(syncformer_model(x[i : i + batch_size]))
        x = torch.cat(outputs, dim=0)  # [b * num_segments, 1, 8, 768]
        x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
        return x
    
    
    def encode_video_with_siglip2(self, x: torch.Tensor, siglip_model, batch_size: int = -1):
        b, t, c, h, w = x.shape
        if batch_size < 0:
            batch_size = b * t
        x = rearrange(x, "b t c h w -> (b t) c h w")
        outputs = []
        for i in range(0, b * t, batch_size):
            outputs.append(siglip_model.get_image_features(pixel_values=x[i : i + batch_size]))
        res = torch.cat(outputs, dim=0)
        res = rearrange(res, "(b t) d -> b t d", b=b)
        return res
    

    def extract_text_features(self, prompt: List[str], max_sequence_length: int = 77, offload: bool = True):
       
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        self.to_device(self.text_encoder)
        
        hash = self.text_encoder.hash({"prompt": prompt})
        if self.text_encoder.enable_cache:
            cached = self.text_encoder.load_cached(hash)
            if cached is not None:
                return cached[0].to(self.device)
        
        inputs = self.text_encoder.tokenizer(prompt, padding=True, return_tensors="pt").to(self.device)
        if not self.text_encoder.model_loaded:
            self.text_encoder.model = self.text_encoder.load_model()
        
        self.to_device(self.text_encoder)
        self.text_encoder.model.text_model.embeddings.token_type_ids = self.text_encoder.model.text_model.embeddings.token_type_ids.to(self.device).to(torch.long)
        
        outputs = self.text_encoder.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        if self.text_encoder.enable_cache:
            self.text_encoder.cache(hash, last_hidden_state)
            
        cond_feat = last_hidden_state[1:, :max_sequence_length]
        uncond_feat = last_hidden_state[:1, :max_sequence_length]
        
        return cond_feat, uncond_feat

    def prepare_latents(self, scheduler, batch_size, num_channels_latents, length, dtype, device, generator=None):
        shape = (batch_size, num_channels_latents, int(length))
        latents = randn_tensor(shape, device=device, dtype=dtype, generator=generator)
    
        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * scheduler.init_noise_sigma
    
        return latents
    
    def run(
        self,
        video: InputVideo,
        prompt: str,
        negative_prompt: str = 'noisy, harsh',
        guidance_scale: float = 4.5,
        num_inference_steps: int = 50,
        seed: int | None = None,
        duration: str | int = 15,
        use_video_duration: bool = True,
        offload: bool = True,
        **kwargs,
    ):
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        batch_size = 1
        
        syncformer_features, audio_length_s = self.extract_syncformer_features(video)
        siglip_features = self.extract_siglip_features(video)
        
        if not use_video_duration:
            if isinstance(duration, str):
                # remove trailing s
                duration = duration.replace("s", "")
                try:
                    duration = float(duration)
                    audio_length_s = duration
                except ValueError:
                    self.logger.warning(f"Invalid duration: {duration}")
                    
        if offload:
            self._offload("siglip_model")
            self._offload("syncformer_model")
            
        prompts = [negative_prompt, prompt]
        
        text_feat, uncond_text_feat = self.extract_text_features(prompts, offload=offload)

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
      
        timesteps, num_inference_steps = self._get_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
        )
        
        latents = self.prepare_latents(
            self.scheduler,
            batch_size=1,
            num_channels_latents=self.audio_vae_latent_dim,
            length=audio_length_s * self.audio_frame_rate,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]
        
        text_feat = text_feat.to(dtype=transformer_dtype)
        uncond_text_feat = uncond_text_feat.to(dtype=transformer_dtype)
        
        
        with self._progress_bar(total=len(timesteps), desc="Denoising steps") as progress_bar:
            for i, t in enumerate(timesteps):
  
                # noise latents
                latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                latent_input = self.scheduler.scale_model_input(latent_input, t)
        
                t_expand = t.repeat(latent_input.shape[0])
        
                # siglip2 features
                siglip2_feat = siglip_features.repeat(batch_size, 1, 1)  # Repeat for batch_size
                uncond_siglip2_feat = self.transformer.get_empty_clip_sequence(
                        bs=batch_size, len=siglip2_feat.shape[1]
                ).to(self.device)
        
                if guidance_scale is not None and guidance_scale > 1.0:
                    siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat], dim=0)
                else:
                    siglip2_feat_input = siglip2_feat
        
                # syncformer features
                syncformer_feat = syncformer_features.repeat(batch_size, 1, 1)  # Repeat for batch_size
                uncond_syncformer_feat = self.transformer.get_empty_sync_sequence(
                        bs=batch_size, len=syncformer_feat.shape[1]
                ).to(self.device)
                if guidance_scale is not None and guidance_scale > 1.0:
                    syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat], dim=0)
                else:
                    syncformer_feat_input = syncformer_feat
        
                # text features
                text_feat_repeated = text_feat.repeat(batch_size, 1, 1)  # Repeat for batch_size
                uncond_text_feat_repeated = uncond_text_feat.repeat(batch_size, 1, 1)  # Repeat for batch_size
                if guidance_scale is not None and guidance_scale > 1.0:
                    text_feat_input = torch.cat([uncond_text_feat_repeated, text_feat_repeated], dim=0)
                else:
                    text_feat_input = text_feat_repeated
    
                noise_pred = self.transformer(
                    x=latent_input.to(dtype=transformer_dtype),
                    t=t_expand,
                    cond=text_feat_input,
                    clip_feat=siglip2_feat_input.to(dtype=transformer_dtype),
                    sync_feat=syncformer_feat_input.to(dtype=transformer_dtype),
                    return_dict=False,
                )[0]
        
                noise_pred = noise_pred.to(dtype=torch.float32)
        
                if guidance_scale is not None and guidance_scale > 1.0:
                    # Perform classifier-free guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if offload:
            self._offload("transformer")
            
        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        
        latents = latents.to(dtype=self.vae.dtype)
        with torch.no_grad():
            audio = self.vae.decode(latents)
            audio = audio.float().cpu()
        
        if offload:
            self._offload("vae")
            
        audio = audio[:, :int(audio_length_s * self.audio_frame_rate)]
        return audio.unbind(0)