import torch 
import torch.nn as nn
from taehv.taehv import TAEHV
from collections import namedtuple
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from src.engine import UniversalEngine

latent = torch.load("latent.pt", weights_only=False)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class TAEW2_2DiffusersWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.device = "mps"
        self.taehv = TAEHV("lighttaew2_2.safetensors").to(self.device).to(self.dtype)
        self.config = DotDict(
            scaling_factor=1.0,
            latents_mean=torch.zeros(self.taehv.latent_channels),
            z_dim=self.taehv.latent_channels,
            latents_std=torch.ones(self.taehv.latent_channels)
        )

    def decode(self, latents, return_dict=None):
        n, c, t, h, w = latents.shape
        # low-memory, set parallel=True for faster + higher memory
        return (self.taehv.decode_video(latents.transpose(1, 2), parallel=False).transpose(1, 2).mul_(2).sub_(1),)


taehv = TAEW2_2DiffusersWrapper()
vp = VideoProcessor()

with torch.no_grad():
    video = taehv.decode(latent.to(torch.float32))[0]

video = vp.postprocess_video(video)
export_to_video(video[0], "video.mp4", fps=24)

engine = UniversalEngine(yaml_path="/Users/tosinkuye/apex-workspace/apex-studio/apps/api/manifest/v0.1.2/video/wan-2.2-5b-text-to-image-to-video.yml", components_to_load=["vae"], selected_components={
    "text_encoder": {
        "variant": "GGUF_Q8_0"
    },
    "transformer": {
        "variant": "GGUF_Q6_K"
    }
}, attention_type="metal_flash")

vae = engine.vae
with torch.no_grad():
    vae.enable_tiling()
    video = vae.decode(latent)[0]

video = vp.postprocess_video(video)
export_to_video(video[0], "video_engine.mp4", fps=24)