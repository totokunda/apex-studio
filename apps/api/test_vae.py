from src.vae.tae.model import TAEHV
from src.vae.hunyuanvideo15.model import AutoencoderKLHunyuanVideo15
import torch as th
import cv2
from safetensors.torch import load_file
from diffusers.utils import export_to_video
from diffusers.pipelines.hunyuan_video1_5.image_processor import (
    HunyuanVideo15ImageProcessor,
)
import json

class VideoTensorReader:
    def __init__(self, video_file_path):
        self.cap = cv2.VideoCapture(video_file_path)
        assert self.cap.isOpened(), f"Could not load {video_file_path}"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    def __iter__(self):
        return self
    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration  # End of video or error
        return th.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) # BGR HWC -> RGB CHW

class VideoTensorWriter:
    def __init__(self, video_file_path, width_height, fps=30):
        self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
        assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"
    def write(self, frame_tensor):
        assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
        self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)) # RGB CHW -> BGR HWC
    def __del__(self):
        if hasattr(self, 'writer'): self.writer.release()


#video_path = "/root/apex-studio/apps/api/result.mp4"
#reader = VideoTensorReader(video_path)
#video = th.stack(list(reader), 0)[None].to("cuda", th.bfloat16).div_(255.0)

model_path = "lighttaehy1_5.safetensors"

vae = TAEHV(model_type="hy15", patch_size=2, latent_channels=32, scaling_factor=1.03682)
vae.load_state_dict(load_file(model_path))
vae.to("cuda", th.bfloat16)
vae.eval()

official_model_path = "/root/apex-diffusion/components/583a3c3320bed0f15628e4aadde45693a574802d4be2d0709481dd7abd03da62_vae-bf16.safetensors"
config = json.load(open('/root/apex-studio/apps/api/configs/HunyuanVideo-1.5/vae/config.json'))
official_vae = AutoencoderKLHunyuanVideo15.from_config(config)
official_vae.load_state_dict(load_file(official_model_path))
official_vae.to("cuda", th.bfloat16)
official_vae.eval()
official_vae.enable_tiling()

# encode video
with th.no_grad():
    #print(video.shape)
    #posterior = vae.encode(video.transpose(1, 2))[0]
    #latents = posterior.mode()
    latents = th.load("latents.pt")
    print(latents.shape)
    denormalized_latents = official_vae.denormalize_latents(latents)
    official_decoded = official_vae.decode(denormalized_latents)[0]
    decoded = vae.decode(latents)[0]
    print(decoded.shape)
    print(official_decoded.shape)
    
# save decoded video
output_path = "output0.mp4"
official_output_path = "official_output0.mp4"

video_processor = HunyuanVideo15ImageProcessor(vae_scale_factor=16)

outvideo = video_processor.postprocess_video(decoded)
official_outvideo = video_processor.postprocess_video(official_decoded)
# export to video
export_to_video(outvideo[0], output_path, fps=24)
export_to_video(official_outvideo[0], official_output_path, fps=24)
