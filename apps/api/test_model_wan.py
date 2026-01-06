from dotenv import load_dotenv
load_dotenv()
import os

from src.engine.registry import UniversalEngine
import torch

torch.set_printoptions(threshold=500, linewidth=300)

yaml_path = "manifest/video/wan2.2-a14b-image-to-video-1.0.0.v1.yml"

engine = UniversalEngine(yaml_path=yaml_path, selected_components={
    "high_noise_transformer": {
        "variant": "GGUF_Q8_0"
    },
    "low_noise_transformer": {
        "variant": "GGUF_Q8_0"
    },
    "text_encoder": {
        "variant": "FP8"
    }
}, auto_memory_management=True)

prompt = """The woman while still holding her phone in her right hand, grabs her shirt and pulls it down to expose her bare breasts."""
image = "woman_edit_qwen.png"

out = engine.run(
    image=image,
    prompt=prompt,
    height=640, 
    width=640,
    duration=81,
    num_inference_steps=4,
    num_images=1,
    high_noise_guidance_scale=1.0,
    low_noise_guidance_scale=1.0,
    chunking_profile="aggressive"
)

from diffusers.utils import export_to_video
export_to_video(out[0], "woman_edit_wan.mp4", fps=16, quality=8)
