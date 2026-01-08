import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["APEX_RUST_DOWNLOAD_MAX_FILES"] = "32"
import torch

torch.set_printoptions(threshold=500, linewidth=300)
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex-studio/apps/api/manifest/video/hunyuanvideo-1.5-i2v-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, selected_components={
    "transformer": {
        "variant": "STEP_DISTILLED"
    }
}, attention_type="flex-block-attn")

prompt = "The woman while still holding her phone in her hand, smiles at the camera and waves gently."
image = "woman_edit_qwen.png"
from diffusers.utils import export_to_video

out = engine.run(
    reference_image=image,
    prompt=prompt,
    height=832,
    width=480,
    duration=121,
    num_inference_steps=8,
    num_videos=1,
    seed=42,
    fps=16,
    guidance_scale=1.0
)

export_to_video(out[0], "hunyuan_image_to_video.mp4", fps=24)