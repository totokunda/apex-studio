import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
from src.engine.registry import UniversalEngine

engine = UniversalEngine(yaml_path="manifest/upscalers/seedvr2-3b.yml", selected_components={
    "transformer": {"variant": "GGUF_Q8_0"},
}, attention_type="sdpa")

video = "/workspace/apex-studio/apps/api/result_6_42.mp4"

out = engine.run(
    video=video,
    height=1024,  # Reduced from 1024 for 12GB VRAM
    width=1024,  # 720p is more realistic for 12GB
    seed=42,
    chunk_frames=17,  # Reduced for lower memory
    chunk_overlap=4,
    use_chunking=True,
    vae_memory_device="cpu",
    vae_conv_max_mem=0.05,  # More aggressive slicing
    vae_norm_max_mem=0.05,
    vae_split_size=4,  # Smaller splits
)

from diffusers.utils import export_to_video
export_to_video(out[0], "result_6_42_upscaled.mp4", fps=16, quality=8.0)