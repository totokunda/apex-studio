from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os

variants = ["GGUF_Q2_K", "GGUF_Q3_K_S", "GGUF_Q3_K_M", "GGUF_Q4_K_S", "GGUF_Q4_K_M", "GGUF_Q5_K_S", "GGUF_Q5_K_M", "GGUF_Q6_K", "GGUF_Q8_0", "FP8", "default"]
variant = "FP8"

engine = UniversalEngine(
    yaml_path="/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-a14b-text-to-video-1.0.0.v1.yml",
    selected_components={
        "low_noise_transformer": {
            "variant": variant
        },
        "high_noise_transformer": {
            "variant": variant
        },
        "text_encoder": {
            "variant": "FP8"
        },
    },
    attention_type="sage",
)

dir_path = "test_t2v_wan_22_a14b"
os.makedirs(dir_path, exist_ok=True)

prompt = "A young dancer performs energetic hip-hop moves in the middle of a busy street in Shibuya, Tokyo. Neon signs glow overhead, crowds flow around the dancer, blurred motion from passing pedestrians and traffic. Vibrant night lights reflect off wet pavement. Dynamic camera movement: smooth tracking shots, slight handheld realism. High-contrast cinematic lighting, rich colors, detailed urban textures. Atmosphere feels lively, modern, and electric, capturing the iconic Shibuya nightlife energy."

video = engine.run(
    prompt=prompt,
    height=480,
    width=832,
    duration=81,
    num_videos=1,
    num_inference_steps=4,
    guidance_scale=[1.0, 1.0],
    boundary_ratio=0.875,
    seed=42,
)

export_to_video(
    video[0],
    os.path.join(
        dir_path, f"test_wan_22_t2v_a14b_{variant.lower()}_lightning_lora_sage.mp4"
    ),
    fps=16,
    quality=8,
)