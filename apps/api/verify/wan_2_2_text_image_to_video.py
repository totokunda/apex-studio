from sympy import N
from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os

variants = ["GGUF_Q2_K", "GGUF_Q3_K_S", "GGUF_Q3_K_M", "GGUF_Q4_K_S", "GGUF_Q4_K_M", "GGUF_Q5_K_S", "GGUF_Q5_K_M", "GGUF_Q6_K", "GGUF_Q8_0"]
variants.reverse()

dir_path = "test_ti2v_wan_22_5b"
os.makedirs(dir_path, exist_ok=True)

image = "/home/tosin_coverquick_co/apex/images/wide.png"
prompt = "A young dancer performs energetic hip-hop moves in the middle of a busy street in Shibuya, Tokyo. Neon signs glow overhead, crowds flow around the dancer, blurred motion from passing pedestrians and traffic. Vibrant night lights reflect off wet pavement. Dynamic camera movement: smooth tracking shots, slight handheld realism. High-contrast cinematic lighting, rich colors, detailed urban textures. Atmosphere feels lively, modern, and electric, capturing the iconic Shibuya nightlife energy."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

for variant in variants:
    print(f"[wan2.2 t2i2v] Running variant={variant}")
    
    engine = UniversalEngine(
        yaml_path="/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-5b-text-to-image-to-video-turbo-1.0.0.v1.yml",
        selected_components={
            "transformer": {
                "variant": variant
            },
            "text_encoder": {
                "variant": "FP8"
            },
        },
        attention_type="sage",
        model_type="i2v"
    )
    
    video = engine.run(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=704,
        width=1280,
        duration=121,
        num_videos=1,
        seed=42,
    )

    export_to_video(
        video[0],
        os.path.join(
            dir_path, f"test_wan_22_ti2v_5b_{variant.lower()}.mp4"
        ),
        fps=24,
        quality=8,
    )
