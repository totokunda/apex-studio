from sympy import N
from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os
import psutil

print(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

variants = ["GGUF_Q4_K_S", "GGUF_Q4_K_M", "GGUF_Q5_K_S", "GGUF_Q5_K_M", "GGUF_Q6_K", "GGUF_Q8_0", "FP8"]

dir_path = "test_vace_wan_21_14b"
os.makedirs(dir_path, exist_ok=True)

mask_video = '/home/tosin_coverquick_co/apex/assets/src_mask.mp4'
reference_image = "/home/tosin_coverquick_co/apex/assets/src_ref_image_1.png"
src_video = '/home/tosin_coverquick_co/apex/assets/src_video.mp4'
prompt = "The man is riding the horse passionately, as he gallops with force and speed."

for variant in variants:
    print(f"[wan2.2 t2i2v] Running variant={variant}")
    
    engine = UniversalEngine(
        yaml_path="/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.1-14b-vace-control-1.0.0.v1.yml",
        selected_components={
            "transformer": {
                "variant": variant
            },
            "text_encoder": {
                "variant": "FP8"
            },
        },
        attention_type="sage"
    )
    
    video = engine.run(
        prompt=prompt,
        video=src_video,
        reference_images=[reference_image],
        mask=mask_video,
        height=480,
        width=832,
        duration=81,
        num_videos=1,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
    )

    export_to_video(
        video[0],
        os.path.join(
            dir_path, f"test_wan_21_vace_14b_{variant.lower()}.mp4"
        ),
        fps=16,
        quality=8,
    )
    break
