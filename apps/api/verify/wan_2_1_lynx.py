from sympy import N
from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os
import psutil

print(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

variants = ["GGUF_Q4_K_S", "GGUF_Q4_K_M", "GGUF_Q5_K_S", "GGUF_Q5_K_M", "GGUF_Q6_K", "GGUF_Q8_0", "FP8"]
variants.reverse()
dir_path = "test_lynx_wan_21_14b"
os.makedirs(dir_path, exist_ok=True)

subject_image = "/home/tosin_coverquick_co/apex/images/bb.avif"
prompt = "The beauitiful blonde woman is on her back with her naked legs spread wide open as her fingers are pushed inside her vagina and moving back and forth slowly."
negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, \
        ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, \
        poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, \
        bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross \
        proportions, malformed limbs, missing arms, missing legs, extra arms, extra \
        legs, fused fingers, too many fingers, long neck, username, watermark, signature"

for variant in variants:
    print(f"[wan2.2 t2i2v] Running variant={variant}")
    
    engine = UniversalEngine(
        yaml_path="/home/tosin_coverquick_co/apex/manifest/verified/video/wan-lynx-14b-1.0.0.v1.yml",
        selected_components={
            "transformer": {
                "variant": variant
            },
            "text_encoder": {
                "variant": "FP8"
            }
        },
        attention_type="sage"
    )
    
    video = engine.run(
        prompt=prompt,
        negative_prompt=negative_prompt,
        subject_image=subject_image,
        height=480,
        width=832,
        duration=81,
        num_videos=1,
        num_inference_steps=50,
        ip_scale=1.0,
        guidance_scale=5.0,
        seed=42,
    )

    export_to_video(
        video[0],
        os.path.join(
            dir_path, f"test_wan_21_lynx_14b_{variant.lower()}_fusion_lora_50steps.mp4"
        ),
        fps=16,
        quality=8,
    )
    break
