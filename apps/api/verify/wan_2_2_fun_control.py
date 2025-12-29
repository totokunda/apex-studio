from dotenv import load_dotenv
import torch
import os
load_dotenv()
from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-fun-a14b-control-1.0.0.v1.yml"
variants = ["GGUF_Q2_K", "GGUF_Q3_K_S", "GGUF_Q3_K_M", "GGUF_Q4_K_S", "GGUF_Q4_K_M", "GGUF_Q5_K_S", "GGUF_Q5_K_M", "GGUF_Q6_K", "GGUF_Q8_0", "FP8", "default"]
variants.reverse()
dir_path = "test_fun_control_wan_22_a14b"
os.makedirs(dir_path, exist_ok=True)

reference_image = "/home/tosin_coverquick_co/apex/Wan2.2/examples/wan_animate/animate/process_results/src_ref.png"
control_video = "/home/tosin_coverquick_co/apex/assets/john_oliver.mp4"
prompt = "The man gestures with his hands as he looks slightly irritated towards the camera."
negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",

for variant in variants:
    out_path = os.path.join(
        dir_path, f"test_wan_22_fun_control_a14b_{variant.lower()}_sage_lightning.mp4"
    )
    engine = UniversalEngine(
            yaml_path=yaml_path, 
            selected_components={
                "low_noise_transformer": {"variant": variant},
                "high_noise_transformer": {"variant": variant},
                "text_encoder": {"variant": "FP8"},
            }, 
            attention_type="sage")
    
    video = engine.run(
            ref_image=reference_image,
            control_video=control_video,
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
            out_path,
            fps=16,
            quality=8,
        )
