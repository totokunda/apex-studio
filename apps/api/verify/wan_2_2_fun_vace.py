from sympy import N
from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os
import psutil

print(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

variants = ["GGUF_Q8_0"]

dir_path = "test_vace_wan_22_fun_a14b"
os.makedirs(dir_path, exist_ok=True)

pose_video = "/home/tosin_coverquick_co/apex/images/pose.mp4"
prompt = "一位年轻女子站在阳光明媚的海岸线上，身穿清爽的白色衬衫与裙子，在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

for variant in variants:
    print(f"[wan2.2 fun vace] Running variant={variant}")

    engine = UniversalEngine(
        yaml_path="/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-fun-a14b-vace-1.0.0.v1.yml",
        selected_components={
            "transformer": {"variant": variant},
            "text_encoder": {"variant": "FP8"},
        },
        attention_type="sage",
    )

    video = engine.run(
        prompt=prompt,
        control_video=pose_video,
        height=832,
        width=480,
        duration=81,
        num_videos=1,
        num_inference_steps=4,
        high_noise_guidance_scale=1.0,
        low_noise_guidance_scale=1.0,
        seed=42,
    )

    export_to_video(
        video[0],
        os.path.join(
            dir_path, f"test_wan_22_fun_vace_a14b_control_{variant.lower()}_4steps.mp4"
        ),
        fps=16,
        quality=8,
    )
