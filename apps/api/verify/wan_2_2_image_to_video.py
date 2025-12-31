from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os

variants = [
    "GGUF_Q2_K",
    "GGUF_Q3_K_S",
    "GGUF_Q3_K_M",
    "GGUF_Q4_K_S",
    "GGUF_Q4_K_M",
    "GGUF_Q5_K_S",
    "GGUF_Q5_K_M",
    "GGUF_Q6_K",
    "GGUF_Q8_0",
]
variants.reverse()
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan2.2-a14b-image-to-video-1.0.0.v1.yml"

dir_path = "test_i2v_wan_22_a14b"
os.makedirs(dir_path, exist_ok=True)

prompt = "A colossal volcanic dragon with obsidian-black scales glowing with fiery red cracks unleashes a massive stream of blazing fire into the smoky air, surrounded by drifting ash, swirling embers, and intense heat distortion; the camera pushes in slowly toward its roaring head as its wings flex from the force of the flame, molten light ripples across its armor-like scales, the background mountains blur in cinematic shallow depth of field, and the entire scene carries a high-fantasy, VFX-level atmosphere filled with smoke, molten reflections, and dramatic orange-against-black contrast, maintaining continuity with the reference image while adding powerful motion and epic energy. Fast motion and dramatic camera movement."
image = "/home/tosin_coverquick_co/apex/images/dragon_first_frame.png"

for variant in variants:
    out_path = os.path.join(
        dir_path, f"test_wan_22_i2v_a14b_{variant.lower()}_lightning_lora_sage.mp4"
    )
    print(f"[wan2.2 i2v] Running variant={variant} -> {out_path}")
    try:
        engine = UniversalEngine(
            yaml_path=yaml_path,
            selected_components={
                "low_noise_transformer": {"variant": variant},
                "high_noise_transformer": {"variant": variant},
                "text_encoder": {"variant": "FP8"},
            },
            attention_type="sage",
        )

        video = engine.run(
            image=image,
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
    except Exception as e:
        print(f"[wan2.2 i2v] FAILED variant={variant}: {e}")
