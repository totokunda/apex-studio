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
    "FP8",
]
variants.reverse()
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan2.2-a14b-first-frame-last-frame-1.0.0.v1.yml"

dir_path = "test_fflf_wan_22_a14b"
os.makedirs(dir_path, exist_ok=True)

prompt = (
    "In a cozy recording studio, a man and a woman are singing together with passion and emotion. The man, with short brown hair, wears a light gray button-up shirt, his expression filled with concentration and warmth. The woman, with long wavy brown hair, dons a sleeveless dress adorned with small polka dots, her eyes closed as she belts out a heartfelt melody. The studio is equipped with professional microphones, and the background features soundproofing panels, creating an intimate and focused atmosphere. A close-up shot captures their expressions and the intensity of their performance.",
)
first_frame = "/home/tosin_coverquick_co/apex/images/dragon_first_frame.png"
last_frame = "/home/tosin_coverquick_co/apex/images/dragon_last_frame.png"

for variant in variants:
    out_path = os.path.join(
        dir_path, f"test_wan_22_fflf_a14b_{variant.lower()}_lightning_lora_sage.mp4"
    )
    print(f"[wan2.2 fflf] Running variant={variant} -> {out_path}")
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
            first_frame=first_frame,
            last_frame=last_frame,
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
        print(f"[wan2.2 fflf] FAILED variant={variant}: {e}")
