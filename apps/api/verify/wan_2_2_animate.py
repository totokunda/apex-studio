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
]
variants.reverse()
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-14b-animate-1.0.0.v1.yml"

dir_path = "test_animate_wan_22_a14b"
os.makedirs(dir_path, exist_ok=True)

image = "/home/tosin_coverquick_co/apex/Wan2.2/examples/wan_animate/animate/process_results/src_ref.png"
pose_video = "/home/tosin_coverquick_co/apex/Wan2.2/examples/wan_animate/animate/process_results/src_pose.mp4"
face_video = "/home/tosin_coverquick_co/apex/Wan2.2/examples/wan_animate/animate/process_results/src_face.mp4"

for variant in variants:
    out_path = os.path.join(
        dir_path,
        f"test_wan_22_animate_a14b_{variant.lower()}_sdpa_default_text_encoder_4_steps.mp4",
    )
    print(f"[wan2.2 animate] Running variant={variant} -> {out_path}")
    try:
        engine = UniversalEngine(
            yaml_path=yaml_path,
            selected_components={
                "transformer": {"variant": variant},
                "text_encoder": {"variant": "default"},
            },
            attention_type="sdpa",
        )

        video = engine.run(
            image=image,
            pose_video=pose_video,
            face_video=face_video,
            num_videos=1,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42,
        )

        export_to_video(
            video[0],
            out_path,
            fps=16,
            quality=8,
        )
    except Exception as e:
        print(f"[wan2.2 animate] FAILED variant={variant}: {e}")
        raise e
