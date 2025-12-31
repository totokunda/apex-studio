from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os
from PIL import Image

variants = ["GGUF_Q2_K", "GGUF_Q3_K_S", "GGUF_Q3_K_M", "GGUF_Q4_K_S"]
variants.reverse()
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.1-14b-multitalk-text-to-video-1.0.0.v1.yml"

dir_path = "test_multitalk_wan_22_a14b"
os.makedirs(dir_path, exist_ok=True)

prompt = (
    "In a cozy recording studio, a man and a woman are singing together with passion and emotion. The man, with short brown hair, wears a light gray button-up shirt, his expression filled with concentration and warmth. The woman, with long wavy brown hair, dons a sleeveless dress adorned with small polka dots, her eyes closed as she belts out a heartfelt melody. The studio is equipped with professional microphones, and the background features soundproofing panels, creating an intimate and focused atmosphere. A close-up shot captures their expressions and the intensity of their performance.",
)
negative_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
image = Image.open("MultiTalk/examples/multi/2/multi2.png")
audio = "MultiTalk/examples/multi/2/1.wav"

for variant in variants:
    out_path = os.path.join(
        dir_path,
        f"test_wan_21_multitalk_a14b_{variant.lower()}_fusion_lora_sage_4steps.mp4",
    )
    print(f"[wan2.2 multitalk] Running variant={variant} -> {out_path}")
    try:
        engine = UniversalEngine(
            yaml_path=yaml_path,
            selected_components={
                "transformer": {"variant": variant},
                "text_encoder": {"variant": "FP8"},
            },
            attention_type="sage",
        )

        video = engine.run(
            audio_paths={"person1": audio, "person2": audio},
            image=image,
            prompt=prompt,
            # negative_prompt=negative_prompt,
            height=448,
            width=896,
            duration=81,
            num_videos=1,
            num_inference_steps=6,
            guidance_scale=1.0,
            audio_guidance_scale=4.0,
            seed=42,
        )

        export_to_video(
            video[0],
            out_path,
            fps=25,
            quality=8,
        )

        audio_file = audio
        muxed_outfile_path = out_path.replace(".mp4", "_with_audio.mp4")
        ffmpeg_cmd = f'ffmpeg -y -i "{out_path}" -i "{audio_file}" -c:v copy -c:a aac -shortest "{muxed_outfile_path}"'
        os.system(ffmpeg_cmd)
        os.replace(muxed_outfile_path, out_path)
    except Exception as e:
        raise e
        print(f"[wan2.2 multitalk] FAILED variant={variant}: {e}")

    break
