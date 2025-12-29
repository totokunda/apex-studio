from src.engine import UniversalEngine
from diffusers.utils import export_to_video
import os
from PIL import Image

variants = ["GGUF_Q2_K", "GGUF_Q3_K_S", "GGUF_Q3_K_M", "GGUF_Q4_K_S", "GGUF_Q8_0"]
variants.reverse()
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.1-14b-infinitetalk-text-to-video-1.0.0.v1.yml"

dir_path = "test_infinitetalk_wan_21_14b"
os.makedirs(dir_path, exist_ok=True)


prompt= "A man barks like a dog and snarls loudly, with a menacing expression on his face.",
negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

audio = "/home/tosin_coverquick_co/apex/assets/bark.mp3"  
input_video = "assets/video/man_talking.mp4"

for variant in variants:
    out_path = os.path.join(
            dir_path, f"test_wan_21_infinitetalk_14b_{variant.lower()}_fusion_lora_sage_4steps_bark.mp4"
    )
    print(f"[wan2.1 infinitetalk] Running variant={variant} -> {out_path}")
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
            audio_paths={
                "person1": audio,
            },
            prompt=prompt,
            video=input_video,
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
        print(f"[wan2.1 infinitetalk] FAILED variant={variant}: {e}")
    break
