from dotenv import load_dotenv

load_dotenv()
from src.engine.registry import UniversalEngine
import torch
import subprocess
import shutil

torch.set_printoptions(threshold=500, linewidth=300)

yaml_path = "/home/tosin_coverquick_co/apex-studio/apps/api/manifest/image/qwenimage-2512-1.0.0.v1.yml"

engine = UniversalEngine(yaml_path=yaml_path)

prompt = """A hyper-realistic, macro cinematic shot of a colossal, floating obsidian compass suspended in the heart of a swirling nebula made of liquid gold and crushed amethyst. The compass face is a complex clockwork mechanism of translucent emerald gears and floating holographic star-charts. Etched into the outer obsidian ring in an elegant, glowing Art Deco font is the text "NAVIGATE THE UNSEEN". Swarms of tiny, bioluminescent mechanical dragonflies orbit the device, casting intricate shadows. The lighting is dramatic and iridescent, featuring ray-traced reflections, 8k resolution, ethereal atmosphere, and a depth of field that emphasizes the crystalline texture of the gears."""
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"


out = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1328, 
    width=1328,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    num_images=1,
    seed=42,
)

out[0].save("output.png")

exit()
ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
    raise RuntimeError(
        "ffmpeg not found on PATH; install ffmpeg to mux audio into the output video."
    )

# Mux (and if needed, transcode) audio into the generated video.
subprocess.run(
    [
        ffmpeg,
        "-y",
        "-i",
        video_only_path,
        "-i",
        audio,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        final_path,
    ],
    check=True,
)
