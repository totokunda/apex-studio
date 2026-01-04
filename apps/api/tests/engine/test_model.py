from dotenv import load_dotenv

load_dotenv()
from src.engine.registry import UniversalEngine
import json
from diffusers.utils import export_to_video
import numpy as np
from typing import Optional
import tempfile
import soundfile as wavfile
import os
import torch
import requests
import io
from huggingface_hub import get_token
import subprocess
import shutil

torch.set_printoptions(threshold=500, linewidth=300)

directory = (
    "/home/tosin_coverquick_co/apex/runs/flux2-dev-text-to-image-edit-turbo-1.0.0.v1"
)

with open(os.path.join(directory, "model_inputs.json"), "r") as f:
    data = json.load(f)

engine_kwargs = data["engine_kwargs"]

inputs = data["inputs"]
for input_key, input_value in inputs.items():
    if isinstance(input_value, str) and input_value.startswith("assets"):
        inputs[input_key] = os.path.join(directory, input_value)

import time

start_time = time.perf_counter()
engine = UniversalEngine(**engine_kwargs)

out = engine.run(**inputs)
out[0].save("output.png")

exit()
audio = inputs["audio"]

video_only_path = "output.video_only.mp4"
final_path = "output_fusionx_lora.mp4"


export_to_video(out[0], video_only_path, fps=25, quality=8)

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
