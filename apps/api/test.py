import torch
from src.engine import UniversalEngine
import json
from src.utils.save_audio_video import save_video_ltx2

torch.set_printoptions(threshold=5000)
torch.set_float32_matmul_precision("high")
run_info = json.load(open("runs/ltx2-19b-text-to-image-to-video-1.0.0.v1/model_inputs.json"))

engine = UniversalEngine(
    **run_info["engine_kwargs"]
)

inputs = run_info["inputs"]

video, audio = engine.run(**inputs)
save_video_ltx2(video, audio, "test_fflf")