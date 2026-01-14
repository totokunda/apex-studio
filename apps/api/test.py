import torch
from src.engine import UniversalEngine
import json
from src.utils.save_audio_video import save_video_ltx2

torch.set_printoptions(threshold=5000)
torch.set_float32_matmul_precision("high")
run_info = json.load(open("/Users/tosinkuye/apex-workspace/apex-studio/apps/api/runs/qwenimage-edit-2511-1.0.0.v1/model_inputs.json"))

engine = UniversalEngine(
    **run_info["engine_kwargs"]
)

inputs = run_info["inputs"]


image = engine.run(**inputs)
image[0].save("qwenimage-edit-2511-1.0.0-768-16-9-4-3085748396-v1.png")