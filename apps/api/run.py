from src.engine import UniversalEngine
import json
from diffusers.utils import export_to_video
from pathlib import Path

path = "/root/apex-studio/apps/api/runs/hunyuanvideo-1.5-t2v-480p-distilled/model_inputs.json"
run_dir = Path(path).parent

data = json.load(open(path))
engine_kwargs = data["engine_kwargs"]
inputs = data["inputs"]

for key, value in inputs.items():
    if isinstance(value, str) and "assets" in value:
        inputs[key] = value.replace("assets/", str(run_dir / "assets/") + "/")

engine = UniversalEngine(
    **engine_kwargs
)

out = engine.run(**inputs)[0]
export_to_video(out, "output1.mp4", fps=24, quality=8)