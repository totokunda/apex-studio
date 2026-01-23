import json 
from src.engine import UniversalEngine
import os
from src.utils.save_audio_video import save_video_ltx2
p = "/home/tosin_coverquick_co/apex-studio/apps/api/runs/ltx2-19b-text-to-image-to-video-distilled-1.0.0.v1"

model_inputs = json.load(open(os.path.join(p, "model_inputs.json")))
engine_kwargs = model_inputs["engine_kwargs"]
inputs = model_inputs["inputs"]
engine = UniversalEngine(**engine_kwargs)
out = engine.run(**inputs)

save_video_ltx2(out[0], out[1], "result")