import torch
from src.engine import UniversalEngine
import json
from src.utils.save_audio_video import save_video_ltx2
import os

torch.set_printoptions(threshold=5000)
torch.set_float32_matmul_precision("high")
run_info = json.load(open("C:\\Users\\diviade\\apex-studio\\apps\\api\\runs\\nunchaku-flux-dev-kontext-1.0.0.v1\\model_inputs.json"))
base_path = "C:\\Users\\diviade\\apex-studio\\apps\\api\\runs\\nunchaku-flux-dev-kontext-1.0.0.v1"

engine = UniversalEngine(
    **run_info["engine_kwargs"]
)
base_path = "C:\\Users\\diviade\\apex-studio\\apps\\api\\runs\\nunchaku-flux-dev-kontext-1.0.0.v1"

inputs = run_info["inputs"]
image_path = inputs["image"]
inputs["image"] = os.path.join(base_path, image_path)


# video, audio = engine.run(**inputs)
# save_video_ltx2(video, audio, "test_fflf")

image = engine.run(**inputs)
image[0].save("test_image.png")