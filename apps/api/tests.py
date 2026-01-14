import torch
from src.engine import UniversalEngine
import json
from src.utils.save_audio_video import save_video_ltx2

torch.set_printoptions(threshold=5000)
torch.set_float32_matmul_precision("high")
run_info = json.load(open("/Users/tosinkuye/apex-workspace/apex-studio/apps/api/runs/ltx2-19b-text-to-image-to-video-distilled-1.0.0.v1/model_inputs.json"))

engine = UniversalEngine(
    **run_info["engine_kwargs"]
)

inputs = run_info["inputs"]

prompt = """
A tight cinematic close-up of a man who was just smiling confidently, but his expression suddenly collapses. The smile fades instantly, his eyes widen with panic, and his face tightens like he’s fighting back tears. He leans forward toward the camera, shoulders tense, hands slightly shaking, breathing unevenly. In a burst of raw desperation, he screams with everything he has: “Hire me! I am begging you! I need a job desperately!” His voice cracks midway, sounding strained and urgent, like he’s reached his breaking point. The moment feels uncomfortably real—an emotional breakdown caught on camera.
Audio direction: Clear male voice, thick Nigerian accent, loud and pleading, strained throat, shaky breath, emotional urgency.
Camera direction: Handheld close-up, subtle shake, fast push-in as he starts yelling, shallow depth of field, sharp facial detail.
Lighting & style: Moody indoor lighting, soft shadows, cinematic contrast, realistic skin texture, dramatic intensity, high realism.
"""

inputs["prompt"] = prompt

video, audio = engine.run(**inputs)

save_video_ltx2(video, audio, "ltx2-19b-text-to-image-to-video-distilled-1.0.0-720-5-exp-v1")