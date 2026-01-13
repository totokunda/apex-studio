from src.engine import UniversalEngine
import json
from src.utils.save_audio_video import save_video_ltx2

run_info = json.load(open("runs/ltx2-19b-text-to-image-to-video-1.0.0.v1/model_inputs.json"))

engine = UniversalEngine(
    **run_info["engine_kwargs"]
    )


inputs = run_info["inputs"]

prompt = """INT. NEO-TOKYO NOODLE STALL – RAINY NIGHT
PROMPT: A cinematic, high-contrast cyberpunk scene drenched in teal and orange neon light. Rain drums rhythmically against the corrugated metal roof, creating a wet, glossy texture on the counter. The camera begins in a gritty handheld medium shot, focused on an elderly chef with a rusty mechanical arm. He is chopping green onions with intense precision, the sound sharp and rhythmic. Steam rises violently from a pot of boiling broth, swirling around him.
The chef stops, wipes his brow with a rag, and slides a steaming ceramic bowl across the counter to a customer who is revealed to be a pristine, white-shelled android wearing a tattered trench coat. Chef (gruff, deep voice): "Spicy Miso. Just like you remember." The Android tilts its head, the steam reflecting in its smooth, featureless black visor. Android (soft, synthesized voice): "I have no memory... but my sensors detect warmth." The camera slowly pushes in past the chef’s shoulder, shifting focus from the steam to the android's reflection in the dark soup broth, ending on a tight, contemplative close-up of the visor.
"""

inputs["prompt"] = prompt
video, audio = engine.run(**inputs)

save_video_ltx2(video, audio, "tests/ltx2-19b-text-to-image-to-video-1.0.0-720-30.v1")