from dotenv import load_dotenv
load_dotenv()

from src.engine.registry import UniversalEngine
import torch
import subprocess
import shutil

torch.set_printoptions(threshold=500, linewidth=300)

yaml_path = "/Users/tosinkuye/apex-workspace/apex-studio/apps/api/manifest/image/zimage-turbo-1.0.0.v1.yml"

engine = UniversalEngine(yaml_path=yaml_path, selected_components={
    "transformer": {
        "variant": "GGUF_Q8_0"
    }
})

prompt = """Playful mirror selfie, full-body view, young curvy brunette woman with freckles across nose/cheeks, long brown hair in two high pigtails. Kneeling upright on white fluffy rug, mischievously sticking tongue out, holding silver iPhone 17 Pro Max (screen active) to take photo of reflection. Wearing cropped rat boi style mint green Henley top (buttons visible) with translucent super deep neck and have M-Z Cup: Bust over 12in larger than band Weighs over 9.5lbs. High-waisted mint green booty cut short shorts with super thick thighs with subtle hubby figure. Sharp focus on subject/reflection, slightly high angle. In Cosy room"""

out = engine.run(
    prompt=prompt,
    height=1024, 
    width=1024,
    num_inference_steps=9,
    num_images=1,
    seed=42,
)

out[0].save("woman_with_iphone.png")
