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

prompt = """A hyper-realistic, macro cinematic shot of a colossal, floating obsidian compass suspended in the heart of a swirling nebula made of liquid gold and crushed amethyst. The compass face is a complex clockwork mechanism of translucent emerald gears and floating holographic star-charts. Etched into the outer obsidian ring in an elegant, glowing Art Deco font is the text "NAVIGATE THE UNSEEN". Swarms of tiny, bioluminescent mechanical dragonflies orbit the device, casting intricate shadows. The lighting is dramatic and iridescent, featuring ray-traced reflections, 8k resolution, ethereal atmosphere, and a depth of field that emphasizes the crystalline texture of the gears."""

out = engine.run(
    prompt=prompt,
    height=1024, 
    width=1024,
    num_inference_steps=9,
    num_images=1,
    seed=42,
)

out[0].save("output_zimage-turbo_gguf_q8.png")
