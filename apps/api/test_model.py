from dotenv import load_dotenv
load_dotenv()

from src.engine.registry import UniversalEngine
import torch
import subprocess
import shutil

torch.set_printoptions(threshold=500, linewidth=300)

yaml_path = "manifest/image/zimage-turbo-1.0.0.v1.yml"

engine = UniversalEngine(yaml_path=yaml_path, selected_components={
    "transformer": {
        "variant": "GGUF_Q8_0"
    }
}, auto_memory_management=False)

prompt = """Full-body view, young curvy blonde woman with sun-kissed skin, bright platinum blonde hair tied up in a messy high bun. Standing leaning casually against a glass railing, giving a peace sign with free hand, holding silver iPhone 17 Pro Max (screen active) to capture reflection in a large leaning metal-framed mirror. Wearing cropped rat boi style baby blue zip-up hoodie (worn open) over a deep-V white ribbed tank top and have M-Z Cup: Bust over 12in larger than band Weighs over 9.5lbs. High-waisted baby blue biker shorts highlighting super thick thighs with subtle hubby figure. Sharp focus on subject/reflection, eye-level angle, bright natural daylight. In luxe penthouse patio setting with large potted palms in background."""

out = engine.run(
    prompt=prompt,
    height=1024, 
    width=1024,
    num_inference_steps=9,
    num_images=1,
    seed=42,
)

out[0].save("woman.png")
