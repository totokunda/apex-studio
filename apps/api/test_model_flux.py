from dotenv import load_dotenv
load_dotenv()

from src.engine.registry import UniversalEngine
import torch
import subprocess
import shutil

torch.set_printoptions(threshold=500, linewidth=300)

yaml_path = "manifest/image/flux2-dev-text-to-image-edit-turbo-1.0.0.v1.yml"

engine = UniversalEngine(yaml_path=yaml_path, selected_components={
    "transformer": {
        "variant": "GGUF_Q5_0"
    },
})

prompt = """Make the woman have much larger breasts, with an F cup size."""
image = "woman.png"

out = engine.run(
    image=[image],
    prompt=prompt,
    height=1024, 
    width=1024,
    num_inference_steps=8,
    guidance_scale=3.5,
    num_images=1,
)

out[0].save("woman_edit_flux.png")
