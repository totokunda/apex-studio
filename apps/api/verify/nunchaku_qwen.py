import os
from src.engine.registry import UniversalEngine
import torch
from diffusers.utils import export_to_video

engine = UniversalEngine(engine_type="qwenimage", yaml_path="/home/tosin_coverquick_co/apex/manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml", selected_components={
    "transformer": {
        "variant": "int4_r128"
    }
})

space_prompt = "Deep space: a velvet-black void studded with shimmering stars. Center frame: a colossal spiral galaxy, its cerulean and rose arms rotating against cosmic darkness. In the foreground, a sleek silver spacecraft hovers silently, its hull reflecting swirling nebulae.  \
Through panoramic windows, two astronauts drift weightlessly, their visors glowing with the light of a nearby supernova. Suddenly, iridescent plasma tendrils burst from a distant pulsar, casting rippling shadows as electric arcs dance like liquid light through glittering interstellar dust. \
Below, a ringed ice planet spins serenely, its pale rings fractured by twin moons that cast elongated shadows across frozen plains. A golden beam of starlight pierces a drifting gas cloud, igniting the spacecraft in a brief halo before fading into silence. Ethereal synths swell as the spiral galaxy pulses ever so faintly, hinting at a living universe that endures long after the scene fades to black."
negative_prompt=" "

image = engine.run(
    prompt=space_prompt + ", Ultra HD, 4K, cinematic composition.",
    negative_prompt=negative_prompt,
    height=928,
    width=1664,
    num_images=1,
    num_inference_steps=50,
    true_cfg_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
)

image[0].save("test_qwenimage_t2i_1_0_0_space.png")