from dotenv import load_dotenv

load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine
import os

variants = ["default"]
yaml_path = "/home/tosin_coverquick_co/apex/manifest/verified/video/hunyuanvideo-1.5-i2v-1.0.0.v1.yml"

dir_path = "test_i2v_hunyuan_video_15"
os.makedirs(dir_path, exist_ok=True)
image = "/home/tosin_coverquick_co/apex/assets/harry_cup.jpg"
prompt = "The camera remains completely static. In the foreground, a clear, empty glass cup sits centered at the bottom of the frame on a cluttered desk, its thick rim and vertical facets catching faint reflections from the screen behind it. In the background, a laptop display fills most of the frame, showing a vivid, high-contrast image of a young wizard with round glasses and intense green eyes. He leans forward from the screen, arm extended, gripping a wand that appears to point directly toward the real glass in the foreground, creating a strong forced-perspective illusion. The wizard’s expression is focused and confident, his lips slightly parted as if mid-spell, with teal and emerald magical light swirling behind him on the screen. After a brief pause, the wizard’s expression softens into a subtle smile. He flicks his wand forward in a smooth, controlled motion. Instantly, a concentrated beam of bright blue magical light shoots from the wand’s tip, visually bridging the boundary between the screen and the real world. The beam strikes the inside of the glass cup, causing a luminous blue liquid to materialize from nothing. The liquid swirls as it forms, glowing intensely as the level steadily rises inside the cup. Soft steam curls upward from the surface, and the blue glow reflects sharply off the glass walls and desk surface. As the spell completes, the wand’s light fades, leaving the glass filled with a calm, radiant blue liquid that continues to shimmer faintly in the ambient light."

for variant in variants:
    num_inference_steps = 50
    out_path = os.path.join(
        dir_path,
        f"test_hunyuan_video_15_i2v_720p_{variant.lower()}_{num_inference_steps}_harry_cup.mp4",
    )
    print(f"[hunyuan video 1.5 t2v] Running variant={variant} -> {out_path}")
    try:
        engine = UniversalEngine(
            yaml_path=yaml_path,
            selected_components={
                "transformer": {"variant": variant},
                "text_encoder": {"variant": variant},
            },
            attention_type="sage",
        )

        video = engine.run(
            prompt=prompt,
            reference_image=image,
            num_inference_steps=num_inference_steps,
            resolution="720p",
            guidance_scale=6.0,
            enable_cache=True,
            duration=73,
            seed=42,
        )

        export_to_video(
            video[0],
            out_path,
            fps=24,
            quality=8,
        )
    except Exception as e:

        print(f"[hunyuan video 1.5 t2v] FAILED variant={variant}: {e}")
        raise e
