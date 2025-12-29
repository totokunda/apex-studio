from src.quantize import TransformerQuantizer
from src.utils.defaults import DEFAULT_SAVE_PATH
import os
from src.quantize.quants import QuantType

model_path = "/home/tosin_coverquick_co/apex-diffusion/components/HunyuanImage-3.0"

save_path = os.path.join(DEFAULT_SAVE_PATH, "gguf")
os.makedirs(save_path, exist_ok=True)

quantizer = TransformerQuantizer(
    output_path=os.path.join(save_path, "hunyuanimage3_q2.gguf"),
    model_path=model_path,
    architecture="hunyuanimage3",
    quantization=QuantType.Q2_K,
)

quantizer.quantize(
    keys_to_exclude=[
        "vae.",
        "vision_model.",
        "vision_aligner.",
        "lm_head.",
        "ln_f."
    ]
)    