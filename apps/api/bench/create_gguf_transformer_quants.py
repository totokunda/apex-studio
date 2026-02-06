from src.quantize import TransformerQuantizer
from src.quantize.quants import QuantType
from src.quantize.transformer import ModelArchitecture


transformer_quantizer = TransformerQuantizer(
    output_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-q8_0",
    model_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-bf16.safetensors",
    quantization=QuantType.Q8_0,
    architecture=ModelArchitecture.FLUX2,
)

transformer_quantizer.quantize()

transformer_quantizer = TransformerQuantizer(
    output_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-q6_k",
    model_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-bf16.safetensors",
    quantization=QuantType.Q6_K,
    architecture=ModelArchitecture.FLUX2,
)

transformer_quantizer.quantize()


transformer_quantizer = TransformerQuantizer(
    output_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-q4_k_m",
    model_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-bf16.safetensors",
    quantization=QuantType.Q4_K_M,
    architecture=ModelArchitecture.FLUX2,
)

transformer_quantizer.quantize()