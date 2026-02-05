from src.quantize import TextEncoderQuantizer
from src.quantize.quants import QuantType

llama_quantize_path = "/home/tosin_coverquick_co/apex-studio/apps/api/llama-b7902/llama-quantize"

text_encoder_quantizer = TextEncoderQuantizer(
    output_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/text_encoder/text_encoder-q8_0",
    model_path="/home/tosin_coverquick_co/apex-diffusion/components/black-forest-labs_FLUX.2-dev/text_encoder",
    tokenizer_path="/home/tosin_coverquick_co/apex-diffusion/components/black-forest-labs_FLUX.2-dev/tokenizer",
    quantization=QuantType.Q8_0,
)

text_encoder_quantizer.quantize(
    llama_quantize_path=llama_quantize_path,
)

text_encoder_quantizer = TextEncoderQuantizer(
    output_path="/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/text_encoder/text_encoder-q4_k_m",
    model_path="/home/tosin_coverquick_co/apex-diffusion/components/black-forest-labs_FLUX.2-dev/text_encoder",
    tokenizer_path="/home/tosin_coverquick_co/apex-diffusion/components/black-forest-labs_FLUX.2-dev/tokenizer",
    quantization=QuantType.Q4_K_M,
)

text_encoder_quantizer.quantize(
    llama_quantize_path=llama_quantize_path,
)