from src.quantize import TextEncoderQuantizer
from src.quantize.quants import QuantType

llama_quantize_path = "/home/tosin_coverquick_co/apex-studio/apps/api/llama-b7902/llama-quantize"

text_encoder_quantizer = TextEncoderQuantizer(
    output_path="/home/tosin_coverquick_co/apex-studio/apps/api/out_t/text_encoder-gguf_q8_0.gguf",
    model_path="/home/tosin_coverquick_co/apex-studio/apps/api/text_encoder",
    tokenizer_path="/home/tosin_coverquick_co/apex-studio/apps/api/tokenizer",
    quantization=QuantType.Q8_0,
)

text_encoder_quantizer.quantize(
    llama_quantize_path=llama_quantize_path,
)