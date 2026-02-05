from src.quantize.quants import quantize, dequantize, _use_ggml_quants
from gguf import GGMLQuantizationType
import numpy as np
from tqdm import tqdm

weights = np.random.rand(5120, 5120).astype(np.float32)

print(_use_ggml_quants())


print(weights.shape)
print(weights.dtype)

# do this 100 iterations
for i in tqdm(range(100)):    
    quantized_weights = quantize(weights, GGMLQuantizationType.Q4_K)
