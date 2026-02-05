p = "/home/tosin_coverquick_co/apex-studio/apps/api/talk/adapters"
from glob import glob
import os
import torch
import numpy as np
import safetensors.torch
from src.quantize.transformer import convert_model, ModelArchitecture, QuantConfig
from src.quantize.scaled_layer import get_fp_maxval, fp8_tensor_quant
files = glob(os.path.join(p, "*.safetensors"))

# open and convert to fp8
os.makedirs("/home/tosin_coverquick_co/apex-studio/apps/api/talk/fp8", exist_ok=True)
max_val = get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1)
for file in files:
    # copy resampler to the output directory
    # quantize to fp8
    state_dict = safetensors.torch.load_file(file)
    quantized_state_dict = {}
    for key, value in state_dict.items():
        if "weight" in key and value.ndim == 2:
            scale = value.abs().max() / max_val
            quantized_value, scale, _ = fp8_tensor_quant(value, scale)
            quantized_state_dict[key] = quantized_value.to(torch.float8_e4m3fn)
            quantized_state_dict[key.replace("weight", "scale_weight")] = scale.to(torch.float32)
        else:
            quantized_state_dict[key] = value
    safetensors.torch.save_file(quantized_state_dict, file.replace(".safetensors", "-fp8_e4m3fn.safetensors"))