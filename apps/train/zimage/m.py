p = "/home/tosin_coverquick_co/apex-studio/apps/train/zimage/comfyui_adapter_model.safetensors"

from safetensors.torch import load_file
model = load_file(p)

print(model.keys())