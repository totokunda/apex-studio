path = "/home/tosin_coverquick_co/.cache/huggingface/hub/models--BigDannyPt--WAN-2.2-SmoothMix-FP16/snapshots/6ecb62ba180d39931f6fb5e3d60766639b09bf9c/smoothmixTxt2vidLow-FP16.safetensors"

from safetensors.torch import load_file, save_file

state_dict = load_file(path)
state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items() if "model.diffusion_model." in k}

# save back to same path
save_file(state_dict, "./smoothmixTxt2vidLow-FP16.safetensors")