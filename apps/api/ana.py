from safetensors.torch import load_file, save_file
import torch

p = "/home/tosin_coverquick_co/apex-studio/apps/api/7B/LQ_proj_in.ckpt"
other_p = "/home/tosin_coverquick_co/apex-studio/apps/api/7B/diffusion_pytorch_model_streaming_dmd.safetensors"
other_state_dict = load_file(other_p)
state_dict = torch.load(p, map_location="cpu")
state_dict = {f"LQ_proj_in.{key}": value for key, value in state_dict.items()}
# merge state_dict and other_state_dict
state_dict.update(other_state_dict)
# save state_dict
save_file(state_dict, "7B/transformer-bf16.safetensors")