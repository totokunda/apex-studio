import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

from src.vae.wan.model import AutoencoderKLWan
import torch
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
device = "mps"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16).to(device)

#x = torch.randn(1, 3, 121, 480, 720, dtype=torch.bfloat16, device=device)
z = torch.randn(1, 48, 31, 30, 45, dtype=torch.bfloat16, device=device)

import time 

torch.mps.synchronize()
start_time = time.time()
with torch.no_grad():
    latent = vae.decode(z, return_dict=False)[0]
    print(latent.shape)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")