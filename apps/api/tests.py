from flash_attn import flash_attn_func
import torch

q = torch.randn(1, 32, 512, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 8, 512, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 8, 512, 128, device='cuda', dtype=torch.bfloat16)

print("q.shape", q.shape)
print("k.shape", k.shape)
print("v.shape", v.shape)

out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))     

print(out.shape)