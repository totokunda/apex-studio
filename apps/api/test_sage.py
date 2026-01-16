from sageattention import sageattn
import torch.nn.functional as F
import torch
B, H, S, D = 1, 32, 1024, 128
q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

out = sageattn(q, k, v)
# compare to 
out_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print(out_torch)
print(out_torch.shape)
print(out.shape)

# mean and max diff
print(torch.mean(torch.abs(out_torch - out)))
print(torch.max(torch.abs(out_torch - out)))