from sageattention import sageattn
# create a real q, k, v tensors and compare to torch sdpa
import torch
import math

# Bigger sequence length and head dimension
b, h, s_q, d = 1, 40, 2048, 64

q = torch.randn(b, h, s_q, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, h, s_q, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, h, s_q, d, device="cuda", dtype=torch.bfloat16)

# Match PyTorch's default scaling: 1 / sqrt(head_dim)
sm_scale = 1.0 / math.sqrt(d)
sage_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=sm_scale)

# Force PyTorch to use math kernel for more direct comparison, and pass scale explicitly
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
    torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, scale=sm_scale)

print(sage_output.shape)
print(torch_output.shape)

# compare the output
print(f"allclose: {torch.allclose(sage_output, torch_output, atol=1e-4)}")
print(f"max diff: {(sage_output - torch_output).abs().max().item()}")
print(f"mean diff: {(sage_output - torch_output).abs().mean().item()}")
