import torch
from src.attention.functions import attention_register
import warnings
warnings.filterwarnings("ignore")
# Configuration: 32 query heads, 8 key/value heads (4:1 ratio)
batch_size = 1
num_q_heads = 32
num_kv_heads = 32
seq_len = 1024*48
head_dim = 128
import time

q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)


torch.cuda.synchronize()
start_time = time.time()
out_metal_flash = attention_register.call(q, k, v, key="flash")
torch.cuda.synchronize()
end_time = time.time()
print(f"Flash Time: {end_time - start_time} seconds, throughput: {seq_len / (end_time - start_time)} tokens/second")


torch.cuda.synchronize()
start_time = time.time()
out_efficient_attention = attention_register.call(q, k, v, key="efficient_dot_product_attention")
torch.cuda.synchronize()
end_time = time.time()
print(f"Efficient Attention Time: {end_time - start_time} seconds, throughput: {seq_len / (end_time - start_time)} tokens/second")