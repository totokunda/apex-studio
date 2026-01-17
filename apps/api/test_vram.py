import torch
import time
import torch 
device = "cuda"
dtype = torch.float16  # 2 bytes/elem (use float32 for 4 bytes/elem)

gb = 15
bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
numel = gb * (1024**3) // bytes_per_elem

try:
    x = torch.empty(numel, device=device, dtype=dtype)
except Exception as e:
    print("Teehee OOM! OOM!")
    torch.cuda.empty_cache()