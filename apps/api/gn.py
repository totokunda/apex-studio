import torch

from torch.nn.functional import group_norm

x = torch.randn((3, 32, 2048, 2048), device="cuda", dtype=torch.bfloat16)
w = torch.randn((32), device="cuda", dtype=torch.bfloat16)
b = torch.randn((32), device="cuda", dtype=torch.bfloat16)
eps = 1e-8

y = group_norm(x, 32, w, b, eps)
print(y.shape)
print(y.mean(), y.std())
import time
time.sleep(20)