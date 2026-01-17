"""
Attention backend smoke tests for Apex Studio (API).

Goal:
  - Import the API attention registry
  - Verify which attention backends are usable in THIS runtime
  - Run small Q/K/V attention forwards to ensure kernels execute and outputs are sane

Run (PowerShell, repo root):
  & "apps/api/.venv/Scripts/python.exe" ".\\attentions_testing.py"
"""


from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import torch

import inspect
import multiprocessing

from src.attention.functions import attention_register

q=torch.randn(1, 32, 1024, 64).to(torch.bfloat16).cuda()
k=torch.randn(1, 32, 1024, 64).to(torch.bfloat16).cuda()
v=torch.randn(1, 32, 1024, 64).to(torch.bfloat16).cuda()

print(q.shape, k.shape, v.shape)
# 
for backend in attention_register.all_available():
    if backend == "flash3":
        continue
    print(backend)
    print(attention_register.call(q, k, v, key=backend))
    print("-"*100)

