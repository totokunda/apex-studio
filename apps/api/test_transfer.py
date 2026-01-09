import torch
import time

# Create a dummy block (1GB of data)
# 256M elements * 4 bytes (float32) = 1GB
data_cpu = torch.randn(256 * 1024 * 1024, device='cpu').pin_memory()
data_gpu = torch.empty_like(data_cpu, device='cuda')

# CUDA Events for hardware-level timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Warmup (Crucial: First transfer is always slower due to context overhead)
data_gpu.copy_(data_cpu)
torch.cuda.synchronize()

# Record the actual transfer
start_event.record()
data_gpu.copy_(data_cpu, non_blocking=True)
end_event.record()

# Wait for GPU to catch up
torch.cuda.synchronize()

# Calculate result
elapsed_time_ms = start_event.elapsed_time(end_event)
bandwidth_gbs = (1.0) / (elapsed_time_ms / 1000.0)

print(f"Transfer Time: {elapsed_time_ms:.2f} ms")
print(f"Effective Bandwidth: {bandwidth_gbs:.2f} GB/s")