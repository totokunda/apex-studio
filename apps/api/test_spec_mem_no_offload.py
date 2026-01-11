import time
import torch
from src.engine.registry import UniversalEngine


engine = UniversalEngine(
    yaml_path="manifest/video/ltx2-19b-text-to-image-to-video-distilled-1.0.0.v1.yml",
    selected_components={
        "transformer": {"variant": "default"},
    },
    attention_type="sdpa",
).engine

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; this script requires a CUDA GPU.")

cuda = torch.device("cuda")
engine.device = cuda

engine.load_component_by_type("transformer")
transformer = engine.transformer
transformer.eval()
transformer.requires_grad_(False)

# Ensure the entire transformer (including blocks) is on GPU: no offloading.
transformer.to(cuda)


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


inputs = torch.load("transformer_input.pt")
inputs = _to_device(inputs, cuda)

print("Performing forward pass (no offloading)...")
with torch.no_grad():
    # Warmup (helps stabilize kernel selection + caches).
    _ = transformer(**inputs)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    start_evt.record()
    out = transformer(**inputs)[0]
    end_evt.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    gpu_ms = start_evt.elapsed_time(end_evt)
    wall_ms = (t1 - t0) * 1000.0
    print(out.shape)
    print(f"Forward time: gpu={gpu_ms:.2f} ms | wall={wall_ms:.2f} ms")


