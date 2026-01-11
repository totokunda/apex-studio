import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import copy
import torch
from src.engine.registry import UniversalEngine
from src.memory_management.utils import (
    analyze_block_to_flat_spec,
    allocate_block_flat_tensor,
    pack_block_to_flat,
    unpack_flat_to_block,
)

engine = UniversalEngine(
    yaml_path="manifest/video/ltx2-19b-text-to-image-to-video-distilled-1.0.0.v1.yml",
    selected_components={
        "transformer": {"variant": "default"},
    },
    attention_type="sdpa",
    auto_memory_management=False,
).engine

engine.device = torch.device("cpu")

cuda = torch.device("cuda")
engine.load_component_by_type("transformer")
transformer = engine.transformer

# Put *everything except* the transformer blocks on GPU, without ever moving
# the blocks to CUDA (they can be huge).

if torch.cuda.is_available():
    # Move all top-level submodules except `transformer_blocks`.
    for name, child in transformer.named_children():
        if name == "transformer_blocks":
            continue
        child.to(cuda)

    # Move any parameters/buffers that live directly on the transformer module
    # (not within child modules).
    for k, p in list(transformer._parameters.items()):
        if p is None:
            continue
        transformer._parameters[k] = torch.nn.Parameter(
            p.to(cuda), requires_grad=p.requires_grad
        )
    for k, b in list(transformer._buffers.items()):
        if b is None:
            continue
        transformer._buffers[k] = b.to(cuda)
else:
    raise RuntimeError("CUDA is not available; this script requires a CUDA GPU.")

blocks: List[torch.nn.Module] = list(transformer.transformer_blocks)
block0 = blocks[0]
spec = analyze_block_to_flat_spec(block0)

# Grouped streaming config (controlled via env vars so we can benchmark quickly):
# - n_buffers: number of resident CUDA "slots" / staging buffers
# - group_size: number of consecutive blocks loaded together per buffer fill
#
# Note: increasing GROUP_SIZE increases *resident* VRAM roughly proportional to:
#   n_buffers * group_size * block_weights
# and can OOM quickly for large blocks.
N_BUFFERS = int(os.environ.get("N_BUFFERS", "2"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "1"))

# Quick sanity log: confirm we are NOT using diffusers group offloading in this script.
print(
    f"[sanity] transformer _apex_group_offloading_enabled="
    f"{getattr(transformer, '_apex_group_offloading_enabled', False)}"
)
print(f"[sanity] block0 spec: entries={len(spec.entries):,} total_bytes={spec.total_bytes:,}")
print(
    f"[sanity] blocks={len(blocks)} total_packed_bytes={spec.total_bytes * len(blocks):,} "
    f"({(spec.total_bytes * len(blocks)) / (1024**3):.2f} GiB)"
)
print(f"[sanity] streamer config: n_buffers={N_BUFFERS} group_size={GROUP_SIZE}")

# Pre-pack blocks into CPU pinned memory (raw bytes) once, then pack into pinned groups.
# (This keeps the experiment controlled: group size only affects transfer granularity.)
packed_blocks_cpu: List[torch.Tensor] = [
    allocate_block_flat_tensor(spec, device="cpu", pin_memory=True) for _ in blocks
]
for i, b in enumerate(blocks):
    pack_block_to_flat(b, packed_blocks_cpu[i], spec)

num_blocks = len(blocks)
group_size = max(1, int(GROUP_SIZE))
num_groups = (num_blocks + group_size - 1) // group_size
packed_groups_cpu: List[torch.Tensor] = []
group_lens: List[int] = []
for g in range(num_groups):
    start = g * group_size
    end = min(num_blocks, (g + 1) * group_size)
    glen = end - start
    group_lens.append(glen)
    buf = torch.empty((spec.total_bytes * glen,), device="cpu", dtype=torch.uint8, pin_memory=True)
    # Concatenate block flats into the group buffer.
    for j in range(glen):
        off = j * spec.total_bytes
        buf[off : off + spec.total_bytes].copy_(packed_blocks_cpu[start + j], non_blocking=False)
    packed_groups_cpu.append(buf)

class _BlockStreamer:
    """
    N-buffered block streamer:
    - CPU pinned packed blocks (bytes)
    - N x flat GPU byte buffers
    - N x resident CUDA "slot" blocks used for execution
    - prefetch stream + events to overlap load with compute
    """

    def __init__(
        self,
        *,
        template_block_cpu: torch.nn.Module,
        packed_groups_cpu: List[torch.Tensor],
        group_lens: List[int],
        spec,
        device: torch.device,
        n_buffers: int = 2,  # Configurable buffering
        group_size: int = 1,
    ):
        self.device = device
        self.packed_groups_cpu = packed_groups_cpu
        self.group_lens = group_lens
        self.group_size = max(1, int(group_size))
        self.num_groups = len(packed_groups_cpu)
        self.spec = spec
        self.n_buffers = max(1, n_buffers)

        # Prefetch runs on a dedicated stream.
        self.prefetch_stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in range(self.n_buffers)]

        # GPU flat byte buffers for H2D staging (max size across groups).
        self.max_group_len = max(group_lens) if group_lens else 0
        self.max_group_bytes = self.spec.total_bytes * self.max_group_len
        self.flat_gpu = [
            torch.empty((self.max_group_bytes,), device=device, dtype=torch.uint8)
            for _ in range(self.n_buffers)
        ]

        # Resident CUDA slot blocks for execution: [buffer][block_in_group]
        self.slot: List[List[torch.nn.Module]] = []
        for _ in range(self.n_buffers):
            group_slots = [
                copy.deepcopy(template_block_cpu).to(device) for _ in range(self.max_group_len)
            ]
            for s in group_slots:
                s.eval()
                s.requires_grad_(False)
            self.slot.append(group_slots)

        # Which group index is currently loaded into each buffer.
        self.loaded_group = [-1] * self.n_buffers
        self._stats = {
            "prefetch_calls": 0,
            "prefetch_unique": 0,
        }
        self._loaded_set = set()
        # Store event pairs and compute timings at the end (after a synchronize)
        # to avoid forcing per-block synchronization.
        self._h2d_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._unpack_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._stall_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._compute_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @torch.no_grad()
    def prefetch(self, group_idx: int) -> None:
        """Schedule loading group `group_idx` into its assigned buffer."""
        if group_idx < 0 or group_idx >= self.num_groups:
            return
        
        # Round-robin mapping
        buf = group_idx % self.n_buffers
        
        # Optimization: If already loaded, do nothing.
        if self.loaded_group[buf] == group_idx:
            return

        self._stats["prefetch_calls"] += 1
        if group_idx not in self._loaded_set:
            self._loaded_set.add(group_idx)
            self._stats["prefetch_unique"] += 1

        with torch.cuda.stream(self.prefetch_stream):
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            unpack_end = torch.cuda.Event(enable_timing=True)

            h2d_start.record(self.prefetch_stream)
            # 1) H2D: packed group bytes -> flat GPU bytes
            group_cpu = self.packed_groups_cpu[group_idx]
            group_bytes = int(group_cpu.numel())
            self.flat_gpu[buf][:group_bytes].copy_(group_cpu, non_blocking=True)
            h2d_end.record(self.prefetch_stream)
            
            # 2) Unpack/alias each block in the group into its resident slot block.
            glen = self.group_lens[group_idx]
            for j in range(glen):
                off = j * self.spec.total_bytes
                flat_view = self.flat_gpu[buf][off : off + self.spec.total_bytes]
                unpack_flat_to_block(flat_view, self.slot[buf][j], self.spec, non_blocking=True)
            unpack_end.record(self.prefetch_stream)
            self.ready[buf].record(self.prefetch_stream)

            self._h2d_events.append((h2d_start, h2d_end))
            self._unpack_events.append((h2d_end, unpack_end))
        self.loaded_group[buf] = group_idx

    @torch.no_grad()
    def run(self, idx: int, *args, **kwargs):
        group_idx = idx // self.group_size
        within = idx % self.group_size
        if group_idx >= self.num_groups:
            raise IndexError(f"block idx {idx} out of range (num_groups={self.num_groups})")

        buf = group_idx % self.n_buffers
        
        # 1. Ensure current block is scheduled
        self.prefetch(group_idx)
        
        # 2. Aggressively prefetch future blocks to fill the pipeline
        #    (Look ahead up to n_buffers - 1 steps)
        for step in range(1, self.n_buffers):
            self.prefetch(group_idx + step)

        # 3. Wait until current buffer/slot has finished loading (this is "stall" time).
        stall_start = torch.cuda.Event(enable_timing=True)
        stall_end = torch.cuda.Event(enable_timing=True)
        stall_start.record()
        torch.cuda.current_stream().wait_event(self.ready[buf])
        stall_end.record()

        # 4. Execute using the resident slot block (compute time on the current stream).
        comp_start = torch.cuda.Event(enable_timing=True)
        comp_end = torch.cuda.Event(enable_timing=True)
        comp_start.record()
        out = self.slot[buf][within](*args, **kwargs)
        comp_end.record()

        # 5. Store timing events; compute totals in `report()` after sync.
        self._stall_events.append((stall_start, stall_end))
        self._compute_events.append((comp_start, comp_end))
        return out

    def report(self) -> Dict[str, float]:
        # Ensure all recorded events have completed before elapsed_time.
        torch.cuda.synchronize()

        def _sum_ms(pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
            total = 0.0
            for s, e in pairs:
                total += s.elapsed_time(e)
            return total

        out = dict(self._stats)
        out["h2d_ms"] = _sum_ms(self._h2d_events)
        out["unpack_ms"] = _sum_ms(self._unpack_events)
        out["stall_ms"] = _sum_ms(self._stall_events)
        out["compute_ms"] = _sum_ms(self._compute_events)
        out["unpack_calls"] = float(len(self._unpack_events))
        out["run_calls"] = float(len(self._compute_events))
        return out


class _StreamingBlock(torch.nn.Module):
    def __init__(self, streamer: _BlockStreamer, idx: int):
        super().__init__()
        self.streamer = streamer
        self.idx = idx

    def forward(self, *args, **kwargs):
        return self.streamer.run(self.idx, *args, **kwargs)

# Build streamer + swap transformer blocks with streaming proxies.
streamer = _BlockStreamer(
    template_block_cpu=blocks[0],
    packed_groups_cpu=packed_groups_cpu,
    group_lens=group_lens,
    spec=spec,
    device=cuda,
    group_size=GROUP_SIZE,
    n_buffers=N_BUFFERS,
)

# "Cheat": warm-load initial blocks before starting compute so first calls don't wait.
for i in range(min(len(blocks), streamer.n_buffers)):
    streamer.prefetch(i)
torch.cuda.synchronize()

transformer.transformer_blocks = torch.nn.ModuleList(
    [_StreamingBlock(streamer, i) for i in range(len(blocks))]
)

inputs = torch.load("transformer_input.pt")

def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x

inputs = _to_device(inputs, cuda)

print("Performing forward pass...")
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

    stats = streamer.report()
    denom = max(1, len(blocks))
    print(
        "[streamer breakdown] "
        f"prefetch_calls={int(stats['prefetch_calls'])} "
        f"prefetch_unique={int(stats['prefetch_unique'])} "
        f"h2d_ms={stats['h2d_ms']:.2f} (avg/block={stats['h2d_ms']/denom:.3f}) "
        f"unpack_ms={stats['unpack_ms']:.2f} (avg/block={stats['unpack_ms']/denom:.3f}) "
        f"stall_ms={stats['stall_ms']:.2f} (avg/block={stats['stall_ms']/denom:.3f}) "
        f"compute_ms={stats['compute_ms']:.2f} (avg/block={stats['compute_ms']/denom:.3f})"
    )