import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import bisect
import copy
import torch
from src.engine.registry import UniversalEngine
from src.memory_management.utils import (
    analyze_block_to_flat_spec,
    allocate_block_flat_tensor,
    pack_block_to_flat,
    unpack_flat_to_block,
)
from src.quantize.ggml_tensor import GGMLTensor

engine = UniversalEngine(
    yaml_path="manifest/video/ltx2-19b-text-to-image-to-video-distilled-1.0.0.v1.yml",
    selected_components={
        "transformer": {"variant": "GGUF_Q8_0"},
        "text_encoder": {"variant": "FP8"},
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

def _set_tensor_by_dotted_name(
    root: torch.nn.Module,
    dotted_name: str,
    *,
    kind: Literal["param", "buffer"],
    value: torch.Tensor,
) -> None:
    parts = dotted_name.split(".")
    parent: torch.nn.Module = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]
    if kind == "param":
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    else:
        parent._buffers[leaf] = value


def _make_empty_like_for_streaming(
    t: torch.Tensor,
    *,
    physical_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create an empty tensor placeholder that preserves GGUF quant metadata when needed.
    This keeps GGML layers on the dequantization path (i.e. `is_quantized()` remains true).
    """
    base = torch.empty((0,), device="cpu", dtype=physical_dtype)
    if isinstance(t, GGMLTensor):
        return GGMLTensor(
            base,
            tensor_type=getattr(t, "tensor_type", None),
            tensor_shape=getattr(t, "tensor_shape", None),
            dequant_dtype=getattr(t, "dequant_dtype", None),
            patches=getattr(t, "patches", None),
            requires_grad=False,
        )
    return base


def make_streaming_slot_template_block(
    template_block: torch.nn.Module,
    *,
    spec,
) -> torch.nn.Module:
    """
    Make a lightweight template block for CUDA slots:
    - keep module structure + GGML tensor metadata
    - replace tensor storages with empty placeholders (so `.to(cuda)` is cheap)
    The real weight/buffer bytes are later rebound by `unpack_flat_to_block()`.
    """
    template_block.eval()
    template_block.requires_grad_(False)

    params = dict(template_block.named_parameters(recurse=True))
    buffers = dict(template_block.named_buffers(recurse=True))

    for e in spec.entries:
        src = params[e.name] if e.kind == "param" else buffers[e.name]
        empty = _make_empty_like_for_streaming(src, physical_dtype=e.physical_dtype)
        _set_tensor_by_dotted_name(template_block, e.name, kind=e.kind, value=empty)
    return template_block


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
        blocks_cpu: List[torch.nn.Module],
        template_blocks_cpu: List[torch.nn.Module],
        packed_blocks_cpu: List[torch.Tensor],
        spec,
        device: torch.device,
        n_buffers: int = 2,  # Configurable buffering
        group_size: int = 1,
    ):
        self.device = device
        self.blocks_cpu = blocks_cpu
        self.packed_blocks_cpu = packed_blocks_cpu
        self.group_size = max(1, int(group_size))
        self.num_blocks = len(packed_blocks_cpu)
        self.spec = spec
        self.n_buffers = max(1, n_buffers)

        # Prefetch runs on a dedicated stream.
        self.prefetch_stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in range(self.n_buffers)]

        # Pinned blocks that permanently live on GPU as *modules* (no unpack per call).
        self.pinned_blocks: Dict[int, torch.nn.Module] = {}

        # Staging buffers are sized based on the current streaming groups.
        # Initialize to 0 so `_set_stream_indices()` can safely grow them.
        self.max_group_len = 0
        self.max_group_bytes = 0
        self.flat_gpu: List[torch.Tensor] = []

        # Build initial streaming groups (all blocks streamed by default).
        self.stream_indices: List[int] = []
        self.stream_pos_by_idx: Dict[int, int] = {}
        self.packed_groups_cpu: List[torch.Tensor] = []
        self.group_lens: List[int] = []
        self.num_groups = 0
        self._set_stream_indices(list(range(self.num_blocks)))

        # Resident CUDA slot blocks for execution: one per buffer.
        #
        # We intentionally avoid `copy.deepcopy()` here. In this codebase, some GGUF/GGML
        # tensor subclasses can behave badly as deepcopy/memo keys (and may also hit
        # torch Storage deepcopy bugs). Instead, we reuse already-distinct CPU blocks
        # as templates (one per buffer), strip their storages to empty placeholders,
        # then move to CUDA.
        if len(template_blocks_cpu) < self.n_buffers:
            raise ValueError(
                f"Need at least n_buffers={self.n_buffers} template blocks, got {len(template_blocks_cpu)}"
            )
        self.slot: List[torch.nn.Module] = []
        for i in range(self.n_buffers):
            b = make_streaming_slot_template_block(template_blocks_cpu[i], spec=self.spec).to(device)
            b.eval()
            b.requires_grad_(False)
            self.slot.append(b)

        # Which group index is currently loaded into each buffer.
        self.loaded_group = [-1] * self.n_buffers

    def _set_stream_indices(self, indices: List[int]) -> None:
        # Keep only indices not pinned.
        indices = [i for i in indices if i not in self.pinned_blocks]
        self.stream_indices = indices
        self.stream_pos_by_idx = {idx: pos for pos, idx in enumerate(indices)}

        # (Re)build packed group buffers in CPU pinned memory.
        self.packed_groups_cpu = []
        self.group_lens = []
        n = len(indices)
        self.num_groups = (n + self.group_size - 1) // self.group_size if n else 0
        for g in range(self.num_groups):
            start = g * self.group_size
            end = min(n, (g + 1) * self.group_size)
            block_ids = indices[start:end]
            glen = len(block_ids)
            self.group_lens.append(glen)
            buf = torch.empty(
                (self.spec.total_bytes * glen,),
                device="cpu",
                dtype=torch.uint8,
                pin_memory=True,
            )
            for j, block_idx in enumerate(block_ids):
                off = j * self.spec.total_bytes
                buf[off : off + self.spec.total_bytes].copy_(
                    self.packed_blocks_cpu[block_idx], non_blocking=False
                )
            self.packed_groups_cpu.append(buf)

        # Reset resident group tracking so prefetch reloads under new grouping.
        self.loaded_group = [-1] * self.n_buffers

        # Resize staging buffers if needed (should only ever shrink, but handle growth safely).
        new_max_group_len = max(self.group_lens) if self.group_lens else 0
        new_max_group_bytes = self.spec.total_bytes * new_max_group_len
        if new_max_group_bytes > self.max_group_bytes:
            self.max_group_len = new_max_group_len
            self.max_group_bytes = new_max_group_bytes
            self.flat_gpu = [
                torch.empty((self.max_group_bytes,), device=self.device, dtype=torch.uint8)
                for _ in range(self.n_buffers)
            ]

    @torch.no_grad()
    def pin_more_blocks_after_warmup(
        self,
        *,
        headroom_bytes: int,
        min_stream_blocks: int = 1,
    ) -> Tuple[int, int]:
        """
        After a warmup run (so allocator/caches are in a steady-ish state), pin as many
        blocks as possible at the beginning and end by moving the *whole block module*
        to GPU. Then rebuild the streaming groups for the remaining middle blocks.

        Returns (n_pinned_left, n_pinned_right).
        """
        if self.num_blocks == 0:
            return (0, 0)

        # Use CUDA driver free bytes; this accounts for memory already held by the process.
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        budget = max(0, int(free_bytes) - int(headroom_bytes))
        per_block = int(self.spec.total_bytes)
        if per_block <= 0:
            return (0, 0)

        # Don't pin everything; keep at least `min_stream_blocks` in the middle (or 0 if requested).
        max_pin = max(0, self.num_blocks - max(0, int(min_stream_blocks)))
        can_pin = min(max_pin, budget // per_block)
        if can_pin <= 0:
            return (0, 0)

        left = (can_pin + 1) // 2
        right = can_pin // 2
        left = min(left, self.num_blocks)
        right = min(right, self.num_blocks - left)

        # Pin from both ends inwards.
        to_pin: List[int] = []
        for i in range(left):
            to_pin.append(i)
        for i in range(right):
            to_pin.append(self.num_blocks - 1 - i)
        to_pin = sorted(set(to_pin))

        # Move whole block modules to GPU once (no unpack per call).
        for idx in to_pin:
            if idx in self.pinned_blocks:
                continue
            b = self.blocks_cpu[idx].to(self.device, non_blocking=True)
            b.eval()
            b.requires_grad_(False)
            self.pinned_blocks[idx] = b
        torch.cuda.synchronize()

        # Rebuild the streaming region as the remaining middle blocks.
        remaining = [i for i in range(self.num_blocks) if i not in self.pinned_blocks]
        self._set_stream_indices(remaining)
        return (left, right)

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

        with torch.cuda.stream(self.prefetch_stream):
            # 1) H2D: packed group bytes -> flat GPU bytes
            group_cpu = self.packed_groups_cpu[group_idx]
            group_bytes = int(group_cpu.numel())
            self.flat_gpu[buf][:group_bytes].copy_(group_cpu, non_blocking=True)
            self.ready[buf].record(self.prefetch_stream)
        self.loaded_group[buf] = group_idx

    @torch.no_grad()
    def run(self, idx: int, *args, **kwargs):
        # Pinned blocks (no H2D).
        if idx in self.pinned_blocks:
            # Keep the streaming pipeline warm even while executing pinned blocks.
            # Otherwise, when we transition from pinned -> streamed, the first streamed
            # block can pay the full H2D latency (pipeline "cold start").
            if self.num_groups > 0 and self.stream_indices:
                # Find the next streamed block index after `idx`.
                pos = bisect.bisect_left(self.stream_indices, idx + 1)
                if pos < len(self.stream_indices):
                    next_streamed_idx = self.stream_indices[pos]
                    next_stream_pos = self.stream_pos_by_idx.get(next_streamed_idx, None)
                    if next_stream_pos is not None:
                        next_group = next_stream_pos // self.group_size
                        self.prefetch(next_group)
                        for step in range(1, self.n_buffers):
                            self.prefetch(next_group + step)

            return self.pinned_blocks[idx](*args, **kwargs)

        # Streamed blocks (H2D + bind).
        stream_pos = self.stream_pos_by_idx.get(idx, None)
        if stream_pos is None:
            raise IndexError(f"block idx {idx} not found in pinned or streaming sets")

        group_idx = stream_pos // self.group_size
        within = stream_pos % self.group_size
        if group_idx >= self.num_groups:
            raise IndexError(
                f"block idx {idx} out of range (num_groups={self.num_groups})"
            )

        buf = group_idx % self.n_buffers
        
        # 1. Ensure current block is scheduled
        self.prefetch(group_idx)
        
        # 2. Aggressively prefetch future blocks to fill the pipeline
        #    (Look ahead up to n_buffers - 1 steps)
        for step in range(1, self.n_buffers):
            self.prefetch(group_idx + step)

        # 3. Wait until current buffer/slot has finished loading.
        torch.cuda.current_stream().wait_event(self.ready[buf])

        # 3.5 Bind/alias the correct block's weights into the resident slot module.
        # This keeps GGML layers on the dequant path while avoiding having N copies
        # of the full module per group.
        glen = self.group_lens[group_idx]
        if within >= glen:
            raise IndexError(
                f"within-group index {within} out of range for group {group_idx} (len={glen})"
            )
        off = within * self.spec.total_bytes
        flat_view = self.flat_gpu[buf][off : off + self.spec.total_bytes]
        unpack_flat_to_block(flat_view, self.slot[buf], self.spec, non_blocking=False)

        # 4. Execute using the resident slot block.
        out = self.slot[buf](*args, **kwargs)
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
    blocks_cpu=blocks,
    # IMPORTANT: `make_streaming_slot_template_block()` mutates its inputs.
    # Use blocks well away from the ones we may pin (typically ends), so pinning
    # can still move the original blocks to CUDA intact.
    template_blocks_cpu=blocks[8 : 8 + N_BUFFERS],
    packed_blocks_cpu=packed_blocks_cpu,
    spec=spec,
    device=cuda,
    group_size=GROUP_SIZE,
    n_buffers=N_BUFFERS,
)

# "Cheat": warm-load initial blocks before starting compute so first calls don't wait.
for g in range(min(streamer.num_groups, streamer.n_buffers)):
    streamer.prefetch(g)
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

    # After warmup, pin as many blocks as we can (from both ends) to shrink streaming.
    headroom_mb = int(os.environ.get("PIN_HEADROOM_MB", "2048"))
    free_b, total_b = torch.cuda.mem_get_info()
    used_gib = (total_b - free_b) / (1024**3)
    print(f"[warmup] vram_used={used_gib:.2f} GiB (before pinning)")
    pinned_left, pinned_right = streamer.pin_more_blocks_after_warmup(
        headroom_bytes=headroom_mb * 1024 * 1024,
        min_stream_blocks=1,
    )
    print(
        f"[pinning] pinned_left={pinned_left} pinned_right={pinned_right} "
        f"stream_blocks={len(streamer.stream_indices)}"
    )

    t0 = time.perf_counter()
    from tqdm import tqdm
    for i in tqdm(range(10)):
        out = transformer(**inputs)[0]
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1000.0
    print(out.shape)
    print(f"Forward time: wall={wall_ms:.2f} ms")