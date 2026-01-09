import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch

from src.engine.registry import UniversalEngine


@dataclass(frozen=True)
class BlockTensorEntry:
    kind: Literal["param", "buffer"]
    name: str
    shape: torch.Size
    numel: int
    dtype: torch.dtype
    start: int
    end: int  # exclusive


@dataclass(frozen=True)
class BlockFlatSpec:
    entries: Tuple[BlockTensorEntry, ...]
    total_numel: int
    dtype: torch.dtype


def _collect_named_tensors(
    block: torch.nn.Module,
    *,
    include_buffers: bool,
) -> List[Tuple[Literal["param", "buffer"], str, torch.Tensor]]:
    tensors: List[Tuple[Literal["param", "buffer"], str, torch.Tensor]] = []
    for name, p in block.named_parameters(recurse=True):
        if p is None:
            continue
        tensors.append(("param", name, p))
    if include_buffers:
        for name, b in block.named_buffers(recurse=True):
            if b is None:
                continue
            tensors.append(("buffer", name, b))
    return tensors


def analyze_block_to_flat_spec(
    block: torch.nn.Module,
    *,
    include_buffers: bool = True,
    require_single_dtype: bool = True,
    skip_names: Optional[Sequence[str]] = None,
) -> BlockFlatSpec:
    """
    Analyze a module "block" and produce a flat packing spec over its tensors.

    The spec provides a deterministic mapping between:
    - the block's parameters (+ optional buffers), and
    - a single 1D flat tensor of length `total_numel`.
    """
    skip = set(skip_names or [])
    named = [
        (kind, name, t)
        for (kind, name, t) in _collect_named_tensors(block, include_buffers=include_buffers)
        if name not in skip and t.numel() > 0
    ]
    if not named:
        return BlockFlatSpec(entries=tuple(), total_numel=0, dtype=torch.float32)

    dtypes = {t.dtype for _, _, t in named}
    if require_single_dtype and len(dtypes) != 1:
        details = ", ".join(sorted({f"{kind}:{name}={t.dtype}" for kind, name, t in named}))
        raise ValueError(
            "Block tensors have multiple dtypes; flat packing expects a single dtype. "
            f"Found: {sorted(map(str, dtypes))}. Details: {details}"
        )
    flat_dtype = next(iter(dtypes)) if len(dtypes) == 1 else torch.float32

    entries: List[BlockTensorEntry] = []
    offset = 0
    for kind, name, t in named:
        n = int(t.numel())
        entries.append(
            BlockTensorEntry(
                kind=kind,
                name=name,
                shape=t.shape,
                numel=n,
                dtype=t.dtype,
                start=offset,
                end=offset + n,
            )
        )
        offset += n
    return BlockFlatSpec(entries=tuple(entries), total_numel=offset, dtype=flat_dtype)


def allocate_block_flat_tensor(
    spec: BlockFlatSpec,
    *,
    device: torch.device | str,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Allocate an *empty* 1D flat tensor sized to hold the entire block (per `spec`)."""
    if spec.total_numel == 0:
        return torch.empty((0,), device=device, dtype=spec.dtype)
    device = torch.device(device)
    if device.type == "cpu":
        return torch.empty((spec.total_numel,), device=device, dtype=spec.dtype, pin_memory=pin_memory)
    return torch.empty((spec.total_numel,), device=device, dtype=spec.dtype)


def pack_block_to_flat(
    block: torch.nn.Module,
    flat: torch.Tensor,
    spec: BlockFlatSpec,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy tensors from `block` into `flat` according to `spec`."""
    if flat.numel() != spec.total_numel:
        raise ValueError(f"Flat tensor has numel={flat.numel()} but spec expects {spec.total_numel}")
    if flat.dtype != spec.dtype:
        raise ValueError(f"Flat tensor dtype={flat.dtype} but spec dtype={spec.dtype}")

    params: Dict[str, torch.Tensor] = dict(block.named_parameters(recurse=True))
    buffers: Dict[str, torch.Tensor] = dict(block.named_buffers(recurse=True))

    with torch.no_grad():
        for e in spec.entries:
            t = params[e.name] if e.kind == "param" else buffers[e.name]
            src = t.detach().reshape(-1)
            dst = flat[e.start : e.end]
            dst.copy_(src, non_blocking=non_blocking)
    return flat


def unpack_flat_to_block(
    flat: torch.Tensor,
    block: torch.nn.Module,
    spec: BlockFlatSpec,
    *,
    non_blocking: bool = False,
) -> None:
    """Copy values from `flat` back into `block` tensors according to `spec`."""
    if flat.numel() != spec.total_numel:
        raise ValueError(f"Flat tensor has numel={flat.numel()} but spec expects {spec.total_numel}")
    if flat.dtype != spec.dtype:
        raise ValueError(f"Flat tensor dtype={flat.dtype} but spec dtype={spec.dtype}")

    params: Dict[str, torch.Tensor] = dict(block.named_parameters(recurse=True))
    buffers: Dict[str, torch.Tensor] = dict(block.named_buffers(recurse=True))

    with torch.no_grad():
        for e in spec.entries:
            t = params[e.name] if e.kind == "param" else buffers[e.name]
            dst = t.reshape(-1)
            src = flat[e.start : e.end]
            dst.copy_(src, non_blocking=non_blocking)


def describe_block_flat_spec(spec: BlockFlatSpec, *, max_rows: int = 25) -> str:
    rows = [f"total_numel={spec.total_numel:,} dtype={spec.dtype} entries={len(spec.entries)}"]
    for i, e in enumerate(spec.entries[:max_rows]):
        rows.append(f"{i:03d} {e.kind:6s} {e.name:60s} shape={tuple(e.shape)} numel={e.numel:,}")
    if len(spec.entries) > max_rows:
        rows.append(f"... ({len(spec.entries) - max_rows} more)")
    return "\n".join(rows)


def time_flat_h2d_copy_ms(
    flat_cpu_pinned: torch.Tensor,
    *,
    iters: int = 20,
    warmup: int = 5,
) -> float:
    """Time CPU(pinned) -> GPU copy for a single flat tensor using CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if flat_cpu_pinned.device.type != "cpu":
        raise ValueError("flat_cpu_pinned must be on CPU")
    if not flat_cpu_pinned.is_pinned():
        raise ValueError("flat_cpu_pinned must be pinned memory for a fair H2D benchmark")

    flat_gpu = torch.empty_like(flat_cpu_pinned, device="cuda")

    for _ in range(warmup):
        flat_gpu.copy_(flat_cpu_pinned, non_blocking=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(iters):
        start.record()
        flat_gpu.copy_(flat_cpu_pinned, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / max(iters, 1)


def time_block_upload_to_gpu_via_flat_ms(
    block: torch.nn.Module,
    *,
    include_buffers: bool = True,
    iters: int = 20,
    warmup: int = 5,
) -> Dict[str, float]:
    """
    Benchmark "put the block on the GPU" by:
    - packing all block tensors into a single pinned CPU flat tensor
    - timing a single contiguous H2D copy into a flat GPU tensor
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Ensure we're measuring upload (CPU -> GPU), not device->device.
    block_cpu = block.to("cpu")

    spec = analyze_block_to_flat_spec(block_cpu, include_buffers=include_buffers)
    flat_cpu = allocate_block_flat_tensor(spec, device="cpu", pin_memory=True)

    t0 = time.perf_counter()
    pack_block_to_flat(block_cpu, flat_cpu, spec)
    pack_ms = (time.perf_counter() - t0) * 1000.0

    h2d_ms = time_flat_h2d_copy_ms(flat_cpu, iters=iters, warmup=warmup)

    bytes_total = spec.total_numel * torch.tensor([], dtype=spec.dtype).element_size()
    gb = bytes_total / (1024**3)
    h2d_gbps = gb / (h2d_ms / 1000.0) if h2d_ms > 0 else float("inf")

    return {
        "pack_ms": pack_ms,
        "h2d_copy_ms": h2d_ms,
        "h2d_gbps": h2d_gbps,
        "total_numel": float(spec.total_numel),
        "total_gb": gb,
    }


def make_gpu_views_from_flat(flat_gpu: torch.Tensor, spec: BlockFlatSpec) -> Dict[str, torch.Tensor]:
    """
    Create GPU tensor views that match the original block tensor shapes.

    Note: these are views into `flat_gpu` (no copies).
    """
    if flat_gpu.device.type != "cuda":
        raise ValueError("flat_gpu must be on CUDA")
    if flat_gpu.numel() != spec.total_numel:
        raise ValueError(f"flat_gpu has numel={flat_gpu.numel()} but spec expects {spec.total_numel}")
    if flat_gpu.dtype != spec.dtype:
        raise ValueError(f"flat_gpu dtype={flat_gpu.dtype} but spec dtype={spec.dtype}")

    out: Dict[str, torch.Tensor] = {}
    for e in spec.entries:
        out[f"{e.kind}:{e.name}"] = flat_gpu[e.start : e.end].view(e.shape)
    return out


def time_upload_blocks_to_gpu_with_prealloc_flat(
    blocks: Sequence[torch.nn.Module],
    *,
    include_buffers: bool = True,
    warmup: int = 1,
    validate_layouts: bool = True,
) -> Dict[str, object]:
    """
    Assumption-driven benchmark:
    - allocate space for ONE block (pinned CPU flat + GPU flat) once
    - for each block: pack into flat_cpu, H2D copy into flat_gpu (timed), then "reshape" by making views

    No `block.to("cpu")` and no `block.to("cuda")`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if len(blocks) == 0:
        return {"per_block": [], "sum_h2d_ms": 0.0, "sum_pack_ms": 0.0, "sum_view_ms": 0.0, "size_gb": 0.0}

    # Spec + prealloc for ONE block.
    spec0 = analyze_block_to_flat_spec(blocks[0], include_buffers=include_buffers)
    flat_cpu = allocate_block_flat_tensor(spec0, device="cpu", pin_memory=True)
    flat_gpu = allocate_block_flat_tensor(spec0, device="cuda")

    # Warmup to avoid first-copy overheads.
    for _ in range(max(warmup, 0)):
        pack_block_to_flat(blocks[0], flat_cpu, spec0)
        flat_gpu.copy_(flat_cpu, non_blocking=True)
    torch.cuda.synchronize()

    per_block: List[Dict[str, float]] = []
    sum_pack_ms = 0.0
    sum_h2d_ms = 0.0
    sum_view_ms = 0.0

    bytes_total = spec0.total_numel * torch.tensor([], dtype=spec0.dtype).element_size()
    size_gb = bytes_total / (1024**3)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if validate_layouts:
        for idx, block in enumerate(blocks):
            spec = analyze_block_to_flat_spec(block, include_buffers=include_buffers)
            if spec.total_numel != spec0.total_numel or spec.dtype != spec0.dtype or len(spec.entries) != len(spec0.entries):
                raise ValueError(f"Block[{idx}] spec differs from Block[0]; cannot reuse single prealloc flat buffers.")
            for a, b in zip(spec.entries, spec0.entries):
                if (a.kind, a.name, tuple(a.shape), a.numel, a.dtype) != (b.kind, b.name, tuple(b.shape), b.numel, b.dtype):
                    raise ValueError(f"Block[{idx}] layout differs from Block[0]; cannot reuse single prealloc flat buffers.")

    for idx, block in enumerate(blocks):

        t0 = time.perf_counter()
        pack_block_to_flat(block, flat_cpu, spec0)
        pack_ms = (time.perf_counter() - t0) * 1000.0

        start.record()
        flat_gpu.copy_(flat_cpu, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        h2d_ms = start.elapsed_time(end)

        t1 = time.perf_counter()
        _views = make_gpu_views_from_flat(flat_gpu, spec0)
        # Touch a value to ensure shapes are realized (still zero-copy).
        if _views:
            _ = next(iter(_views.values())).shape
        view_ms = (time.perf_counter() - t1) * 1000.0

        bw = (size_gb / (h2d_ms / 1000.0)) if h2d_ms > 0 else float("inf")
        per_block.append(
            {
                "idx": float(idx),
                "pack_ms": pack_ms,
                "h2d_ms": h2d_ms,
                "view_ms": view_ms,
                "size_gb": size_gb,
                "h2d_gbps": bw,
            }
        )
        sum_pack_ms += pack_ms
        sum_h2d_ms += h2d_ms
        sum_view_ms += view_ms

    overall_bw = (size_gb * len(per_block)) / (sum_h2d_ms / 1000.0) if sum_h2d_ms > 0 else float("inf")
    return {
        "per_block": per_block,
        "size_gb": size_gb,
        "sum_pack_ms": sum_pack_ms,
        "sum_h2d_ms": sum_h2d_ms,
        "sum_view_ms": sum_view_ms,
        "overall_h2d_gbps": overall_bw,
    }


def time_blocks_to_cuda_and_back_ms(
    blocks: Sequence[torch.nn.Module],
    *,
    include_buffers: bool = True,
    warmup: int = 1,
) -> Dict[str, object]:
    """
    Baseline comparison: for each block, time:
    - CPU -> CUDA via `block.to("cuda")`
    - CUDA -> CPU via `block.to("cpu")`

    We move each block back to CPU immediately to avoid accumulating GPU memory.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if len(blocks) == 0:
        return {"per_block": [], "sum_to_cuda_ms": 0.0, "sum_to_cpu_ms": 0.0, "sum_gb": 0.0}

    # Warmup (first .to can pay extra setup).
    for _ in range(max(warmup, 0)):
        blocks[0].to("cuda")
        torch.cuda.synchronize()
        blocks[0].to("cpu")
        torch.cuda.synchronize()

    per_block: List[Dict[str, float]] = []
    sum_to_cuda_ms = 0.0
    sum_to_cpu_ms = 0.0
    sum_gb = 0.0

    for idx, block in enumerate(blocks):
        gb = _block_size_gb(block, include_buffers=include_buffers)
        sum_gb += gb

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        block.to("cuda")
        torch.cuda.synchronize()
        to_cuda_ms = (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        block.to("cpu")
        torch.cuda.synchronize()
        to_cpu_ms = (time.perf_counter() - t1) * 1000.0

        per_block.append(
            {
                "idx": float(idx),
                "gb": gb,
                "to_cuda_ms": to_cuda_ms,
                "to_cpu_ms": to_cpu_ms,
                "to_cuda_gbps": (gb / (to_cuda_ms / 1000.0)) if to_cuda_ms > 0 else float("inf"),
                "to_cpu_gbps": (gb / (to_cpu_ms / 1000.0)) if to_cpu_ms > 0 else float("inf"),
            }
        )
        sum_to_cuda_ms += to_cuda_ms
        sum_to_cpu_ms += to_cpu_ms

    return {
        "per_block": per_block,
        "sum_to_cuda_ms": sum_to_cuda_ms,
        "sum_to_cpu_ms": sum_to_cpu_ms,
        "sum_gb": sum_gb,
        "overall_to_cuda_gbps": (sum_gb / (sum_to_cuda_ms / 1000.0)) if sum_to_cuda_ms > 0 else float("inf"),
        "overall_to_cpu_gbps": (sum_gb / (sum_to_cpu_ms / 1000.0)) if sum_to_cpu_ms > 0 else float("inf"),
    }


def _block_size_gb(block: torch.nn.Module, *, include_buffers: bool = True) -> float:
    spec = analyze_block_to_flat_spec(block, include_buffers=include_buffers, require_single_dtype=False)
    # Use each entry dtype for bytes to support mixed dtypes (FP8 etc).
    total_bytes = 0
    for e in spec.entries:
        total_bytes += e.numel * torch.tensor([], dtype=e.dtype).element_size()
    return total_bytes / (1024**3)


def time_move_blocks_to_gpu_ms(
    blocks: Sequence[torch.nn.Module],
    *,
    include_buffers: bool = True,
) -> Dict[str, object]:
    """
    Time true "onload" behavior: loop blocks and call `block.to("cuda")` (in-place).

    Returns per-block timings and aggregate totals.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    per_block: List[Dict[str, float]] = []
    total_ms_sum = 0.0
    total_gb_sum = 0.0

    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()

    for idx, block in enumerate(blocks):
        # Ensure the block starts on CPU for upload timing.
        block.to("cpu")
        gb = _block_size_gb(block, include_buffers=include_buffers)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        block.to("cuda")
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0

        bw = (gb / (ms / 1000.0)) if ms > 0 else float("inf")
        per_block.append(
            {
                "idx": float(idx),
                "ms": ms,
                "gb": gb,
                "gbps": bw,
            }
        )
        total_ms_sum += ms
        total_gb_sum += gb

    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    overall_gbps = (total_gb_sum / (total_ms_sum / 1000.0)) if total_ms_sum > 0 else float("inf")

    return {
        "per_block": per_block,
        "sum_ms": total_ms_sum,
        "sum_gb": total_gb_sum,
        "sum_gbps": overall_gbps,
        "wall_ms": wall_ms,
    }


def _main() -> None:
    engine = UniversalEngine(
        yaml_path="manifest/video/ovi-10b-5s-1.0.0.v1.yml",
        selected_components={
            "transformer": {"variant": "FP8"},
            "text_encoder": {"variant": "FP8"},
        },
        attention_type="sdpa",
    ).engine

    engine.load_component_by_type("transformer")
    transformer = engine.transformer
    
    block = transformer.video_model.blocks[0]

    # --- Time how long it takes to put the block on the GPU (weights upload) ---
    if torch.cuda.is_available():
        stats = time_block_upload_to_gpu_via_flat_ms(block, include_buffers=True, iters=20, warmup=5)
        print(
            "block upload via flat: "
            f"pack={stats['pack_ms']:.2f}ms, "
            f"h2d_copy={stats['h2d_copy_ms']:.2f}ms, "
            f"size={stats['total_gb']:.3f}GB, "
            f"bw={stats['h2d_gbps']:.2f}GB/s"
        )

        # --- Assumption: prealloc one-block space, only copy flat -> GPU, then reshape (views) ---
        blocks = list(transformer.video_model.blocks)
        print(f"timing blocks flat->gpu w/ prealloc (no block.to): num_blocks={len(blocks)}")

        # Baseline: block.to("cuda") then block.to("cpu") (per-block, no accumulation on GPU)
        baseline = time_blocks_to_cuda_and_back_ms(blocks, include_buffers=True, warmup=1)

        # Your path: prealloc one-block flat buffers, then pack -> H2D copy -> view/reshape
        results = time_upload_blocks_to_gpu_with_prealloc_flat(
            blocks, include_buffers=True, warmup=1, validate_layouts=True
        )

        print("--- per-block comparison (upload) ---")
        for b_row, f_row in zip(baseline["per_block"], results["per_block"]):
            flat_total_ms = f_row["pack_ms"] + f_row["h2d_ms"] + f_row["view_ms"]
            speedup = (b_row["to_cuda_ms"] / flat_total_ms) if flat_total_ms > 0 else float("inf")
            print(
                f"block[{int(b_row['idx'])}] "
                f"to_cuda={b_row['to_cuda_ms']:.2f}ms "
                f"flat(pack+h2d+view)={flat_total_ms:.2f}ms "
                f"speedup={speedup:.2f}x"
            )

        print("--- details: flat path (per-block) ---")
        for row in results["per_block"]:
            print(
                f"block[{int(row['idx'])}] "
                f"pack={row['pack_ms']:.2f}ms "
                f"h2d={row['h2d_ms']:.2f}ms "
                f"view={row['view_ms']:.2f}ms "
                f"size={row['size_gb']:.4f}GB "
                f"h2d_bw={row['h2d_gbps']:.2f}GB/s"
            )

        print("--- details: block.to path (per-block) ---")
        for row in baseline["per_block"]:
            print(
                f"block[{int(row['idx'])}] "
                f"to_cuda={row['to_cuda_ms']:.2f}ms "
                f"to_cpu={row['to_cpu_ms']:.2f}ms "
                f"size={row['gb']:.4f}GB "
                f"to_cuda_bw={row['to_cuda_gbps']:.2f}GB/s "
                f"to_cpu_bw={row['to_cpu_gbps']:.2f}GB/s"
            )

        print(
            "blocks aggregate (prealloc flat): "
            f"sum_pack_ms={results['sum_pack_ms']:.2f} "
            f"sum_h2d_ms={results['sum_h2d_ms']:.2f} "
            f"sum_view_ms={results['sum_view_ms']:.2f} "
            f"size_per_block={results['size_gb']:.4f}GB "
            f"overall_h2d_bw={results['overall_h2d_gbps']:.2f}GB/s"
        )
        print(
            "blocks aggregate (block.to): "
            f"sum_to_cuda_ms={baseline['sum_to_cuda_ms']:.2f} "
            f"sum_to_cpu_ms={baseline['sum_to_cpu_ms']:.2f} "
            f"sum_size={baseline['sum_gb']:.4f}GB "
            f"overall_to_cuda_bw={baseline['overall_to_cuda_gbps']:.2f}GB/s "
            f"overall_to_cpu_bw={baseline['overall_to_cpu_gbps']:.2f}GB/s"
        )

        flat_total_ms = results["sum_pack_ms"] + results["sum_h2d_ms"] + results["sum_view_ms"]
        speedup_total = (baseline["sum_to_cuda_ms"] / flat_total_ms) if flat_total_ms > 0 else float("inf")
        print(f"aggregate upload speedup (block.to_cuda / flat_total) = {speedup_total:.2f}x")
    else:
        print("CUDA not available; skipping block upload timing.")



if __name__ == "__main__":
    _main()