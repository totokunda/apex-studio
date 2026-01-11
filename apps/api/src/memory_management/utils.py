from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, cast
import torch

@dataclass(frozen=True)
class BlockTensorEntry:
    kind: Literal["param", "buffer"]
    name: str
    shape: torch.Size
    numel: int
    logical_dtype: torch.dtype
    physical_dtype: torch.dtype
    byte_start: int
    byte_end: int  # exclusive


@dataclass(frozen=True)
class BlockFlatSpec:
    entries: Tuple[BlockTensorEntry, ...]
    total_bytes: int
    flat_dtype: torch.dtype = torch.uint8


def _as_physical_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Return a view of `t` as a plain base `torch.Tensor` so dtype overrides on
    tensor subclasses (e.g. GGMLTensor, FPScaledTensor/Parameter) don't affect
    packing/unpacking.
    """
    # For real tensor subclasses, `.as_subclass(torch.Tensor)` preserves storage but
    # removes overridden properties like `.dtype`.
    try:
        return t.as_subclass(torch.Tensor)
    except Exception:
        # Fallback: many torch.Tensor-like objects still are torch.Tensor instances.
        return cast(torch.Tensor, t)


def _physical_dtype(t: torch.Tensor) -> torch.dtype:
    """
    Best-effort "true storage dtype" for tensor subclasses that expose logical dtypes.
    """
    dt = getattr(t, "physical_dtype", None)
    if isinstance(dt, torch.dtype):
        return dt
    dt = getattr(t, "base_dtype", None)
    if isinstance(dt, torch.dtype):
        return dt
    # `.as_subclass(torch.Tensor).dtype` is the actual storage dtype for real subclasses.
    return _as_physical_tensor(t).dtype


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
    require_single_dtype: bool = False,
    skip_names: Optional[Sequence[str]] = None,
    alignment: int = 128,
) -> BlockFlatSpec:
    """
    Analyze a module "block" and produce a flat packing spec over its tensors.

    The spec provides a deterministic mapping between:
    - the block's parameters (+ optional buffers), and
    - a single 1D *byte* flat tensor of length `total_bytes` (dtype uint8),
      containing the *raw storage bytes* of each tensor in order.

    Why bytes?
    - Blocks may contain mixed dtypes (FP16/BF16/FP8/etc).
    - Some tensor subclasses intentionally override `.dtype` to expose a logical
      compute dtype (e.g. GGMLTensor, FPScaledParameter). Packing raw bytes avoids
      any casting and preserves quantized/FP8 storage exactly.
    """
    skip = set(skip_names or [])
    named = [
        (kind, name, t)
        for (kind, name, t) in _collect_named_tensors(block, include_buffers=include_buffers)
        if name not in skip and t.numel() > 0
    ]
    if not named:
        return BlockFlatSpec(entries=tuple(), total_bytes=0)

    physical_dtypes = {_physical_dtype(t) for _, _, t in named}
    if require_single_dtype and len(physical_dtypes) != 1:
        details = ", ".join(
            sorted(
                {
                    f"{kind}:{name}=logical:{t.dtype}/physical:{_physical_dtype(t)}"
                    for kind, name, t in named
                }
            )
        )
        raise ValueError(
            "Block tensors have multiple *physical storage* dtypes; packing was configured "
            "to require a single dtype. "
            f"Found: {sorted(map(str, physical_dtypes))}. Details: {details}"
        )

    entries: List[BlockTensorEntry] = []
    byte_offset = 0
    for kind, name, t in named:
        base = _as_physical_tensor(t)
        
        # Align byte_offset to `alignment` bytes (default 128 for cache/DMA efficiency)
        if alignment > 1:
            padding = (alignment - (byte_offset % alignment)) % alignment
            byte_offset += padding

        n = int(base.numel())
        nbytes = n * base.element_size()
        if nbytes == 0:
            continue
        entries.append(
            BlockTensorEntry(
                kind=kind,
                name=name,
                shape=base.shape,
                numel=n,
                logical_dtype=t.dtype,
                physical_dtype=base.dtype,
                byte_start=byte_offset,
                byte_end=byte_offset + nbytes,
            )
        )
        byte_offset += nbytes
    return BlockFlatSpec(entries=tuple(entries), total_bytes=byte_offset)


def allocate_block_flat_tensor(
    spec: BlockFlatSpec,
    *,
    device: torch.device | str,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Allocate an *empty* 1D byte flat tensor sized to hold the entire block (per `spec`)."""
    if spec.total_bytes == 0:
        return torch.empty((0,), device=device, dtype=spec.flat_dtype)
    device = torch.device(device)
    if device.type == "cpu":
        return torch.empty(
            (spec.total_bytes,), device=device, dtype=spec.flat_dtype, pin_memory=pin_memory
        )
    return torch.empty((spec.total_bytes,), device=device, dtype=spec.flat_dtype)


def pack_block_to_flat(
    block: torch.nn.Module,
    flat: torch.Tensor,
    spec: BlockFlatSpec,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy raw storage bytes from `block` into `flat` according to `spec`."""
    if flat.numel() != spec.total_bytes:
        raise ValueError(f"Flat tensor has numel={flat.numel()} but spec expects {spec.total_bytes}")
    if flat.dtype != spec.flat_dtype:
        raise ValueError(f"Flat tensor dtype={flat.dtype} but spec dtype={spec.flat_dtype}")

    params: Dict[str, torch.Tensor] = dict(block.named_parameters(recurse=True))
    buffers: Dict[str, torch.Tensor] = dict(block.named_buffers(recurse=True))

    with torch.no_grad():
        for e in spec.entries:
            t = params[e.name] if e.kind == "param" else buffers[e.name]
            base = _as_physical_tensor(t)
            src_bytes = base.detach().reshape(-1).view(torch.uint8)
            dst_bytes = flat[e.byte_start : e.byte_end]
            if src_bytes.numel() != dst_bytes.numel():
                raise ValueError(
                    f"Entry {e.kind}:{e.name} byte size mismatch: "
                    f"tensor has {src_bytes.numel()} bytes but spec expects {dst_bytes.numel()} bytes"
                )
            dst_bytes.copy_(src_bytes, non_blocking=non_blocking)
    return flat


def unpack_flat_to_block(
    flat: torch.Tensor,
    block: torch.nn.Module,
    spec: BlockFlatSpec,
    *,
    non_blocking: bool = False,
) -> None:
    """
    Copy raw storage bytes from `flat` back into `block` tensors according to `spec`.

    Optimization: If alignment permits, this will alias `block`'s tensors to `flat`'s storage
    (zero-copy) using `set_()`. This is instant and avoids D2D copies.
    """
    if flat.numel() != spec.total_bytes:
        raise ValueError(f"Flat tensor has numel={flat.numel()} but spec expects {spec.total_bytes}")
    if flat.dtype != spec.flat_dtype:
        raise ValueError(f"Flat tensor dtype={flat.dtype} but spec dtype={spec.flat_dtype}")

    params: Dict[str, torch.Tensor] = dict(block.named_parameters(recurse=True))
    buffers: Dict[str, torch.Tensor] = dict(block.named_buffers(recurse=True))

    with torch.no_grad():
        for e in spec.entries:
            t = params[e.name] if e.kind == "param" else buffers[e.name]
            base = _as_physical_tensor(t)
            
            # Check if we can use zero-copy aliasing (set_)
            # Conditions: Byte offset is divisible by element size (alignment)
            elem_size = base.element_size()
            
            if (e.byte_start % elem_size == 0) and (base.numel() * elem_size == (e.byte_end - e.byte_start)):
                # Aliasing path: Set `t` storage to point into `flat`.
                offset_elem = e.byte_start // elem_size
                # Use untyped_storage() to access the raw bytes of flat.
                # set_ creates a contiguous view of the storage segment.
                t.set_(flat.untyped_storage(), offset_elem, base.size())
            else:
                # Copy path (fallback for unaligned or special layouts)
                dst_bytes = base.reshape(-1).view(torch.uint8)
                src_bytes = flat[e.byte_start : e.byte_end]
                if dst_bytes.numel() != src_bytes.numel():
                    raise ValueError(
                        f"Entry {e.kind}:{e.name} byte size mismatch: "
                        f"tensor expects {dst_bytes.numel()} bytes but spec provides {src_bytes.numel()} bytes"
                    )
                dst_bytes.copy_(src_bytes, non_blocking=non_blocking)


def describe_block_flat_spec(spec: BlockFlatSpec, *, max_rows: int = 25) -> str:
    rows = [
        f"total_bytes={spec.total_bytes:,} flat_dtype={spec.flat_dtype} entries={len(spec.entries)}"
    ]
    for i, e in enumerate(spec.entries[:max_rows]):
        rows.append(
            f"{i:03d} {e.kind:6s} {e.name:60s} "
            f"shape={tuple(e.shape)} "
            f"numel={e.numel:,} "
            f"logical_dtype={e.logical_dtype} "
            f"physical_dtype={e.physical_dtype} "
            f"bytes={(e.byte_end - e.byte_start):,}"
        )
    if len(spec.entries) > max_rows:
        rows.append(f"... ({len(spec.entries) - max_rows} more)")
    return "\n".join(rows)

