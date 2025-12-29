import torch
from typing import Iterable, List, Tuple, Sequence

from .ggml_tensor import GGMLTensor
from .scaled_layer import FPScaledTensor # type: ignore


def ggml_cat(tensors: Iterable[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    A cat helper that preserves GGML metadata when inputs are GGMLTensor.

    - If all inputs are GGMLTensor, the output is a GGMLTensor and metadata is
      copied from the first tensor, with `tensor_shape` updated to the new shape.
    - Otherwise, this behaves exactly like `torch.cat` and returns a plain Tensor.
    """
    tensors = list(tensors)
    out = torch.cat(tensors, dim=dim)

    # ---- GGML path --------------------------------------------------------
    ggml_inputs: List[GGMLTensor] = [t for t in tensors if isinstance(t, GGMLTensor)]
    if ggml_inputs:
        # Use metadata from the first GGML tensor
        template = ggml_inputs[0]
        wrapped = torch.Tensor._make_subclass(
            GGMLTensor, out, require_grad=out.requires_grad
        )
        wrapped.tensor_type = getattr(template, "tensor_type", None)
        wrapped.dequant_dtype = getattr(template, "dequant_dtype", None)
        wrapped.patches = list(getattr(template, "patches", []))

        # Compute logical tensor_shape based on existing tensor_shapes along `dim`
        base_shape = list(getattr(template, "tensor_shape", template.shape))
        ndims = len(base_shape)
        dim_normalized = dim if dim >= 0 else dim + ndims
        if not 0 <= dim_normalized < ndims:
            # Fallback: mirror runtime shape if something is inconsistent
            wrapped.tensor_shape = torch.Size(out.shape)
            return wrapped

        new_dim = 0
        for t in tensors:
            if isinstance(t, GGMLTensor) and hasattr(t, "tensor_shape"):
                shape = list(t.tensor_shape)
            else:
                shape = list(t.shape)
            new_dim += int(shape[dim_normalized])

        base_shape[dim_normalized] = new_dim
        wrapped.tensor_shape = torch.Size(base_shape)
        return wrapped

    # ---- FP8ScaledTensor path --------------------------------------------
    fp8_inputs: List[FPScaledTensor] = [
        t for t in tensors if isinstance(t, FPScaledTensor)
    ]
    if fp8_inputs:
        template = fp8_inputs[0]
        wrapped_fp8 = torch.Tensor._make_subclass(
            FPScaledTensor, out, require_grad=out.requires_grad
        )
        wrapped_fp8.logical_dtype = getattr(template, "logical_dtype", None)
        return wrapped_fp8

    # Fallback: no special tensor subclasses, return plain tensor
    return out


def ggml_chunk(
    input: torch.Tensor, chunks: int, dim: int = 0
) -> Tuple[torch.Tensor, ...]:
    """
    A chunk helper that preserves GGML metadata when the input is GGMLTensor.

    - If `input` is GGMLTensor, each chunk is a GGMLTensor with metadata copied
      from `input`, and `tensor_shape` updated to the chunk's shape.
    - Otherwise, this behaves exactly like `torch.chunk` and returns plain tensors.
    """
    parts = torch.chunk(input, chunks, dim=dim)

    # ---- GGML path --------------------------------------------------------
    if isinstance(input, GGMLTensor):
        out_chunks: List[GGMLTensor] = []

        # Base logical shape from the input tensor
        base_shape = list(getattr(input, "tensor_shape", input.shape))
        ndims = len(base_shape)
        dim_normalized = dim if dim >= 0 else dim + ndims

        for p in parts:
            wrapped = torch.Tensor._make_subclass(
                GGMLTensor, p, require_grad=p.requires_grad
            )
            wrapped.tensor_type = getattr(input, "tensor_type", None)
            wrapped.dequant_dtype = getattr(input, "dequant_dtype", None)
            wrapped.patches = list(getattr(input, "patches", []))

            # Update logical tensor_shape for this chunk using its size along `dim`
            if 0 <= dim_normalized < ndims:
                logical_shape = list(base_shape)
                logical_shape[dim_normalized] = int(p.shape[dim_normalized])
                wrapped.tensor_shape = torch.Size(logical_shape)
            else:
                # Fallback to runtime shape if dim is somehow inconsistent
                wrapped.tensor_shape = torch.Size(p.shape)

            out_chunks.append(wrapped)

        return tuple(out_chunks)

    # ---- FP8ScaledTensor path --------------------------------------------
    if isinstance(input, FPScaledTensor):
        out_chunks_fp8: List[FPScaledTensor] = []
        for p in parts:
            wrapped_fp8 = torch.Tensor._make_subclass(
                FPScaledTensor, p, require_grad=p.requires_grad
            )
            wrapped_fp8.logical_dtype = getattr(input, "logical_dtype", None)
            out_chunks_fp8.append(wrapped_fp8)
        return tuple(out_chunks_fp8)

    # Fallback: plain tensors
    return parts


def ggml_split(
    input: torch.Tensor, split_size_or_sections, dim: int = 0
) -> Tuple[torch.Tensor, ...]:
    """
    A split helper that preserves GGML metadata when the input is GGMLTensor.

    - If `input` is GGMLTensor, each split piece is a GGMLTensor with metadata
      copied from `input`, and `tensor_shape` updated using the logical shape.
    - Otherwise, this behaves exactly like `torch.split` and returns plain tensors.
    """
    parts = torch.split(input, split_size_or_sections, dim=dim)

    # ---- GGML path --------------------------------------------------------
    if isinstance(input, GGMLTensor):
        out_splits: List[GGMLTensor] = []

        # Base logical shape from the input tensor
        base_shape = list(getattr(input, "tensor_shape", input.shape))
        ndims = len(base_shape)
        dim_normalized = dim if dim >= 0 else dim + ndims

        for p in parts:
            wrapped = torch.Tensor._make_subclass(
                GGMLTensor, p, require_grad=p.requires_grad
            )
            wrapped.tensor_type = getattr(input, "tensor_type", None)
            wrapped.dequant_dtype = getattr(input, "dequant_dtype", None)
            wrapped.patches = list(getattr(input, "patches", []))

            # Update logical tensor_shape for this split using its size along `dim`
            if 0 <= dim_normalized < ndims:
                logical_shape = list(base_shape)
                logical_shape[dim_normalized] = int(p.shape[dim_normalized])
                wrapped.tensor_shape = torch.Size(logical_shape)
            else:
                # Fallback to runtime shape if dim is somehow inconsistent
                wrapped.tensor_shape = torch.Size(p.shape)

            out_splits.append(wrapped)

        return tuple(out_splits)

    # ---- FP8ScaledTensor path --------------------------------------------
    if isinstance(input, FPScaledTensor):
        out_splits_fp8: List[FPScaledTensor] = []
        for p in parts:
            wrapped_fp8 = torch.Tensor._make_subclass(
                FPScaledTensor, p, require_grad=p.requires_grad
            )
            wrapped_fp8.logical_dtype = getattr(input, "logical_dtype", None)
            out_splits_fp8.append(wrapped_fp8)
        return tuple(out_splits_fp8)

    # Fallback: plain tensors
    return parts
