import torch
from typing import Any, Optional, Iterable


class GGMLTensor(torch.Tensor):
    """
    A safe torch.Tensor subclass that carries GGUF metadata:

      - tensor_type: gguf.GGMLQuantizationType (or None for F16/F32)
      - tensor_shape: original logical shape for the tensor (torch.Size)
      - dequant_dtype: preferred dtype when dequantizing (torch.dtype)
      - patches: optional patch metadata list

    IMPORTANT: Construct with an existing tensor `data` so storage exists.
    """

    # ---- creation ---------------------------------------------------------
    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        *,
        tensor_type: Any,
        tensor_shape: Optional[Iterable[int]] = None,
        dequant_dtype: Optional[torch.dtype] = None,
        patches: Optional[Iterable[Any]] = None,
        requires_grad: bool = False,
    ):
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)  # ensure a tensor
        # make a real subclass that shares the same storage
        obj = torch.Tensor._make_subclass(cls, data, require_grad=requires_grad)
        obj.tensor_type = tensor_type
        obj.tensor_shape = (
            torch.Size(tuple(tensor_shape))
            if tensor_shape is not None
            else torch.Size(data.shape)
        )
        obj.dequant_dtype = dequant_dtype
        obj.patches = list(patches) if patches is not None else []
        return obj

    # ---- helpers to re-wrap outputs and preserve metadata -----------------
    def _wrap_like(self, base: torch.Tensor) -> "GGMLTensor":
        out = torch.Tensor._make_subclass(
            GGMLTensor, base, require_grad=base.requires_grad
        )
        out.tensor_type = getattr(self, "tensor_type", None)
        out.tensor_shape = torch.Size(getattr(self, "tensor_shape", base.shape))
        out.dequant_dtype = getattr(self, "dequant_dtype", None)
        out.patches = list(getattr(self, "patches", []))

        return out

    @property
    def base_dtype(self) -> torch.dtype:
        return super().dtype

    # ---- tensor API overrides (preserve metadata) -------------------------
    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        """
        Expose the *logical* dtype instead of the storage dtype.

        Quantized GGML tensors may store data as uint8/int8/etc., but all high-level
        PyTorch code should see their dequantization dtype (e.g. float16/float32).
        This makes checks like `hidden_states.dtype` in HF modules behave as if the
        tensor were already dequantized, avoiding spurious casts based on uint8.
        """
        return getattr(self, "dequant_dtype", None) or super().dtype

    def to(self, *args, **kwargs):
        base = super().to(*args, **kwargs)
        # Keep it as GGMLTensor to preserve metadata
        return self._wrap_like(base)

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device=None, non_blocking=False, **kwargs):
        return self.to(
            device if device is not None else "cuda", non_blocking=non_blocking
        )

    def clone(self, *args, **kwargs):
        base = super().clone(*args, **kwargs)
        return self._wrap_like(base)

    def detach(self):
        base = super().detach()
        return self._wrap_like(base)

    def requires_grad_(self, requires_grad: bool = True):
        base = super().requires_grad_(requires_grad)
        # _make_subclass already returned self, no need to re-wrap
        return self

    def new_empty(self, size, *args, **kwargs):
        base = super().new_empty(size, *args, **kwargs)
        # preserve metadata on empties (Intel Arc / odd backends friendliness)
        out = self._wrap_like(base)
        out.tensor_shape = torch.Size(size)
        return out
