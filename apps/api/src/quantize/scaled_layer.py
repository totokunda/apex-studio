import math
from typing import Optional, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm

# Optional Transformer Engine (TE) support
try:  # pragma: no cover - optional dependency
    import transformer_engine.pytorch as te  # type: ignore

    _HAS_TE = True
except Exception:  # pragma: no cover - optional dependency
    te = None  # type: ignore
    _HAS_TE = False

# -------------------- FP8 quantization helpers --------------------


def get_fp_maxval(
    bits: int = 8, mantissa_bit: int = 3, sign_bits: int = 1
) -> torch.Tensor:
    """
    Compute the maximum representable value for a custom FP format.
    Defaults to FP8 E4M3 (bits=8, mantissa_bit=3, sign_bits=1).
    """
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1.0
    for i in range(mantissa_bit - 1):
        mantissa += 1.0 / (2 ** (i + 1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    return maxval


def quantize_to_fp8(
    x: torch.Tensor,
    bits: int = 8,
    mantissa_bit: int = 3,
    sign_bits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize-dequantize `x` to an FP8-like grid (default E4M3) and return:
      - qdq_out: quantize-dequantized tensor (same dtype as input)
      - log_scales: per-value log2 scaling factors
    """
    bits_t = torch.tensor(bits)
    mantissa_bit_t = torch.tensor(mantissa_bit)
    sign_bits_t = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(mantissa_bit_t), 1, bits_t - sign_bits_t)
    E = bits_t - sign_bits_t - M
    bias = 2 ** (E - 1) - 1

    mantissa = 1.0
    for i in range(mantissa_bit - 1):
        mantissa += 1.0 / (2 ** (i + 1))

    maxval = mantissa * 2 ** (2**E - 1 - bias)
    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)

    input_clamp = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp(
        (torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0
    )
    log_scales = 2.0 ** (log_scales - M - bias.type(x.dtype))

    # Dequantize back to the original dtype/grid
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales


def quantize_to_fp4(
    x: torch.Tensor,
    *,
    per_channel_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Block-FP4 / int4-like quantization with a shared pow2 scale:
      - `per_channel_dim` controls along which dimension the scale is computed.
        For Linear/Conv weights shaped [out, in, ...], use the default 0.
      - Codes are 4-bit signed in [-7, 7].

    Returns:
      - q: int8 tensor (only 4 bits used)
      - scales: pow2 scale tensor, broadcastable to `x`
    """
    device = x.device
    dtype = x.dtype

    # Choose per-channel or per-tensor amax
    if per_channel_dim is None:
        amax = x.abs().max()
    else:
        amax = x.abs().amax(dim=per_channel_dim, keepdim=True)

    # Avoid degenerate all-zero / tiny scales
    eps = torch.finfo(dtype).tiny
    amax = torch.clamp(amax, min=eps)

    max_code = 7.0  # 4-bit signed range [-7, 7]

    # Block-floating pow2 scale: 2^round(log2(amax / max_code))
    raw_scale = amax / max_code
    log2_scale = torch.round(torch.log2(raw_scale))
    scales = torch.pow(2.0, log2_scale).to(dtype).to(device)

    # Quantize and clamp to 4-bit range
    q = torch.round(x / scales).to(torch.int8)
    q = torch.clamp(q, min=-7, max=7)

    return q, scales


def dequantize_from_fp4(
    q: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    q: int8 tensor containing FP4 codes
    scales: same shape as q (or broadcastable)
    """
    return q.to(out_dtype) * scales.to(out_dtype)


def fp8_tensor_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int = 8,
    mantissa_bit: int = 3,
    sign_bits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize `x` by `scale`, then quantize-dequantize to FP8 grid.
    Returns:
      - quant_dequant_x: qdq(x / scale)
      - scale: broadcasted scale used
      - log_scales: per-value log2 scaling factors
    """
    for _ in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)

    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(
        new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits
    )
    return quant_dequant_x, scale, log_scales


def fp8_activation_dequant(
    qdq_out: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequantize FP8 activations/weights back to `dtype` using `scale`.

    `qdq_out` is assumed to already live on an FP8-like grid (e.g. float8_e4m3fn),
    and `scale` is the (possibly per-channel) scaling tensor.
    """
    qdq_out = qdq_out.to(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x


class FPScaledTensor(torch.Tensor):
    """
    Tensor subclass used for FP8-scaled weights.

    It exposes a *logical* compute dtype via ``.dtype`` while still keeping the
    underlying storage dtype (FP8) available via ``.physical_dtype`` for
    low-level dequantization logic.
    """

    logical_dtype: Optional[torch.dtype] = None

    @staticmethod
    def _from_base(
        base: torch.Tensor, *, logical_dtype: Optional[torch.dtype]
    ) -> "FPScaledTensor":
        out = torch.Tensor._make_subclass(
            FPScaledTensor, base, require_grad=base.requires_grad
        )
        out.logical_dtype = logical_dtype
        return out

    def _wrap_like(self, base: torch.Tensor) -> "FPScaledTensor":
        out = torch.Tensor._make_subclass(
            FPScaledTensor, base, require_grad=base.requires_grad
        )
        out.logical_dtype = getattr(self, "logical_dtype", None)
        return out

    @property
    def physical_dtype(self) -> torch.dtype:
        """
        True storage dtype, ignoring any logical/compute dtype override.
        """
        return super().dtype

    @property  # type: ignore[override]
    def dtype(self) -> torch.dtype:
        """
        Logical/compute dtype exposed to high-level code.

        This is what external libraries (e.g. attention blocks that inspect
        ``weight.dtype``) will see, allowing us to "pretend" the weights live
        in ``compute_dtype`` while they are actually stored as FP8.
        """
        return self.logical_dtype or super().dtype

    def to(self, *args, **kwargs):
        base = super().to(*args, **kwargs)
        return self._wrap_like(base)

    def clone(self, *args, **kwargs):
        base = super().clone(*args, **kwargs)
        return self._wrap_like(base)


class FPScaledParameter(nn.Parameter):
    """
    Parameter subclass for FP8-scaled weights.

    Mirrors `FP8ScaledTensor` semantics but ensures the tensor is still
    registered as an `nn.Parameter` on modules.
    """

    logical_dtype: Optional[torch.dtype] = None

    @staticmethod
    def _from_base(
        base: torch.Tensor,
        *,
        logical_dtype: Optional[torch.dtype],
        requires_grad: bool,
    ) -> "FPScaledParameter":
        out = torch.Tensor._make_subclass(
            FPScaledParameter, base, require_grad=requires_grad
        )
        out.logical_dtype = logical_dtype
        return out

    def _wrap_like(self, base: torch.Tensor) -> "FPScaledParameter":
        out = torch.Tensor._make_subclass(
            FPScaledParameter, base, require_grad=base.requires_grad
        )
        out.logical_dtype = getattr(self, "logical_dtype", None)
        return out

    @property
    def physical_dtype(self) -> torch.dtype:
        """
        True storage dtype, ignoring any logical/compute dtype override.
        """
        return super().dtype

    @property  # type: ignore[override]
    def dtype(self) -> torch.dtype:
        """
        Logical/compute dtype exposed to high-level code.
        """
        return self.logical_dtype or super().dtype

    def to(self, *args, **kwargs):
        base = super().to(*args, **kwargs)
        return self._wrap_like(base)

    def clone(self, *args, **kwargs):
        base = super().clone(*args, **kwargs)
        return self._wrap_like(base)

    def dequant(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Return a dequantized / compute-dtype view of this parameter for arithmetic.

        For FP8/FP4 physical storage this returns a regular (base) Tensor cast to a
        higher-precision dtype suitable for compute (typically the logical dtype).

        Note:
        - This does *not* apply any external scaling (e.g. `scale_weight` on
          FPScaledLayer weights). It only casts from the physical storage dtype.
        """
        base = self.as_subclass(torch.Tensor)
        target_dtype = dtype or getattr(self, "logical_dtype", None) or base.dtype
        return base.to(target_dtype)

    # Make common arithmetic ops "safe" by dequantizing to the logical dtype.
    # This is important for parameters like `scale_shift_table` that participate
    # in addition/subtraction/multiplication with FP32 tensors.
    def __add__(self, other):  # type: ignore[override]
        return self.dequant(dtype=other.dtype) + other

    def __radd__(self, other):  # type: ignore[override]
        return other + self.dequant(dtype=other.dtype)

    def __sub__(self, other):  # type: ignore[override]
        return self.dequant(dtype=other.dtype) - other

    def __rsub__(self, other):  # type: ignore[override]
        return other - self.dequant(dtype=other.dtype)

    def __mul__(self, other):  # type: ignore[override]
        return self.dequant(dtype=other.dtype) * other

    def __rmul__(self, other):  # type: ignore[override]
        return other * self.dequant(dtype=other.dtype)

    def __matmul__(self, other):  # type: ignore[override]
        return self.dequant(dtype=other.dtype) @ other

    def __rmatmul__(self, other):  # type: ignore[override]
        return other @ self.dequant(dtype=other.dtype)


def restore_fpscaled_parameters(
    model: nn.Module,
    *,
    default_compute_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Re-wrap FP-scaled weights as FPScaledParameter after state_dict loading.

    Some `load_state_dict` pathways may replace custom Parameter subclasses with
    plain `nn.Parameter` instances. This helper re-applies FPScaledParameter to
    all FPScaled* layers, preserving logical vs. physical dtype semantics.
    """

    for name, module in model.named_modules():
        if isinstance(module, FPScaledLayer):
            weight = getattr(module, "weight", None)
            # Skip if there's no weight or it's already an FPScaledParameter
            if not isinstance(weight, nn.Parameter) or isinstance(
                weight, FPScaledParameter
            ):
                continue

            # Prefer the module's compute_dtype, then an explicit default, then
            # fall back to the current weight dtype.
            logical_dtype = (
                getattr(module, "compute_dtype", None)
                or default_compute_dtype
                or weight.dtype
            )

            module.weight = _wrap_fpscaled_weight_parameter(
                weight, logical_dtype=logical_dtype
            )

    for name, param in model.named_parameters():
        if (
            param.dtype
            in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2]
            and ".weight" not in name
            and ".bias" not in name
            and ".scale_weight" not in name
        ):
            wrapped_param = FPScaledParameter._from_base(
                param, logical_dtype=param.dtype, requires_grad=param.requires_grad
            )
            # Navigate to the module and set the parameter
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], wrapped_param)


def _wrap_fpscaled_weight_parameter(
    param: nn.Parameter,
    *,
    logical_dtype: Optional[torch.dtype],
) -> nn.Parameter:
    """
    Wrap a weight parameter in FPScaledTensor so that:

      * ``param.dtype`` appears as ``logical_dtype`` to callers, and
      * the underlying storage dtype remains intact for FP8 handling.
    """
    base = param.detach()
    wrapped = FPScaledParameter._from_base(
        base, logical_dtype=logical_dtype, requires_grad=param.requires_grad
    )
    return wrapped


class FPScaledLayer(nn.Module):
    """
    Base class for FP-scaled layers.

    Assumptions:
      - The *stored* weights are in FP8 (e4m3fn / e5m2) or another low-precision dtype.
      - A `scale_weight` tensor (usually per-out-feature) exists in the state dict.
      - On every forward we:
          1. Cast weights from FP8 → `compute_dtype`
          2. Apply `scale_weight`
          3. Run the corresponding op (linear / conv / embedding / etc.)

    The model should be patched *before* loading an FP-scaled state_dict so that
    `scale_weight` parameters are correctly created and populated.
    """

    # Preferred compute dtype. If None, we fall back to the input dtype,
    # defaulting to float16 for FP inputs.
    compute_dtype: Optional[torch.dtype] = None

    def __init__(self, *, compute_dtype: Optional[torch.dtype] = None) -> None:
        # NOTE: Do not call super().__init__() here because in multiple
        # inheritance subclasses like FPScaledLinear(FPScaledLayer, nn.Linear)
        # the next class in the MRO is nn.Linear, whose __init__ expects
        # (in_features, out_features, ...). Calling nn.Module.__init__
        # directly avoids that TypeError while still initializing the
        # module base class.
        nn.Module.__init__(self)
        if compute_dtype is not None:
            self.compute_dtype = compute_dtype

    # -------------------- helpers --------------------

    def _effective_compute_dtype(
        self, x: Optional[torch.Tensor], requested: Optional[torch.dtype] = None
    ) -> torch.dtype:
        if requested is not None:
            return requested
        if getattr(self, "compute_dtype", None) is not None:
            return self.compute_dtype  # type: ignore[return-value]
        if x is not None:
            # If input is already a "good" compute dtype, keep it, otherwise
            # default to float16 which is a safe/small compute type.
            if x.dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                return x.dtype

        return torch.float16

    # NOTE:
    # ----
    # Some FP checkpoints may store `scale_weight` as a 0-d scalar tensor
    # (shape `[]`) while our modules register it as a length-1 tensor
    # (shape `[1]`), or vice versa.  PyTorch's `load_state_dict` is strict
    # about shape mismatches and would normally error in this case.
    #
    # To avoid halting model loading, we normalize these cases by allowing
    # 0-d and 1-d length-1 tensors to load interchangeably for
    # `{prefix}scale_weight`, reshaping the incoming tensor to match the
    # module parameter shape when they both contain a single element.
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "scale_weight"
        if key in state_dict:
            tensor = state_dict[key]
            param = getattr(self, "scale_weight", None)
            # Only handle the simple "single-element tensor" case and leave all
            # other behaviour to the default Module implementation.
            if isinstance(param, torch.nn.Parameter):
                try:
                    if (
                        tensor.numel() == 1
                        and param.numel() == 1
                        and tensor.shape != param.shape
                    ):
                        # View the incoming checkpoint tensor as the current
                        # parameter shape (e.g. [] <-> [1]).
                        state_dict[key] = tensor.view_as(param)
                except Exception:
                    # If anything goes wrong, fall back to default behaviour
                    # and let the underlying implementation surface an error.
                    pass

        key = prefix + "weight"

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _scale_and_cast_weight(
        self,
        weight: Optional[torch.Tensor],
        scale_weight: Optional[torch.Tensor],
        *,
        target_dtype: torch.dtype,
        per_out_feature_dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Cast `weight` to `target_dtype` and apply `scale_weight` (if present).

        - `per_out_feature_dim` is the dimension that corresponds to the "output"
          features/channels (0 for Linear/Conv, -1 for Embedding).
        """
        if weight is None:
            return None

        # Fast path: FP8 weights + explicit scale tensor → use the provided
        # FP8 dequant helper directly. This matches the user-provided
        # fp8_activation_dequant semantics.
        #
        # NOTE: when using FP8ScaledTensor, ``weight.dtype`` may expose the
        # logical compute dtype; for dequant we must check the physical dtype.
        physical_dtype = getattr(weight, "physical_dtype", weight.dtype)
        if (
            physical_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and scale_weight is not None
        ):
            return fp8_activation_dequant(weight, scale_weight, target_dtype)
        if (
            physical_dtype in (torch.float4_e2m1fn_x2, torch.uint8)
            and scale_weight is not None
        ):
            return dequantize_from_fp4(weight, scale_weight, target_dtype)

        # Dequantize / cast from FP8 (or any low-precision) to target_dtype.
        w = weight.to(target_dtype)

        if scale_weight is not None:
            s = scale_weight.to(target_dtype)
            # Common cases:
            #   - scalar
            #   - per-out-feature: [out_features]
            if s.numel() == 1:
                w = w * s
            else:
                # Broadcast along the out-feature dimension
                if per_out_feature_dim < 0:
                    per_out_feature_dim = w.dim() + per_out_feature_dim

                # View scale as [out_features, 1, 1, ...]
                shape = [1] * w.dim()
                shape[per_out_feature_dim] = -1
                s_view = s.view(*shape)
                w = w * s_view

        return w


class FPScaledLinear(FPScaledLayer, nn.Linear):
    """
    Linear layer with FP weights and `scale_weight` support.

    If Transformer Engine is available, we still keep the public API identical
    (so that FP checkpoints load cleanly) but we optionally wrap the matmul in
    TE's `fp8_autocast` context for optimized kernels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Linear.__init__(
            self,
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        # Wrap the weight parameter so that external code sees the *compute*
        # dtype (logical) via ``weight.dtype``, while we still retain the true
        # FP storage dtype for dequantization via ``weight.physical_dtype``.
        logical_dtype = getattr(self, "compute_dtype", None) or self.weight.dtype
        self.weight = _wrap_fpscaled_weight_parameter(
            self.weight, logical_dtype=logical_dtype
        )

        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    @property
    def effective_dtype(self):
        return self.compute_dtype or self.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)

        x_c = x.to(compute_dtype)
        w = self._scale_and_cast_weight(
            self.weight, getattr(self, "scale_weight", None), target_dtype=compute_dtype
        )
        b = self.bias
        if b is not None and b.dtype != compute_dtype:
            b = b.to(compute_dtype)

        def _linear(inp: torch.Tensor, w_: torch.Tensor, b_: Optional[torch.Tensor]):
            return F.linear(inp, w_, b_)

        if _HAS_TE:
            # We only use TE for the matmul kernel; weights are already dequantized
            # and scaled here.
            with te.fp8_autocast(enabled=False):  # keep semantics explicit
                out = _linear(x_c, w, b)
        else:
            out = _linear(x_c, w, b)

        # Always return in compute dtype (float16/bfloat16/float32), not FP8.
        return out


class FPScaledConv2d(FPScaledLayer, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        # Wan FP checkpoints store `scale_weight` as a scalar; we rely on
        # broadcasting in `_scale_and_cast_weight` to apply it per out-channel.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)

        x_c = x.to(compute_dtype)
        w = self._scale_and_cast_weight(
            self.weight,
            getattr(self, "scale_weight", None),
            target_dtype=compute_dtype,
            per_out_feature_dim=0,
        )
        b = self.bias
        if b is not None and b.dtype != compute_dtype:
            b = b.to(compute_dtype)

        out = F.conv2d(
            x_c,
            w,
            b,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # Always return in compute dtype (float16/bfloat16/float32), not FP8.
        return out


class FPScaledConv1d(FPScaledLayer, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        # Wan FP checkpoints store `scale_weight` as a scalar; we rely on
        # broadcasting in `_scale_and_cast_weight` to apply it per out-channel.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)

        x_c = x.to(compute_dtype)
        w = self._scale_and_cast_weight(
            self.weight,
            getattr(self, "scale_weight", None),
            target_dtype=compute_dtype,
            per_out_feature_dim=0,
        )
        b = self.bias
        if b is not None and b.dtype != compute_dtype:
            b = b.to(compute_dtype)

        out = F.conv1d(
            x_c,
            w,
            b,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        # Always return in compute dtype (float16/bfloat16/float32), not FP8.
        return out


class FPScaledEmbedding(FPScaledLayer, nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.Embedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )
        # Wan FP checkpoints store `scale_weight` as a scalar; we rely on
        # broadcasting in `_scale_and_cast_weight` to apply it per embedding-dim.
        self.scale_weight = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(None)
        w = self._scale_and_cast_weight(
            self.weight,
            getattr(self, "scale_weight", None),
            target_dtype=compute_dtype,
            per_out_feature_dim=-1,
        )
        return F.embedding(
            x,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class FPScaledLayerNorm(FPScaledLayer, nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.LayerNorm.__init__(
            self,
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if elementwise_affine:
            logical_dtype = getattr(self, "compute_dtype", None) or self.weight.dtype
            self.weight = _wrap_fpscaled_weight_parameter(
                self.weight, logical_dtype=logical_dtype
            )
            self.scale_weight = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)
        x_c = x.to(compute_dtype)

        if self.elementwise_affine:
            w = self._scale_and_cast_weight(
                self.weight,
                getattr(self, "scale_weight", None),
                target_dtype=compute_dtype,
                per_out_feature_dim=-1,
            )
            b = self.bias
            if b is not None and b.dtype != compute_dtype:
                b = b.to(compute_dtype)
        else:
            w = None
            b = None

        return F.layer_norm(x_c, self.normalized_shape, w, b, self.eps)


class FPScaledGroupNorm(FPScaledLayer, nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.GroupNorm.__init__(
            self,
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )
        if affine:
            logical_dtype = getattr(self, "compute_dtype", None) or self.weight.dtype
            self.weight = _wrap_fpscaled_weight_parameter(
                self.weight, logical_dtype=logical_dtype
            )
            self.scale_weight = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)
        x_c = x.to(compute_dtype)

        if self.affine:
            w = self._scale_and_cast_weight(
                self.weight,
                getattr(self, "scale_weight", None),
                target_dtype=compute_dtype,
                per_out_feature_dim=0,
            )
            b = self.bias
            if b is not None and b.dtype != compute_dtype:
                b = b.to(compute_dtype)
        else:
            w = None
            b = None

        return F.group_norm(x_c, self.num_groups, w, b, self.eps)


class FPScaledBatchNorm2d(FPScaledLayer, nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        if affine:
            logical_dtype = getattr(self, "compute_dtype", None) or self.weight.dtype
            self.weight = _wrap_fpscaled_weight_parameter(
                self.weight, logical_dtype=logical_dtype
            )
            self.scale_weight = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)
        x_c = x.to(compute_dtype)

        if self.affine:
            w = self._scale_and_cast_weight(
                self.weight,
                getattr(self, "scale_weight", None),
                target_dtype=compute_dtype,
                per_out_feature_dim=0,
            )
            b = self.bias
            if b is not None and b.dtype != compute_dtype:
                b = b.to(compute_dtype)
        else:
            w = None
            b = None

        return F.batch_norm(
            x_c,
            self.running_mean,
            self.running_var,
            w,
            b,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class FPScaledBatchNorm3d(FPScaledLayer, nn.BatchNorm3d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.BatchNorm3d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        if affine:
            logical_dtype = getattr(self, "compute_dtype", None) or self.weight.dtype
            self.weight = _wrap_fpscaled_weight_parameter(
                self.weight, logical_dtype=logical_dtype
            )
            self.scale_weight = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(x)
        x_c = x.to(compute_dtype)

        if self.affine:
            w = self._scale_and_cast_weight(
                self.weight,
                getattr(self, "scale_weight", None),
                target_dtype=compute_dtype,
                per_out_feature_dim=0,
            )
            b = self.bias
            if b is not None and b.dtype != compute_dtype:
                b = b.to(compute_dtype)
        else:
            w = None
            b = None

        return F.batch_norm(
            x_c,
            self.running_mean,
            self.running_var,
            w,
            b,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class FPScaledRMSNorm(FPScaledLayer, nn.RMSNorm):
    """
    FPScaled wrapper for torch.nn.RMSNorm.

    Important: this uses *multiple inheritance* (like FPScaledLinear/Conv/Embedding)
    rather than containing an inner submodule, so the state_dict keys remain
    compatible (`weight`, `eps`, etc.) with the original `nn.RMSNorm`.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        compute_dtype: Optional[torch.dtype] = None,
        device=None,
        dtype=None,
    ) -> None:
        FPScaledLayer.__init__(self, compute_dtype=compute_dtype)
        nn.RMSNorm.__init__(
            self,
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

        if self.elementwise_affine and self.weight is not None:
            logical_dtype = getattr(self, "compute_dtype", None) or self.weight.dtype
            self.weight = _wrap_fpscaled_weight_parameter(
                self.weight, logical_dtype=logical_dtype
            )
            self.scale_weight = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._effective_compute_dtype(hidden_states)
        x = hidden_states.to(compute_dtype)

        # Manual RMSNorm (so we can apply scaled weight cleanly)
        nd = len(self.normalized_shape)
        dims = tuple(range(-nd, 0)) if nd > 0 else (-1,)
        variance = x.to(torch.float32).pow(2).mean(dims, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        if self.elementwise_affine and self.weight is not None:
            # Apply optional scale_weight and broadcast safely across normalized dims.
            w = self.weight.to(compute_dtype)
            s = getattr(self, "scale_weight", None)
            if s is not None:
                s = s.to(compute_dtype)
                if s.numel() == 1:
                    w = w * s
                elif s.shape == w.shape:
                    w = w * s
                elif s.numel() == w.shape[-1] and w.dim() == 1:
                    w = w * s
                else:
                    w = w * s.view_as(w) if s.numel() == w.numel() else (w * s)
            x = x * w

        return x


_TYPE_MAP = {
    nn.Linear: FPScaledLinear,
    nn.Conv2d: FPScaledConv2d,
    nn.Conv1d: FPScaledConv1d,
    nn.Embedding: FPScaledEmbedding,
    nn.LayerNorm: FPScaledLayerNorm,
    nn.GroupNorm: FPScaledGroupNorm,
    nn.BatchNorm2d: FPScaledBatchNorm2d,
    nn.BatchNorm3d: FPScaledBatchNorm3d,
    nn.RMSNorm: FPScaledRMSNorm,
    RMSNorm: FPScaledRMSNorm,
}


def patch_fpscaled_model(
    model: nn.Module,
    name_filter: Optional[Callable[[str], bool]] = None,
    *,
    default_compute_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    In-place patch of a model to use FPScaled* layers.

    This should be called *before* loading an FP-scaled checkpoint whose
    state_dict contains both `{name}.weight` (in FP) and `{name}.scale_weight`.

    The function prefers a Transformer Engine-backed pathway when
    `transformer_engine.pytorch` is importable (via the FPScaled* layers using
    TE matmul kernels where applicable) and otherwise falls back to pure PyTorch
    implementations.
    """

    stack = [("", model)]
    while stack:
        prefix, mod = stack.pop()
        for name, child in list(mod._modules.items()):
            qname = f"{prefix}{name}"
            t = type(child)

            if t in _TYPE_MAP and (name_filter is None or name_filter(qname)):
                # Recreate the module as an FPScaled* of the appropriate type,
                # copying over the existing (non-FP) weights. The FP / scale
                # weights will then be loaded from the FP8 checkpoint.
                cls = _TYPE_MAP[t]

                if isinstance(child, nn.Linear):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.Conv2d):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.Conv1d):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.Embedding):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.num_embeddings,
                        child.embedding_dim,
                        padding_idx=child.padding_idx,
                        max_norm=child.max_norm,
                        norm_type=child.norm_type,
                        scale_grad_by_freq=child.scale_grad_by_freq,
                        sparse=child.sparse,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                    with torch.no_grad():
                        new_mod.weight.copy_(child.weight)
                elif isinstance(child, nn.LayerNorm):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.normalized_shape,
                        eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                        bias=child.bias is not None,
                        compute_dtype=default_compute_dtype,
                        device=(
                            child.weight.device if child.elementwise_affine else None
                        ),
                        dtype=child.weight.dtype if child.elementwise_affine else None,
                    )
                    with torch.no_grad():
                        if child.elementwise_affine:
                            new_mod.weight.copy_(child.weight)
                            if child.bias is not None:
                                new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.GroupNorm):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.num_groups,
                        child.num_channels,
                        eps=child.eps,
                        affine=child.affine,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device if child.affine else None,
                        dtype=child.weight.dtype if child.affine else None,
                    )
                    with torch.no_grad():
                        if child.affine:
                            new_mod.weight.copy_(child.weight)
                            if child.bias is not None:
                                new_mod.bias.copy_(child.bias)
                elif isinstance(child, nn.BatchNorm2d):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.num_features,
                        eps=child.eps,
                        momentum=child.momentum,
                        affine=child.affine,
                        track_running_stats=child.track_running_stats,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device if child.affine else None,
                        dtype=child.weight.dtype if child.affine else None,
                    )
                    with torch.no_grad():
                        if child.affine:
                            new_mod.weight.copy_(child.weight)
                            if child.bias is not None:
                                new_mod.bias.copy_(child.bias)
                        if child.track_running_stats:
                            new_mod.running_mean.copy_(child.running_mean)
                            new_mod.running_var.copy_(child.running_var)
                            new_mod.num_batches_tracked.copy_(child.num_batches_tracked)
                elif isinstance(child, nn.BatchNorm3d):
                    new_mod = cls(  # type: ignore[call-arg]
                        child.num_features,
                        eps=child.eps,
                        momentum=child.momentum,
                        affine=child.affine,
                        track_running_stats=child.track_running_stats,
                        compute_dtype=default_compute_dtype,
                        device=child.weight.device if child.affine else None,
                        dtype=child.weight.dtype if child.affine else None,
                    )
                    with torch.no_grad():
                        if child.affine:
                            new_mod.weight.copy_(child.weight)
                            if child.bias is not None:
                                new_mod.bias.copy_(child.bias)
                        if child.track_running_stats:
                            new_mod.running_mean.copy_(child.running_mean)
                            new_mod.running_var.copy_(child.running_var)
                            new_mod.num_batches_tracked.copy_(child.num_batches_tracked)
                elif isinstance(child, nn.RMSNorm) or isinstance(child, RMSNorm):
                    # torch.nn.RMSNorm uses `normalized_shape` (tuple-like), not `.dim`
                    new_mod = cls(  # type: ignore[call-arg]
                        child.normalized_shape if isinstance(child, nn.RMSNorm) else child.dim,
                        eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                        compute_dtype=default_compute_dtype,
                        device=(
                            child.weight.device
                            if child.elementwise_affine and child.weight is not None
                            else None
                        ),
                        dtype=(
                            child.weight.dtype
                            if child.elementwise_affine and child.weight is not None
                            else None
                        ),
                    )
                    with torch.no_grad():
                        if child.elementwise_affine and child.weight is not None:
                            new_mod.weight.copy_(child.weight)
                else:  # pragma: no cover - defensive
                    continue

                mod._modules[name] = new_mod
                child = new_mod

            if child is not None and len(child._modules) > 0:
                stack.append((qname + ".", child))


def _infer_fpscaled_module_names_from_state_dict(state_dict: dict) -> set[str]:
    """
    Infer which module qualified names should be patched to FPScaled* layers
    based on checkpoint keys / dtypes.

    We patch a module `foo.bar` when the state_dict contains:
      - `foo.bar.scale_weight`, or
      - `foo.bar.weight` with an FP8/FP4 physical dtype.
    """
    names: set[str] = set()
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue

        if k.endswith(".scale_weight"):
            names.add(k[: -len(".scale_weight")])
            continue

        if k.endswith(".weight"):
            # Some checkpoints may provide FPScaledTensor/Parameter wrappers;
            # prefer the physical dtype if present.
            dtype = getattr(v, "physical_dtype", None) or getattr(v, "dtype", None)
            if dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float4_e2m1fn_x2,
                torch.uint8,
            ):
                names.add(k[: -len(".weight")])
                continue

    return names


def make_fpscaled_name_filter_from_state_dict(
    state_dict: dict,
) -> Callable[[str], bool]:
    """
    Create a name_filter for `patch_fpscaled_model()` that patches only modules
    that appear to be FP-scaled in the incoming checkpoint.
    """
    names = _infer_fpscaled_module_names_from_state_dict(state_dict)
    return lambda qname: qname in names


def patch_fpscaled_model_from_state_dict(
    model: nn.Module,
    state_dict: dict,
    *,
    default_compute_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Patch only the subset of modules that are FP-scaled in `state_dict`.

    This is the recommended entry point when you have the checkpoint available
    at patch time (e.g. in LoaderMixin).
    """
    patch_fpscaled_model(
        model,
        name_filter=make_fpscaled_name_filter_from_state_dict(state_dict),
        default_compute_dtype=default_compute_dtype,
    )
