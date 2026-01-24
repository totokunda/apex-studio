from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
import torch
from typing import Optional
from warnings import warn

from src.transformer import TRANSFORMERS_REGISTRY


@TRANSFORMERS_REGISTRY("qwenimage.nunchaku")
class QwenImageTransformer2DModel(NunchakuQwenImageTransformer2DModel):
    """
    Apex wrapper around Nunchaku's quantized QwenImage transformer.

    - Provides a stable `.to()` override that avoids the recursion bug in the
      upstream implementation while preserving its semantics.
    - Exposes a convenience hook for enabling Nunchaku's block offloading.
    """

    def enable_group_offload(
        self,
        onload_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
        offload_type: str = "block_level",
        num_blocks_per_group: Optional[int] = None,
        non_blocking: bool = False,
        use_stream: bool = False,
        record_stream: bool = False,
        low_cpu_mem_usage: bool = True,
        offload_to_disk_path: Optional[str] = None,
    ):
        # Delegate to Nunchaku's offload manager. We currently support block-level
        # offloading via `num_blocks_per_group` → `num_blocks_on_gpu`.
        super().set_offload(
            offload=True,
            use_pin_memory=low_cpu_mem_usage,
            num_blocks_on_gpu=num_blocks_per_group,
        )

    def to(self, *args, **kwargs):
        """
        Safe `.to()` implementation for the quantized transformer.

        Mirrors the intent of `NunchakuQwenImageTransformer2DModel.to`:
        - Disallow changing dtype after quantization (`_is_initialized`).
        - When `offload` is enabled, ignore device moves (but still allow dtype-only
          casts during initial construction).
        """

        # Detect whether a device / dtype is being requested.
        device_arg_or_kwarg_present = (
            any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        )
        dtype_present_in_args = "dtype" in kwargs

        # Strings that can be parsed as a device also count as a device request.
        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        # Block dtype changes once the quantized blocks have been patched in.
        if dtype_present_in_args and getattr(self, "_is_initialized", False):
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the "
                "dtype of unquantized layers, please use the `torch_dtype` argument "
                "when loading the model using `from_pretrained` or `from_single_file`."
            )

        # When offload is enabled, keep the model on its current device and only allow
        # non-device-related calls (e.g. dtype-only casts during construction).
        if getattr(self, "offload", False) and device_arg_or_kwarg_present:
            warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
            return self

        # Important: bypass the buggy upstream `to()` that uses `super(type(self), self)`
        # and delegate directly to the base `torch.nn.Module.to`.
        return torch.nn.Module.to(self, *args, **kwargs)
