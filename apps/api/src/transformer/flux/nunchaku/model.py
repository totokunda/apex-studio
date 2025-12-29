from nunchaku.models.transformers.transformer_flux import (
    NunchakuFluxTransformer2dModel,
)
import torch
from typing import Optional
from warnings import warn
from src.transformer import TRANSFORMERS_REGISTRY

@TRANSFORMERS_REGISTRY("flux.nunchaku")
class FluxTransformer2dModel(NunchakuFluxTransformer2dModel):
    """
    Apex wrapper around Nunchaku's quantized FLUX transformer.

    - Adds a stable `.to()` override that avoids potential recursion issues and
      mirrors the semantics used in the QwenImage wrapper.
    - Exposes a helper for enabling Nunchaku's block offloading.
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
        Safe `.to()` implementation for the quantized FLUX transformer.

        Matches the behavior of our QwenImage wrapper:
        - Disallow changing dtype after quantization (if the upstream model tracks
          this via `_is_initialized`).
        - When offload is enabled, ignore device moves but still allow dtype-only
          casts during initial construction.
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

        # Some Nunchaku models track a post-quantization flag; if present, enforce
        # the same "no dtype change after quantization" rule.
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

        # Delegate directly to the base `torch.nn.Module.to` instead of any custom
        # `to()` in the upstream class to avoid recursion issues.
        return torch.nn.Module.to(self, *args, **kwargs)
