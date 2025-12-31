import torch
import torch.nn as nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.lora import layer as lora_layer

from src.quantize.scaled_layer import FPScaledLayer
from src.quantize.ggml_layer import GGMLLayer


class QuantizedLinearLora(lora_layer.Linear):
    """
    LoRA wrapper for quantized linear layers (GGML / FPScaled).

    Differences from the default PEFT `Linear` wrapper:
    - Adapter weights (lora_A/B, etc.) are always kept in a *compute* dtype
      (e.g. float16 / bfloat16) even when the base layer stores FP / GGML
      weights.
    - This avoids running LoRA matmuls in unsupported dtypes like float8,
      while still calling the quantized base layer for the main projection.
    """

    def _move_adapter_to_device_of_base_layer(  # type: ignore[override]
        self, adapter_name: str, device: torch.device | None = None
    ) -> None:
        """
        Move this adapter's parameters to the base layer's device, but choose a
        *safe* compute dtype instead of blindly mirroring the base weight dtype.
        """
        base_layer = self.get_base_layer()

        # Resolve device once from the base layer
        if device is None:
            if isinstance(base_layer, nn.MultiheadAttention):
                base_layer = base_layer.out_proj

            # Prefer any weight-like attribute to infer device
            for weight_name in ("weight", "qweight"):
                w = getattr(base_layer, weight_name, None)
                if w is not None:
                    device = w.device
                    break
            else:
                # Could not determine device
                return

        # Choose a compute dtype for the adapters:
        # - For FP8Scaled layers, use their effective compute dtype.
        # - For GGML layers, use their dequant dtype or fallback to float16.
        # - Otherwise, mirror the base layer's weight dtype (PEFT default).
        if isinstance(base_layer, FPScaledLayer):
            try:
                compute_dtype = base_layer._effective_compute_dtype(None)
            except Exception:
                compute_dtype = torch.bfloat16
        elif isinstance(base_layer, GGMLLayer):
            compute_dtype = getattr(base_layer, "dequant_dtype", None) or torch.bfloat16
        else:
            # Fallback to the original behavior
            w = getattr(base_layer, "weight", None) or getattr(
                base_layer, "qweight", None
            )
            if w is None:
                return
            compute_dtype = w.dtype

        meta = torch.device("meta")

        # Move only this adapter's parameters; other adapters may live elsewhere.
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(
                adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)
            ):
                continue
            if adapter_name not in adapter_layer:
                continue
            if any(p.device == meta for p in adapter_layer.parameters()):
                continue

            if compute_dtype.is_floating_point or compute_dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(
                    device, dtype=compute_dtype
                )
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    def forward(self, x, *args, **kwargs):
        """
        Custom forward that avoids doing math in float8:

        - Run the quantized base layer as-is to get `base_out` (often FP8).
        - Upcast `base_out` to the LoRA compute dtype.
        - Compute LoRA delta in the same compute dtype and add it there.
        - Cast the final result back to the original base dtype.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # Defer to stock paths for disabled/merged/mixed-batch cases.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if adapter_names is not None:
            # Rare mixed-batch inference path – keep the original behavior.
            return super().forward(x, *args, adapter_names=adapter_names, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        # Base projection through quantized layer (may return FP8).
        base_out = self.base_layer(x, *args, **kwargs)

        # Determine a compute dtype from any existing LoRA adapter.
        if len(self.lora_A) == 0:
            return base_out
        first_key = next(iter(self.lora_A.keys()))
        compute_dtype = self.lora_A[first_key].weight.dtype

        # Work in compute dtype for both base and LoRA delta accumulation.
        result = base_out.to(compute_dtype)
        lora_A_keys = self.lora_A.keys()

        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)

            if active_adapter not in self.lora_variant:  # vanilla LoRA
                delta = lora_B(lora_A(dropout(x_cast))) * scaling
                result = result + delta.to(compute_dtype)
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x_cast,
                    result=result,
                )

        # Return in compute dtype so downstream ops (norms, activations) never see FP8.
        return result


def _dispatch_default_with_quant(
    target: nn.Module,
    adapter_name: str,
    lora_config,
    parameter_name: str | None = None,
    **kwargs,
) -> nn.Module | None:
    """
    Drop-in replacement for `peft.tuners.lora.layer.dispatch_default` that
    routes FP8Scaled / GGML-backed linears to `QuantizedLinearLora`.
    """
    # If the target is already a BaseTunerLayer, unwrap to its base.
    if isinstance(target, lora_layer.BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # If we're targeting an nn.Parameter (ParamWrapper path), delegate to PEFT.
    if parameter_name is not None:
        return lora_layer.ParamWrapper(
            target,
            adapter_name,
            parameter_name=parameter_name,
            **kwargs,
        )

    # Embedding / Conv / MHA paths: keep PEFT defaults.
    if isinstance(target_base_layer, nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        return lora_layer.Embedding(target, adapter_name, **embedding_kwargs)

    if isinstance(target_base_layer, nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        return lora_layer.Conv2d(target, adapter_name, **kwargs)

    if isinstance(target_base_layer, nn.Conv3d):
        kwargs.update(lora_config.loftq_config)
        return lora_layer.Conv3d(target, adapter_name, **kwargs)

    if isinstance(target_base_layer, nn.Conv1d):
        kwargs.update(lora_config.loftq_config)
        return lora_layer.Conv1d(target, adapter_name, **kwargs)

    if isinstance(target_base_layer, nn.MultiheadAttention):
        kwargs.update(lora_config.loftq_config)
        return lora_layer.MultiheadAttention(target, adapter_name, **kwargs)

    # --- Linear family (this is where we hook GGML / FP8Scaled) ---
    if isinstance(target_base_layer, nn.Linear):
        # If this is one of our quantized/scaled linear types, use custom LoRA.
        if isinstance(target_base_layer, (FPScaledLayer, GGMLLayer)):
            # PEFT expects fan_in_fan_out=False for torch.nn.Linear
            if kwargs.get("fan_in_fan_out", False):
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            kwargs.update(lora_config.loftq_config)
            return QuantizedLinearLora(target, adapter_name, **kwargs)

        # Fallback to stock behavior for regular linears.
        if kwargs.get("fan_in_fan_out", False):
            import warnings

            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        return lora_layer.Linear(target, adapter_name, **kwargs)

    # Conv1D from transformers: same as PEFT default.
    if isinstance(target_base_layer, lora_layer.Conv1D):
        if not kwargs.get("fan_in_fan_out", False):
            import warnings

            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        return lora_layer.Linear(
            target, adapter_name, is_target_conv_1d_layer=True, **kwargs
        )

    # No compatible type found.
    return None


def patch_peft_for_quantized_lora() -> None:
    """
    Monkey-patch PEFT so that GGML / FP8Scaled linears use `QuantizedLinearLora`.

    Safe to call multiple times; the assignment is idempotent.
    """
    # Patch the layer-level dispatcher.
    if (
        getattr(lora_layer.dispatch_default, "__name__", "")
        != "_dispatch_default_with_quant"
    ):
        lora_layer.dispatch_default = _dispatch_default_with_quant  # type: ignore[assignment]

    # Also patch the LoraModel-level reference used in `_create_new_module`.
    try:
        from peft.tuners.lora import model as lora_model  # type: ignore

        if (
            getattr(lora_model, "dispatch_default", None)
            is not _dispatch_default_with_quant
        ):
            lora_model.dispatch_default = _dispatch_default_with_quant  # type: ignore[assignment]
    except Exception:
        # If PEFT internals change, we don't want to hard-crash; in that case
        # only the layer-level patch will be active.
        pass
