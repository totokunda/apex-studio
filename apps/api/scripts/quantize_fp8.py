import argparse
import importlib
from pathlib import Path
from typing import Optional
from typing import Literal
import torch
from src.quantize.scaled_layer import get_fp_maxval, fp8_tensor_quant
from src.engine import UniversalEngine
import safetensors.torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize large Linear layers of a model to torch.float8_e4m3fn or "
            "torch.float8_e5m2.\n\n"
            "By default, this is configured for a UMT5EncoderModel text encoder "
            "from Wan 2.2 as an example, but it can be pointed at any compatible "
            "Hugging Face-style directory."
        )
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        default=(
            "/home/tosin_coverquick_co/apex/manifest/verified/video/wan-2.2-a14b-text-to-video-1.0.0.v1.yml"
        ),
        help=(
            "Path to the model directory (Hugging Face-style, with config.json). "
            "Defaults to the Wan 2.2 I2V A14B UMT5EncoderModel text encoder path."
        ),
    )
    parser.add_argument(
        "--component-type",
        type=str,
        default="text_encoder",
        help=(
            "Type of component to load, e.g. 'text_encoder', 'transformer', 'vae', 'scheduler'."
        ),
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=(
            "Where to save the quantized model. "
            "If omitted, saves to `<model-path>-fp8-<dtype_tag>`."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="e4m3fn",
        choices=("e4m3fn", "e5m2"),
        help=(
            "Target float8 storage dtype for Linear weights. "
            "`e4m3fn` → torch.float8_e4m3fn, `e5m2` → torch.float8_e5m2."
        ),
    )
    parser.add_argument(
        "--min-numel",
        type=int,
        default=4096,
        help=(
            "Only Linear layers with `weight.numel() >= min_numel` will be quantized. "
            "Smaller layers are left at their original dtype."
        ),
    )
    parser.add_argument(
        "--ignore-keys",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of substrings; any Linear layer whose qualified name "
            "contains one of these will be skipped. "
            "Example: --ignore-keys lm_head final_layer_norm"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which layers would be quantized without modifying or saving the model.",
    )
    return parser.parse_args()


def _resolve_fp8_dtype(tag: str) -> torch.dtype:
    tag = tag.lower()
    if tag == "e4m3fn":
        return torch.float8_e4m3fn
    if tag == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8 dtype tag: {tag!r}")


def _mantissa_bits_for_dtype(dtype: torch.dtype) -> int:
    """
    Map a target float8 dtype to the corresponding mantissa bits used by
    the FP8 grid helpers in `scaled_layer`.
    """
    if dtype is torch.float8_e4m3fn:
        return 3
    if dtype is torch.float8_e5m2:
        return 2
    raise ValueError(f"Unsupported FP8 dtype for mantissa mapping: {dtype!r}")



def quantize_large_linear_layers_to_fp8(
    model: torch.nn.Module,
    *,
    target_dtype: torch.dtype,
    min_numel: int,
    ignore_keys: Optional[list[str]] = None,
    dry_run: bool = False,
) -> None:
    """
    In-place convert large Linear weights to FP8 storage.

    - Only affects `torch.nn.Linear` modules whose `weight` has at least
      `min_numel` elements.
    - Bias tensors are left at their original dtype.
    - For each quantized weight we:
        * use the FP8 helpers from `src.quantize.scaled_layer` to project
          onto an FP8-like grid, and
        * attach a `scale_weight` parameter storing the scalar scale used.
    """
    num_linear = 0
    num_converted = 0

    ignore_keys = ignore_keys or []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        num_linear += 1

        if any(k in name for k in ignore_keys):
            # Explicitly skip any modules whose qualified name matches a pattern.
            print(f"[info] Skipping Linear layer '{name}' (matched ignore-keys).")
            continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue

        if weight.numel() < min_numel:
            continue

        # Determine FP8 grid parameters from the requested storage dtype.
        mantissa_bits = _mantissa_bits_for_dtype(target_dtype)
        bits = 8
        sign_bits = 1

        # Compute a scalar scale so that max(|w / scale|) fits within the FP8 grid.
        with torch.no_grad():
            max_abs = weight.detach().abs().max()
            if max_abs == 0:
                scale_val = torch.tensor(1.0, device=weight.device, dtype=torch.float32)
            else:
                fp_max = get_fp_maxval(
                    bits=bits, mantissa_bit=mantissa_bits, sign_bits=sign_bits
                ).to(device=weight.device, dtype=max_abs.dtype)
                scale_val = (max_abs / fp_max).clamp_min(1e-8).to(torch.float32)

        # Quantize onto an FP8-like grid using the shared helper.
        qdq_weight, scale_tensor, _ = fp8_tensor_quant(
            weight,
            scale=scale_val,
            bits=bits,
            mantissa_bit=mantissa_bits,
            sign_bits=sign_bits,
        )

        if dry_run:
            print(
                f"[dry-run] Would quantize Linear layer '{name}' with "
                f"shape={tuple(weight.shape)} to {target_dtype} "
                f"using scale={scale_val.item():.6e}"
            )
            num_converted += 1
            continue

        with torch.no_grad():
            # Store FP8-grid-snapped weights in the requested float8 dtype.
            module.weight = torch.nn.Parameter(qdq_weight.to(target_dtype))

            # Attach a learnable-but-frozen scale parameter so that downstream
            # loaders (e.g. FP8Scaled* layers) can dequantize correctly.
            module.scale_weight = torch.nn.Parameter(
                scale_tensor.to(torch.float32), requires_grad=False
            )
        num_converted += 1
        print(
            f"[info] Quantized Linear layer '{name}' "
            f"shape={tuple(weight.shape)} to {target_dtype}"
        )

    print(
        f"[summary] Visited {num_linear} Linear layers; "
        f"quantized {num_converted} with min_numel={min_numel}."
    )


def save_quantized_model(model: torch.nn.Module, save_path: str) -> None:
    safetensors.torch.save_model(model, save_path)
    print(f"[info] Saved quantized model to {save_path}")


def main() -> None:
    args = parse_args()

    target_dtype = _resolve_fp8_dtype(args.dtype)
    engine = UniversalEngine(yaml_path=args.yaml_path)
    engine.engine.load_component_by_type(args.component_type)
    text_encoder_model = engine.engine.text_encoder.load_model(no_weights=False)

    quantize_large_linear_layers_to_fp8(
        text_encoder_model,
        target_dtype=target_dtype,
        min_numel=args.min_numel,
        ignore_keys=args.ignore_keys,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("[dry-run] No changes were saved.")
        return

    save_path = args.save_path
    if save_path is None:
        # Default: sibling directory derived from model_path.
        mp = Path(args.yaml_path)
        dtype_tag = "fp8-" + args.dtype
        if mp.is_dir():
            save_path = str(mp.with_name(mp.name + f"-{dtype_tag}"))
        else:
            save_path = str(mp.parent / f"{mp.stem}-{dtype_tag}")

    save_quantized_model(text_encoder_model, save_path)


if __name__ == "__main__":
    main()


