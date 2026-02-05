from safetensors.torch import load_file, save_file
from src.quantize.scaled_layer import fp8_tensor_quant, get_fp_maxval
import torch
from tqdm import tqdm

# Substrings (matched against lowercased state_dict keys) that should remain in
# BF16/BF32. For Flux2, the key difference vs Flux1 is `time_guidance_embed`.
#
# Note: we *do not* exclude the large input projections (`x_embedder`,
# `context_embedder`) for Flux2 so they can be quantized too.
_EXCLUDE_SUBSTRINGS = (
    # Output normalization (AdaLN) is sensitive and not part of the heavy matmuls.
    "norm_out",
    # Flux2 combined timestep + guidance embedding stack.
    "time_guidance_embed",
    # Flux1 legacy names (kept for compatibility with older checkpoints/scripts).
    "distilled_guidance_layer",
    "time_text_embed",
)


def _should_quantize_linear_weight(key: str, value) -> bool:
    # Only quantize weights (not biases) for 2D linear layers.
    if not key.endswith(".weight"):
        return False
    if not hasattr(value, "ndim") or value.ndim != 2:
        return False
    # Skip non-float tensors.
    if hasattr(value, "is_floating_point") and not value.is_floating_point():
        return False
    k = key.lower()
    return not any(substr in k for substr in _EXCLUDE_SUBSTRINGS)



file_path = "/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-bf16.safetensors"
save_path = "/home/tosin_coverquick_co/apex-studio/apps/api/weights/FLUX.2-dev/transformer/transformer-fp8_e4m3fn.safetensors"

state_dict = load_file(file_path)

new_state_dict = {}

len_state_dict = len(state_dict)
for i, (key, value) in tqdm(enumerate(state_dict.items()), total=len_state_dict):
    if _should_quantize_linear_weight(key, value):
        # Linear weights are typically shaped [out_features, in_features].
        # Use per-out-feature scaling for better fidelity.
        maxval = get_fp_maxval()
        scale = torch.max(torch.abs(value.flatten())) / maxval
        linear_weight, scale, log_scales = fp8_tensor_quant(value, scale)
        linear_weight = linear_weight.to(torch.float8_e4m3fn)
        new_state_dict[key] = linear_weight
        new_state_dict[key.replace(".weight", ".scale_weight")] = scale
    else:
        new_state_dict[key] = value

print(f"Saved {len(new_state_dict)} tensors to {save_path}")
save_file(new_state_dict, save_path)