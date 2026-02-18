from safetensors.torch import load_file, save_file
from src.quantize.scaled_layer import fp8_tensor_quant, get_fp_maxval
import torch
from tqdm import tqdm



_EXCLUDE_SUBSTRINGS = (
    # Embeddings
    ".embeddings.",
    "embeddings.",
    ".embedding.",
    "embedding.",
    "embed_tokens",
    "embed_positions",
    "token_embedding",
    "tok_embeddings",
    "position_embedding",
    "pos_embedding",
    "pos_embed",
    "word_embeddings",
    "wte",
    "wpe",
    # Norms
    "norm",
    "layer_norm",
    "layernorm",
    "rms_norm",
    "rmsnorm",
    "batchnorm",
    "groupnorm",
    # Other common non-core params we don't want to FP8
    "lora",
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



file_path = "/home/tosin_coverquick_co/apex-studio/apps/api/out_t/text_encoder-bf16.safetensors"
save_path = "/home/tosin_coverquick_co/apex-studio/apps/api/out_t/text_encoder-fp8_e4m3fn.safetensors"

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