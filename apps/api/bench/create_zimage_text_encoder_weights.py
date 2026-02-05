"""
Build ZImage text-encoder weights under ./weights/ZImage/text_encoder.

Outputs (in the target folder):
  - text_encoder-bf16.safetensors
  - fp8_e4m3fn.safetensors
  - text_encoder-q8_0.gguf

Notes:
  - The default text-encoder weights for ZImage are provided as a single BF16
    safetensors file (Comfy-Org). We keep that as the source for BF16/FP8.
  - For GGUF export we stage a minimal HF-like folder containing:
      - Tongyi-MAI/Z-Image-Turbo/text_encoder/config.json
      - model.safetensors (copied from the Comfy-Org weight file)
    plus the tokenizer downloaded from Tongyi-MAI/Z-Image-Turbo/tokenizer.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]  # apps/api/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.quantize.scaled_layer import fp8_tensor_quant, get_fp_maxval
from src.quantize.quantize import TextEncoderQuantizer
from src.quantize.quants import QuantType


ZIMAGE_TEXT_ENCODER_WEIGHTS_REPO = "Comfy-Org/z_image_turbo"
ZIMAGE_TEXT_ENCODER_WEIGHTS_FILE = "split_files/text_encoders/qwen_3_4b.safetensors"

ZIMAGE_TOKENIZER_REPO = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_TOKENIZER_SUBFOLDER = "tokenizer"
ZIMAGE_TEXT_ENCODER_CONFIG_REPO = "Tongyi-MAI/Z-Image-Turbo"


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


def _repo_root() -> Path:
    return _ROOT


def _out_dir() -> Path:
    return _repo_root() / "weights" / "ZImage" / "text_encoder"


def _tokenizer_out_dir() -> Path:
    return _repo_root() / "weights" / "ZImage" / "tokenizer"


def _should_quantize_linear_weight(key: str, value: torch.Tensor) -> bool:
    if not key.endswith(".weight"):
        return False
    if not hasattr(value, "ndim") or value.ndim != 2:
        return False
    if hasattr(value, "is_floating_point") and not value.is_floating_point():
        return False
    k = key.lower()
    return not any(substr in k for substr in _EXCLUDE_SUBSTRINGS)


def _download_tokenizer(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ZIMAGE_TOKENIZER_REPO,
        local_dir=str(dest_dir),
        allow_patterns=[f"{ZIMAGE_TOKENIZER_SUBFOLDER}/*", "LICENSE", "README*"],
    )
    tok_dir = dest_dir / ZIMAGE_TOKENIZER_SUBFOLDER
    if not tok_dir.is_dir():
        raise FileNotFoundError(f"Expected tokenizer dir at {tok_dir}")
    return tok_dir


def _download_text_encoder_config(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ZIMAGE_TEXT_ENCODER_CONFIG_REPO,
        local_dir=str(dest_dir),
        allow_patterns=["text_encoder/*.json", "text_encoder/*.index.json", "LICENSE", "README*"],
    )
    te_dir = dest_dir / "text_encoder"
    if not te_dir.is_dir():
        raise FileNotFoundError(f"Expected text_encoder dir at {te_dir}")
    return te_dir


def _download_base_weights_file() -> Path:
    p = hf_hub_download(
        repo_id=ZIMAGE_TEXT_ENCODER_WEIGHTS_REPO,
        filename=ZIMAGE_TEXT_ENCODER_WEIGHTS_FILE,
    )
    return Path(p)


def _create_fp8_e4m3fn_weights(bf16_safetensors_path: Path, out_path: Path) -> None:
    state_dict = load_file(str(bf16_safetensors_path))
    new_state_dict: dict[str, torch.Tensor] = {}

    for key, value in tqdm(state_dict.items(), desc="FP8 quantizing", unit="tensor"):
        if _should_quantize_linear_weight(key, value):
            maxval = get_fp_maxval()
            scale = torch.max(torch.abs(value.flatten())) / maxval
            linear_weight, scale, _log_scales = fp8_tensor_quant(value, scale)
            linear_weight = linear_weight.to(torch.float8_e4m3fn)
            new_state_dict[key] = linear_weight
            new_state_dict[key.replace(".weight", ".scale_weight")] = scale
        else:
            new_state_dict[key] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(out_path))


def main() -> None:
    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download/copy base weights into the expected bf16 filename
    bf16_path = out_dir / "text_encoder-bf16.safetensors"
    if not bf16_path.exists():
        src = _download_base_weights_file()
        shutil.copyfile(src, bf16_path)

    # 2) Create FP8 weights
    fp8_path = out_dir / "fp8_e4m3fn.safetensors"
    if not fp8_path.exists():
        _create_fp8_e4m3fn_weights(bf16_path, fp8_path)

    # 3) GGUF Q8_0
    llama_quantize_path = _repo_root() / "llama-b7902" / "llama-quantize"
    if not llama_quantize_path.exists():
        raise FileNotFoundError(f"llama-quantize not found at {llama_quantize_path}")

    # Stage a minimal HF-like folder for the GGUF exporter.
    staging_root = out_dir / "_staging"
    te_config_dir = _download_text_encoder_config(staging_root / "Z-Image-Turbo")
    tok_dir = _download_tokenizer(_tokenizer_out_dir())

    staged_model_file = te_config_dir / "model.safetensors"
    if not staged_model_file.exists():
        shutil.copyfile(bf16_path, staged_model_file)

    gguf_out = out_dir / "text_encoder-q8_0.gguf"
    if not gguf_out.exists():
        te_quantizer = TextEncoderQuantizer(
            output_path=str(out_dir / "text_encoder-q8_0"),
            model_path=str(te_config_dir),
            tokenizer_path=str(tok_dir),
            quantization=QuantType.Q8_0,
        )
        out_path = Path(
            te_quantizer.quantize(
                llama_quantize_path=str(llama_quantize_path),
            )
        )
        if out_path != gguf_out:
            out_path.replace(gguf_out)

    print("Done. Outputs:")
    print(f" - {bf16_path}")
    print(f" - {fp8_path}")
    print(f" - {gguf_out}")


if __name__ == "__main__":
    main()

