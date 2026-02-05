"""
Build LTX2 (Gemma3) text-encoder weights under ./weights/LTX2/text_encoder.

Outputs (in the target folder):
  - text_encoder-bf16.safetensors
  - fp8_e4m3fn.safetensors
  - text_encoder-q8_0.gguf

This script:
  1) Downloads the *base* Gemma3 text-encoder weights (HF safetensors shards) and tokenizer
  2) Consolidates shards into a single BF16 safetensors file
  3) Creates FP8(E4M3FN) weights for linear layer matrices (plus per-tensor scales)
  4) Exports a GGUF and quantizes it to Q8_0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm


_ROOT = Path(__file__).resolve().parents[1]  # apps/api/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.quantize.scaled_layer import fp8_tensor_quant, get_fp_maxval
from src.quantize.quantize import TextEncoderQuantizer
from src.quantize.quants import QuantType


MODEL_REPO = "blanchon/LTX-2-Distilled-diffusers"
MODEL_SUBFOLDER = "text_encoder"

TOKENIZER_REPO = "Lightricks/LTX-2"
TOKENIZER_SUBFOLDER = "tokenizer"


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


def _should_quantize_linear_weight(key: str, value: torch.Tensor) -> bool:
    if not key.endswith(".weight"):
        return False
    if not hasattr(value, "ndim") or value.ndim != 2:
        return False
    if hasattr(value, "is_floating_point") and not value.is_floating_point():
        return False
    k = key.lower()
    return not any(substr in k for substr in _EXCLUDE_SUBSTRINGS)


def _repo_root() -> Path:
    return _ROOT


def _out_dir() -> Path:
    return _repo_root() / "weights" / "LTX2" / "text_encoder"


def _tokenizer_out_dir() -> Path:
    return _repo_root() / "weights" / "LTX2" / "tokenizer"


def _download_model_and_tokenizer(staging_dir: Path) -> tuple[Path, Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(staging_dir / "model"),
        allow_patterns=[
            f"{MODEL_SUBFOLDER}/*.json",
            f"{MODEL_SUBFOLDER}/*.safetensors",
            f"{MODEL_SUBFOLDER}/*.index.json",
            "LICENSE",
            "README*",
        ],
    )
    model_dir = staging_dir / "model" / MODEL_SUBFOLDER
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Expected model dir at {model_dir}")

    snapshot_download(
        repo_id=TOKENIZER_REPO,
        local_dir=str(staging_dir / "tokenizer"),
        allow_patterns=[f"{TOKENIZER_SUBFOLDER}/*", "LICENSE", "README*"],
    )
    tokenizer_dir = staging_dir / "tokenizer" / TOKENIZER_SUBFOLDER
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Expected tokenizer dir at {tokenizer_dir}")

    return model_dir, tokenizer_dir


def _consolidate_safetensors_shards_to_single_file(shards: list[Path], out_path: Path) -> None:
    """
    Consolidate multiple safetensors shards into a single safetensors file.

    NOTE: This materializes the full state dict in RAM.
    """
    state: dict[str, torch.Tensor] = {}
    for shard in tqdm(shards, desc="Loading safetensors shards", unit="shard"):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_path))


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

    staging_dir = out_dir / "_staging"
    model_dir, tokenizer_dir = _download_model_and_tokenizer(staging_dir)

    # 1) bf16 safetensors (consolidated)
    bf16_path = out_dir / "text_encoder-bf16.safetensors"
    if not bf16_path.exists():
        shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors shards found under {model_dir}")
        _consolidate_safetensors_shards_to_single_file(shards, bf16_path)

    # 2) fp8 safetensors
    fp8_path = out_dir / "fp8_e4m3fn.safetensors"
    if not fp8_path.exists():
        _create_fp8_e4m3fn_weights(bf16_path, fp8_path)

    # 3) q8_0 gguf
    llama_quantize_path = _repo_root() / "llama-b7902" / "llama-quantize"
    if not llama_quantize_path.exists():
        raise FileNotFoundError(f"llama-quantize not found at {llama_quantize_path}")

    gguf_path = out_dir / "text_encoder-q8_0.gguf"
    if not gguf_path.exists():
        te_quantizer = TextEncoderQuantizer(
            output_path=str(out_dir / "text_encoder-q8_0"),
            model_path=str(model_dir),
            tokenizer_path=str(tokenizer_dir),
            quantization=QuantType.Q8_0,
        )
        out_path = Path(
            te_quantizer.quantize(
                llama_quantize_path=str(llama_quantize_path),
            )
        )
        if out_path != gguf_path:
            out_path.replace(gguf_path)

    # Also copy tokenizer into weights for convenience/reproducibility.
    tok_out = _tokenizer_out_dir()
    if not tok_out.exists():
        tok_out.parent.mkdir(parents=True, exist_ok=True)
        # snapshot_download already placed the tokenizer in staging; just mirror it
        # by copying the folder (small relative to weights).
        import shutil

        shutil.copytree(tokenizer_dir, tok_out, dirs_exist_ok=True)

    print("Done. Outputs:")
    print(f" - {bf16_path}")
    print(f" - {fp8_path}")
    print(f" - {gguf_path}")


if __name__ == "__main__":
    main()

