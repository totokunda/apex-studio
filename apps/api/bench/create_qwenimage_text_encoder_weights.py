"""
Build Qwen-Image text-encoder weights under ./weights/Qwen-Image/text_encoder.

Outputs (in the target folder):
  - text_encoder-bf16.safetensors
  - fp8_e4m3fn.safetensors
  - text_encoder-q8_0.gguf   (merged: contains both text + vision/mmproj tensors)

This script:
  1) Downloads the *base* text-encoder weights (HF safetensors shards) and tokenizer
  2) Consolidates shards into a single BF16 safetensors file
  3) Creates FP8(E4M3FN) weights for linear layer matrices (plus per-tensor scales)
  4) Exports a GGUF and quantizes it to Q8_0
  5) Downloads Unsloth's mmproj GGUF and merges its tensors + vision metadata into
     the text GGUF so "vision weights are in the same GGUF file".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]  # apps/api/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

import gguf

from src.quantize.scaled_layer import fp8_tensor_quant, get_fp_maxval
from src.quantize.quantize import TextEncoderQuantizer
from src.quantize.quants import QuantType


HF_REPO_ID = "Qwen/Qwen-Image"
MM_PROJ_REPO_ID = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
MM_PROJ_FILENAME = "mmproj-BF16.gguf"


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


def _repo_root() -> Path:
    # apps/api/bench/ -> apps/api/
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    return _repo_root() / "weights" / "Qwen-Image" / "text_encoder"


def _download_qwenimage_text_encoder_and_tokenizer(staging_dir: Path) -> tuple[Path, Path]:
    """
    Downloads to a local staging dir and returns:
      - model_dir: <staging>/text_encoder
      - tokenizer_dir: <staging>/tokenizer
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(staging_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "text_encoder/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.index.json",
            "tokenizer/*",
            "LICENSE",
            "README*",
        ],
    )

    model_dir = staging_dir / "text_encoder"
    tokenizer_dir = staging_dir / "tokenizer"

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Expected model dir at {model_dir}")
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Expected tokenizer dir at {tokenizer_dir}")

    return model_dir, tokenizer_dir


def _consolidate_safetensors_shards_to_single_file(
    shards: list[Path], out_path: Path
) -> None:
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


def _build_text_encoder_q8_0_gguf(
    *,
    model_dir: Path,
    tokenizer_dir: Path,
    out_path_prefix: Path,
    llama_quantize_path: Path,
) -> Path:
    """
    Builds a Q8_0 text-encoder GGUF (text tensors only) and returns the GGUF path.
    """
    te_quantizer = TextEncoderQuantizer(
        output_path=str(out_path_prefix),
        model_path=str(model_dir),
        tokenizer_path=str(tokenizer_dir),
        quantization=QuantType.Q8_0,
    )
    return Path(
        te_quantizer.quantize(
            llama_quantize_path=str(llama_quantize_path),
        )
    )


def _download_mmproj_gguf(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mmproj_path = hf_hub_download(
        repo_id=MM_PROJ_REPO_ID,
        filename=MM_PROJ_FILENAME,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    return Path(mmproj_path)


def _field_to_python_value(field: gguf.gguf_reader.ReaderField) -> Any:
    """
    Convert a GGUF reader field to a python scalar or list.
    """
    types = field.types
    if len(types) == 1 and types[0] == gguf.GGUFValueType.STRING:
        return bytes(field.parts[-1]).decode("utf-8")
    if len(types) == 1 and types[0] == gguf.GGUFValueType.BOOL:
        return bool(field.parts[-1][0])
    if len(types) == 1 and types[0] in {
        gguf.GGUFValueType.INT8,
        gguf.GGUFValueType.INT16,
        gguf.GGUFValueType.INT32,
        gguf.GGUFValueType.INT64,
        gguf.GGUFValueType.UINT8,
        gguf.GGUFValueType.UINT16,
        gguf.GGUFValueType.UINT32,
        gguf.GGUFValueType.UINT64,
    }:
        return int(field.parts[-1][0])
    if len(types) == 1 and types[0] in {
        gguf.GGUFValueType.FLOAT32,
        gguf.GGUFValueType.FLOAT64,
    }:
        return float(field.parts[-1][0])

    if len(types) == 2 and types[0] == gguf.GGUFValueType.ARRAY:
        subtype = types[1]
        if subtype == gguf.GGUFValueType.STRING:
            return [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]
        if subtype == gguf.GGUFValueType.BOOL:
            return [bool(field.parts[idx][0]) for idx in field.data]
        if subtype in {
            gguf.GGUFValueType.INT8,
            gguf.GGUFValueType.INT16,
            gguf.GGUFValueType.INT32,
            gguf.GGUFValueType.INT64,
            gguf.GGUFValueType.UINT8,
            gguf.GGUFValueType.UINT16,
            gguf.GGUFValueType.UINT32,
            gguf.GGUFValueType.UINT64,
        }:
            return [int(field.parts[idx][0]) for idx in field.data]
        if subtype in {gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64}:
            return [float(field.parts[idx][0]) for idx in field.data]

    raise NotImplementedError(f"Unsupported GGUF field types: {types}")


def _add_field_to_writer(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    if isinstance(value, str):
        writer.add_string(key, value)
    elif isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        # Use uint32 for most ids/counts; fall back if too large/negative.
        if 0 <= value <= (2**32 - 1):
            writer.add_uint32(key, value)
        elif -(2**31) <= value <= (2**31 - 1):
            writer.add_int32(key, value)
        else:
            writer.add_int64(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, (list, tuple)):
        writer.add_array(key, list(value))
    else:
        raise TypeError(f"Unsupported field value type for {key}: {type(value)}")


def _merge_text_gguf_with_mmproj(
    *,
    text_gguf_path: Path,
    mmproj_gguf_path: Path,
    out_path: Path,
) -> None:
    """
    Create a new GGUF that contains:
      - All text GGUF tensors + metadata
      - All mmproj tensors
      - Vision/mmproj-related metadata keys from the mmproj file
    """
    text_reader = gguf.GGUFReader(str(text_gguf_path))
    mm_reader = gguf.GGUFReader(str(mmproj_gguf_path))

    arch_field = text_reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError("text gguf missing general.architecture")
    arch = bytes(arch_field.parts[-1]).decode("utf-8")

    # Merge KVs: start from text, then add/override vision keys from mmproj.
    kv: dict[str, Any] = {}

    def is_reserved(k: str) -> bool:
        return k.startswith("GGUF.")

    def is_vision_key(k: str) -> bool:
        return k.startswith(("clip.", "vision.", "mmproj.", "comfy.clip.", "comfy.vision."))

    for k, field in text_reader.fields.items():
        if is_reserved(k) or k == "general.architecture":
            continue
        kv[k] = _field_to_python_value(field)

    for k, field in mm_reader.fields.items():
        if is_reserved(k) or k == "general.architecture":
            continue
        v = _field_to_python_value(field)
        if (k not in kv) or is_vision_key(k):
            kv[k] = v

    # Ensure "has vision encoder" is set if present in mmproj.
    if "clip.has_vision_encoder" in mm_reader.fields:
        kv["clip.has_vision_encoder"] = _field_to_python_value(
            mm_reader.fields["clip.has_vision_encoder"]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch, use_temp_file=False)

    for k, v in kv.items():
        _add_field_to_writer(writer, k, v)

    def add_all_tensors(reader: gguf.GGUFReader, label: str) -> None:
        for t in tqdm(reader.tensors, desc=f"Adding tensors from {label}", unit="tensor"):
            # `t.data` is a numpy view (often memmap-backed). Preserve raw dtype for quantized tensors.
            raw = t.data
            if not isinstance(raw, np.ndarray):
                raw = np.asarray(raw)
            # For quantized tensors, GGUFWriter expects the *byte shape* (raw.shape)
            # plus the quantization type. For F16/F32, do not pass raw_dtype.
            if t.tensor_type in {
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F32,
            }:
                writer.add_tensor(t.name, raw)
            else:
                writer.add_tensor(
                    t.name,
                    raw,
                    raw_shape=raw.shape,
                    raw_dtype=t.tensor_type,
                )

    add_all_tensors(text_reader, text_gguf_path.name)
    add_all_tensors(mm_reader, mmproj_gguf_path.name)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def main() -> None:
    out_dir = _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    staging_root = out_dir / "_staging" / "Qwen-Qwen-Image"
    model_dir, tokenizer_dir = _download_qwenimage_text_encoder_and_tokenizer(staging_root)

    # 1) bf16 safetensors
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

    # 3) q8_0 gguf (text-only, then we merge mmproj)
    llama_quantize_path = _repo_root() / "llama-b7902" / "llama-quantize"
    if not llama_quantize_path.exists():
        raise FileNotFoundError(f"llama-quantize not found at {llama_quantize_path}")

    text_q8_gguf = out_dir / "text_encoder-q8_0.gguf"
    if not text_q8_gguf.exists():
        gguf_path = _build_text_encoder_q8_0_gguf(
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            out_path_prefix=out_dir / "text_encoder-q8_0",
            llama_quantize_path=llama_quantize_path,
        )
        # quantizer returns a full path; normalize name to expected output.
        if gguf_path != text_q8_gguf:
            gguf_path.replace(text_q8_gguf)

    # 4) merge mmproj into same gguf file
    mmproj_path = _download_mmproj_gguf(out_dir / "_staging" / "mmproj")
    merged_tmp = out_dir / "text_encoder-q8_0.merged.tmp.gguf"
    _merge_text_gguf_with_mmproj(
        text_gguf_path=text_q8_gguf,
        mmproj_gguf_path=mmproj_path,
        out_path=merged_tmp,
    )
    merged_tmp.replace(text_q8_gguf)

    print("Done. Outputs:")
    print(f" - {bf16_path}")
    print(f" - {fp8_path}")
    print(f" - {text_q8_gguf}")


if __name__ == "__main__":
    main()

