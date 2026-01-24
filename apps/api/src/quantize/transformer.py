# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
from tqdm import tqdm
from typing import Iterator, Optional, List
from collections.abc import Mapping
import psutil
from safetensors.torch import load_file as st_load_file, safe_open
from glob import glob
import numpy as np
from loguru import logger
from enum import Enum
from src.quantize.quants import quantize, QuantConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

QUANTIZATION_THRESHOLD = 4096
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4


class ModelArchitecture(Enum):
    WAN = "wan"
    COSMOS = "cosmos"
    COGVIDEO = "cogvideo"
    HUNYUAN = "hunyuan"
    HUNYUANIMAGE3 = "hunyuanimage3"
    SKYREELS = "skyreels"
    LTX = "ltx"
    MAGI = "magi"
    MOCHI = "mochi"
    STEPVIDEO = "stepvideo"


FP32_WEIGHTS_PRESERVE_DTYPE = {
    # WAN-family transformers (wan.base, wan.fun, wan.causal, wan.vace, wan.multitalk, wan.apex_framepack)
    ModelArchitecture.WAN: [
        "patch_embedding",
        "audio_proj",
        "text_embedder",
        "image_embedder",
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "norm_q",
        "norm_k",
        "norm_added_k",
        "pos_embed",
    ],
    # Cosmos transformers
    ModelArchitecture.COSMOS: [
        "patch_embed",
        "time_embed",
        "learnable_pos_embed",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "norm_q",
        "norm_k",
    ],
    # CogVideoX transformers
    ModelArchitecture.COGVIDEO: [
        "patch_embed",
        "time_proj",
        "time_embedding",
        "ofs_proj",
        "ofs_embedding",
        "norm1",
        "norm2",
        "norm_out",
        "norm_final",
    ],
    # Hunyuan transformers (base, avatar, framepack)
    ModelArchitecture.HUNYUAN: [
        "context_embedder",
        "time_text_embed",
        "audio_projection",
        "ref_latents_proj",
        "norm1",
        "norm2",
        "norm_out",
        "norm1_context",
        "norm2_context",
    ],
    # Hunyuan Image 3 transformers
    ModelArchitecture.HUNYUANIMAGE3: [
        # Embedding / projection modules
        "wte",
        "timestep_emb",
        "time_embed",
        "time_embed_2",
        "patch_embed",
        "final_layer",
        # Normalization layers in the decoder and attention
        "input_layernorm",
        "post_attention_layernorm",
        "query_layernorm",
        "key_layernorm",
    ],
    # SkyReels transformers
    ModelArchitecture.SKYREELS: [
        "time_embedder",
        "fps_embedding",
        "fps_projection",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "norm_q",
        "norm_k",
        "pos_embed",
    ],
    # LTX-Video transformers
    ModelArchitecture.LTX: [
        "time_embed",
        "scale_shift_table",
        "caption_projection",
        "norm1",
        "norm2",
        "norm_out",
        "scale_shift_table",
    ],
    # MAGI transformers
    ModelArchitecture.MAGI: [
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "norm_q",
        "norm_k",
        "timestep_embedding",
        "patch_embedding",
        "y_proj_xattn",
        "y_proj_adaln",
        "null_caption_embedding",
        "rope",
        "proj_out",
    ],
    # Mochi transformers
    ModelArchitecture.MOCHI: [
        "patch_embed",
        "time_embed",
        "norm1",
        "norm2",
        "norm3",
        "norm4",
        "norm_out",
        "norm2_context",
        "norm3_context",
        "norm4_context",
    ],
    # StepVideo transformers
    ModelArchitecture.STEPVIDEO: [
        "pos_embed",
        "adaln_single",
        "caption_norm",
        "caption_projection",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
    ],
}


def get_f32_weights_preserve_dtype(model_architecture: ModelArchitecture):
    return FP32_WEIGHTS_PRESERVE_DTYPE[model_architecture]


class LazySafetensorsDict(Mapping):
    """Lazy, read-only Mapping over a `.safetensors` file.

    - Does not materialize all tensors up front
    - Tensors are loaded on access via `safe_open`
    """

    def __init__(
        self, file_path: str, device: str = "cpu", dtype: Optional[torch.dtype] = None
    ):
        self._file_path = file_path
        self._device = device
        self._dtype = dtype
        self._keys_cache: Optional[list[str]] = None

    def _ensure_keys(self) -> list[str]:
        if self._keys_cache is None:
            with safe_open(self._file_path, framework="pt", device=self._device) as f:
                # materialize keys only (small)
                self._keys_cache = list(f.keys())
        return self._keys_cache

    def __getitem__(self, key: str) -> torch.Tensor:
        with safe_open(self._file_path, framework="pt", device=self._device) as f:
            if key not in f.keys():
                raise KeyError(key)
            tensor = f.get_tensor(key)
        if self._dtype is not None:
            tensor = tensor.to(self._dtype)
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self._ensure_keys())

    def __len__(self) -> int:
        return len(self._ensure_keys())


def load_file(file_path: str):
    """Load a state dict from file in a memory-aware way.

    - For `.safetensors`: return a lazy mapping if RAM is insufficient; otherwise load eagerly.
    - For `.pt/.pth/.bin`: use torch's memory-mapped loading to avoid fully materializing weights.
    """
    file_path = str(file_path)
    lower_path = file_path.lower()

    if lower_path.endswith(".safetensors"):
        # Decide eager vs lazy based on available RAM vs file size
        try:
            file_size_bytes = os.path.getsize(file_path)
        except OSError:
            file_size_bytes = 0

        avail_bytes = psutil.virtual_memory().available
        safety_margin = 1.1  # 10% overhead for metadata, fragmentation

        if file_size_bytes > 0 and file_size_bytes * safety_margin > avail_bytes:
            # Not enough free RAM -> lazy mapping
            return LazySafetensorsDict(file_path, device="cpu")
        # Enough RAM -> eager load
        return st_load_file(file_path, device="cpu")

    elif lower_path.endswith((".bin", ".pth", ".pt")):
        # Use memory-mapped storages; this avoids materializing the whole tensor data in RAM
        # weights_only reduces deserialization overhead when present
        try:
            return torch.load(
                file_path, map_location="cpu", mmap=True, weights_only=True
            )
        except TypeError:
            # Older torch without weights_only
            return torch.load(file_path, map_location="cpu", mmap=True)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def check_key_in_preserve_weights_dtype(key: str, preserve_weights_dtype: list[str]):
    for preserve_weight_dtype in preserve_weights_dtype:
        if preserve_weight_dtype in key:
            return True
    return False


def _prepare_and_quantize_tensor(args):
    """Quantize a single tensor and return the result without writing to file.
    Args is a tuple of (key, data, preserve_weights_dtype, keys_to_exclude, qconfig)"""
    key, data, preserve_weights_dtype, keys_to_exclude, qconfig = args

    if keys_to_exclude:
        skip = False
        for exclude_key in keys_to_exclude:
            if exclude_key in key:
                skip = True
                break
        if skip:
            return None

    old_dtype = data.dtype
    if data.dtype == torch.bfloat16:
        data = data.to(torch.float32)
    elif data.dtype in [
        getattr(torch, "float8_e4m3fn", "_invalid"),
        getattr(torch, "float8_e5m2", "_invalid"),
    ]:
        data = data.to(torch.float16)

    data = data.numpy()
    n_dims = len(data.shape)
    if old_dtype == torch.bfloat16:
        data_qtype = gguf.GGMLQuantizationType.BF16
    elif old_dtype == torch.float32:
        data_qtype = gguf.GGMLQuantizationType.F32
    else:
        data_qtype = qconfig.qtype
    num_elements = int(np.prod(data.shape))
    if n_dims == 1:
        data_qtype = gguf.GGMLQuantizationType.F32
    elif n_dims == 5:
        data_qtype = (
            gguf.GGMLQuantizationType.F32
            if old_dtype == torch.float32
            else (
                gguf.GGMLQuantizationType.BF16
                if old_dtype == torch.bfloat16
                else gguf.GGMLQuantizationType.F16
            )
        )
    elif num_elements <= QUANTIZATION_THRESHOLD:
        data_qtype = gguf.GGMLQuantizationType.F32
    elif check_key_in_preserve_weights_dtype(key, preserve_weights_dtype):
        data_qtype = gguf.GGMLQuantizationType.F32
    else:
        data_qtype = qconfig.qtype

    try:
        data = quantize(data, data_qtype)
    except (AttributeError, gguf.QuantError) as e:
        logger.warning(f"Falling back to F16: {e}")
        data_qtype = gguf.GGMLQuantizationType.F16
        data = quantize(data, data_qtype)

    return (key, data, data_qtype)


def _quantize_data(
    data: torch.Tensor,
    key: str,
    preserve_weights_dtype: list[str],
    keys_to_exclude: list[str],
    qconfig: QuantConfig,
    writer: gguf.GGUFWriter,
):
    result = _prepare_and_quantize_tensor(
        (key, data, preserve_weights_dtype, keys_to_exclude, qconfig)
    )
    if result is not None:
        key, data, data_qtype = result
        writer.add_tensor(
            name=key,
            tensor=data,
            raw_dtype=data_qtype,
        )


def convert_model(
    model_path: str,
    output_path: str,
    model_architecture: ModelArchitecture,
    split_max_tensors: int = 0,
    split_max_size: int = 0,
    dry_run: bool = False,
    small_first_shard: bool = False,
    bigendian: bool = False,
    keys_to_exclude: List[str] = None,
    qconfig: QuantConfig = QuantConfig(
        gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16
    ),
    num_workers: int = None,
):
    preserve_weights_dtype = get_f32_weights_preserve_dtype(model_architecture)

    file_pattern = "**/*.safetensors"
    bin_pattern = "**/*.bin"
    pt_pattern = "**/*.pt"
    pth_pattern = "**/*.pth"

    writer = gguf.GGUFWriter(
        path=None,
        arch=model_architecture.value,
        split_max_tensors=split_max_tensors,
        split_max_size=split_max_size,
        dry_run=dry_run,
        small_first_shard=small_first_shard,
        endianess=gguf.GGUFEndian.LITTLE if not bigendian else gguf.GGUFEndian.BIG,
    )

    writer.add_type("diffusion")

    files_to_load = glob(os.path.join(model_path, file_pattern), recursive=True)

    files_to_load += glob(os.path.join(model_path, bin_pattern), recursive=True)

    files_to_load += glob(os.path.join(model_path, pt_pattern), recursive=True)

    files_to_load += glob(os.path.join(model_path, pth_pattern), recursive=True)

    # Parallel quantization with ProcessPoolExecutor for true parallel CPU processing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Quantizing with {num_workers} workers")
    if num_workers == 1:
        for idx, file_path in enumerate(
            tqdm(files_to_load, desc="Loading model files")
        ):
            state_dict = load_file(file_path)
            for key, data in tqdm(
                state_dict.items(),
                desc=f"Quantizing file {idx + 1}/{len(files_to_load)}",
            ):
                _quantize_data(
                    data, key, preserve_weights_dtype, keys_to_exclude, qconfig, writer
                )
    else:
        for idx, file_path in enumerate(
            tqdm(files_to_load, desc="Loading model files")
        ):
            state_dict = load_file(file_path)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks for tensors in this file
                futures = {
                    executor.submit(
                        _prepare_and_quantize_tensor,
                        (key, data, preserve_weights_dtype, keys_to_exclude, qconfig),
                    ): key
                    for key, data in state_dict.items()
                }

                # Process results as they complete with live progress updates
            with tqdm(
                total=len(futures),
                desc=f"Quantizing file {idx + 1}/{len(files_to_load)}",
            ) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        key, data, data_qtype = result
                        writer.add_tensor(
                            name=key,
                            tensor=data,
                            raw_dtype=data_qtype,
                        )
                    pbar.update(1)

    writer.add_file_type(ftype=qconfig.ftype)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.write_header_to_file(path=output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
