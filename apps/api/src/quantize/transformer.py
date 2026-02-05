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
from gguf import quants as gguf_quants

QUANTIZATION_THRESHOLD = 4096
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4


class ModelArchitecture(Enum):
    # NOTE: These values are written into GGUF metadata as `arch=...` and are also
    # used to pick architecture-specific "preserve dtype" rules during quantization.
    #
    # Keep these aligned with our manifest `components[type=transformer].base` families:
    # e.g. `wan.base` -> "wan", `flux2.base` -> "flux2", `zimage.base` -> "zimage", etc.
    #
    # Some sub-families intentionally get their own ids when the dtype-preservation
    # rules need to be more specific than the base family (e.g. "foley").
    WAN = "wan"
    COSMOS = "cosmos"
    COGVIDEO = "cogvideo"
    HUNYUAN = "hunyuan"
    HUNYUANVIDEO = "hunyuanvideo"
    FOLEY = "foley"
    HUNYUANVIDEO15 = "hunyuanvideo15"
    HUNYUANIMAGE = "hunyuanimage"
    HUNYUANIMAGE3 = "hunyuanimage3"
    SKYREELS = "skyreels"
    LTX = "ltx"
    LTX2 = "ltx2"
    MAGI = "magi"
    MOCHI = "mochi"
    STEPVIDEO = "stepvideo"
    FLUX = "flux"
    FLUX2 = "flux2"
    CHROMA = "chroma"
    ZIMAGE = "zimage"
    QWENIMAGE = "qwenimage"
    ANIMA = "anima"
    SEEDVR = "seedvr"
    # Alias: some tooling/checkpoints refer to the same family as "seedvr2".
    SEEDVR2 = "seedvr2"
    OVIS = "ovis"
    KANDINSKY5 = "kandinsky5"
    LONGCAT = "longcat"
    HIDREAM = "hidream"
    FIBO = "fibo"


#
# Dtype preservation rules (for GGUF export)
# -----------------------------------------
#
# We quantize most large tensors, but keeping certain sensitive tensors in BF16/F16
# tends to significantly improve output quality with negligible size impact:
# - normalization parameters (LayerNorm/RMSNorm, QK norms)
# - embedding/projection modules (pos/time/rope, conditioning projections, etc.)
#
# The rules are implemented as a list of *substrings*; if a tensor name contains
# any of these substrings, we will keep it in BF16 (see `_prepare_and_quantize_tensor`).
#
COMMON_PRESERVE_SUBSTRINGS: list[str] = [
    # Norms are almost always sensitive and tiny.
    "norm",
    # Position / RoPE related buffers or weights.
    "pos_embed",
    "rope",
    # Time embeddings / timestep projections.
    "time_embed",
    "time_proj",
    "timestep",
    # Diffusion-specific modulation tables.
    "scale_shift_table",
]

FP32_WEIGHTS_PRESERVE_DTYPE = {
    # WAN-family transformers (wan.base, wan.fun, wan.causal, wan.vace, wan.multitalk, wan.apex_framepack)
    ModelArchitecture.WAN: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "patch_embedding",
        "audio_proj",
        "text_embedder",
        "image_embedder",
        "time_embedder",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "norm_q",
        "norm_k",
        "norm_added_k",
        "pos_embed",
        "proj_out",
    ],
    # Cosmos transformers
    ModelArchitecture.COSMOS: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "patch_embed",
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
        *COMMON_PRESERVE_SUBSTRINGS,
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
        *COMMON_PRESERVE_SUBSTRINGS,
        "context_embedder",
        "time_text_embed",
        "audio_projection",
        "ref_latents_proj",
        "norm1",
        "norm2",
        "norm_out",
        "norm1_context",
        "norm2_context",
        "proj_out",
    ],
    # HunyuanVideo transformers (hunyuanvideo.base, hunyuanvideo.avatar, hunyuanvideo.framepack)
    ModelArchitecture.HUNYUANVIDEO: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "x_embedder",
        "context_embedder",
        "time_text_embed",
        "text_embedder",
        "guidance_embedder",
        "caption_projection",
        "proj_out",
    ],
    # HunyuanVideo-Foley transformers (hunyuanvideo.foley)
    ModelArchitecture.FOLEY: [
        *COMMON_PRESERVE_SUBSTRINGS,
        # Embedding / projection stacks
        "audio_embedder",
        "visual_proj",
        "cond_in",
        "time_in",
        "sync_in",
        # Learnable empty-condition params
        "empty_clip_feat",
        "empty_sync_feat",
        "sync_pos_emb",
        # Output head
        "final_layer",
    ],
    # HunyuanVideo 1.5 transformers (hunyuanvideo15.base)
    ModelArchitecture.HUNYUANVIDEO15: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "x_embedder",
        "image_embedder",
        "context_embedder",
        "time_embed",
        "time_text_embed",
        "cond_type_embed",
        "proj_in",
        "rope",
        "proj_out",
    ],
    # HunyuanImage transformers (hunyuanimage.base)
    ModelArchitecture.HUNYUANIMAGE: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "x_embedder",
        "context_embedder",
        "time_guidance_embed",
        "time_text_embed",
        "rope",
        "proj_out",
    ],
    # Hunyuan Image 3 transformers
    ModelArchitecture.HUNYUANIMAGE3: [
        *COMMON_PRESERVE_SUBSTRINGS,
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
        *COMMON_PRESERVE_SUBSTRINGS,
        "time_embedder",
        "fps_embedding",
        "fps_projection",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "norm_q",
        "norm_k",
        "pos_embed",
        "proj_out",
    ],
    # LTX-Video transformers
    ModelArchitecture.LTX: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "time_embed",
        "caption_projection",
        "norm1",
        "norm2",
        "norm_out",
        "proj_out",
    ],
    # LTX-2 transformers (ltx2.base / ltx2.base2)
    ModelArchitecture.LTX2: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "patchify_proj",
        "adaln_single",
        "caption_projection",
        "proj_out",
        # Audio/video multi-modal stack
        "audio_patchify_proj",
        "audio_adaln_single",
        "audio_caption_projection",
        "audio_proj_out",
        "audio_norm_out",
        "audio_scale_shift_table",
        "av_ca_",
        # Arg preprocessors are mostly small, but sensitive.
        "video_args_preprocessor",
        "audio_args_preprocessor",
    ],
    # MAGI transformers
    ModelArchitecture.MAGI: [
        *COMMON_PRESERVE_SUBSTRINGS,
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
        *COMMON_PRESERVE_SUBSTRINGS,
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
        "proj_out",
    ],
    # StepVideo transformers
    ModelArchitecture.STEPVIDEO: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "adaln_single",
        "caption_norm",
        "caption_projection",
        "norm1",
        "norm2",
        "norm3",
        "norm_out",
        "proj_out",
    ],
    ModelArchitecture.FLUX: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "context_embedder",
        "x_embedder",
        "distilled_guidance_layer",
        "time_text_embed",
        "norm_out",
        "proj_out",
    ],
    ModelArchitecture.FLUX2: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "norm_out",
        # Flux2 combined timestep + guidance embedding stack.
        "time_guidance_embed",
        # Flux1 legacy names (kept for compatibility with older checkpoints/scripts).
        "distilled_guidance_layer",
        "time_text_embed",
        "x_embedder",
        "proj_out",
        # Flux2 modulation stacks are sensitive.
        "double_stream_modulation",
        "single_stream_modulation",
    ],
    # Chroma is a Flux-style transformer variant (chroma.base)
    ModelArchitecture.CHROMA: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "time_text_embed",
        "distilled_guidance_layer",
        "context_embedder",
        "x_embedder",
        "norm_out",
        "proj_out",
    ],
    # Z-Image (zimage.base / zimage.control)
    ModelArchitecture.ZIMAGE: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "t_embedder",
        "cap_embedder",
        "rope_embedder",
        "all_x_embedder",
        "all_final_layer",
        "final_layer",
        "norm_final",
        "x_pad_token",
        "cap_pad_token",
        "siglip_pad_token",
        "siglip_embedder",
        # Control variant uses these module names.
        "control_layers",
        "control_all_x_embedder",
        "before_proj",
        "after_proj",
        # Block norms (naming differs from other DiTs).
        "attention_norm",
        "ffn_norm",
        "adaLN_modulation",
    ],
    # Qwen-Image (qwenimage.base)
    ModelArchitecture.QWENIMAGE: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "time_text_embed",
        "txt_norm",
        "img_in",
        "txt_in",
        "norm_out",
        "proj_out",
    ],
    # Anima (anima.base)
    ModelArchitecture.ANIMA: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "t_embedder",
        "t_embedding_norm",
        "x_embedder",
        "final_layer",
        "pos_embedder",
        "extra_pos_embedder",
        "layer_norm",
        "q_norm",
        "k_norm",
        "adaln_modulation",
    ],
    # SeedVR (seedvr.base)
    ModelArchitecture.SEEDVR: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "vid_in",
        "txt_in",
        "emb_in",
        "emb_scale",
        "vid_out",
        # base_v2 optionally adds an output norm + AdaLN modulation stack.
        "vid_out_norm",
        "vid_out_ada",
    ],
    # SeedVR2 (seedvr.base / seedvr.base_v2) alias
    ModelArchitecture.SEEDVR2: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "vid_in",
        "txt_in",
        "emb_in",
        "emb_scale",
        "vid_out",
        "vid_out_norm",
        "vid_out_ada",
    ],
    # Ovis (ovis.base)
    ModelArchitecture.OVIS: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "timestep_embedder",
        "context_embedder_norm",
        "context_embedder",
        "x_embedder",
        "norm_out",
        "proj_out",
    ],
    # Kandinsky 5 (kandinsky5.base)
    ModelArchitecture.KANDINSKY5: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "time_embeddings",
        "text_embeddings",
        "pooled_text_embeddings",
        "visual_embeddings",
        "text_rope_embeddings",
        "visual_rope_embeddings",
        "modulation",
        "out_layer",
    ],
    # LongCat (longcat.base)
    ModelArchitecture.LONGCAT: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "x_embedder",
        "t_embedder",
        "y_embedder",
        "final_layer",
        "adaLN_modulation",
        "mod_norm",
        "pre_crs_attn_norm",
    ],
    # HiDream (hidream.base)
    ModelArchitecture.HIDREAM: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "t_embedder",
        "p_embedder",
        "x_embedder",
        "pe_embedder",
        "caption_projection",
        "final_layer",
    ],
    # Bria Fibo (fibo.base)
    ModelArchitecture.FIBO: [
        *COMMON_PRESERVE_SUBSTRINGS,
        "pos_embed",
        "time_embed",
        "guidance_embed",
        "context_embedder",
        "x_embedder",
        "caption_projection",
        "norm_out",
        "proj_out",
    ],
}


def get_f32_weights_preserve_dtype(model_architecture: ModelArchitecture):
    # Be permissive: when new architectures are added upstream, falling back to
    # the common rules is better than crashing with a KeyError.
    return FP32_WEIGHTS_PRESERVE_DTYPE.get(model_architecture, COMMON_PRESERVE_SUBSTRINGS)


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
        data_qtype = gguf.GGMLQuantizationType.BF16
    else:
        data_qtype = qconfig.qtype
    num_elements = int(np.prod(data.shape))
    if n_dims == 1:
        data_qtype = gguf.GGMLQuantizationType.BF16
    elif n_dims == 5:
        data_qtype = (
            gguf.GGMLQuantizationType.BF16
            if old_dtype == torch.float32
            else (
                gguf.GGMLQuantizationType.BF16
                if old_dtype == torch.bfloat16
                else gguf.GGMLQuantizationType.F16
            )
        )
    elif num_elements <= QUANTIZATION_THRESHOLD:
        data_qtype = gguf.GGMLQuantizationType.BF16
    elif check_key_in_preserve_weights_dtype(key, preserve_weights_dtype):
        data_qtype = gguf.GGMLQuantizationType.BF16
    else:
        data_qtype = qconfig.qtype

    # Guard against writing invalid block-quantized tensors.
    #
    # Many GGUF quant types (Q8_0, Q4_K, Q6_K, ...) are block-quantized along the
    # last dimension. That last dimension must be a multiple of the quant block size.
    # Conv kernels (e.g. Conv1d with kernel_size=3) often have shape[..., 3] and are
    # therefore incompatible. If we write those tensors as quantized, the resulting
    # GGUF becomes unreadable by gguf readers.
    try:
        block_size, _ = gguf_quants.GGML_QUANT_SIZES.get(data_qtype, (1, 0))
    except Exception:
        block_size = 1
    if block_size > 1 and (data.shape[-1] % block_size) != 0:
        data_qtype = (
            gguf.GGMLQuantizationType.BF16
            if old_dtype == torch.bfloat16
            else gguf.GGMLQuantizationType.F16
        )

    try:
        data = quantize(data, data_qtype)
    except (AttributeError, gguf.QuantError, ValueError) as e:
        logger.warning(f"Falling back to F16: {e}")
        data_qtype = (
            gguf.GGMLQuantizationType.BF16
            if old_dtype == torch.bfloat16
            else gguf.GGMLQuantizationType.F16
        )
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
    num_workers: int = 1,
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
    
    # check if model_path is a file 
    if os.path.isfile(model_path):
        files_to_load = [model_path]

    # Parallel quantization with ProcessPoolExecutor for true parallel CPU processing
    if num_workers is None or num_workers < 1:
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
