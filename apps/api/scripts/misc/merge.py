from pathlib import Path
import gguf
import numpy as np
from typing import Any
from tqdm import tqdm


t = Path("apex-studio/apps/api/qwi/text_encoder-q8_0.gguf")
m = Path("apex-studio/apps/api/qwi/mmproj-BF16.gguf")
o = Path("apex-studio/apps/api/qwi/text_encoder-q8_0-mmproj.gguf")


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



_merge_text_gguf_with_mmproj(text_gguf_path=t, mmproj_gguf_path=m, out_path=o)