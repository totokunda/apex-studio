import gguf
from typing import Literal
import warnings
from tqdm import tqdm
import torch
from typing import Dict
from src.quantize.ggml_tensor import GGMLTensor
from src.quantize.dequant import is_quantized, dequantize_tensor
from src.utils.dtype import convert_str_dtype
from loguru import logger

T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

STEP_SD_MAP = {
    # layers
    "blk.": "transformer.layers.",
    # attention norms
    "attn_norm": "attention_norm",
    # attention projections (unfused path for GGUF)
    "attn_q": "attention.wq",
    "attn_k": "attention.wk",
    "attn_v": "attention.wv",
    "attn_output": "attention.wo",
    # ffn norms
    "ffn_norm": "ffn_norm",
    # feed-forward weights (unfused path for GGUF)
    "ffn_gate": "feed_forward.ffn_gate",
    "ffn_up": "feed_forward.ffn_up",
    "ffn_down": "feed_forward.ffn_down",
    # embeddings
    "token_embd": "tok_embeddings.word_embeddings",
}

MISTRAL_SD_MAP = {
    # Mistral "language_model.*" HF-style key layout (e.g. Mistral3 / Pixtral-style wrappers)
    # NOTE: GGUF stores dims reversed; loader already reverses shapes, so we only rename keys here.
    "blk.": "language_model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "patch_embd": "patch_conv",
    # Common non-block tensors (kept for completeness / parity with LLAMA_SD_MAP)
    "token_embd": "language_model.embed_tokens",
    "output_norm": "language_model.norm",
    "output.weight": "language_model.lm_head.weight",
    "mm.1": "multi_modal_projector.linear_1",
    "mm.2": "multi_modal_projector.linear_2",
    "mm.input_norm": "multi_modal_projector.norm",
    "mm.patch_merger": "multi_modal_projector.patch_merger.merging_layer",
}

MISTRAL_VISION_SD_MAP = {
    # Pixtral-style vision tower ("v.*" in GGUF)
    "v.blk.": "vision_tower.transformer.layers.",
    "v.patch_embd": "vision_tower.patch_conv",
    "v.pre_ln": "vision_tower.ln_pre",
    "v.token_embd.img_break": "vision_tower.token_embd.img_break",
    # attention
    "attn_q": "attention.q_proj",
    "attn_k": "attention.k_proj",
    "attn_v": "attention.v_proj",
    "attn_out": "attention.o_proj",
    # norms (Pixtral HF naming)
    "ln1": "attention_norm",
    "ln2": "ffn_norm",
    # feed-forward
    "ffn_up": "feed_forward.up_proj",
    "ffn_down": "feed_forward.down_proj",
    "ffn_gate": "feed_forward.gate_proj",
}

QWEN_VL_SD_MAP = {
    "blk.": "model.layers.",
    "token_embd.": "model.embed_tokens.",
    ".attn_norm.": ".input_layernorm.",
    ".ffn_norm.": ".post_attention_layernorm.",
    ".attn_k": ".self_attn.k_proj",
    ".attn_q": ".self_attn.q_proj",
    ".attn_v": ".self_attn.v_proj",
    ".attn_output": ".self_attn.o_proj",
    ".ffn_down": ".mlp.down_proj",
    ".ffn_up": ".mlp.up_proj",
    ".ffn_gate": ".mlp.gate_proj",
    ".output": ".lm_head",
    "output_norm.": "model.norm.",
}

QWEN_VL_VISION_SD_MAP = {
    "mm.0": "visual.merger.mlp.0",
    "mm.2": "visual.merger.mlp.2",
    "v.patch_embd.weight": "visual.patch_embed.proj.weight",
    "v.patch_embd.weight.1": "visual.patch_embed.proj.weight.1",
    "v.blk": "visual.blocks",
    ".attn_q": ".attn.q",
    ".attn_k": ".attn.k",
    ".attn_v": ".attn.v",
    ".attn_out.": ".attn.proj.",
    ".ffn_down": ".mlp.down_proj",
    ".ffn_up": ".mlp.up_proj",
    ".ffn_gate": ".mlp.gate_proj",
    ".ln1.": ".norm1.",
    ".ln2.": ".norm2.",
    "v.post_ln": "visual.merger.ln_q"
}

GEMMA3_SD_MAP = {
    "token_embd.": "model.embed_tokens.",
    "blk.": "model.layers.",
    "attn_k.": "self_attn.k_proj.",
    "attn_q.": "self_attn.q_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_norm.": "input_layernorm.",
    "ffn_down.": "mlp.down_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_gate.": "mlp.gate_proj.",
    "post_attention_norm.": "post_attention_layernorm.",
    "post_ffw_norm.": "post_feedforward_layernorm.",
    "ffn_norm.": "pre_feedforward_layernorm.",
    "output_norm.": "model.norm.",
}

GEMMA3_VISION_SD_MAP = {
    "mm.input_projection.weight": "multi_modal_projector.mm_input_projection_weight",
    "mm.soft_emb_norm.": "multi_modal_projector.mm_soft_emb_norm.",
    "v.patch_embd.bias": "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "v.patch_embd.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "v.position_embd.weight": "vision_tower.vision_model.embeddings.position_embedding.weight",
    "v.post_ln.": "vision_tower.vision_model.post_layernorm.",
    "v.blk.": "vision_tower.vision_model.encoder.layers.",
    "ln": "layer_norm",
    "attn_k": "self_attn.k_proj",
    "attn_out": "self_attn.out_proj",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "ffn_down": "mlp.fc2",
    "ffn_up": "mlp.fc1",
}


def remap_key(key: str, key_map: Literal["t5", "llama", "step", "mistral", "qwen_vl", "gemma3"] = "t5"):

    if key_map == "t5":
        key_map = T5_SD_MAP
    elif key_map == "llama":
        key_map = LLAMA_SD_MAP
    elif key_map == "step":
        key_map = STEP_SD_MAP
    elif key_map == "mistral":
        # Mistral multimodal GGUF can contain both language and vision keys.
        # Vision keys are prefixed with "v." and use different HF-style submodule names.
        key_map = MISTRAL_VISION_SD_MAP if key.startswith("v.") else MISTRAL_SD_MAP
    elif key_map == "qwen_vl":
        key_map = QWEN_VL_VISION_SD_MAP if (key.startswith("v.") or key.startswith("mm.")) else QWEN_VL_SD_MAP
    elif key_map == "gemma3":
        key_map = GEMMA3_VISION_SD_MAP if (key.startswith("v.") or key.startswith("mm.")) else GEMMA3_SD_MAP
    else:
        raise ValueError(f"Invalid key map: {key_map}")

    for k, v in key_map.items():
        key = key.replace(k, v)
    return key


def load_text_encoder_gguf(
    path: str,
    key_map: Literal["t5", "llama", "step", "mistral", "qwen_vl"] = "t5",
    dequant_dtype: torch.dtype | str = torch.float16,
    device: str = "cpu",
    **kwargs,
):

    if key_map is None:
        key_map = "t5"

    if isinstance(dequant_dtype, str):
        dequant_dtype = convert_str_dtype(dequant_dtype)
    reader = gguf.GGUFReader(path)
    state_dict: Dict[str, GGMLTensor] = {}
    qtype_dict: Dict[str, int] = {}
    dev = torch.device(device)

    for tensor in tqdm(reader.tensors):
        name = remap_key(tensor.name, key_map)

        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            # Map quantized GGUF buffers to int8 to avoid downstream frameworks casting activations to uint8
            torch_tensor = torch.from_numpy(tensor.data)
            if dev.type != "cpu" and torch_tensor.device.type == "cpu":
                if dev.type == "cuda":
                    try:
                        torch_tensor = torch_tensor.pin_memory().to(
                            dev, non_blocking=True
                        )
                    except Exception:
                        torch_tensor = torch_tensor.to(dev)
                else:
                    torch_tensor = torch_tensor.to(dev)

            ggml_tensor = GGMLTensor(
                (
                    torch_tensor.view(torch.int8)
                    if is_quantized(torch_tensor)
                    else torch_tensor
                ),
                tensor_type=tensor.tensor_type,
                tensor_shape=shape,
                dequant_dtype=dequant_dtype,
            )

        state_dict[name] = ggml_tensor
        if tensor.name == "token_embd.weight":
            state_dict[name] = dequantize_tensor(
                ggml_tensor, dequant_dtype=dequant_dtype
            )
            if key_map == "t5":  # We duplicate the token embedding for t5
                state_dict["encoder.embed_tokens.weight"] = state_dict[name]

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    return state_dict, qtype_dict


def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}"
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def load_transformer_gguf(
    path: str,
    dequant_dtype: torch.dtype | str = torch.float16,
    device: str = "cpu",
):
    if isinstance(dequant_dtype, str):
        dequant_dtype = convert_str_dtype(dequant_dtype)

    dev = torch.device(device)
    reader = gguf.GGUFReader(path)
    state_dict: Dict[str, GGMLTensor] = {}
    qtype_dict: Dict[str, int] = {}

    for tensor in tqdm(reader.tensors):
        name = tensor.name
        shape = get_orig_shape(reader, name)
        if shape is None:
            # GGUF stores dims reversed
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            base = torch.from_numpy(tensor.data)

            # For F16/F32, present the logical shape now; quantized shapes are handled by dequant
            if tensor.tensor_type in {
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
            }:
                base = base.view(*shape)

            # Move the *raw storage* tensor before wrapping as GGMLTensor.
            # Calling `.to(device)` on GGMLTensor is unsafe for quantized tensors because
            # GGMLTensor overrides `.dtype` to report `dequant_dtype`, which can trigger
            # an unwanted cast and corrupt packed quantized bytes.
            if dev.type != "cpu" and base.device.type == "cpu":
                if dev.type == "cuda":
                    # Best-effort pinned H2D copy for speed; falls back to regular copy.
                    try:
                        base = base.pin_memory().to(dev, non_blocking=True)
                    except Exception:
                        base = base.to(dev)
                else:
                    base = base.to(dev)

            ggml_tensor = GGMLTensor(
                base,
                tensor_type=tensor.tensor_type,
                tensor_shape=shape,
                dequant_dtype=dequant_dtype,
                patches=[],
                requires_grad=False,
            )

        state_dict[name] = ggml_tensor
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    return state_dict, qtype_dict


def load_gguf(
    path: str,
    type: Literal["text_encoder", "transformer"],
    key_map: Literal["t5", "llama", "step", "mistral", "qwen_vl"] | None = None,
    device: str = "cpu",
    **kwargs,
):
    if type == "text_encoder":
        if key_map is None:
            key_map = "t5"
        # NOTE: text encoder loader currently doesn't accept `device`; it materializes
        # tensors on CPU and the model code moves/dequantizes as needed.
        return load_text_encoder_gguf(path, key_map=key_map, **kwargs)
    elif type == "transformer":
        return load_transformer_gguf(path, device=device, **kwargs)
    else:
        raise ValueError(f"Invalid type: {type}")
