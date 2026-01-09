from typing import Dict, Any
from src.quantize.ggml_ops import ggml_cat, ggml_stack
from src.converters.base_converter import BaseConverter
from typing import List

class TextEncoderConverter(BaseConverter):
    pass


class Gemma3TextEncoderConverter(TextEncoderConverter):
    """
    Converter for Gemma3 text encoder checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            "^language_model.model": "model.language_model",
            "^vision_tower": "model.vision_tower",
            "^vision_model.": "model.vision_tower.vision_model.",
            "model.embed_tokens": "model.language_model.model.embed_tokens",
            "model.layers": "model.language_model.model.layers",
            "model.norm": "model.language_model.model.norm",
            "^multi_modal_projector": "model.multi_modal_projector",
            "^language_model.lm_head": "lm_head",
        }

class Qwen2_5_VLTextEncoderConverter(TextEncoderConverter):
    """
    Converter for Qwen2.5 VL text encoder checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            r"^visual": "model.visual",
            r"^model(?!\.(language_model|visual))": "model.language_model",
        }
        self.special_keys_map = {
            ".attn.q": self.merge_qkv_inplace,
            ".attn.k": self.merge_qkv_inplace,
            ".attn.v": self.merge_qkv_inplace,
            "model.visual.patch_embed.proj": self.combine_patch_embed_inplace,
        }
        
    @staticmethod
    def combine_patch_embed_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Combine the patch embed weights into a single tensor.
        """
        if key not in state_dict:
            return
        
       
        if key.replace("weight", "weight.1") in state_dict:
            weight = state_dict.pop(key)
            weight_1 = state_dict.pop(key.replace("weight", "weight.1"))
            weight = ggml_stack([weight, weight_1], dim=2)
            state_dict[key] = weight

    
    @staticmethod
    def merge_qkv_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Merge the Q, K, and V tensors into a single tensor.
        """
        if key not in state_dict:
            return
        # merge q k v into a single tensor
        
        if ".attn.q." in key:
            q = state_dict.pop(key)
            k = state_dict.pop(key.replace(".attn.q.", ".attn.k."))
            v = state_dict.pop(key.replace(".attn.q.", ".attn.v."))
            qkv = ggml_cat([q, k, v], dim=0)
            state_dict[key.replace(".attn.q.", ".attn.qkv.")] = qkv
        elif ".attn.k." in key:
            k = state_dict.pop(key)
            q = state_dict.pop(key.replace(".attn.k.", ".attn.q."))
            v = state_dict.pop(key.replace(".attn.k.", ".attn.v."))
            qkv = ggml_cat([q, k, v], dim=0)
            state_dict[key.replace(".attn.k.", ".attn.qkv.")] = qkv
        elif ".attn.v." in key:
            v = state_dict.pop(key)
            q = state_dict.pop(key.replace(".attn.v.", ".attn.q."))
            k = state_dict.pop(key.replace(".attn.v.", ".attn.k."))
            qkv = ggml_cat([q, k, v], dim=0)
            state_dict[key.replace(".attn.v.", ".attn.qkv.")] = qkv
        

class T5TextEncoderConverter(TextEncoderConverter):
    """
    Converter for T5-style text encoder checkpoints.

    Uses the same rename/convert mechanics as `TextEncoderConverter`, with a
    `rename_dict` mirroring `T5_SD_MAP` from `src.quantize.load`.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            # ---- Legacy quantized T5-style checkpoints (T5_SD_MAP) ----
            # Kept for backwards compatibility with older GGUF/SD mappings.
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
            # ---- New Wan 2.x text-encoder layout -> HF UMT5EncoderModel ----
            # Top-level embeddings / final norm
            "token_embedding": "shared",
            "norm.weight": "encoder.final_layer_norm.weight",
            "norm.bias": "encoder.final_layer_norm.bias",
            # Blocks prefix
            "blocks": "encoder.block",  # older style
            "block.": "encoder.block.",  # Wan 2.2 style: block.<idx>.*
            # Self-attention inside each block
            "attn.q": "layer.0.SelfAttention.q",
            "attn.k": "layer.0.SelfAttention.k",
            "attn.v": "layer.0.SelfAttention.v",
            "attn.o": "layer.0.SelfAttention.o",
            "attn.norm": "layer.0.layer_norm",
            "attn.rel_b": "layer.0.SelfAttention.relative_attention_bias",
            # Some checkpoints use sparse-dot style names without the "attn." prefix.
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            # Per-block norms (Wan 2.2: norm1 / norm2)
            ".norm1": ".layer.0.layer_norm",
            ".norm2": ".layer.1.layer_norm",
            # Per-block position / relative bias
            "pos_embedding.embedding": "layer.0.SelfAttention.relative_attention_bias",
            # Feed-forward (Wan: ffn.fc1 / ffn.fc2 / ffn.gate.0)
            ".ffn.fc1": ".layer.1.DenseReluDense.wi_1",
            ".ffn.fc2": ".layer.1.DenseReluDense.wo",
            ".ffn.gate.0": ".layer.1.DenseReluDense.wi_0",
        }
        # No pre-special handling for now.
        self.pre_special_keys_map: Dict[str, Any] = {}
        # Special handling to mirror T5's tied input embeddings:
        # HF models expect both `shared.weight` and `encoder.embed_tokens.weight`.
        # The Wan checkpoints only provide a single token embedding weight tensor.
        self.special_keys_map: Dict[str, Any] = {
            "shared.weight": self._duplicate_shared_to_embed_tokens_inplace,
        }

    @staticmethod
    def _duplicate_shared_to_embed_tokens_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Ensure that `encoder.embed_tokens.weight` exists and shares weights with `shared.weight`.

        UMT5-style encoders typically tie embeddings by setting:
          - `shared.weight`
          - `encoder.embed_tokens.weight` (same tensor)
        The Wan checkpoints only carry a single embedding matrix; we reuse it.
        """
        if key not in state_dict:
            return
        if "encoder.embed_tokens.weight" not in state_dict:
            state_dict["encoder.embed_tokens.weight"] = state_dict[key]

    @staticmethod
    def _normalize_double_encoder_prefix_inplace(state_dict: Dict[str, Any]) -> None:
        """
        Defensive cleanup: collapse accidental `encoder.encoder.` prefixes.

        This can happen if upstream code runs conversion multiple times or if
        legacy mappings are combined with Wan-style mappings.
        """
        for key in list(state_dict.keys()):
            if "encoder.encoder." not in key:
                continue
            new_key = key
            while "encoder.encoder." in new_key:
                new_key = new_key.replace("encoder.encoder.", "encoder.")
            if new_key != key and key in state_dict:
                state_dict[new_key] = state_dict.pop(key)

    @staticmethod
    def _already_hf_umt5_layout(state_dict: Dict[str, Any]) -> bool:
        """
        Heuristic: return True if `state_dict` keys already match HF UMT5/T5 encoder layout.

        This is intentionally conservative and avoids using `rename_dict` markers because
        some source fragments (e.g. `block.` / `norm.weight`) are substrings of *target*
        keys (e.g. `encoder.block.*` / `layer_norm.weight`), which would defeat the base
        converter's generic heuristic.
        """
        if not state_dict:
            return True

        keys = list(state_dict.keys())

        # Strong positive signals of HF-style UMT5/T5 keys.
        has_target = any(
            k.startswith("encoder.block.") or k.startswith("encoder.final_layer_norm.")
            for k in keys
        )
        if not has_target:
            return False

        # Strong signals of *source* (Wan / legacy) layouts that need conversion.
        # Use exact/prefix checks to avoid collisions with HF keys like `layer_norm.weight`.
        has_source = any(
            k.startswith("enc.")
            or k.startswith("blocks.")
            or k.startswith("block.")
            or k == "norm.weight"
            or k == "norm.bias"
            or k.startswith("token_embedding")
            or "token_embd" in k
            or "output_norm" in k
            or ".blk." in k
            or ".attn." in k
            or ".ffn." in k
            for k in keys
        )
        return not has_source

    def convert(self, state_dict: Dict[str, Any], model_keys: List[str] = None):
        """
        Custom convert that allows multiple rename rules to apply to the same key.

        The base `TransformerConverter.convert` updates the state dict inside the
        inner rename loop, which effectively limits each key to one replacement.
        For T5/UMT5 we often need to rewrite both the block prefix and the
        inner-attention/FFN subkeys (e.g. `blocks.0.attn.q` ->
        `encoder.block.0.layer.0.SelfAttention.q`). This override applies all
        applicable replacements first, then performs a single in-place update.
        """
        
        if self._already_converted(state_dict, model_keys):
            return state_dict
            
        # Keep the same ordering semantics as the base class.
        self._sort_rename_dict()

        # Apply any pre-special handlers first (these may drop or reshape keys).
        for key in list(state_dict.keys()):
            for (
                pre_special_key,
                handler_fn_inplace,
            ) in self.pre_special_keys_map.items():
                if pre_special_key in key:
                    handler_fn_inplace(key, state_dict)

        # If this already looks like an HF UMT5/T5 encoder state dict, do NOT apply
        # substring-based renames (they can corrupt already-correct keys like
        # `encoder.final_layer_norm.weight` because `norm.weight` is a substring of
        # `layer_norm.weight`). We still run special handlers (e.g., tying embeddings).
        if self._already_hf_umt5_layout(state_dict):
            self._normalize_double_encoder_prefix_inplace(state_dict)
            for key in list(state_dict.keys()):
                for special_key, handler_fn_inplace in self.special_keys_map.items():
                    if special_key in key:
                        handler_fn_inplace(key, state_dict)
            return state_dict

        # Apply *all* rename_dict rules to each key before updating the dict.
        for key in list(state_dict.keys()):
            # Never rewrite already-HF keys. This keeps conversion safe even for
            # partially-converted state dicts where both source and target keys
            # coexist (we only want to convert the source-shaped keys).
            if key.startswith("encoder.") or key.startswith("shared."):
                continue
            new_key = key
            for replace_key, rename_key in self.rename_dict.items():
                # Handle legacy "enc." prefix carefully: only normalize true
                # `enc.*` prefixes and avoid re-touching already-correct
                # `encoder.*` keys.
                if replace_key == "enc.":
                    if new_key.startswith("enc."):
                        new_key = new_key.replace("enc.", rename_key, 1)
                    continue

                # Avoid accidental substring collisions for the global final norm mapping:
                # `norm.weight` is a substring of `layer_norm.weight` (HF keys), so only
                # map it when the *entire* key matches.
                if replace_key in {"norm.weight", "norm.bias"}:
                    if new_key == replace_key:
                        new_key = rename_key
                    continue

                # Only treat `blocks` / `block.` as source prefixes, not generic substrings,
                # to avoid rewriting already-HF keys like `encoder.block.*`.
                if replace_key == "blocks":
                    # Replace a `blocks` path segment, either at the start or after a prefix.
                    if new_key.startswith("blocks."):
                        new_key = new_key.replace("blocks", rename_key, 1)
                    elif ".blocks." in new_key:
                        new_key = new_key.replace(".blocks.", f".{rename_key}.")
                    continue
                if replace_key == "block.":
                    # Replace a `block.` path segment, either at the start or after a prefix.
                    if new_key.startswith("block."):
                        new_key = new_key.replace("block.", rename_key, 1)
                    elif ".block." in new_key:
                        new_key = new_key.replace(".block.", f".{rename_key}")
                    continue

                if replace_key in new_key:
                    new_key = new_key.replace(replace_key, rename_key)

            # Collapse any accidental double "encoder.encoder." prefixes that
            # may be introduced when combining legacy `enc.` rules with newer
            # Wan-style `blocks` → `encoder.block` mappings.
            while "encoder.encoder." in new_key:
                new_key = new_key.replace("encoder.encoder.", "encoder.")

            if new_key != key and key in state_dict:
                state_dict[new_key] = state_dict.pop(key)

        # Finally, run any special key handlers (e.g., tying embeddings).
        for key in list(state_dict.keys()):
            for special_key, handler_fn_inplace in self.special_keys_map.items():
                if special_key in key:
                    handler_fn_inplace(key, state_dict)


class LlamaTextEncoderConverter(TextEncoderConverter):
    """
    Converter for LLaMA-style text encoder checkpoints.

    The mapping here mirrors `LLAMA_SD_MAP` from `src.quantize.load`.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
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
        self.pre_special_keys_map: Dict[str, Any] = {}
        self.special_keys_map: Dict[str, Any] = {}


class StepTextEncoderConverter(TextEncoderConverter):
    """
    Converter for STEP-style text encoder checkpoints.

    The mapping here mirrors `STEP_SD_MAP` from `src.quantize.load`.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
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
        self.pre_special_keys_map: Dict[str, Any] = {}
        self.special_keys_map: Dict[str, Any] = {}


class MistralTextEncoderConverter(TextEncoderConverter):
    """
    Converter for Mistral-style text encoder checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            # Prefix all keys with `model.` unless they already start with it.
            # Uses regex because TransformerConverter supports regex-based rename rules.
            r"^(?!model\.)": "model.",
        }

        self.special_keys_map = {
            "language_model.model": self._rename_model_inplace,
            "language_model.lm_head.weight": self._duplicate_lm_head_weight_inplace,
            "model.vision_tower.token_embd.img_break": self._remove_img_break_token_inplace,
        }

    @staticmethod
    def _rename_model_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Rename the `model.` key to `model.language_model.model.`.
        """
        if key not in state_dict:
            return
        state_dict[key.replace("language_model.model.", "language_model.")] = (
            state_dict.pop(key)
        )

    @staticmethod
    def _rename_model_embed_tokens_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Rename the `model.embed_tokens.` key to `model.language_model.model.embed_tokens.`.
        """
        if key not in state_dict:
            return
        state_dict[
            key.replace("model.embed_tokens.", "model.language_model.embed_tokens.")
        ] = state_dict.pop(key)

    @staticmethod
    def _rename_model_layers_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Rename the `model.layers.` key to `model.language_model.model.layers.`.
        """
        if key not in state_dict:
            return
        state_dict[key.replace("model.layers.", "model.language_model.layers.")] = (
            state_dict.pop(key)
        )

    @staticmethod
    def _duplicate_lm_head_weight_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Ensure that `lm_head.weight` exists and shares weights with `language_model.lm_head.weight`.
        """
        if key not in state_dict:
            return
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict.pop(key)

    @staticmethod
    def _remove_img_break_token_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Remove the `model.vision_tower.token_embd.img_break` token from the state dict.
        """
        if key not in state_dict:
            return
        if "model.vision_tower.token_embd.img_break" in state_dict:
            state_dict.pop("model.vision_tower.token_embd.img_break")
