from typing import Dict, Any
from src.converters.utils import update_state_dict_
from src.converters.transformer_converters import TransformerConverter
from typing import List


class VAEConverter(TransformerConverter):
    pass


class LTX2VAEConverter(VAEConverter):
    def __init__(self):
        super().__init__()
        # NOTE:
        # `BaseConverter._apply_rename_dict()` applies rename rules sequentially on the
        # same string. For LTX2 VAE, multiple rules intentionally remap indices and some
        # targets overlap with other sources (e.g. `...downsamplers.0 -> down_blocks.1`
        # and also `down_blocks.1 -> down_blocks.2`). If we map directly, we get
        # accidental cascading renames.
        #
        # To prevent that, we first map overlapping sources to short placeholder tokens
        # (kept intentionally short so they sort last), then expand placeholders to the
        # final targets.
        # LTX2 VAE checkpoints come in (at least) two layouts:
        #
        # 1) Nested diffusers-style:
        #    - encoder/decoder blocks include `downsamplers` / `upsamplers`
        #    - a `mid_block` sits between encoder and decoder
        #    We "flatten" those modules into a sequential `down_blocks.{i}` /
        #    `up_blocks.{i}` indexing used by our runtime model.
        #
        # 2) Already-flat (apex-studio / ltx-core style):
        #    - keys already look like `encoder.down_blocks.{i}....` / `decoder.up_blocks.{i}....`
        #    - no `downsamplers` / `upsamplers` / `mid_block` modules
        #    In this case, applying the flattening/index-shift rules would corrupt indices
        #    and cause widespread shape mismatches.
        #
        # We keep both rename maps and select at convert-time based on what the checkpoint contains.
        # This mapping converts a diffusers-style LTX2 VAE state dict:
        #   - encoder.down_blocks.{0..3} + encoder.mid_block
        #   - decoder.mid_block + decoder.up_blocks.{0..2}
        # into our alternating-block layout:
        #   - encoder.down_blocks.{0..8}
        #   - decoder.up_blocks.{0..6}
        #
        # IMPORTANT: These rules must be prefix-specific. In particular, diffusers has
        # both `encoder.mid_block` and `decoder.mid_block`, and we must map them to
        # *different* targets.
        self._nested_rename_dict = {
            # Encoder (placeholders to avoid cascaded remaps)
            "encoder.down_blocks.0.downsamplers.0": "@enc_ds0@",
            "encoder.down_blocks.1.downsamplers.0": "@enc_ds1@",
            "encoder.down_blocks.2.downsamplers.0": "@enc_ds2@",
            "encoder.down_blocks.3.downsamplers.0": "@enc_ds3@",
            "encoder.down_blocks.1": "@enc_b1@",
            "encoder.down_blocks.2": "@enc_b2@",
            "encoder.down_blocks.3": "@enc_b3@",
            "encoder.mid_block": "@enc_mid@",
            # Decoder (placeholders to avoid cascaded remaps)
            "decoder.mid_block": "@dec_mid@",
            "decoder.up_blocks.0.upsamplers.0": "@dec_us0@",
            "decoder.up_blocks.0": "@dec_b0@",
            "decoder.up_blocks.1.upsamplers.0": "@dec_us1@",
            "decoder.up_blocks.1": "@dec_b1@",
            "decoder.up_blocks.2.upsamplers.0": "@dec_us2@",
            "decoder.up_blocks.2": "@dec_b2@",
            # Common
            "time_embedder": "last_time_embedder",
            "scale_shift_table": "last_scale_shift_table",
            # 3D ResNet blocks
            #
            # Our LTX2 VAE implementation (`src/vae/ltx2`) uses `resnets` (diffusers-style).
            # Some external checkpoints may use `res_blocks` instead; normalize those to `resnets`.
            "res_blocks": "resnets",
            # Per-channel statistics / latents buffers
            #
            # Some checkpoints store latent normalization stats under `per_channel_statistics.*`
            # (e.g. Comfy exports). Our model uses top-level buffers: `latents_mean` / `latents_std`.
            "per_channel_statistics.mean-of-means": "latents_mean",
            "per_channel_statistics.std-of-means": "latents_std",
            # Placeholder expansions (must run after the sources above)
            "@enc_ds0@": "encoder.down_blocks.1",
            "@enc_b1@": "encoder.down_blocks.2",
            "@enc_ds1@": "encoder.down_blocks.3",
            "@enc_b2@": "encoder.down_blocks.4",
            "@enc_ds2@": "encoder.down_blocks.5",
            "@enc_b3@": "encoder.down_blocks.6",
            "@enc_ds3@": "encoder.down_blocks.7",
            "@enc_mid@": "encoder.down_blocks.8",
            "@dec_mid@": "decoder.up_blocks.0",
            "@dec_us0@": "decoder.up_blocks.1",
            "@dec_b0@": "decoder.up_blocks.2",
            "@dec_us1@": "decoder.up_blocks.3",
            "@dec_b1@": "decoder.up_blocks.4",
            "@dec_us2@": "decoder.up_blocks.5",
            "@dec_b2@": "decoder.up_blocks.6",
        }

        # Minimal mapping for already-flat checkpoints: avoid any index shifts.
        self._flat_rename_dict = {
            "time_embedder": "last_time_embedder",
            "scale_shift_table": "last_scale_shift_table",
            "res_blocks": "resnets",
            "per_channel_statistics.mean-of-means": "latents_mean",
            "per_channel_statistics.std-of-means": "latents_std",
        }

        # Default to the flat mapping; `convert()` will select the proper map.
        self.rename_dict = dict(self._flat_rename_dict)

        # Drop auxiliary per-channel stats keys that don't correspond to model buffers.
        self.special_keys_map = {
            "per_channel_statistics.channel": self.remove_keys_inplace,
            "per_channel_statistics.mean-of-stds": self.remove_keys_inplace,
            "per_channel_statistics.mean-of-stds_over_std-of-means": self.remove_keys_inplace,
        }

    def convert(self, state_dict: Dict[str, Any], model_keys: List[str] = None):
        # Select rename scheme based on checkpoint layout.
        keys = list(state_dict.keys()) if state_dict else []
        is_nested = any(
            (".downsamplers." in k) or (".upsamplers." in k) or ("mid_block" in k)
            for k in keys
        )
        self.rename_dict = dict(self._nested_rename_dict if is_nested else self._flat_rename_dict)
        self._sort_rename_dict()

        if self._already_converted(state_dict, model_keys):
            return state_dict

        # Apply pre-special keys map (in-place) first.
        for key in list(state_dict.keys()):
            for pre_special_key, handler_fn_inplace in self.pre_special_keys_map.items():
                if pre_special_key in key:
                    handler_fn_inplace(key, state_dict)

        # IMPORTANT:
        # BaseConverter.convert() performs renames in-place while iterating a snapshot
        # of keys. For LTX2 VAE, many rules remap `*.down_blocks.{i}` -> `*.down_blocks.{j}`,
        # which can temporarily collide with existing source keys that haven't been
        # processed yet (e.g. renaming `encoder.down_blocks.1.*` to `encoder.down_blocks.2.*`
        # while an original `encoder.down_blocks.2.*` key still exists).
        #
        # Those collisions can cause later iterations to move the wrong tensor into the
        # final key, producing hard-to-debug size mismatches.
        #
        # To avoid this, we build a fresh state dict with renamed keys (collision-free)
        # and then swap it into place.
        items = list(state_dict.items())
        renamed: Dict[str, Any] = {}
        for key, value in items:
            new_key = self._apply_rename_dict(key)
            renamed[new_key] = value

        state_dict.clear()
        state_dict.update(renamed)

        # Apply special keys map (in-place) on the renamed dict.
        for key in list(state_dict.keys()):
            for special_key, handler_fn_inplace in self.special_keys_map.items():
                if special_key not in key:
                    continue
                handler_fn_inplace(key, state_dict)

        self._strip_known_prefixes_inplace(state_dict, model_keys=model_keys)
        return state_dict

        
    

class LTXVAEConverter(VAEConverter):
    def __init__(self, version: str | None = None):
        super().__init__()
        self.rename_dict = {
            # decoder
            "up_blocks.0": "mid_block",
            "up_blocks.1": "up_blocks.0",
            "up_blocks.2": "up_blocks.1.upsamplers.0",
            "up_blocks.3": "up_blocks.1",
            "up_blocks.4": "up_blocks.2.conv_in",
            "up_blocks.5": "up_blocks.2.upsamplers.0",
            "up_blocks.6": "up_blocks.2",
            "up_blocks.7": "up_blocks.3.conv_in",
            "up_blocks.8": "up_blocks.3.upsamplers.0",
            "up_blocks.9": "up_blocks.3",
            # encoder
            "down_blocks.0": "down_blocks.0",
            "down_blocks.1": "down_blocks.0.downsamplers.0",
            "down_blocks.2": "down_blocks.0.conv_out",
            "down_blocks.3": "down_blocks.1",
            "down_blocks.4": "down_blocks.1.downsamplers.0",
            "down_blocks.5": "down_blocks.1.conv_out",
            "down_blocks.6": "down_blocks.2",
            "down_blocks.7": "down_blocks.2.downsamplers.0",
            "down_blocks.8": "down_blocks.3",
            "down_blocks.9": "mid_block",
            # common
            "conv_shortcut": "conv_shortcut.conv",
            "res_blocks": "resnets",
            "norm3.norm": "norm3",
            "per_channel_statistics.mean-of-means": "latents_mean",
            "per_channel_statistics.std-of-means": "latents_std",
        }

        self.special_keys_map = {
            "per_channel_statistics.channel": self.remove_keys_inplace,
            "per_channel_statistics.mean-of-means": self.remove_keys_inplace,
            "per_channel_statistics.mean-of-stds": self.remove_keys_inplace,
            "model.diffusion_model": self.remove_keys_inplace,
        }

        additional_rename_dict = {
            "0.9.1": {
                "up_blocks.0": "mid_block",
                "up_blocks.1": "up_blocks.0.upsamplers.0",
                "up_blocks.2": "up_blocks.0",
                "up_blocks.3": "up_blocks.1.upsamplers.0",
                "up_blocks.4": "up_blocks.1",
                "up_blocks.5": "up_blocks.2.upsamplers.0",
                "up_blocks.6": "up_blocks.2",
                "up_blocks.7": "up_blocks.3.upsamplers.0",
                "up_blocks.8": "up_blocks.3",
                # common
                "last_time_embedder": "time_embedder",
                "last_scale_shift_table": "scale_shift_table",
            },
            "0.9.5": {
                "up_blocks.0": "mid_block",
                "up_blocks.1": "up_blocks.0.upsamplers.0",
                "up_blocks.2": "up_blocks.0",
                "up_blocks.3": "up_blocks.1.upsamplers.0",
                "up_blocks.4": "up_blocks.1",
                "up_blocks.5": "up_blocks.2.upsamplers.0",
                "up_blocks.6": "up_blocks.2",
                "up_blocks.7": "up_blocks.3.upsamplers.0",
                "up_blocks.8": "up_blocks.3",
                # encoder
                "down_blocks.0": "down_blocks.0",
                "down_blocks.1": "down_blocks.0.downsamplers.0",
                "down_blocks.2": "down_blocks.1",
                "down_blocks.3": "down_blocks.1.downsamplers.0",
                "down_blocks.4": "down_blocks.2",
                "down_blocks.5": "down_blocks.2.downsamplers.0",
                "down_blocks.6": "down_blocks.3",
                "down_blocks.7": "down_blocks.3.downsamplers.0",
                "down_blocks.8": "mid_block",
                # common
                "last_time_embedder": "time_embedder",
                "last_scale_shift_table": "scale_shift_table",
            },
        }

        additional_rename_dict["0.9.7"] = additional_rename_dict["0.9.5"].copy()

        if version is not None:
            self.rename_dict.update(additional_rename_dict[version])

    @staticmethod
    def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
        state_dict.pop(key)


class MagiVAEConverter(VAEConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
            # Encoder components
            "encoder.patch_embed.proj": "encoder.patch_embedding",
            "encoder.norm": "encoder.norm_out",
            "encoder.last_layer": "encoder.linear_out",
            # Decoder components
            "decoder.norm": "decoder.norm_out",
            "decoder.last_layer": "decoder.conv_out",
            # Attention components (for both encoder and decoder)
            "attn.proj": "attn.to_out.0",
            # MLP components (for both encoder and decoder)
            "mlp.fc1": "proj_out.net.0.proj",
            "mlp.fc2": "proj_out.net.2",
        }

        self.special_keys_map = {
            "attn.qkv": self.handle_qkv_chunking,
        }

    @staticmethod
    def handle_qkv_chunking(key: str, state_dict: Dict[str, Any]):
        """Handle chunking of QKV weights and biases for attention layers"""
        tensor = state_dict.pop(key)

        # Get the base key without the qkv suffix
        base_key = key.replace("attn.qkv", "attn")

        if key.endswith(".weight"):
            q_weight, k_weight, v_weight = tensor.chunk(3, dim=0)
            state_dict[f"{base_key}.to_q.weight"] = q_weight
            state_dict[f"{base_key}.to_k.weight"] = k_weight
            state_dict[f"{base_key}.to_v.weight"] = v_weight
        elif key.endswith(".bias"):
            q_bias, k_bias, v_bias = tensor.chunk(3, dim=0)
            state_dict[f"{base_key}.to_q.bias"] = q_bias
            state_dict[f"{base_key}.to_k.bias"] = k_bias
            state_dict[f"{base_key}.to_v.bias"] = v_bias


class MMAudioVAEConverter(VAEConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {}
        self.special_keys_map = {}

    def convert(self, state_dict: Dict[str, Any], model_keys: List[str] = None):
        if self._already_converted(state_dict, model_keys):
            return state_dict
        keys = list(state_dict.keys())
        generator_state = None
        decoder_state = None

        if len(keys) == 1 and keys[0] == "generator":
            generator_state = state_dict["generator"]
            for key in list(generator_state.keys()):
                update_state_dict_(generator_state, key, f"tod.vocoder.vocoder.{key}")
        elif "data_mean" in keys:
            decoder_state = state_dict.copy()
            for key in list(decoder_state.keys()):
                update_state_dict_(decoder_state, key, f"tod.vae.{key}")

        state_dict.clear()
        if generator_state is not None:
            state_dict.update(generator_state)
        elif decoder_state is not None:
            state_dict.update(decoder_state)
        else:
            raise ValueError("No generator or decoder state found in the state dict")


class TinyWANVAEConverter(VAEConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
            r"^decoder\.": "taehv.decoder.",
        }
        self.special_keys_map = {}
