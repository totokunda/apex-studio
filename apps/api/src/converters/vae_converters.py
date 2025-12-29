from typing import Dict, Any
from src.converters.utils import update_state_dict_
from src.converters.transformer_converters import TransformerConverter


class VAEConverter(TransformerConverter):
    pass


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

    def convert(self, state_dict: Dict[str, Any]):
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
            "^decoder\.": "taehv.decoder.",
        }
        self.special_keys_map = {}


class NoOpVAEConverter(VAEConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {}
        self.special_keys_map = {}

    def convert(self, state_dict: Dict[str, Any]):
        return None
