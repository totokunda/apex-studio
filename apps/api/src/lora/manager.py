import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from src.converters.convert import (
    get_transformer_converter_by_model_name,
    strip_common_prefix,
)
from src.lora.lora_converter import LoraConverter
import torch
from loguru import logger
from diffusers.loaders import PeftAdapterMixin
from safetensors.torch import load_file
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import DEFAULT_LORA_SAVE_PATH
from urllib.parse import urlencode
import torch.nn as nn
from safetensors.torch import safe_open

# Ensure PEFT is patched to handle GGML / FP8-scaled linears with a custom LoRA wrapper.
from src.lora.quantized_lora import patch_peft_for_quantized_lora
from src.lora.key_remap import remap_embedding_lora_keys

patch_peft_for_quantized_lora()

try:
    from nunchaku.lora.flux.compose import compose_lora
except ImportError:
    compose_lora = None


@dataclass
class LoraItem:
    # A single LoRA entry that may consist of 1+ files (.safetensors/.bin and optional .json config)
    source: str
    local_paths: List[str]
    scale: float = 1.0
    name: Optional[str] = None
    component: Optional[str] = None


class LoraManager(DownloadMixin):
    def __init__(self, save_dir: str = DEFAULT_LORA_SAVE_PATH) -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # cache source->LoraItem to avoid repeated downloads
        self._cache: Dict[str, LoraItem] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def resolve(
        self,
        source: str,
        prefer_name: Optional[str] = None,
        progress_callback: Optional[
            Callable[[int, Optional[int], Optional[str]], None]
        ] = None,
    ) -> LoraItem:
        """
        Resolve and download a LoRA from any supported source:
        - HuggingFace repo or file path
        - CivitAI model or file URL (e.g., https://civitai.com/api/download/models/<id>)
        - Generic direct URL
        - Local path (dir or file)
        Returns a LoraItem with one or more local file paths.
        """
        if source in self._cache:
            return self._cache[source]

        # Local path – nothing to download, so no progress_callback needed
        if os.path.exists(source):
            paths = self._collect_lora_files(source)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or os.path.basename(source),
            )
            self._cache[source] = item
            return item

        # CivitAI integration: support plain download links and model/file ids
        if self._is_civitai(source):
            # Model-id style spec, delegate to helper with progress
            if source.startswith("civitai:"):
                local_path = self._download_from_civitai_spec(
                    source,
                    progress_callback=progress_callback,
                )
            else:
                # Direct CivitAI download URL
                local_path = self._download(
                    source,
                    self.save_dir,
                    progress_callback=progress_callback,
                )
            paths = self._collect_lora_files(local_path)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or self._infer_name(source, local_path),
            )
            self._cache[source] = item
            return item

        # Generic URL / HF / cloud / path handled by DownloadMixin with progress
        local_path = self._download(
            source,
            self.save_dir,
            progress_callback=progress_callback,
        )
        paths = self._collect_lora_files(local_path)
        item = LoraItem(
            source=source,
            local_paths=paths,
            name=prefer_name or self._infer_name(source, local_path),
        )
        self._cache[source] = item
        return item

    def _looks_like_hf_file(self, text: str) -> bool:
        # matches org/repo/…/file.safetensors or similar
        return bool(re.match(r"^[\w\-]+/[\w\-]+/.+\.[A-Za-z0-9]+$", text))

    def _is_civitai(self, url: str) -> bool:
        return "civitai.com" in url or url.startswith("civitai:")

    def _infer_name(self, source: str, local_path: str) -> str:
        if os.path.isdir(local_path):
            return os.path.basename(local_path.rstrip("/"))
        return os.path.splitext(os.path.basename(local_path))[0]

    def _collect_lora_files(self, path: str) -> List[str]:
        """Return a list of LoRA weight files for a given path (dir or file)."""
        if not path:
            return []
        if os.path.isdir(path):
            files = []
            for root, _dirs, fnames in os.walk(path):
                for fn in fnames:
                    path = os.path.join(root, fn)
                    if self._is_lora_file(path):
                        files.append(path)
            return sorted(files)
        if os.path.isfile(path) and self._is_lora_file(path):
            return [path]
        return []

    def _is_lora_file(self, filename: str) -> bool:
        """
        Heuristic check for whether a file is *likely* a LoRA weights file.
        Designed to be very cheap in CPU usage:
        - Only checks the extension.
        - Optionally sniffs a tiny header to rule out obvious HTML/text error pages.
        """
        try:
            with open(filename, "rb") as f:
                head = f.read(2048)
        except OSError:
            return False

        if not head:
            return False

        # If this decodes cleanly to mostly text and looks like HTML or an error,
        # treat it as "not a LoRA file". This keeps CPU usage tiny while avoiding
        # obviously-wrong files.
        text_sample = head.decode("utf-8", errors="ignore").lstrip().lower()
        html_markers = ("<!doctype html", "<html", "<head", "<body")
        if text_sample.startswith(html_markers):
            return False

        # Common text error indicators near the top of the file.
        for marker in ("error", "not found", "access denied"):
            if marker in text_sample[:512]:
                return False

        return True

    def _clean_adapter_name(self, name: str) -> str:
        if len(name) > 64:
            name = name[:64]
        if "." in name or "/" in name:
            name = name.replace(".", "_").replace("/", "_")
        return name

    def _get_prefix_key(self, keys: List[str]):
        prefix = None
        if keys[0].startswith("transformer.") and keys[-1].startswith("transformer."):
            prefix = "transformer"
        elif keys[0].startswith("diffusion_model.") and keys[-1].startswith(
            "diffusion_model."
        ):
            prefix = "diffusion_model"
        elif keys[0].startswith("model.") and keys[-1].startswith("model."):
            prefix = "model"
        elif keys[0].startswith("unet.") and keys[-1].startswith("unet."):
            prefix = "unet"
        return prefix

    @staticmethod
    def _build_lora_config_metadata_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """
        Diffusers' PEFT loader infers LoRA rank from keys containing "lora_B".
        That misses embedding LoRA keys ("lora_embedding_*"), which can have a much
        smaller effective rank (e.g. num_embeddings=3 => max rank=3).

        We build an explicit LoraConfig kwargs dict (metadata) including `rank_pattern`
        so modules with non-default rank (like embeddings) are injected with the
        correct rank and can load without shape mismatches.
        """
        import collections

        per_module_rank: Dict[str, int] = {}

        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor) or v.ndim < 2:
                continue

            if k.endswith(".lora_B.weight"):
                module = k[: -len(".lora_B.weight")]
                per_module_rank[module] = int(v.shape[1])
            elif k.endswith(".lora_embedding_B"):
                # Expected (embed_dim, r)
                module = k[: -len(".lora_embedding_B")]
                per_module_rank[module] = int(v.shape[1])
            elif k.endswith(".lora_embedding_A"):
                # Expected (r, num_embeddings) – only use if we don't already have rank for this module
                module = k[: -len(".lora_embedding_A")]
                per_module_rank.setdefault(module, int(v.shape[0]))

        if not per_module_rank:
            return None

        r = collections.Counter(per_module_rank.values()).most_common(1)[0][0]
        rank_pattern = {m: rr for m, rr in per_module_rank.items() if rr != r}

        target_modules = sorted(
            {name.split(".lora")[0] for name in state_dict.keys() if ".lora" in name}
        )
        use_dora = any("lora_magnitude_vector" in k for k in state_dict.keys())
        lora_bias = any(
            ("lora_B" in k and k.endswith(".bias")) for k in state_dict.keys()
        )

        return {
            "r": int(r),
            "lora_alpha": int(r),
            "rank_pattern": rank_pattern,
            "alpha_pattern": {},
            "target_modules": target_modules,
            "use_dora": use_dora,
            "lora_bias": lora_bias,
        }

    def load_into(
        self,
        model: Union[torch.nn.Module, PeftAdapterMixin],
        loras: List[Union[str, LoraItem, Tuple[Union[str, LoraItem], float]]],
        adapter_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
    ) -> List[LoraItem]:
        """
        Load multiple LoRAs into a PEFT-enabled model. Supports per-adapter scaling.
        - loras can be strings (sources), LoraItem, or tuples of (source|LoraItem, scale)
        - adapter_names optionally overrides adapter naming
        - scales optionally overrides per-adapter scale values
        Returns resolved LoraItem objects in load order.
        """

        if not hasattr(model, "set_adapters"):
            raise ValueError(
                "Model doesn't support PEFT/LoRA. Ensure transformer inherits PeftAdapterMixin."
            )

        # verify the state dict is correct or convert it to the correct format

        resolved: List[LoraItem] = []
        final_names: List[str] = []
        final_scales: List[float] = []
        loaded_resolved: List[LoraItem] = []
        for idx, entry in enumerate(loras):
            scale: float = 1.0
            if isinstance(entry, tuple):
                src_or_item, scale = entry
            else:
                src_or_item = entry

            if isinstance(src_or_item, LoraItem):
                item = src_or_item
            else:
                item = self.resolve(str(src_or_item))

            # override from global scales list if provided
            if scales is not None and idx < len(scales) and scales[idx] is not None:
                scale = float(scales[idx])
            item.scale = float(scale)
            resolved.append(item)

        if False:
            composed_lora = []
            for i, item in enumerate(resolved):
                for local_path in item.local_paths:
                    composed_lora.append((local_path, item.scale))
            composed_lora = compose_lora(composed_lora)
            model.update_lora_params(composed_lora)

        else:
            for i, item in enumerate(resolved):
                adapter_name = (
                    adapter_names[i]
                    if adapter_names and i < len(adapter_names) and adapter_names[i]
                    else item.name or f"lora_{self._hash(item.source)}"
                )
                final_names.append(adapter_name)
                final_scales.append(item.scale)
                # diffusers supports str or dict mapping for multiple files; we load one-by-one if multiple

                for local_path in item.local_paths:
                    class_name = model.__class__.__name__
                    local_path_state_dict = self.maybe_convert_state_dict(
                        local_path, class_name
                    )

                    local_path_state_dict = strip_common_prefix(
                        local_path_state_dict, model.state_dict()
                    )

                    # Normalize keys that include an embedded adapter name, e.g.:
                    # "vace_blocks.0.attn2.to_k.lora_B.default.weight"
                    # becomes "vace_blocks.0.attn2.to_k.lora_B.weight"

                    local_path_state_dict = self._strip_adapter_name_from_keys(
                        local_path_state_dict
                    )

                    # Embedding LoRA uses lora_embedding_* keys (no ".weight") and swaps A/B roles.
                    local_path_state_dict = remap_embedding_lora_keys(
                        local_path_state_dict, model  # type: ignore[arg-type]
                    )

                    keys = list(local_path_state_dict.keys())
                    prefix = self._get_prefix_key(keys)
                    # ensure adapter name is not too long and does not have . or / in it if so remove it
                    adapter_name = self._clean_adapter_name(adapter_name)
                    # convert to dtype of the model
                    model_dtype = next(model.parameters()).dtype

                    metadata = self._build_lora_config_metadata_from_state_dict(
                        local_path_state_dict
                    )
                    if metadata is not None and prefix is not None:
                        # diffusers filters metadata keys by prefix and strips it, so prefix these keys to keep them.
                        metadata = {f"{prefix}.{k}": v for k, v in metadata.items()}

                    model.load_lora_adapter(
                        local_path_state_dict,
                        adapter_name=adapter_name,
                        prefix=prefix,
                        metadata=metadata,
                    )
                    logger.info(f"Loaded LoRA {adapter_name} from {local_path}")

            # Activate all adapters with their weights in one call
            try:
                model.set_adapters(final_names, weights=final_scales)
                loaded_resolved.extend(resolved)
            except Exception as e:
                # print the full stack trace
                import traceback

                traceback.print_exc()
                logger.warning(
                    f"Failed to activate adapters {final_names} with scales {final_scales}: {e}"
                )
                import traceback

                traceback.print_exc()

        return loaded_resolved

    def maybe_convert_state_dict(self, local_path: str, model_name: str):
        state_dict = self.load_file(local_path)
        converter = get_transformer_converter_by_model_name(model_name)
        lora_converter = LoraConverter()
        lora_converter.convert(state_dict)

        if converter is not None:
            converter.convert(state_dict)
        return state_dict

    def _format_to_extension(self, format: str) -> str:
        format = format.lower()
        if format == "safetensors" or format == "safetensor":
            return "safetensors"
        elif (
            format == "pickletensor"
            or format == "pickle"
            or format == "pt"
            or format == "pth"
        ):
            return "pt"
        else:
            return "safetensors"  # default to safetensors

    def _download_from_civitai_spec(
        self,
        spec: str,
        progress_callback: Optional[
            Callable[[int, Optional[int], Optional[str]], None]
        ] = None,
    ) -> str:
        """
        Support strings like:
          - "civitai:MODEL_ID" -> fetch model metadata, pick first LoRA SafeTensor file
          - "civitai-file:FILE_ID" -> download that specific file id
        Returns a local path (file) to the downloaded artifact.
        """
        import requests

        api_key = os.getenv("CIVITAI_API_KEY", None)
        headers = {"User-Agent": "apex-lora-manager/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def download_file_id(
            file_id: Union[int, str], format: Optional[str] = None
        ) -> str:
            url = f"https://civitai.com/api/download/models/{file_id}"
            url_params = {}
            if format is not None and format == "safetensors" or format == "pt":
                url_params["type"] = "Model"
                url_params["format"] = (
                    "SafeTensor"
                    if format == "safetensors" or format == "safetensor"
                    else "PickleTensor"
                )
            api_key = os.getenv("CIVITAI_API_KEY", None)
            if api_key:
                url_params["token"] = api_key
            url_params = urlencode(url_params)
            url = f"{url}?{url_params}"
            local_path = self.download_from_url(
                url,
                self.save_dir,
                progress_callback=progress_callback,
                filename=f"{file_id}.{format}" if format else None,
            )
            return local_path

        if spec.startswith("civitai-file:"):
            file_id = spec.split(":", 1)[1]
            return download_file_id(file_id, format)

        # civitai:MODEL_ID
        model_id = spec.split(":", 1)[1]
        meta_url = f"https://civitai.com/api/v1/models/{model_id}"
        # Metadata fetch is typically small; no need to wire byte-level progress here.
        resp = requests.get(meta_url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Find a file that looks like a LoRA in SafeTensor format
        format = None
        for version in data.get("modelVersions", []):
            file_id = version.get("id")
            for f in version.get("files", []):
                fname = f.get("name") or ""
                format = f.get("metadata", {}).get("format", "").lower()
                format = self._format_to_extension(format)
                if (
                    fname.lower().endswith((".safetensors", ".pt", ".bin"))
                    and file_id is not None
                ):
                    return download_file_id(file_id, format)

        raise RuntimeError(f"No downloadable files found for CivitAI model {model_id}")

    def load_file(self, local_path: str) -> Dict[str, torch.Tensor]:
        if local_path.endswith(".safetensors"):
            return load_file(local_path)
        else:
            torch.load(local_path)

    def _strip_adapter_name_from_keys(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Some PEFT exports include the adapter name in the key, e.g.:
          "vace_blocks.0.attn2.to_k.lora_B.default.weight"
        where "default" (or any other string) is the adapter name.
        This method removes that adapter-name segment so the key becomes:
          "vace_blocks.0.attn2.to_k.lora_B.weight".
        """
        new_state: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            parts = key.split(".")
            # Look for the specific pattern: ... lora_A|lora_B.<adapter_name>.weight|bias|alpha
            if (
                len(parts) >= 3
                and parts[-3] in ("lora_A", "lora_B")
                and parts[-1] in ("weight", "bias", "alpha")
                and parts[-2] not in ("lora_A", "lora_B")
            ):
                # Drop the adapter name (the penultimate component)
                parts.pop(-2)
                key = ".".join(parts)
            new_state[key] = value

        del state_dict
        return new_state
