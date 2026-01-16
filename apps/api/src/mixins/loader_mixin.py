from __future__ import annotations

from src.utils.defaults import DEFAULT_HEADERS
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any
import math
import os
import json
from typing import Callable
from logging import Logger
from src.utils.module import find_class_recursive
import importlib
import inspect
from loguru import logger
from src.utils.yaml import LoaderWithInclude
from src.manifest.loader import validate_and_normalize
from io import BytesIO
from typing import List
import tempfile
from glob import glob
from src.mixins.download_mixin import DownloadMixin
# Import pretrained config from transformers
from src.utils.defaults import DEFAULT_CONFIG_SAVE_PATH
from src.types import InputImage, InputVideo, InputAudio
import types
import numpy as np

class _LazyModule(types.ModuleType):
    """
    Minimal lazy-import proxy for heavyweight third-party modules.
    We use this to keep importing `LoaderMixin` fast in contexts like setup/boot.
    """

    def __init__(self, module_name: str):
        super().__init__(module_name)
        self.__dict__["_module_name"] = module_name
        self.__dict__["_loaded"] = None

    def _load(self):
        loaded = self.__dict__.get("_loaded")
        if loaded is not None:
            return loaded
        mod = importlib.import_module(self.__dict__["_module_name"])
        self.__dict__["_loaded"] = mod
        return mod

    def __getattr__(self, item: str):
        return getattr(self._load(), item)


def _lazy_import(module_name: str):
    return _LazyModule(module_name)


# Heavy deps are resolved lazily on first use.
requests = _lazy_import("requests")
yaml = _lazy_import("yaml")
torch = _lazy_import("torch")
np = _lazy_import("numpy")
PIL = _lazy_import("PIL")
Image = _lazy_import("PIL.Image")
cv2 = _lazy_import("cv2")
mx = _lazy_import("mlx.core")
librosa = _lazy_import("librosa")
gguf = _lazy_import("gguf")
pydash = _lazy_import("pydash")


def tqdm(*args, **kwargs):
    return _lazy_import("tqdm").tqdm(*args, **kwargs)


def init_empty_weights():
    # used as: `with init_empty_weights():`
    return _lazy_import("accelerate").init_empty_weights()


def _PreTrainedModel():
    from transformers.modeling_utils import PreTrainedModel

    return PreTrainedModel


def _PretrainedConfig():
    from transformers.configuration_utils import PretrainedConfig

    return PretrainedConfig

IMAGE_EXTS = [
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "ico",
    "webp",
    "avif",
]
VIDEO_EXTS = [
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm",
    "flv",
    "wmv",
    "mpg",
    "mpeg",
    "m4v",
]


class LoaderMixin(DownloadMixin):
    logger: Logger = logger

    def fetch_config(
        self,
        config_path: str,
        config_save_path: str = DEFAULT_CONFIG_SAVE_PATH,
        return_path: bool = False,
    ):
        path = self._download(config_path, config_save_path)
        if return_path:
            return path
        else:
            return self._load_config_file(path)

    def _load_model(
        self,
        component: Dict[str, Any],
        getter_fn: Callable | None = None,
        module_name: str = "diffusers",
        load_dtype: torch.dtype | mx.Dtype | None = None,
        load_device: str = "cpu",
        no_weights: bool = False,
        key_map: Dict[str, str] | None = None,
        extra_kwargs: Dict[str, Any] | None = None,
    ):

        if not self.logger:
            self.logger = logger

        if extra_kwargs is None:
            extra_kwargs = {}

        # Detect if memory management is configured for this component on the owning engine.
        # We intentionally keep this lazy and optional so that LoaderMixin can be used
        # independently of BaseEngine. Only enable for PyTorch engines, since group offloading
        # relies on torch.nn.Module hooks provided by diffusers.
        mm_config = None
        engine_type = getattr(self, "engine_type", "torch")
        if engine_type == "torch":
            resolve_mm_cfg = getattr(self, "_resolve_memory_config_for_component", None)
            if callable(resolve_mm_cfg):
                try:
                    mm_config = resolve_mm_cfg(component)
                except Exception:
                    mm_config = None
                    
        

        model_base = component.get("base")
        model_path = component.get("model_path")
        
        

        if mm_config is not None:
            # Should be cpu often times since the model is loaded on the cpu
            load_device = "cpu"


        if getter_fn:
            model_class = getter_fn(model_base)
        else:
            model_class = find_class_recursive(
                importlib.import_module(module_name), model_base
            )
        if model_class is None:
            raise ValueError(f"Model class for base '{model_base}' not found")
        config_path = component.get("config_path")
        config = {}

        if "nunchaku" in model_base:
            return model_class.from_pretrained(model_path, torch_dtype=load_dtype)

        if config_path:
            pydash.merge(config, self.fetch_config(config_path))

        if component.get("config"):
            pydash.merge(config, component.get("config"))

        # Lazy import here as well to avoid circular imports.
        from src.converters.convert import (
            get_transformer_converter,
            get_vae_converter,
            NoOpConverter,
            get_text_encoder_converter,
        )

        # Decide which converter (if any) is needed for this component.
        if component.get("type") == "vae":
            converter = get_vae_converter(model_base)
        elif component.get("type") == "transformer":
            converter = get_transformer_converter(model_base)
        elif component.get("type") == "text_encoder":
            converter = get_text_encoder_converter(model_base)
        else:
            converter = NoOpConverter()
            
        
        if os.path.isdir(model_path) and not config_path:
            # look for a config.json file
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                pydash.merge(config, self._load_json(config_path))

        with init_empty_weights():
            # Check the constructor signature to determine what it expects
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.values())

            # Skip 'self' parameter
            if params and params[0].name == "self":
                params = params[1:]

            # Check if the first parameter expects a PretrainedConfig object
            expects_pretrained_config = False
            if params:
                first_param = params[0]
                PretrainedConfig = _PretrainedConfig()
                PreTrainedModel = _PreTrainedModel()
                if (
                    first_param.annotation == PretrainedConfig
                    or (
                        hasattr(first_param.annotation, "__name__")
                        and "Config" in first_param.annotation.__name__
                    )
                    or first_param.name in ["config"]
                    and issubclass(model_class, PreTrainedModel)
                ):
                    expects_pretrained_config = True

            if expects_pretrained_config:
                # Use the model's specific config class if available, otherwise fall back to PretrainedConfig
                config_class = getattr(model_class, "config_class", PretrainedConfig)
                conf = config_class(**config)
                if hasattr(model_class, "_from_config"):
                    model = model_class._from_config(conf, **extra_kwargs)
                else:
                    model = model_class.from_config(conf, **extra_kwargs)
            else:
                if hasattr(model_class, "_from_config"):
                    model = model_class._from_config(config, **extra_kwargs)
                else:
                    model = model_class.from_config(config, **extra_kwargs)

        if no_weights:
            return model
        
        model_keys = list(model.state_dict().keys())

        files_to_load = []
        if os.path.isdir(model_path):
            extensions = component.get(
                "extensions", ["safetensors", "bin", "pt", "ckpt", "gguf", "pth"]
            )
            if "gguf" not in extensions:
                extensions = list(extensions) + ["gguf"]
            self.logger.info(f"Loading model from {model_path}")
            for ext in extensions:
                files_to_load.extend(glob(os.path.join(model_path, f"*.{ext}")))
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            extensions = component.get(
                "extensions", ["safetensors", "bin", "pt", "ckpt", "gguf", "pth"]
            )
            if "gguf" not in extensions:
                extensions = list(extensions) + ["gguf"]
            # ensure model_path ends with one of the extensions
            if any(model_path.endswith(ext) for ext in extensions):
                files_to_load = [model_path]
        extra_model_paths = component.get("extra_model_paths", [])

        if isinstance(extra_model_paths, str):
            extra_model_paths = [extra_model_paths]
        if extra_kwargs.get("load_extra_model_paths", True):
            files_to_load.extend(extra_model_paths)
        # Track whether we've already patched this model for FP-scaled weights

        patched_for_fpscaled = False
        gguf_kwargs = component.get("gguf_kwargs", {})
        for file_path in tqdm(
            files_to_load, desc="Loading weights", total=len(files_to_load)
        ):
            if str(file_path).endswith(".gguf"):
                # GGUF follows the same "files_to_load" pathway as other weight files.
                if hasattr(self, "engine_type") and self.engine_type == "mlx":
                    # Can load gguf directly into mlx model; no need to convert.
                    from src.utils.mlx import check_mlx_convolutional_weights

                    gguf_weights = mx.load(file_path)
                    check_mlx_convolutional_weights(gguf_weights, model)
                    model.load_weights(gguf_weights)
                    continue

                logger.info(f"Loading GGUF model from {file_path}")
                from src.quantize.ggml_layer import (
                    patch_model_from_state_dict as patch_model_ggml_from_state_dict,
                )
                from src.quantize.load import load_gguf

                state_dict, _ = load_gguf(
                    file_path,
                    type=component.get("type"),
                    dequant_dtype=load_dtype,
                    device=load_device,
                    **gguf_kwargs,
                )
                converter.convert(state_dict, model_keys)
                # Load GGMLTensors without replacing nn.Parameters by copying data
                patch_model_ggml_from_state_dict(
                    model,
                    state_dict,
                    default_dequant_dtype=load_dtype,
                )

                for key, value in state_dict.items():
                    if load_dtype:
                        if getattr(value, "tensor_type", None) in {
                            gguf.GGMLQuantizationType.F32,
                            gguf.GGMLQuantizationType.F16,
                        }:
                            state_dict[key] = value.to(load_dtype)
                            
               
                model.load_state_dict(state_dict, assign=True, strict=False)
                continue

            from src.utils.safetensors import load_safetensors

            file_path_str = str(file_path)
            is_safetensors = file_path_str.lower().endswith(".safetensors")
            if is_safetensors:
                state_dict = load_safetensors(
                    file_path,
                    device=load_device,
                    dtype=load_dtype,
                    framework=(
                        "np"
                        if hasattr(self, "engine_type") and self.engine_type == "mlx"
                        else "pt"
                    ),
                )
            else:
                state_dict = torch.load(
                    file_path, map_location=load_device, weights_only=True, mmap=True
                )

            # remap keys if key_map is provided replace part of existing key with new key
            if key_map:
                new_state_dict = {}
                for k, v in key_map.items():
                    for k2, v2 in state_dict.items():
                        if k in k2:
                            new_state_dict[k2.replace(k, v)] = v2
                        else:
                            new_state_dict[k2] = v2
                state_dict = new_state_dict

            converter.convert(state_dict, model_keys)
            if load_dtype and not is_safetensors:
                for k, v in state_dict.items():
                    state_dict[k] = v.to(load_dtype)
            # Detect FP-scaled checkpoints (e.g., Wan2.2 FP e4m3fn scaled)
            # and patch the model with FPScaled* layers *before* loading
            # the state dict. We only do this once per model.
            if hasattr(self, "engine_type") and self.engine_type == "mlx":
                from src.utils.mlx import check_mlx_convolutional_weights

                check_mlx_convolutional_weights(state_dict, model)
            if (
                not patched_for_fpscaled
                and getattr(self, "engine_type", "torch") == "torch"
                and isinstance(state_dict, dict)
                and (
                    any(k.endswith("scale_weight") for k in state_dict.keys())
                    or any(
                        (v.dtype == torch.float8_e4m3fn or v.dtype == torch.float8_e5m2)
                        for v in state_dict.values()
                    )
                )
            ):
                from src.quantize.scaled_layer import (
                    patch_fpscaled_model_from_state_dict,
                )

                # Prefer the explicit load_dtype (if it's a torch dtype)
                # as the compute dtype for FP dequantization; otherwise
                # let the scaled layers infer it from their inputs.
                default_compute_dtype = (
                    load_dtype if isinstance(load_dtype, torch.dtype) else None
                )
                self.logger.info(
                    "Detected FP-scaled checkpoint (found '*.scale_weight' "
                    "keys). Patching model with FPScaled layers."
                )
                patch_fpscaled_model_from_state_dict(
                    model,
                    state_dict,
                    default_compute_dtype=default_compute_dtype,
                )
                # Mark the model so we can treat leftover meta scale_weight
                # parameters more leniently in the post-load meta check, and
                # stash the default compute dtype so we can restore FP
                # parameter subclasses after loading.
                setattr(model, "_patched_for_fpscaled", True)
                setattr(
                    model,
                    "_fpscaled_default_compute_dtype",
                    default_compute_dtype,
                )
                patched_for_fpscaled = True
            if hasattr(model, "load_state_dict"):

                model.load_state_dict(
                    state_dict, strict=False, assign=True
                )  # must be false as we are iteratively loading the state dict
            elif hasattr(model, "load_weights"):
                model.load_weights(state_dict, strict=False)
            else:
                raise ValueError(
                    f"Model {model} does not have a load_state_dict or load_weights method"
                )

        if getattr(self, "engine_type", "torch") == "torch":
            has_meta_params = False
            patched_for_fpscaled = getattr(model, "_patched_for_fpscaled", False)

            # Some HF causal LMs may legally omit `lm_head.*` from checkpoints when
            # weights are tied to input embeddings. When we build the module under
            # `init_empty_weights()`, those params start on the meta device and will
            # remain meta if the state dict didn't contain them. Before treating
            # meta params as a hard error, attempt to materialize/tie lm_head.
            PreTrainedModel = _PreTrainedModel()
            if isinstance(model, PreTrainedModel):
                try:
                    out_emb = (
                        model.get_output_embeddings()
                        if hasattr(model, "get_output_embeddings")
                        else None
                    )
                    needs_out_emb_fix = (
                        out_emb is not None
                        and hasattr(out_emb, "weight")
                        and getattr(out_emb.weight, "device", None) is not None
                        and out_emb.weight.device.type == "meta"
                    )
                    if needs_out_emb_fix:
                        # First try HF's standard tying logic.
                        if hasattr(model, "tie_weights") and callable(
                            getattr(model, "tie_weights")
                        ):
                            try:
                                model.tie_weights()
                            except Exception:
                                pass

                        # Re-fetch after tie attempt.
                        out_emb = (
                            model.get_output_embeddings()
                            if hasattr(model, "get_output_embeddings")
                            else out_emb
                        )
                        still_meta = (
                            out_emb is not None
                            and hasattr(out_emb, "weight")
                            and getattr(out_emb.weight, "device", None) is not None
                            and out_emb.weight.device.type == "meta"
                        )

                        if still_meta:
                            in_emb = (
                                model.get_input_embeddings()
                                if hasattr(model, "get_input_embeddings")
                                else None
                            )
                            # If input embeddings are real, tie output weight to them.
                            if (
                                in_emb is not None
                                and hasattr(in_emb, "weight")
                                and getattr(in_emb.weight, "device", None) is not None
                                and in_emb.weight.device.type != "meta"
                            ):
                                try:
                                    out_emb.weight = in_emb.weight
                                except Exception:
                                    pass

                        # If it is *still* meta, allocate parameters so downstream
                        # code can run (e.g., some checkpoints truly omit lm_head).
                        out_emb = (
                            model.get_output_embeddings()
                            if hasattr(model, "get_output_embeddings")
                            else out_emb
                        )
                        if (
                            out_emb is not None
                            and hasattr(out_emb, "weight")
                            and getattr(out_emb.weight, "device", None) is not None
                            and out_emb.weight.device.type == "meta"
                        ):
                            import torch.nn as nn

                            init_range = getattr(
                                getattr(model, "config", None),
                                "initializer_range",
                                0.02,
                            )
                            param_dtype = (
                                load_dtype
                                if isinstance(load_dtype, torch.dtype)
                                else torch.float32
                            )
                            w = torch.empty(
                                tuple(out_emb.weight.shape),
                                device=load_device,
                                dtype=param_dtype,
                            )
                            # Match HF Linear default init: normal_(0, initializer_range)
                            nn.init.normal_(w, mean=0.0, std=float(init_range))
                            out_emb.weight = nn.Parameter(w, requires_grad=False)

                            # Bias is uncommon for lm_head, but handle it if present.
                            if hasattr(out_emb, "bias") and out_emb.bias is not None:
                                b = out_emb.bias
                                if (
                                    getattr(b, "device", None) is not None
                                    and b.device.type == "meta"
                                ):
                                    out_emb.bias = nn.Parameter(
                                        torch.zeros(
                                            tuple(b.shape),
                                            device=load_device,
                                            dtype=param_dtype,
                                        ),
                                        requires_grad=False,
                                    )
                except Exception as e:
                    # Never fail loading solely due to this best-effort fix; the
                    # meta-param check below will still catch real issues.
                    if hasattr(self, "logger"):
                        self.logger.debug(f"lm_head meta fix skipped: {e}")

            # If this is an FP-scaled model, re-wrap any FP weights as
            # FPScaledParameter after loading, since some load_state_dict
            # code paths may strip custom Parameter subclasses.
            if patched_for_fpscaled:
                from src.quantize.scaled_layer import restore_fpscaled_parameters

                default_compute_dtype = getattr(
                    model, "_fpscaled_default_compute_dtype", None
                )
                restore_fpscaled_parameters(
                    model, default_compute_dtype=default_compute_dtype
                )

            # Final FP8 sanity pass: if we ended up with raw `nn.Linear` modules
            # whose weights are stored in FP8 dtypes, swap them to our FPScaledLinear
            # implementation so forward passes remain stable (it casts/dequantizes
            # weights to a compute dtype and avoids relying on float8 matmul support).
            #
            # This is intentionally cheap (type checks + dtype inspection) and only
            # runs when weights are loaded.
 
       
            if not no_weights:
                try:
                    import torch.nn as nn

                    from src.quantize.scaled_layer import FPScaledLinear, FPScaledLayer

                    fp8_dtypes = tuple(
                        dt
                        for dt in (
                            getattr(torch, "float8_e4m3fn", None),
                            getattr(torch, "float8_e5m2", None),
                            # Newer PyTorch builds may include *_fnuz variants
                            getattr(torch, "float8_e4m3fnuz", None),
                            getattr(torch, "float8_e5m2fnuz", None),
                        )
                        if dt is not None
                    )

                    def _physical_dtype(p: torch.nn.Parameter) -> torch.dtype:
                        # FPScaledParameter exposes true storage dtype via `.physical_dtype`.
                        try:
                            return getattr(p, "physical_dtype")  # type: ignore[return-value]
                        except Exception:
                            return p.dtype

                    converted = 0
                    stack = [("", model)]
                    default_compute_dtype = getattr(
                        model, "_fpscaled_default_compute_dtype", None
                    )
                    while stack:
                        prefix, mod = stack.pop()
                        for child_name, child in list(mod._modules.items()):
                            qname = f"{prefix}{child_name}"
                            if child is None:
                                continue
                            # Recurse first so we can safely replace leaf modules.
                            stack.append((f"{qname}.", child))

                            if isinstance(child, nn.Linear) and not isinstance(
                                child, FPScaledLayer
                            ):
                                w = getattr(child, "weight", None)
                                if w is None:
                                    continue
                                try:
                                    phys = _physical_dtype(w)
                                except Exception:
                                    phys = w.dtype
                                if fp8_dtypes and phys in fp8_dtypes:
                                    new_mod = FPScaledLinear(
                                        child.in_features,
                                        child.out_features,
                                        bias=child.bias is not None,
                                        compute_dtype=default_compute_dtype,
                                        device=child.weight.device,
                                        dtype=child.weight.dtype,
                                    )
                                    with torch.no_grad():
                                        new_mod.weight.copy_(child.weight)
                                        if child.bias is not None:
                                            new_mod.bias.copy_(child.bias)
                                        # Preserve scale_weight if present on the original module
                                        if hasattr(child, "scale_weight") and hasattr(
                                            new_mod, "scale_weight"
                                        ):
                                            try:
                                                new_mod.scale_weight.copy_(
                                                    child.scale_weight
                                                )
                                            except Exception:
                                                pass
                                    mod._modules[child_name] = new_mod
                                    converted += 1

                    if converted and hasattr(self, "logger"):
                        self.logger.info(
                            f"Converted {converted} Linear layers to FPScaledLinear based on FP8 weight dtypes."
                        )
                except Exception as e:
                    # Best-effort only; never fail loading due to a final FP8 fixup.
                    if hasattr(self, "logger"):
                        self.logger.debug(f"FP8 final linear check skipped: {e}")

            for name, param in model.named_parameters():
                if param.device.type == "meta":
                    # If this is an FP-scaled model and the offending parameter
                    # is a residual `scale_weight` that never got real weights
                    # loaded, we can safely drop it instead of erroring out.
                    if patched_for_fpscaled and name.endswith("scale_weight"):
                        self.logger.warning(
                            f"Dropping unused meta-device scale_weight parameter '{name}' "
                            "on FP-scaled model."
                        )
                        # Remove the parameter from the owning module so it no
                        # longer appears as a meta device parameter.
                        module_name, _, param_name = name.rpartition(".")
                        owner = (
                            model.get_submodule(module_name) if module_name else model
                        )
                        owner.register_parameter(param_name, None)
                        continue

                    # For all other parameters on meta, this is still an error.
                    self.logger.error(f"Parameter {name} is on meta device")
                    has_meta_params = True

            if has_meta_params:
                raise ValueError(
                    "Model has parameters on meta device, this is not supported"
                )

        if (
            mm_config is not None
            and not no_weights
            and component.get("type") != "transformer"
            and component.get("type") != "text_encoder"
        ):
            apply_group_offloading = getattr(self, "_apply_group_offloading", None)
            if callable(apply_group_offloading):
                label = (
                    component.get("name")
                    or component.get("type")
                    or type(model).__name__
                )
                offloading_module = component.get("offloading_module", None)
                ignore_offloading_modules = component.get("ignore_offloading_modules", None)
                block_modules = component.get("block_modules", None)
                mm_config.ignore_modules = ignore_offloading_modules
                mm_config.block_modules = block_modules
                if offloading_module:
                    model_to_offload = model.get_submodule(offloading_module)
                else:
                    model_to_offload = model
                try:
                    apply_group_offloading(
                        model_to_offload, mm_config, module_label=label
                    )
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"Failed to enable group offloading for '{label}': {e}"
                        )

        # Optionally compile the fully initialized module according to config.
        if (
            not no_weights
            and component.get("type") != "transformer"
            and component.get("type") != "text_encoder"
        ):
            maybe_compile = getattr(self, "_maybe_compile_module", None)
            if callable(maybe_compile):
                model = maybe_compile(model, component)

        return model.eval()

    def _load_config_file(self, file_path: str | Path):
        try:
            return self._load_json(file_path)
        except json.JSONDecodeError:
            return self._load_yaml(file_path)
        except Exception:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_json(self, file_path: str | Path):
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_yaml(self, file_path: str | Path):
        file_path = Path(file_path)
        text = file_path.read_text()

        # --- PASS 1: extract `shared:` (legacy) or `spec.shared` (v1) with a loader that skips !include tags ---
        prelim = yaml.load(text, Loader=yaml.FullLoader)
        # Collect shared entries from both legacy and v1 shapes
        shared_entries = []
        if isinstance(prelim, dict):
            shared_entries.extend(prelim.get("shared", []) or [])
            spec = prelim.get("spec", {}) or {}
            if isinstance(spec, dict):
                shared_entries.extend(spec.get("shared", []) or [])

        # build alias → manifest Path
        shared_manifests = {}
        for entry in shared_entries:
            p = (file_path.parent / entry).resolve()
            # assume e.g. "shared_wan.yml" → alias "wan"
            try:
                alias = p.stem.split("_", 1)[1]
            except Exception:
                alias = p.stem
            shared_manifests[alias] = p

        # attach it to our custom loader
        LoaderWithInclude.shared_manifests = shared_manifests

        # --- PASS 2: real load with !include expansion ---
        loaded = yaml.load(text, Loader=LoaderWithInclude)

        # Validate and normalize if this is a v1 manifest
        try:
            loaded = validate_and_normalize(loaded)
        except Exception as e:
            raise

        return loaded

    def _load_scheduler(self, component: Dict[str, Any]) -> Any:

        if "scheduler_options" in component:

            component_base = component["scheduler_options"][0].get("base")
            if not component_base:
                raise ValueError("Scheduler component base not specified.")
        else:
            component_base = component.get("base")
            if not component_base:
                raise ValueError("Scheduler component base not specified.")

        component_split = component_base.split(".")
        if len(component_split) > 1:
            module_name = ".".join(component_split[:-1])
            class_name = component_split[-1]
        else:
            module_name = "diffusers"
            class_name = component_base

        try:
            base_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(
                f"Could not import the base module '{module_name}' from component '{component_base}'"
            )

        component_class = find_class_recursive(base_module, class_name)

        if component_class is None:
            raise ValueError(
                f"Could not find scheduler class '{class_name}' in module '{module_name}' or its submodules."
            )

        config_path = component.get("config_path")
        config = component.get("config")
        if config_path and config:
            fetched_config = self.fetch_config(config_path)
            config = {**fetched_config, **config}
        elif config_path:
            config = self.fetch_config(config_path)
        else:
            config = component.get("config", {})

        # Determine which config entries can be passed to the component constructor
        try:
            init_signature = inspect.signature(component_class.__init__)
            init_params = list(init_signature.parameters.values())
            if init_params and init_params[0].name == "self":
                init_params = init_params[1:]
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params
            )
            init_param_names = {p.name for p in init_params}
        except (TypeError, ValueError):
            # Fallback if signature introspection fails
            accepts_var_kwargs = True
            init_param_names = set()

        if accepts_var_kwargs:
            init_kwargs = dict(config)
            config_to_register = {}
        else:
            init_kwargs = {
                k: v for k, v in (config or {}).items() if k in init_param_names
            }
            config_to_register = {
                k: v for k, v in (config or {}).items() if k not in init_param_names
            }

        component = component_class(**init_kwargs)

        # Register remaining config to the component if supported
        if (
            config_to_register
            and hasattr(component, "register_to_config")
            and callable(getattr(component, "register_to_config"))
        ):
            try:
                component.register_to_config(**config_to_register)
            except Exception as e:
                self.logger.warning(
                    f"Failed to register extra config for {component_class}: {e}"
                )

        return component

    def save_component(
        self,
        component: Any,
        model_path: str,
        component_type: str,
        **save_kwargs: Dict[str, Any],
    ):
        try:
            from diffusers import ModelMixin  # type: ignore
        except Exception as e:
            raise ValueError(
                "Saving components requires Diffusers to be installed."
            ) from e

        if component_type == "transformer":
            if issubclass(type(component), ModelMixin):
                component.save_pretrained(model_path, **save_kwargs)
            else:
                raise ValueError(f"Unsupported component type: {component_type}")
        elif component_type == "vae":
            if issubclass(type(component), ModelMixin):
                component.save_pretrained(model_path, **save_kwargs)
            else:
                raise ValueError(f"Unsupported component type: {component_type}")
        else:
            raise ValueError(f"Unsupported component type: {component_type}")

    def _load_image(
        self,
        image: "InputImage",
        convert_method: Callable[[Image.Image], Image.Image] | None = None,
    ) -> Image.Image:
        if isinstance(image, Image.Image):
            out_image = image
        elif isinstance(image, str):
            if self._is_url(image):
                out_image = Image.open(
                    BytesIO(
                        requests.get(image, timeout=10, headers=DEFAULT_HEADERS).content
                    )
                )
            else:
                out_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            out_image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            out_image = Image.fromarray(image.numpy())
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

        out_image = PIL.ImageOps.exif_transpose(out_image)
        if convert_method is not None:
            out_image = convert_method(out_image)
        else:
            out_image = out_image.convert("RGB")
        return out_image

    def _load_video(
        self,
        video_input: "InputVideo",
        fps: int | None = None,
        num_frames: int | None = None,
        reverse: bool = False,
        return_fps: bool = False,
        convert_method: Callable[[Image.Image], Image.Image] | None = None,
    ) -> List[Image.Image]:

        def _finalize_frames(frames, inferred_fps=None):
            # Ensure we are always working with a concrete list
            frames = list(frames)

            # Apply temporal direction first so num_frames semantics are
            # always with respect to the final playback order.
            if reverse:
                frames = list(reversed(frames))

            if num_frames is not None and len(frames) > 0:
                current_len = len(frames)

                if num_frames < current_len:
                    # If we need fewer frames than we have, simply truncate.
                    frames = frames[:num_frames]
                elif num_frames > current_len:
                    # If we need more frames than we have, resample by
                    # evenly duplicating frames across the sequence so
                    # motion still appears smooth.
                    indices = np.linspace(0, current_len - 1, num_frames)
                    frames = [frames[int(round(idx))] for idx in indices]

            if return_fps:
                return frames, inferred_fps
            return frames

        if isinstance(video_input, List):
            out_frames = []
            for v in video_input:
                out_frames.append(self._load_image(v, convert_method=convert_method))
            return _finalize_frames(out_frames, fps)

        if isinstance(video_input, str):
            video_path = video_input
            tmp_file_path = None

            if self._is_url(video_input):
                try:
                    response = requests.get(
                        video_input, timeout=10, headers=DEFAULT_HEADERS
                    )
                    response.raise_for_status()
                    contents = response.content
                    suffix = Path(urlparse(video_input).path).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp_file:
                        tmp_file.write(contents)
                        tmp_file_path = tmp_file.name
                    video_path = tmp_file_path
                except requests.RequestException as e:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                    raise IOError(f"Failed to download video from {video_input}") from e
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {video_path}")

                original_fps = cap.get(cv2.CAP_PROP_FPS)
                requested_fps = None
                if fps is not None:
                    requested_fps = abs(fps) if fps < 0 else fps

                frames = []
                frame_count = 0

                if fps is None:
                    # No fps specified, extract all frames
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Check if frame is grayscale or color
                        if len(frame.shape) == 2 or (
                            len(frame.shape) == 3 and frame.shape[2] == 1
                        ):
                            # Grayscale frame, no color conversion needed
                            frame_rgb = frame
                        else:
                            # Color frame, convert from BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(
                            convert_method(Image.fromarray(frame_rgb))
                            if convert_method
                            else Image.fromarray(frame_rgb)
                        )
                        frame_count += 1
                else:
                    # Extract frames at specified fps using *time-based* resampling.
                    # This correctly handles both:
                    # - downsampling (e.g. 24 -> 16 fps): skip frames
                    # - upsampling (e.g. 16 -> 24 fps): duplicate frames as needed
                    target_fps = requested_fps

                    if target_fps is None or target_fps == 0:
                        raise ValueError(f"Invalid target fps: {fps}")

                    # If OpenCV can't infer fps (0/NaN/inf), fall back to extracting all frames.
                    if (
                        original_fps is None
                        or not math.isfinite(float(original_fps))
                        or float(original_fps) <= 0
                    ):
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if len(frame.shape) == 2 or (
                                len(frame.shape) == 3 and frame.shape[2] == 1
                            ):
                                frame_rgb = frame
                            else:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(
                                convert_method(Image.fromarray(frame_rgb))
                                if convert_method
                                else Image.fromarray(frame_rgb)
                            )
                        return _finalize_frames(frames, target_fps)

                    orig_dt = 1.0 / float(original_fps)
                    target_dt = 1.0 / float(target_fps)
                    next_sample_t = 0.0

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Source frame covers [t0, t1) in seconds.
                        t1 = (frame_count + 1) * orig_dt

                        # Convert the frame once, then duplicate references as needed.
                        if len(frame.shape) == 2 or (
                            len(frame.shape) == 3 and frame.shape[2] == 1
                        ):
                            frame_rgb = frame
                        else:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = (
                            convert_method(Image.fromarray(frame_rgb))
                            if convert_method
                            else Image.fromarray(frame_rgb)
                        )

                        # Emit 0..N frames depending on how many target sample times
                        # fall within this source-frame interval.
                        while next_sample_t < (t1 + 1e-9):
                            frames.append(pil_frame)
                            next_sample_t += target_dt

                        frame_count += 1
                return _finalize_frames(
                    frames, requested_fps if requested_fps is not None else original_fps
                )
            finally:
                if "cap" in locals() and cap.isOpened():
                    cap.release()
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        if isinstance(video_input, np.ndarray):
            frames = [
                (
                    convert_method(Image.fromarray(frame))
                    if convert_method
                    else Image.fromarray(frame).convert("RGB")
                )
                for frame in video_input
            ]
            return _finalize_frames(frames, fps)

        if isinstance(video_input, torch.Tensor):
            tensor = video_input.cpu()
            if tensor.ndim == 5:
                if tensor.shape[1] == 3 or tensor.shape[1] == 1:
                    # Means shape is (B, C, F, H, W)
                    tensor = tensor.permute(0, 2, 1, 3, 4).squeeze(0)
                    frames = []
                elif tensor.shape[2] == 1 or tensor.shape[2] == 3:
                    # Means shape is (B, C, F, H, W)
                    tensor = tensor.squeeze(0)
                    frames = []
                else:
                    raise ValueError(f"Invalid tensor shape: {tensor.shape}")

                for frame in tensor:
                    frame = frame.permute(1, 2, 0).numpy()
                    # check if frame is between 0 and 1
                    if frame.mean() <= 1:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    # check if frame is grayscale if so then don't convert to RGB
                    if frame.shape[2] == 1:
                        frames.append(
                            convert_method(Image.fromarray(frame.squeeze(2)))
                            if convert_method
                            else Image.fromarray(frame.squeeze(2))
                        )
                    else:
                        frames.append(
                            convert_method(Image.fromarray(frame))
                            if convert_method
                            else Image.fromarray(frame).convert("RGB")
                        )
                return _finalize_frames(frames, fps)

            if tensor.ndim == 4 and (
                tensor.shape[1] == 3 or tensor.shape[1] == 1
            ):  # NCHW to NHWC
                tensor = tensor.permute(0, 2, 3, 1)

            numpy_array = tensor.numpy()
            if numpy_array.mean() <= 1:
                numpy_array = (numpy_array * 255).clip(0, 255).astype(np.uint8)

            frames = [
                (
                    convert_method(Image.fromarray(frame))
                    if convert_method
                    else (
                        Image.fromarray(frame).convert("RGB")
                        if frame.shape[2] == 3
                        else (
                            convert_method(Image.fromarray(frame))
                            if convert_method
                            else Image.fromarray(frame)
                        )
                    )
                )
                for frame in numpy_array
            ]
            return _finalize_frames(frames, fps)
        raise ValueError(f"Invalid video type: {type(video_input)}")

    def _load_audio(
        self,
        audio_input: "InputAudio",
        sample_rate: int = 16000,
        normalize: bool = True,
    ) -> np.ndarray:
        """Robustly load audio from various inputs and optionally normalize.

        Supports:
        - Local file paths (audio or video files)
        - Remote URLs (audio or video files, downloaded to a temporary file)
        - NumPy arrays
        - Torch tensors
        - Lists of the above (concatenated into a single 1D array)
        """

        # If we already have an array, just normalize/return it.
        if isinstance(audio_input, np.ndarray):
            audio_array = audio_input
            if normalize:
                audio_array = self._normalize_audio(audio_array, sample_rate)
            return audio_array

        # Handle torch tensors by converting to NumPy.
        if isinstance(audio_input, torch.Tensor):
            tensor = audio_input.detach().cpu()
            # Squeeze extra dimensions if needed (e.g., (1, N) -> (N,))
            if tensor.ndim > 1:
                tensor = tensor.squeeze()
            audio_array = tensor.numpy()
            if normalize:
                audio_array = self._normalize_audio(audio_array, sample_rate)
            return audio_array

        # Handle lists by loading each element and concatenating.
        if isinstance(audio_input, list):
            arrays = [
                self._load_audio(item, sample_rate=sample_rate, normalize=normalize)
                for item in audio_input
            ]
            # Ensure all items are 1D arrays for concatenation.
            arrays = [np.ravel(a) for a in arrays]
            return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

        # From here on we expect a string-like input (path or URL).
        if not isinstance(audio_input, str):
            raise ValueError(f"Invalid audio type: {type(audio_input)}")

        audio_path = audio_input
        tmp_file_path = None

        # If the input is a URL, download it to a temporary file first.
        if self._is_url(audio_input):
            try:
                response = requests.get(
                    audio_input,
                    timeout=10,
                    headers=DEFAULT_HEADERS,
                )
                response.raise_for_status()
                contents = response.content
                suffix = Path(urlparse(audio_input).path).suffix or ".wav"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp_file:
                    tmp_file.write(contents)
                    tmp_file_path = tmp_file.name
                audio_path = tmp_file_path
            except requests.RequestException as e:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                raise IOError(f"Failed to download audio from {audio_input}") from e

        try:
            # Check if it's a video file by extension.
            video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
            if any(audio_path.lower().endswith(ext) for ext in video_extensions):
                audio_array = self._extract_audio_from_video(audio_path, sample_rate)
            else:
                # Load audio file directly
                audio_array, sr = librosa.load(audio_path, sr=sample_rate)
                if normalize:
                    audio_array = self._normalize_audio(audio_array, sr)
            
            return audio_array
        finally:
            # Clean up any temporary file we created for remote inputs.
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def _extract_audio_from_video(
        self, video_path: str, sample_rate: int, normalize: bool = True
    ) -> np.ndarray:
        """Extract audio from video file."""
        import subprocess
        from src.utils.ffmpeg import get_ffmpeg_path

        # Create temporary file for extracted audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name

        try:
            # Use ffmpeg to extract audio
            ffmpeg_command = [
                get_ffmpeg_path(),
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                temp_audio_path,
            ]
            subprocess.run(ffmpeg_command, check=True, capture_output=True)

            # Load the extracted audio
            audio_array, sr = librosa.load(temp_audio_path, sr=sample_rate)
            if normalize:
                audio_array = self._normalize_audio(audio_array, sr)
            return audio_array

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def _normalize_audio(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio loudness."""
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio_array)
            if abs(loudness) < 100:  # Valid loudness measurement
                normalized_audio = pyln.normalize.loudness(audio_array, loudness, -23)
                return normalized_audio
        except ImportError:
            pass  # pyloudnorm not available, skip normalization

        return audio_array

    @staticmethod
    def get_media_type(media_path: str) -> str:
        media_path = media_path.lower()
        if media_path.endswith(tuple(VIDEO_EXTS)):
            return "video"
        elif media_path.endswith(tuple(IMAGE_EXTS)):
            return "image"
        else:
            raise ValueError(f"Invalid media type: {media_path}")
