import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import mlx.core as mx
from safetensors import safe_open
from tqdm import tqdm
from mlx.utils import tree_flatten


def _mx_dtype_from_str(dtype: Optional[str] | Optional[mx.Dtype]) -> Optional[mx.Dtype]:
    if isinstance(dtype, mx.Dtype):
        return dtype
    if dtype is None or dtype == "auto":
        return None
    d = dtype.lower()
    mapping = {
        "float16": mx.float16,
        "fp16": mx.float16,
        "half": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
    }
    if d not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[d]


def _flatten_leaf_arrays(
    obj: Any,
    prefix: str = "",
    names_filter: Optional[Set[str]] = None,
) -> Dict[str, mx.array]:
    """
    Fast flatten of leaf mx.array tensors in a nested object graph.

    - Iterative stack-based traversal
    - Only visits dicts, lists/tuples, and objects with __dict__
    - Skips dir() reflection to avoid exploring class descriptors
    - Tracks visited objects to prevent cycles
    - If names_filter is provided, only records keys present in the filter
    """
    result: Dict[str, mx.array] = {}
    visited: Set[int] = set()
    stack: List[tuple[Any, str]] = [(obj, prefix)]

    while stack:
        current, name = stack.pop()
        oid = id(current)
        if oid in visited:
            continue
        visited.add(oid)

        if isinstance(current, mx.array):
            key = name if name else ""
            if not names_filter or key in names_filter:
                result[key] = current
            continue

        if isinstance(current, (list, tuple)):
            for idx, item in enumerate(current):
                child_name = f"{name}.{idx}" if name else str(idx)
                stack.append((item, child_name))
            continue

        if isinstance(current, dict):
            for key, item in current.items():
                child_name = f"{name}.{key}" if name else str(key)
                stack.append((item, child_name))
            continue

        # Only traverse instance attributes to avoid heavy reflection
        if hasattr(current, "__dict__"):
            for attr, value in current.__dict__.items():
                if attr.startswith("_"):
                    continue
                if callable(value):
                    continue
                child_name = f"{name}.{attr}" if name else attr
                stack.append((value, child_name))

    return result


def _set_by_path(root: Any, key_path: str, value: mx.array) -> bool:
    parts = key_path.split(".")
    obj = root
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        # List index
        if part.isdigit():
            index = int(part)
            if not isinstance(obj, (list, tuple)):
                return False
            if index >= len(obj):
                return False
            if is_last:
                # Cannot assign to list element directly here if it's not an array
                if isinstance(obj, list) and isinstance(obj[index], mx.array):
                    obj[index] = value
                    return True
                return False
            obj = obj[index]
            continue
        # Attribute
        if not hasattr(obj, part):
            return False
        if is_last:
            try:
                setattr(obj, part, value)
                return True
            except Exception:
                return False
        obj = getattr(obj, part)
    return False


def _find_index_json(dir_path: Path) -> Optional[Path]:
    candidates = list(dir_path.glob("*.safetensors.index.json"))
    if not candidates:
        return None
    # Prefer model.safetensors.index.json if present
    for c in candidates:
        if c.name == "model.safetensors.index.json":
            return c
    return candidates[0]


class FromModelMixin:

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Path,
        *,
        dtype: Optional[str] = None,
        strict: bool = True,
        subfolder: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        base_path = Path(pretrained_model_path)
        if subfolder:
            base_path = base_path / subfolder
        if not base_path.exists():
            raise FileNotFoundError(f"Path not found: {base_path}")

        # If a single file is passed, delegate
        if base_path.is_file():
            return cls.from_single_file(
                base_path, dtype=dtype, config=config, strict=strict
            )

        # Load config
        if config is None:
            cfg_path = base_path / "config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"Missing config.json in {base_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        # Instantiate model
        model = cls(**config)

        # Locate index json and shards
        index_path = _find_index_json(base_path)
        shard_map: Dict[str, List[str]] = {}
        if index_path is not None:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map: Dict[str, str] = index_data.get("weight_map", {})
            for tensor_name, shard_name in weight_map.items():
                shard_map.setdefault(shard_name, []).append(tensor_name)
        else:
            # Fallback: scan all safetensors and load all keys
            for shard in sorted(base_path.glob("*.safetensors")):
                shard_map[shard.name] = []  # empty means load all keys

        target_dtype = _mx_dtype_from_str(dtype)

        # Collect model leaves for strict checking (use known names if index is present)
        names_filter: Optional[Set[str]] = None
        if index_path is not None:
            # Build a set of all tensor names we expect to assign for faster flattening
            all_names: Set[str] = set()
            for shard_names in shard_map.values():
                if shard_names:
                    all_names.update(shard_names)
            # If weight_map is empty (None entries), we don't know names -> skip filter
            names_filter = all_names if len(all_names) > 0 else None

        leaves = _flatten_leaf_arrays(model, names_filter=names_filter)
        seen: set[str] = set()

        keep_in_fp32_modules = getattr(model, "_keep_in_fp32_modules", [])
        state_dict = tree_flatten(model.parameters(), destination={})

        # Load shards
        for shard_name, tensor_names in tqdm(shard_map.items(), desc="Loading shards"):
            shard_path = base_path / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            with safe_open(str(shard_path), framework="np", device="cpu") as f:
                names = tensor_names or list(f.keys())
                for name in names:
                    if name not in f.keys():
                        continue
                    np_arr = f.get_tensor(name)
                    arr = (
                        mx.array(np_arr, dtype=target_dtype)
                        if target_dtype
                        else mx.array(np_arr)
                    )
                    for mod in keep_in_fp32_modules:
                        if mod in name:
                            arr = arr.astype(mx.float32)
                            break

                    if arr.ndim == 5:
                        # check same shape as model
                        if arr.shape != state_dict[name].shape:
                            arr = arr.transpose(0, 2, 3, 4, 1)
                            if arr.shape != state_dict[name].shape:
                                shape = state_dict[name].shape
                                raise ValueError(
                                    f"Weight {name} has shape {arr.shape} but expected {shape}"
                                )
                    elif arr.ndim == 4:
                        if arr.shape != state_dict[name].shape:
                            arr = arr.transpose(0, 2, 3, 1)
                            if arr.shape != state_dict[name].shape:
                                shape = state_dict[name].shape
                                raise ValueError(
                                    f"Weight {name} has shape {arr.shape} but expected {shape}"
                                )

                    assigned = _set_by_path(model, name, arr)
                    if assigned:
                        seen.add(name)

        if strict:
            missing_keys = [k for k in leaves.keys() if k not in seen]
            if missing_keys:
                raise RuntimeError(
                    f"Missing keys in checkpoint for strict load: {missing_keys[:20]}... (total {len(missing_keys)})"
                )
        return model

    @classmethod
    def from_single_file(
        cls,
        weights_path: str | Path,
        *,
        dtype: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # If directory: delegate to from_pretrained
        if weights_path.is_dir():
            return cls.from_pretrained(
                weights_path, dtype=dtype, strict=strict, config=config
            )

        # Require config
        if config is None:
            # Try sibling config.json
            cfg_path = weights_path.parent / "config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(
                    "Config is required for from_single_file; 'config.json' not found next to the weights file"
                )
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        model = cls(**config)

        target_dtype = _mx_dtype_from_str(dtype)
        leaves = _flatten_leaf_arrays(model)
        seen: set[str] = set()

        if weights_path.suffix.lower() == ".safetensors":
            with safe_open(str(weights_path), framework="np", device="cpu") as f:
                for name in f.keys():
                    np_arr = f.get_tensor(name)
                    arr = (
                        mx.array(np_arr, dtype=target_dtype)
                        if target_dtype
                        else mx.array(np_arr)
                    )

                    if _set_by_path(model, name, arr):
                        seen.add(name)
        else:
            # Support torch weights by loading with safetensors interface is not possible; skipped intentionally
            raise ValueError(
                "from_single_file currently supports only .safetensors files for MLX models"
            )

        if strict:
            missing_keys = [k for k in leaves.keys() if k not in seen]
            if missing_keys:
                raise RuntimeError(
                    f"Missing keys in checkpoint for strict load: {missing_keys[:20]}... (total {len(missing_keys)})"
                )
        return model
