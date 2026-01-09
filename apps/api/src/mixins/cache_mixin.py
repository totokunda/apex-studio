import hashlib
import os
import re
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from safetensors.torch import safe_open
from safetensors.torch import save_file
from src.utils.defaults import DEFAULT_CACHE_PATH
import pickle


class CacheMixin:
    enable_cache: bool = True
    cache_file: str | None = None
    max_cache_size: int | None = None
    model_path: str | None = None

    def str_encode(self, item: Any) -> str:
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy().tobytes().hex()
        elif isinstance(item, np.ndarray):
            return item.tobytes().hex()
        elif isinstance(item, bytes):
            return item.hex()
        elif isinstance(item, str):
            return item
        else:
            return str(item)

    def hash_prompt(self, kwargs: Dict[str, Any]) -> str:
        # Compatibility with old hash_prompt
        return self.hash(kwargs)

    def hash(self, kwargs: Dict[str, Any]) -> str:
        """Return a deterministic hash for a kwargs dict.

        - Ignores ordering of kwargs themselves.
        - Also normalizes nested dicts / sets / lists so their internal ordering
          does not affect the hash.
        - Uses `str_encode` for tensors / ndarrays / bytes / strings so that
          those values are represented in a stable way.
        """

        def canonicalize(obj: Any) -> Any:
            # Normalize mappings by sorting keys and canonicalizing values
            if isinstance(obj, dict):
                return tuple(
                    (k, canonicalize(v))
                    for k, v in sorted(obj.items(), key=lambda kv: kv[0])
                )
            # Normalize sequences by canonicalizing each element
            elif isinstance(obj, (list, tuple)):
                return tuple(canonicalize(v) for v in obj)
            # Normalize sets by sorting their canonicalized elements
            elif isinstance(obj, set):
                return tuple(sorted(canonicalize(v) for v in obj))
            # Use stable string encodings for common tensor/array/byte/string types
            elif isinstance(obj, (torch.Tensor, np.ndarray, bytes, str)):
                return self.str_encode(obj)
            # Primitive scalars (int, float, bool, None, etc.) are already stable
            else:
                return obj

        canonical = canonicalize(kwargs)
        data = pickle.dumps(canonical, protocol=5)
        return hashlib.sha256(data).hexdigest()

    def get_cached_keys_for_prompt(
        self, hash: str, cache_file: str | None = None
    ) -> List[str]:
        """Return the most recent cache keys for a given hash.

        - Supports an arbitrary number of tensors per hash.
        - Keys are indexed positionally so that tensors can be returned
          in the same order they were cached.
        """
        if not self.enable_cache:
            return []

        cache_file_eff = cache_file or self.cache_file
        if cache_file_eff is None:
            # Back-compat default for existing TextEncoder cache usage
            if not getattr(self, "model_path", None):
                return []
            cache_file_eff = os.path.join(
                DEFAULT_CACHE_PATH,
                f"text_encoder_{self.model_path.replace('/', '_')}.safetensors",
            )

        if not os.path.exists(cache_file_eff):
            return []

        # Key format (new, generic):
        #   "{timestamp_ms}_{prompt_hash}_{index}"
        key_pattern = re.compile(
            r"^(?:(?P<ts>\d{13,})_)?(?P<hash>[a-f0-9\.]+)_(?P<index>\d+)$"
        )

        def parse_entry_key(key: str) -> Tuple[int, str, int] | None:
            match = key_pattern.match(key)
            if not match:
                return None
            ts_str = match.group("ts")
            ts = int(ts_str) if ts_str is not None else 0
            index = int(match.group("index"))
            return ts, match.group("hash"), index

        latest_by_index: Dict[int, Tuple[int, str]] = {}

        try:
            with safe_open(cache_file_eff, framework="pt", device="cpu") as f:
                for key in f.keys():
                    parsed = parse_entry_key(key)
                    if parsed is None:
                        continue
                    ts, hsh, index = parsed
                    if hsh != hash:
                        continue
                    current = latest_by_index.get(index)
                    if current is None or ts >= current[0]:
                        latest_by_index[index] = (ts, key)
        except Exception:
            return []

        # Return keys ordered by their positional index
        return [latest_by_index[i][1] for i in sorted(latest_by_index.keys())]

    def load_cached(
        self, hash: str, cache_file: str | None = None
    ) -> Tuple[torch.Tensor, ...] | None:
        """Load cached tensors for the given prompt hash if present.

        Returns a tuple of tensors in the same order they were cached,
        or None if no valid cached entry exists.
        """
        keys = self.get_cached_keys_for_prompt(hash, cache_file=cache_file)
        if not keys:
            return None
        try:
            cache_file_eff = cache_file or self.cache_file
            if cache_file_eff is None:
                return None
            with safe_open(cache_file_eff, framework="pt", device="cpu") as f:
                tensors: List[torch.Tensor] = []
                for key in keys:
                    tensors.append(f.get_tensor(key))
            return tuple(tensors)
        except Exception:
            return None

    def cache(
        self,
        hash: str,
        *tensors: torch.Tensor,
        cache_file: str | None = None,
        max_cache_size: int | None = None,
    ) -> None:
        """Persist an arbitrary number of tensors with LRU-style eviction.

        - Accepts any number of positional tensors (`*tensors`).
        - Stores tensors in a single safetensors file under timestamped keys to track recency.
        - Removes any pre-existing entries for the same hash (acts like an update/move-to-front).
        - If the number of unique cached prompts exceeds `max_cache_size`, evicts the oldest prompts first.

        Key format (new, generic):
            "{timestamp_ms}_{hash}_{index}"
        """
        if not self.enable_cache:
            return

        if not tensors:
            return

        # Ensure cache path exists
        cache_file_eff = cache_file or self.cache_file
        if cache_file_eff is None:
            # Back-compat default for existing TextEncoder cache usage
            if not getattr(self, "model_path", None):
                return
            cache_file_eff = os.path.join(
                DEFAULT_CACHE_PATH,
                f"text_encoder_{self.model_path.replace('/', '_')}.safetensors",
            )
        os.makedirs(os.path.dirname(cache_file_eff), exist_ok=True)

        # Load existing cache tensors (best-effort)
        existing_tensors: Dict[str, torch.Tensor] = {}
        if os.path.exists(cache_file_eff):
            try:
                with safe_open(cache_file_eff, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # Lazy: only load when needed for rewrite; we need all to rewrite cleanly
                        existing_tensors[key] = f.get_tensor(key)
            except Exception:
                # Corrupt or unreadable cache file; reset it
                existing_tensors = {}

        # Build index of entries by hash with their latest timestamp
        # Recognize only the new generic key format
        key_pattern = re.compile(
            r"^(?:(?P<ts>\d{13,})_)?(?P<hash>[a-f0-9\.]+)_(?P<index>\d+)$"
        )

        def parse_entry_key(key: str) -> Tuple[int, str, int] | None:
            match = key_pattern.match(key)
            if not match:
                return None
            ts_str = match.group("ts")
            ts = int(ts_str) if ts_str is not None else 0
            index = int(match.group("index"))
            return ts, match.group("hash"), index

        entries_by_hash: Dict[str, Dict[int, Tuple[int, str]]] = {}
        # maps: prompt_hash -> { index: (ts, key) }
        for key in list(existing_tensors.keys()):
            parsed = parse_entry_key(key)
            if parsed is None:
                continue
            ts, hsh, index = parsed
            indices = entries_by_hash.setdefault(hsh, {})
            # Keep the most recent key per index for that hash
            if index not in indices or ts >= indices[index][0]:
                indices[index] = (ts, key)

        # Remove previous entries for this prompt hash
        if hash in entries_by_hash:
            for _index, (_ts, key) in list(entries_by_hash[hash].items()):
                existing_tensors.pop(key, None)
            entries_by_hash.pop(hash, None)

        # Insert the new entry with a fresh timestamp
        timestamp_ms = int(time.time() * 1000)
        # Insert the new entries with fresh timestamps and positional indices
        new_indices: Dict[int, Tuple[int, str]] = {}
        for idx, tensor in enumerate(tensors):
            key = f"{timestamp_ms}_{hash}_{idx}"
            existing_tensors[key] = tensor.detach().to("cpu")
            new_indices[idx] = (timestamp_ms, key)

        # Update index with the new entries
        entries_by_hash[hash] = new_indices

        # Enforce max_cache_size (unique prompt hashes)
        unique_hashes = list(entries_by_hash.keys())
        num_prompts = len(unique_hashes)
        max_cache_size_eff = (
            max_cache_size if max_cache_size is not None else self.max_cache_size
        )
        if (
            max_cache_size_eff is not None
            and max_cache_size_eff > 0
            and num_prompts > max_cache_size_eff
        ):
            # Compute recency per hash (latest timestamp across kinds)
            hash_to_latest_ts = {
                hsh: max(ts_index[0] for ts_index in indices.values())
                for hsh, indices in entries_by_hash.items()
            }
            # Sort hashes by recency ascending (oldest first)
            eviction_order = sorted(hash_to_latest_ts.items(), key=lambda x: x[1])
            # Do not evict the newly added prompt; prioritize evicting others first
            eviction_candidates = [h for h, _ in eviction_order if h != hash]
            num_to_evict = num_prompts - max_cache_size_eff
            for hsh in eviction_candidates[:num_to_evict]:
                indices = entries_by_hash.pop(hsh, {})
                for _index, (_ts, key) in indices.items():
                    existing_tensors.pop(key, None)

        # Rewrite cache file atomically
        try:
            save_file(existing_tensors, cache_file_eff)
        except Exception:
            # As a fallback, avoid crashing the caller; drop caching if write fails
            pass

    def load_cached_prompt(self, hash: str) -> Tuple[torch.Tensor, ...] | None:
        return self.load_cached(hash)

    def cache_prompt(self, hash: str, *tensors: torch.Tensor) -> None:
        return self.cache(hash, *tensors)
