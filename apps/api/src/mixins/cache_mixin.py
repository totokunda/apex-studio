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


def sanitize_path_for_filename(path: str) -> str:
    """Sanitize a file path to be safe for use in a filename.
    
    Replaces path separators and invalid filename characters with underscores.
    Handles Windows drive letters and UNC paths properly.
    """
    if not path:
        return "unknown"
    
    # Replace both forward and back slashes
    sanitized = path.replace("\\", "_").replace("/", "_")
    
    # Remove Windows drive letter colon if present (e.g., "C:" -> "C")
    # This handles cases where the path starts with a drive letter
    if len(sanitized) >= 2 and sanitized[1] == ":":
        sanitized = sanitized[0] + sanitized[2:]
    
    # Remove or replace other invalid filename characters
    # Windows: < > : " | ? * \
    # Unix: / (already handled above)
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    
    # Remove leading/trailing dots and spaces (Windows doesn't allow these)
    sanitized = sanitized.strip(". ")
    
    # Collapse multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    
    return sanitized if sanitized else "unknown"


def normalize_cache_file_path(cache_file: str) -> str:
    """Normalize a cache file path so it's safe to use on the current OS.

    This is primarily a Windows hardening layer: callers sometimes accidentally
    embed a full Windows path (e.g. "C:\\...") into what is meant to be a *filename*.
    That can create illegal path segments like "vae_encode_C:" and crash with:
      WinError 123: The filename, directory name, or volume label syntax is incorrect
    """
    if not cache_file:
        return cache_file

    # Only Windows has the "colon is illegal except after drive letter" issue.
    if os.name != "nt":
        return cache_file

    # If any path segment contains ":" (except the leading drive "C:"), the path is unsafe.
    # Example bad input:
    #   <cache_dir>\\vae_encode_C:\\Users\\me\\...\\.safetensors
    parts = re.split(r"[\\/]+", str(cache_file))
    unsafe = False
    for i, part in enumerate(parts):
        if not part:
            continue
        # Allow the leading drive token: "C:"
        if i == 0 and re.fullmatch(r"[A-Za-z]:", part):
            continue
        if ":" in part:
            unsafe = True
            break

    if unsafe:
        filename = sanitize_path_for_filename(str(cache_file))
        if not filename.lower().endswith(".safetensors"):
            filename = f"{filename}.safetensors"
        return os.path.join(DEFAULT_CACHE_PATH, filename)

    # Also sanitize the basename if it contains Windows-invalid characters.
    base = os.path.basename(cache_file)
    if re.search(r'[<>:"|?*]', base):
        safe_base = sanitize_path_for_filename(base)
        if not safe_base.lower().endswith(".safetensors") and base.lower().endswith(".safetensors"):
            safe_base = f"{safe_base}.safetensors"
        return os.path.join(os.path.dirname(cache_file), safe_base)

    return cache_file


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
                f"text_encoder_{sanitize_path_for_filename(self.model_path)}.safetensors",
            )

        cache_file_eff = normalize_cache_file_path(cache_file_eff)

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
            cache_file_eff = normalize_cache_file_path(cache_file_eff)
            with safe_open(cache_file_eff, framework="pt", device="cpu") as f:
                tensors: List[torch.Tensor] = []
                for key in keys:
                    t = f.get_tensor(key)
                    # On Windows, safetensors often returns mmap-backed CPU tensors.
                    # Keeping those alive can prevent later cache rewrites with:
                    #   "file with a user-mapped section open (os error 1224)".
                    # Clone forces a real in-memory copy so the mmap can be released.
                    if os.name == "nt":
                        t = t.clone()
                    tensors.append(t)
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
                f"text_encoder_{sanitize_path_for_filename(self.model_path)}.safetensors",
            )

        cache_file_eff = normalize_cache_file_path(cache_file_eff)

        # Safely create directory - ensure dirname is valid and not empty
        cache_dir = os.path.dirname(cache_file_eff)
        if cache_dir and cache_dir.strip():
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                # Last-resort fallback: remap to a safe filename under DEFAULT_CACHE_PATH.
                cache_file_eff = normalize_cache_file_path(cache_file_eff)
                cache_dir = os.path.dirname(cache_file_eff)
                if cache_dir and cache_dir.strip():
                    os.makedirs(cache_dir, exist_ok=True)

        # Load existing cache tensors (best-effort)
        existing_tensors: Dict[str, torch.Tensor] = {}
        if os.path.exists(cache_file_eff):
            try:
                with safe_open(cache_file_eff, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        # Lazy: only load when needed for rewrite; we need all to rewrite cleanly
                        t = f.get_tensor(key)
                        if os.name == "nt":
                            t = t.clone()
                        existing_tensors[key] = t
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

        # Rewrite cache file atomically (temp write + replace).
        # Avoid in-place overwrite which can fail on Windows if the file is/was
        # memory-mapped (os error 1224).
        try:
            timestamp_ms = int(time.time() * 1000)
            tmp_path = f"{cache_file_eff}.tmp.{os.getpid()}.{timestamp_ms}"
            save_file(existing_tensors, tmp_path)
            os.replace(tmp_path, cache_file_eff)
        except Exception as e:
            print(e)
            # Best-effort cleanup of temp file
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            # As a fallback, avoid crashing the caller; drop caching if write fails
            pass

    def load_cached_prompt(self, hash: str) -> Tuple[torch.Tensor, ...] | None:
        return self.load_cached(hash)

    def cache_prompt(self, hash: str, *tensors: torch.Tensor) -> None:
        return self.cache(hash, *tensors)
