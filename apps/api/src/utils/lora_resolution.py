from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.utils.config_store import (
    config_store_lock,
    read_json_dict,
    write_json_dict_atomic,
)
from src.utils.defaults import get_config_store_path, get_lora_path

LORA_RESOLUTION_STORE_KEY = "lora_resolutions"


def _now_s() -> float:
    try:
        return float(time.time())
    except Exception:
        return 0.0


def _is_complete_file(path: str) -> bool:
    try:
        if not path or not os.path.isfile(path):
            return False
        if path.endswith(".part"):
            return False
        try:
            return os.path.getsize(path) > 0
        except Exception:
            return False
    except Exception:
        return False


def _hash_stable_id(stable_id: str) -> str:
    return hashlib.sha256(str(stable_id).encode("utf-8")).hexdigest()


def _scan_save_dir_for_hash_prefix(save_dir: str, sha256_hex: str) -> List[str]:
    """
    Return candidate downloaded paths for files named like:
      <sha256>_<base_name>
    """
    try:
        out: List[str] = []
        if not save_dir or not sha256_hex:
            return out
        if not os.path.isdir(save_dir):
            return out
        prefix = f"{sha256_hex}_"
        for name in os.listdir(save_dir):
            if not isinstance(name, str):
                continue
            if not name.startswith(prefix):
                continue
            p = os.path.join(save_dir, name)
            if _is_complete_file(p):
                out.append(p)
        return sorted(set(out))
    except Exception:
        return []


def _normalize_civitai_fmt(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    f = str(fmt).strip().lower()
    if f in ("safetensor", "safetensors"):
        return "safetensors"
    if f in ("pt", "pth", "pickle", "pickletensor"):
        return "pt"
    if f in ("bin",):
        return "bin"
    # Keep unknown tokens as-is (may still be used in stable_id)
    return f if f else None


def _stable_ids_for_civitai_version_id(
    version_id: str, preferred_fmt: Optional[str] = None
) -> List[str]:
    """
    Mirrors `LoraManager._download_from_civitai_spec()` stable_id behavior:
      stable_id = f"civitai:{file_id}:{fmt or ''}"

    Here file_id is the "download id" used in the CivitAI download endpoint, which
    for model-version downloads is the modelVersion id.
    """
    version_id = str(version_id).strip()
    if not version_id:
        return []

    preferred = _normalize_civitai_fmt(preferred_fmt)
    order = []
    if preferred:
        order.append(preferred)
    # Fallbacks (common)
    for ext in ("safetensors", "pt", "bin", ""):
        if ext not in order:
            order.append(ext)
    return [f"civitai:{version_id}:{ext}" for ext in order]


def _parse_air_urn_minimal(
    text: str,
) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """
    Minimal AIR URN parser for our use-case (civitai resolution) without importing
    heavy model/runtime deps.

    Returns (source, version_id, fmt), where:
    - source: e.g. "civitai"
    - version_id: the token after '@' in the id segment, if present
    - fmt: the token after '.' in the id segment, if present
    """
    if not isinstance(text, str) or not text.startswith("urn:air:"):
        return None
    rest = text[len("urn:air:") :]
    parts = rest.split(":")
    if len(parts) < 3:
        return None

    known_resource_types = {"model", "lora", "embedding", "hypernet"}

    # Variant with ecosystem: urn:air:{ecosystem}:{type}:{source}:{id}...
    # Variant without:        urn:air:{type}:{source}:{id}...
    if len(parts) >= 4 and str(parts[1]).lower() in known_resource_types:
        source = str(parts[2]).strip().lower()
        remainder = ":".join(parts[3:])
    else:
        source = str(parts[1]).strip().lower()
        remainder = ":".join(parts[2:])

    remainder = remainder.split("?", 1)[0]

    urn_format: Optional[str] = None
    if "." in remainder:
        before, maybe_fmt = remainder.rsplit(".", 1)
        if maybe_fmt and re.match(r"^[A-Za-z0-9_]+$", maybe_fmt):
            urn_format = maybe_fmt.strip().lower()
            remainder = before

    # Optional ":layer" (ignored)
    if ":" in remainder:
        remainder = remainder.split(":", 1)[0]

    version: Optional[str] = None
    rid = remainder.strip()
    if "@" in rid:
        _rid, ver = rid.split("@", 1)
        version = ver.strip() if ver and ver.strip() else None

    return (source, version, urn_format)


def _try_resolve_civitai_source_to_local_paths(
    source: str, *, save_dir: str
) -> List[str]:
    """
    Best-effort local resolution for civitai URNs and civitai:* specs without network I/O.

    Supports:
      - urn:air:*:*:civitai:<model_id>@<version_id>[.<format>]
      - civitai:<model_id>@<version_id>
      - civitai-file:<file_id>[.<format>]
    """
    src = (source or "").strip()
    if not src:
        return []

    preferred_fmt: Optional[str] = None
    version_id: Optional[str] = None

    # AIR URN (preferred): version id is explicit after '@'
    if src.startswith("urn:air:"):
        try:
            parsed = _parse_air_urn_minimal(src)
            if parsed:
                urn_source, urn_version, urn_fmt = parsed
                if urn_source == "civitai":
                    version_id = urn_version
                    preferred_fmt = urn_fmt
        except Exception:
            version_id = None

    # civitai:MODEL@VERSION
    if version_id is None and src.startswith("civitai:"):
        try:
            rest = src.split(":", 1)[1].strip()
            if "@" in rest:
                _model_id, v = rest.split("@", 1)
                version_id = v.strip() if v and v.strip() else None
        except Exception:
            version_id = None

    # civitai-file:FILE_ID[.fmt]
    if version_id is None and src.startswith("civitai-file:"):
        try:
            rest = src.split(":", 1)[1].strip()
            if "." in rest:
                rest, ext = rest.rsplit(".", 1)
                preferred_fmt = ext
            version_id = rest.strip() if rest and rest.strip() else None
        except Exception:
            version_id = None

    if not version_id:
        return []

    paths: List[str] = []
    for stable_id in _stable_ids_for_civitai_version_id(version_id, preferred_fmt):
        h = _hash_stable_id(stable_id)
        found = _scan_save_dir_for_hash_prefix(save_dir, h)
        if found:
            paths.extend(found)
    # Keep only existing complete files
    paths = [p for p in paths if _is_complete_file(p)]
    # Deduplicate but preserve deterministic order
    return sorted(set(paths))


def _collect_lora_weight_files(path: str) -> List[str]:
    """
    Lightweight file collector (no torch import). Used only for UI "what files do we have?"
    """
    try:
        if not path:
            return []
        if os.path.isfile(path):
            return [path] if _is_complete_file(path) else []
        if os.path.isdir(path):
            out: List[str] = []
            exts = (".safetensors", ".pt", ".pth", ".bin", ".ckpt")
            for root, _dirs, files in os.walk(path):
                for fn in files:
                    if not isinstance(fn, str):
                        continue
                    if not fn.lower().endswith(exts):
                        continue
                    p = os.path.join(root, fn)
                    if _is_complete_file(p):
                        out.append(p)
            return sorted(set(out))
        return []
    except Exception:
        return []


def _read_store() -> Dict[str, Any]:
    p = Path(get_config_store_path())
    try:
        with config_store_lock(p):
            return read_json_dict(p)
    except Exception:
        return {}


def _write_store(data: Dict[str, Any]) -> None:
    p = Path(get_config_store_path())
    try:
        with config_store_lock(p):
            write_json_dict_atomic(p, data, indent=2)
    except Exception:
        return


def set_lora_resolution(source: str, local_paths: List[str]) -> None:
    """
    Persist a mapping from the user-facing LoRA `source` (URN/spec/url) to the resolved
    downloaded file paths so other processes (API vs Ray workers) can reuse it.
    """
    try:
        src = (source or "").strip()
        if not src:
            return
        paths = [p for p in (local_paths or []) if isinstance(p, str) and p.strip()]
        # Normalize + filter to existing complete files
        cleaned: List[str] = []
        for p in paths:
            p2 = p.strip()
            if _is_complete_file(p2):
                cleaned.append(p2)
        if not cleaned:
            return

        data = _read_store()
        section = data.get(LORA_RESOLUTION_STORE_KEY)
        if not isinstance(section, dict):
            section = {}
        section[src] = {"local_paths": cleaned, "updated_at": _now_s()}
        data[LORA_RESOLUTION_STORE_KEY] = section
        _write_store(data)
    except Exception as e:
        logger.debug(f"Failed to persist LoRA resolution for '{source}': {e}")


def delete_lora_resolution(source: str) -> None:
    try:
        src = (source or "").strip()
        if not src:
            return
        data = _read_store()
        section = data.get(LORA_RESOLUTION_STORE_KEY)
        if not isinstance(section, dict):
            return
        if src in section:
            del section[src]
            data[LORA_RESOLUTION_STORE_KEY] = section
            _write_store(data)
    except Exception:
        return


def resolve_lora_local_paths(
    source: str, *, save_dir: Optional[str] = None, update_store: bool = True
) -> List[str]:
    """
    Resolve a LoRA `source` (URN/spec/url/local path) into local downloaded file paths,
    without performing any network I/O.
    """
    src = (source or "").strip()
    if not src:
        return []

    save_dir = save_dir or get_lora_path()

    # 1) If source is already a local path, return its files.
    if os.path.exists(src):
        return _collect_lora_weight_files(src)

    # 2) Persisted mapping (cross-process)
    try:
        data = _read_store()
        section = data.get(LORA_RESOLUTION_STORE_KEY)
        if isinstance(section, dict):
            entry = section.get(src)
            if isinstance(entry, dict):
                lp = entry.get("local_paths")
                if isinstance(lp, list):
                    cleaned = [
                        p for p in lp if isinstance(p, str) and _is_complete_file(p)
                    ]
                    if cleaned:
                        return cleaned
    except Exception:
        pass

    # 3) Deterministic civitai stable_id lookup for URNs/ids that include a version/file id
    civ_paths = _try_resolve_civitai_source_to_local_paths(src, save_dir=save_dir)
    if civ_paths:
        if update_store:
            set_lora_resolution(src, civ_paths)
        return civ_paths

    # 4) Fall back to generic DownloadMixin expected destinations (URLs, HF, etc.)
    try:
        from src.mixins.download_mixin import DownloadMixin

        p = DownloadMixin.is_downloaded(src, save_dir)
        if isinstance(p, str) and p:
            files = _collect_lora_weight_files(p)
            if files and update_store:
                set_lora_resolution(src, files)
            return files
    except Exception:
        pass

    return []


def resolve_lora_download_status(source: str) -> Tuple[bool, List[str]]:
    paths = resolve_lora_local_paths(source, update_store=False)
    return (len(paths) > 0, paths)
