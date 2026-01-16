"""
Ray tasks for preprocessor operations
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
from urllib.parse import urlparse
import ray
import traceback
from loguru import logger
from src.memory_management import MemoryConfig
from src.preprocess.aux_cache import AuxillaryCache
import importlib
import hashlib
import os
import re
import shutil
from src.utils.save_audio_video import save_video_ovi, save_video_ltx2
import torch
import inspect
import yaml
import numpy as np
import json
from src.utils.cache import empty_cache
from src.utils.warm_pool import EngineWarmPool, stable_hash_dict
from src.api.preprocessor_registry import get_preprocessor_info
from src.engine.registry import UniversalEngine
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path, get_lora_path, get_preprocessor_path
from diffusers.utils import export_to_video
from src.lora.manager import LoraManager
from src.api.manifest import get_manifest, MANIFEST_BASE_PATH
from src.lora.manager import LoraManager
from src.api.manifest import get_manifest, MANIFEST_BASE_PATH
import gc
import ctypes
import ctypes.util
import time
import struct

def _get_warm_pool() -> EngineWarmPool:
    """
    Lazily create the warm pool inside the Ray worker process.

    Important: Ray pickles remote functions (including module globals) when submitting
    from the driver. A module-level EngineWarmPool instance contains an RLock which
    is not picklable. This accessor avoids creating the pool until after deserialization.
    """
    if not hasattr(_get_warm_pool, "_pool"):
        _get_warm_pool._pool = EngineWarmPool.from_env()  # type: ignore[attr-defined]
    return _get_warm_pool._pool  # type: ignore[attr-defined]


_WIN_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def _safe_fs_component(value: Any, *, fallback: str = "job", max_len: int = 120) -> str:
    """
    Convert an arbitrary value into a filesystem-safe *single path component*.

    Motivation:
    - Windows forbids characters like ":" in directory names. The test-suite uses job_ids like
      "test_suite:<name>", which would otherwise crash when we mkdir under DEFAULT_CACHE_PATH.
    """
    try:
        s = str(value) if value is not None else ""
    except Exception:
        s = ""
    s = s.strip()
    if not s:
        return fallback

    # Windows filename hardening: replace invalid characters and strip trailing dots/spaces.
    if os.name == "nt":
        s = _WIN_INVALID_PATH_CHARS_RE.sub("_", s)
        s = s.rstrip(" .")

    # Collapse underscores and trim.
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return fallback

    # Keep path segments reasonably short to reduce MAX_PATH issues on Windows.
    # If we truncate, keep uniqueness by appending a stable hash suffix.
    if len(s) > max_len:
        digest = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]
        s = f"{s[: max_len - 11]}_{digest}"

    return s


def _persist_run_config(
    manifest_path: str,
    engine_kwargs: Dict[str, Any],
    inputs: Dict[str, Any],
    run_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Persist a snapshot of the engine invocation into a structured `runs` directory.

    Layout:
        <project_root>/runs/<manifest_stem>/assets/<media files...>
        <project_root>/runs/<manifest_stem>/inputs.json

    - Copies any local image/audio/video files referenced in `inputs` into `assets/`
    - Rewrites those input values to use relative `assets/<filename>` paths in the
      persisted JSON, without mutating the original `inputs` dict used by the engine.
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        base_runs_dir = run_root or (project_root / "runs")

        # logger.info(f"\n\nBase runs directory: {base_runs_dir}\n\n")
        # logger.info(f"\n\nproject_root: {project_root}\n\n")
        # logger.info(f"\n\nrun_root: {run_root}\n\n")

        manifest_stem = Path(manifest_path).stem
        run_dir = base_runs_dir / manifest_stem
        assets_dir = run_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        def _extract_media_candidate(value: Any) -> tuple[Optional[str], Optional[str]]:
            """
            Return (path, field_key) where field_key is the dict key that held the path
            ('input_path' / 'src' / 'path') or None if `value` is a bare string.
            """
            if isinstance(value, dict):
                for k in ("input_path", "src", "path"):
                    v = value.get(k)
                    if isinstance(v, str):
                        return v, k
                return None, None
            if isinstance(value, str):
                return value, None
            return None, None

        def _is_media_file(path_str: str) -> bool:
            lower = path_str.lower()
            media_exts = (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".bmp",
                ".tiff",
                ".mp4",
                ".mov",
                ".mkv",
                ".avi",
                ".webm",
                ".wav",
                ".mp3",
                ".flac",
                ".m4a",
                ".ogg",
            )
            return any(lower.endswith(ext) for ext in media_exts)

        persisted_inputs: Dict[str, Any] = {}

        from random import randint

        for key, value in inputs.items():
            media_path, field_key = _extract_media_candidate(value)
            new_value = value

            if media_path and _is_media_file(media_path):
                try:
                    src_path = Path(media_path)
                    if src_path.is_file():
                        if src_path.name == "result.mp4":
                            dest_path = (
                                assets_dir / f"_{randint(1, 100000000)}_{src_path.name}"
                            )
                        else:
                            dest_path = assets_dir / src_path.name
                        if src_path.resolve() != dest_path.resolve():
                            shutil.copy2(src_path, dest_path)
                        rel_path = f"assets/{dest_path.name}"

                        if isinstance(value, dict) and field_key:
                            updated = dict(value)
                            updated[field_key] = rel_path
                            new_value = updated
                        elif isinstance(value, str):
                            new_value = rel_path
                except Exception as copy_err:
                    logger.warning(
                        f"Failed to copy media input '{media_path}' for key '{key}': {copy_err}"
                    )

            persisted_inputs[key] = new_value

        persisted_engine_kwargs: Dict[str, Any] = dict(engine_kwargs or {})
        persisted_engine_kwargs["yaml_path"] = manifest_path

        payload = {
            "engine_kwargs": persisted_engine_kwargs,
            "inputs": persisted_inputs,
        }

        json_path = run_dir / "model_inputs.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        logger.info(f"Persisted run configuration to {json_path}")
        return json_path
    except Exception as e:
        logger.warning(f"Failed to persist run configuration: {e}")
        return None


def _derive_lora_name_from_source(source: str) -> str:
    """
    Derive a human-readable LoRA name/label from an arbitrary source string
    (local path, HF repo path, or URL).
    """
    try:
        src = (source or "").strip()
        parsed = urlparse(src)
        path = parsed.path if parsed.scheme else src
        path = path.rstrip("/") or src
        base = os.path.basename(path) or path
        # Drop query fragments if they somehow slipped through
        if "?" in base:
            base = base.split("?", 1)[0]
        if "#" in base:
            base = base.split("#", 1)[0]
        # Trim extension, keep stem only
        if "." in base:
            base = base.split(".", 1)[0]
        return base or src
    except Exception:
        return source


def _load_manifest_yaml(yaml_path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort YAML loader that never raises; logs and returns None on failure."""
    try:
        if not yaml_path.exists():
            logger.warning(f"Manifest YAML not found at {yaml_path}")
            return None
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Manifest YAML at {yaml_path} is not a mapping")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to load manifest YAML {yaml_path}: {e}")
        return None


def _save_manifest_yaml(yaml_path: Path, doc: Dict[str, Any]) -> None:
    """Best-effort YAML dumper; logs on failure and keeps existing file untouched."""
    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_text = yaml.safe_dump(doc, sort_keys=False)
        yaml_path.write_text(yaml_text, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write updated manifest YAML {yaml_path}: {e}")


def _aggressive_ram_cleanup(*, clear_torch_cache: bool = True) -> None:
    """
    Best-effort cleanup to reduce *resident* CPU memory in long-lived Ray workers.

    Notes:
    - `gc.collect()` drops Python object graphs.
    - `empty_cache()` clears torch CUDA/MPS caches.
    - On Linux/glibc, `malloc_trim(0)` can return freed heap pages back to the OS.
      This is best-effort and safe to ignore if unavailable.
    """
    try:
        gc.collect()
    except Exception:
        pass
    if clear_torch_cache:
        try:
            empty_cache()
        except Exception:
            pass
        # Extra CUDA cleanup (beyond empty_cache) when available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    # Best-effort: return heap pages to OS (Linux/glibc)
    try:
        libc_path = ctypes.util.find_library("c")
        if libc_path:
            libc = ctypes.CDLL(libc_path)
            mt = getattr(libc, "malloc_trim", None)
            if mt is not None:
                mt(0)
    except Exception:
        pass


def _remove_lora_from_manifest(
    lora_name: Optional[str], manifest_id: Optional[str] = None
) -> bool:
    """
    Remove a LoRA entry from the given manifest.

    `lora_name` is treated as a generic identifier and may match:
    - a raw string entry in `spec.loras`
    - a dict entry's `name` / `label`
    - a dict entry's `source` / `path` / `url` / `remote_source`

    Returns True if at least one entry was removed, otherwise False.
    """
    try:
        if not lora_name:
            return False
        manifest = get_manifest(manifest_id)
        manifest_path = MANIFEST_BASE_PATH / Path(manifest.get("full_path"))
        doc = _load_manifest_yaml(manifest_path)
        if doc is None:
            return False

        spec = doc.get("spec") or {}
        loras = spec.get("loras") or []
        if not isinstance(loras, list):
            return False

        new_loras: List[Any] = []
        removed = False
        for entry in loras:
            if isinstance(entry, str):
                if entry == lora_name:
                    removed = True
                    continue
                new_loras.append(entry)
                continue
            if isinstance(entry, dict):
                entry_id_candidates = (
                    entry.get("name"),
                    entry.get("label"),
                    entry.get("source"),
                    entry.get("path"),
                    entry.get("url"),
                    entry.get("remote_source"),
                )
                if any(c == lora_name for c in entry_id_candidates if isinstance(c, str)):
                    removed = True
                    continue
                new_loras.append(entry)
                continue
            # Unknown entry type; keep it untouched
            new_loras.append(entry)

        if not removed:
            return False

        spec["loras"] = new_loras
        doc["spec"] = spec
        _save_manifest_yaml(manifest_path, doc)
        logger.info(f"Removed LoRA '{lora_name}' from manifest {manifest_path.name}")
        return True
    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Failed to remove LoRA '{lora_name}' from manifest: {e}")
        return False


def _engine_pool_key(
    *,
    manifest_path: str,
    engine_type: Any,
    model_type: Any,
    selected_components: Dict[str, Any],
    engine_kwargs: Dict[str, Any],
    attention_type: Any,
    auto_memory_management: Any,
) -> str:
    # Include a manifest fingerprint so edits (e.g. changing LoRAs) don't reuse a stale engine.
    try:
        st = os.stat(manifest_path)
        manifest_fingerprint = {"mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
    except Exception:
        manifest_fingerprint = None
    payload = {
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": manifest_fingerprint,
        "engine_type": str(engine_type),
        "model_type": str(model_type),
        "selected_components": selected_components or {},
        "engine_kwargs": engine_kwargs or {},
        "attention_type": str(attention_type) if attention_type is not None else None,
        "auto_memory_management": bool(auto_memory_management),
    }
    h = stable_hash_dict(payload)
    return f"{payload['engine_type']}/{payload['model_type']}:{h}"


@ray.remote
def warmup_engine_from_manifest(
    manifest_path: str,
    selected_components: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "engine",
) -> Dict[str, Any]:
    """
    Best-effort warmup that won't interfere with a running engine.

    - mode="disk": warm OS page cache for weight files only (bounded via env).
    - mode="engine": instantiate engine into the warm pool (no inference).
    - mode="both": do both.
    """
    t0 = time.time()
    selected_components = selected_components or {}
    mode_lc = str(mode or "engine").strip().lower()

    try:
        from src.utils.yaml import load_yaml as load_manifest_yaml
        from src.manifest.loader import validate_and_normalize
        from src.mixins.loader_mixin import LoaderMixin
        from glob import glob

        raw = load_manifest_yaml(manifest_path)
        config = validate_and_normalize(raw)
        engine_type = config.get("engine") or (config.get("spec") or {}).get("engine")
        model_type = config.get("type") or (config.get("spec") or {}).get("model_type")
        if isinstance(model_type, list):
            model_type = model_type[0] if model_type else None

        attention_type = None
        if isinstance(selected_components, dict):
            attention_type = selected_components.get("attention", {}).get("name", None)
            # mirror run path: don't let attention selection pollute other selections
            selected_components = dict(selected_components)
            selected_components.pop("attention", None)

        engine_kwargs = (config.get("engine_kwargs", {}) or {})
        auto_mm = os.environ.get("AUTO_MEMORY_MANAGEMENT", False)

        pool_key = _engine_pool_key(
            manifest_path=manifest_path,
            engine_type=engine_type,
            model_type=model_type,
            selected_components=selected_components or {},
            engine_kwargs=engine_kwargs,
            attention_type=attention_type,
            auto_memory_management=auto_mm,
        )

        warmed_disk = False
        warmed_engine = False

        if mode_lc in {"disk", "both"}:
            default_max_bytes = 256 * 1024 * 1024
            max_bytes = default_max_bytes
            try:
                max_bytes_env = os.environ.get("APEX_DISK_PREWARM_MAX_BYTES")
                if max_bytes_env is not None and str(max_bytes_env).strip() != "":
                    v = int(str(max_bytes_env).strip())
                    if v > 0:
                        max_bytes = v
                    else:
                        max_bytes = None
            except Exception:
                max_bytes = default_max_bytes

            tmp = LoaderMixin()
            exts_default = ["safetensors", "bin", "pt", "ckpt", "gguf", "pth"]
            for comp in (config.get("components") or []):
                extensions = comp.get("extensions", exts_default) or exts_default
                if "gguf" not in [str(x).lstrip(".").lower() for x in extensions]:
                    extensions = list(extensions) + ["gguf"]

                def _iter_weight_files(path: Any) -> List[str]:
                    if not path:
                        return []
                    if isinstance(path, dict):
                        path = path.get("path") or path.get("model_path") or path.get(
                            "file"
                        )
                    if not path:
                        return []
                    path_str = os.fspath(path)
                    if os.path.isdir(path_str):
                        files: List[str] = []
                        for ext in extensions:
                            ext = str(ext).lstrip(".")
                            files.extend(glob(os.path.join(path_str, f"*.{ext}")))
                        return files
                    path_lower = path_str.lower()
                    for ext in extensions:
                        ext = str(ext).lstrip(".").lower()
                        if path_lower.endswith(f".{ext}"):
                            return [path_str]
                    return []

                for fp in _iter_weight_files(comp.get("model_path")):
                    tmp._prewarm_model(fp, max_bytes=max_bytes)
                extra = comp.get("extra_model_paths") or []
                if isinstance(extra, str):
                    extra = [extra]
                for p in extra:
                    for fp in _iter_weight_files(p):
                        tmp._prewarm_model(fp, max_bytes=max_bytes)
            warmed_disk = True

        if mode_lc in {"engine", "both"}:

            def _factory():
                kwargs = {
                    "engine_type": engine_type,
                    "yaml_path": manifest_path,
                    "model_type": model_type,
                    "selected_components": selected_components or {},
                    "auto_memory_management": auto_mm,
                    **engine_kwargs,
                }
                if attention_type:
                    kwargs["attention_type"] = attention_type
                return UniversalEngine(**kwargs)

            eng, pooled = _get_warm_pool().acquire(pool_key, _factory, allow_pool=True)
            if pooled:
                _get_warm_pool().release(pool_key)
                warmed_engine = True
            else:
                try:
                    if hasattr(eng, "offload_engine"):
                        eng.offload_engine()
                except Exception:
                    pass

        return {
            "status": "ok",
            "mode": mode_lc,
            "pool_key": pool_key,
            "warmed_disk": warmed_disk,
            "warmed_engine": warmed_engine,
            "pool": _get_warm_pool().stats(),
            "duration_s": time.time() - t0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "duration_s": time.time() - t0}


def _cleanup_lora_artifacts_if_remote(lora_item: Any) -> None:
    """
    Best-effort cleanup of downloaded LoRA artifacts when verification fails.

    Only removes files that:
      - belong to the given `lora_item.local_paths`
      - live under our configured LoRA directory
      - and the original `lora_item.source` looks like a remote URL (not a local path).
    """
    try:
        source = getattr(lora_item, "source", None)
        local_paths = getattr(lora_item, "local_paths", None)
        if not source or not isinstance(local_paths, list) or not local_paths:
            return

        dm = DownloadMixin()
        source_str = str(source)
        # Treat some non-URL sources as remote identifiers (AIR URNs, civitai specs).
        # Otherwise assume it's a user-managed local path and never delete it here.
        source_lc = source_str.strip().lower()
        is_remote_identifier = (
            dm._is_url(source_str)
            or source_lc.startswith("civitai:")
            or source_lc.startswith("civitai-file:")
            or source_lc.startswith("urn:air:")
        )
        if not is_remote_identifier:
            return

        base_dir = Path(get_lora_path()).resolve()

        for path_str in list(local_paths):
            try:
                p = Path(path_str).resolve()
            except Exception:
                continue

            # Only operate inside our managed LoRA directory
            try:
                common = os.path.commonpath([str(base_dir), str(p)])
            except Exception:
                continue
            if common != str(base_dir):
                continue

            try:
                if p.is_file():
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to delete LoRA path '{p}': {e}")

            # Try to prune now-empty parents up to (but not including) base_dir
            parent = p.parent
            while True:
                try:
                    if parent == base_dir:
                        break
                    if not parent.exists() or any(parent.iterdir()):
                        break
                    parent.rmdir()
                except Exception:
                    break
                parent = parent.parent
    except Exception as e:
        logger.warning(f"LoRA cleanup failed: {e}")


def _file_looks_like_html(path: str) -> bool:
    """
    Detect the common failure mode where a LoRA "download" succeeds but the saved
    file is actually an HTML error page (auth, rate limit, Cloudflare, etc.).
    """
    try:
        p = Path(path)
        if not p.is_file():
            return False
        # Only need a small sniff (avoid reading the entire file into memory)
        with p.open("rb") as f:
            head = f.read(4096)
        if not head:
            return False
        # Heuristic: lots of CivitAI/HTML errors start with these.
        lowered = head.lstrip().lower()
        if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
            return True
        # If it's text-like and contains an html tag early, treat as html.
        if b"<html" in lowered[:512] or b"</html" in lowered[:512]:
            return True
        return False
    except Exception:
        return False


def _is_valid_safetensors_file(path: str) -> bool:
    """
    Best-effort validation for `.safetensors` files.
    Safetensors format: [u64 header_len LE][header JSON bytes][tensor data...]
    """
    try:
        p = Path(path)
        if not p.is_file():
            return False
        size = p.stat().st_size
        if size < 16:
            return False
        with p.open("rb") as f:
            header_len_raw = f.read(8)
            if len(header_len_raw) != 8:
                return False
            header_len = struct.unpack("<Q", header_len_raw)[0]
            # Guardrails: header must fit in file and be a sane size
            if header_len <= 1 or header_len > min(10_000_000, size - 8):
                return False
            header_bytes = f.read(int(header_len))
        # Header is JSON mapping
        header = json.loads(header_bytes.decode("utf-8"))
        return isinstance(header, dict)
    except Exception:
        return False


def _cleanup_downloaded_lora_paths(local_paths: List[str]) -> None:
    """
    Remove downloaded LoRA artifacts by explicit local paths, but only inside our
    managed LoRA directory.
    """
    try:
        base_dir = Path(get_lora_path()).resolve()
    except Exception:
        return

    for path_str in list(local_paths or []):
        try:
            p = Path(path_str).resolve()
        except Exception:
            continue
        try:
            common = os.path.commonpath([str(base_dir), str(p)])
        except Exception:
            continue
        if common != str(base_dir):
            continue
        try:
            if p.is_file():
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

        # Prune empty parents up to base_dir
        parent = p.parent
        while True:
            try:
                if parent == base_dir:
                    break
                if not parent.exists() or any(parent.iterdir()):
                    break
                parent.rmdir()
            except Exception:
                break
            parent = parent.parent


def _ensure_lora_registered_in_manifests(
    source: str, manifest_id: Optional[str] = None, lora_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Ensure that a LoRA entry for `source` exists in all engine manifests.

    Behaviour:
      - For each manifest YAML:
          - Ensures `spec.loras` exists as a list.
          - If an entry with matching `source` is absent, appends:
                - source: <source>
                - scale: 1.0
                - name: <derived name>
                - label: <derived name>
                - verified: false   (verification happens after download)
          - If entry already exists, leaves the rest of its fields untouched.
    """
    try:
        manifest = get_manifest(manifest_id)
        manifest_path = MANIFEST_BASE_PATH / Path(manifest.get("full_path"))
        doc = _load_manifest_yaml(manifest_path)
        if doc is None:
            return

        spec = doc.setdefault("spec", {})
        loras = spec.setdefault("loras", [])
        if not isinstance(loras, list):
            logger.warning(
                f"spec.loras is not a list in manifest {manifest_id}; leaving unchanged"
            )
            return
        # Skip if already registered in this manifest
        already_present = False
        for entry in loras:
            if isinstance(entry, dict) and (
                entry.get("source") == source
                or entry.get("path") == source
                or entry.get("url") == source
                or entry.get("remote_source") == source
                or entry.get("name") == lora_name
            ):
                already_present = True
                break
            if isinstance(entry, str) and entry == source:
                already_present = True
                break
        if already_present:
            return entry
        name = lora_name or _derive_lora_name_from_source(source)
        new_entry: Dict[str, Any] = {
            "scale": 1.0,
            "name": name,
            "label": name,
            "verified": False,
            "source": source,
        }

        loras.append(new_entry)
        spec["loras"] = loras
        doc["spec"] = spec
        _save_manifest_yaml(manifest_path, doc)
        logger.info(
            f"Registered new LoRA '{source}' in manifest {doc.get('full_path')}"
        )
        return new_entry
    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Failed to register LoRA '{source}' in manifests: {e}")
        return None


def _is_transformer_downloaded_for_manifest(
    doc: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Return the transformer component dict for this manifest with `model_path`
    set to a concrete local string path if any transformer weights are already
    downloaded.

    If nothing is downloaded or no transformer component is present, returns None.
    """
    try:
        spec = doc.get("spec") or {}
        components = spec.get("components") or []
        if not isinstance(components, list):
            return None
        base_dir = get_components_path()

        for component in components:
            if not isinstance(component, dict):
                continue
            if component.get("type") != "transformer":
                continue

            raw_model_paths = component.get("model_path")
            candidate_paths: List[str] = []
            if isinstance(raw_model_paths, str):
                candidate_paths.append(raw_model_paths)
            elif isinstance(raw_model_paths, dict):
                path_val = raw_model_paths.get("path")
                if isinstance(path_val, str):
                    candidate_paths.append(path_val)
            elif isinstance(raw_model_paths, list):
                for item in raw_model_paths:
                    if isinstance(item, str):
                        candidate_paths.append(item)
                    elif isinstance(item, dict):
                        path_val = item.get("path")
                        if isinstance(path_val, str):
                            candidate_paths.append(path_val)

            local_main: Optional[str] = None
            for p in candidate_paths:
                local = DownloadMixin.is_downloaded(str(p), base_dir)
                if local:
                    local_main = local
                    break
            if not local_main:
                continue

            comp_copy: Dict[str, Any] = dict(component)
            comp_copy["model_path"] = local_main

            extra_model_paths = component.get("extra_model_paths") or []
            if extra_model_paths:
                if not isinstance(extra_model_paths, list):
                    return None
                local_extra_model_paths: List[str] = []
                for extra in extra_model_paths:
                    if not isinstance(extra, dict):
                        return None
                    p = extra.get("path")
                    if not isinstance(p, str):
                        return None
                    local = DownloadMixin.is_downloaded(str(p), base_dir)
                    if not local:
                        return None
                    local_extra_model_paths.append(local)
                comp_copy["extra_model_paths"] = local_extra_model_paths

            return comp_copy

        return None

    except Exception as e:
        logger.warning(
            f"Failed to determine transformer download status for manifest: {e}"
        )
        return None


def _mark_lora_verified_in_manifests(
    source: str,
    manifest_id: Optional[str] = None,
    lora_name: Optional[str] = None,
    local_paths: Optional[List[str]] = None,
) -> Optional[bool]:
    """
    Update manifest entries for `source` with a `verified` flag.

    - For each manifest that contains a matching LoRA `source` and whose transformer
      is already downloaded, set `verified: true` on that LoRA entry.

    Returns:
        - True  -> LoRA verified and kept
        - False -> LoRA determined invalid and removed from manifest
        - None  -> No changes made (e.g. transformer not available / no matching entries)
    """
    try:
        manifest = get_manifest(manifest_id)
        manifest_path = MANIFEST_BASE_PATH / Path(manifest.get("full_path"))
        doc = _load_manifest_yaml(manifest_path)
        if doc is None:
            return None
        spec = doc.get("spec") or {}
        loras = spec.get("loras") or []
        if not isinstance(loras, list):
            return None
        target_index: Optional[int] = None
        target_entry: Any = None
        target_source_in_yaml: Optional[str] = None

        def _entry_source_like(e: Any) -> Optional[str]:
            if isinstance(e, str):
                return e
            if isinstance(e, dict):
                for k in ("source", "path", "url", "remote_source"):
                    v = e.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
            return None

        def _entry_matches(e: Any) -> bool:
            nonlocal target_source_in_yaml
            if isinstance(e, str):
                if e == source or (lora_name and e == lora_name):
                    target_source_in_yaml = e
                    return True
                return False
            if isinstance(e, dict):
                if lora_name and (
                    e.get("name") == lora_name or e.get("label") == lora_name
                ):
                    target_source_in_yaml = _entry_source_like(e)
                    return True
                src_like = _entry_source_like(e)
                if isinstance(src_like, str) and src_like == source:
                    target_source_in_yaml = src_like
                    return True
            return False

        for i, entry in enumerate(loras):
            if _entry_matches(entry):
                target_index = i
                target_entry = entry
                break

        if target_index is None:
            return None

        # Already verified -> nothing to do
        if isinstance(target_entry, dict) and bool(target_entry.get("verified")):
            return True

        # Only attempt verification if the manifest's transformer is present locally
        transformer_component = _is_transformer_downloaded_for_manifest(doc)
        logger.info(
            f"Transformer component for validation before engine creation: {transformer_component}"
        )
        if not transformer_component:
            return None
        # Build engine once per manifest for validation
        try:
            engine = UniversalEngine(
                yaml_path=str(manifest_path),
                should_download=False,
                auto_apply_loras=False,
                auto_memory_management=False,
            ).engine
        except Exception as e:
            logger.warning(
                f"Failed to create engine for manifest during LoRA validation: {e}"
            )
            return None

        # Build a set of local candidates to validate against (preferred) without
        # rewriting the manifest entry source/path/url.
        validation_candidates: List[str] = []
        if isinstance(local_paths, list):
            for p in local_paths:
                if isinstance(p, str) and p and p not in validation_candidates:
                    validation_candidates.append(p)
                    try:
                        parent = str(Path(p).parent)
                        if parent and parent not in validation_candidates:
                            validation_candidates.append(parent)
                    except Exception:
                        pass

        # Fallback: use whatever the YAML had (may trigger a download, but keeps correctness)
        if not validation_candidates:
            if target_source_in_yaml:
                validation_candidates.append(target_source_in_yaml)
            else:
                validation_candidates.append(source)

        is_valid = False
        last_err: Optional[str] = None
        for cand in validation_candidates:
            try:
                logger.info(
                    f"Validating LoRA candidate '{cand}' against transformer component: {transformer_component}"
                )
                if engine.validate_lora_path(cand, transformer_component):
                    is_valid = True
                    break
            except Exception as ve:
                last_err = str(ve)
                continue

        # Mutate only the target entry; leave everything else untouched.
        new_loras = list(loras)
        if not is_valid:
            logger.warning(
                f"LoRA verification failed for '{source}' (name={lora_name}) in manifest {manifest_path}: {last_err}"
            )
            # Remove the single target entry
            try:
                new_loras.pop(int(target_index))
            except Exception:
                pass
            spec["loras"] = new_loras
            doc["spec"] = spec
            _save_manifest_yaml(manifest_path, doc)
            return False

        # Verified: set verified=True; ensure name/label exist; DO NOT overwrite source/path/url.
        existing = target_entry
        derived_name = lora_name or _derive_lora_name_from_source(
            target_source_in_yaml or source
        )
        if isinstance(existing, str):
            updated_entry: Dict[str, Any] = {
                "source": existing,
                "scale": 1.0,
                "name": derived_name,
                "label": derived_name,
                "verified": True,
            }
        elif isinstance(existing, dict):
            updated_entry = dict(existing)
            updated_entry.setdefault("name", derived_name)
            updated_entry.setdefault("label", updated_entry.get("name") or derived_name)
            updated_entry["verified"] = True
            # Only set a source if the entry had none of the supported identifiers
            if _entry_source_like(updated_entry) is None:
                updated_entry["source"] = source
        else:
            # Unknown type; replace with a conservative dict entry
            updated_entry = {
                "source": target_source_in_yaml or source,
                "scale": 1.0,
                "name": derived_name,
                "label": derived_name,
                "verified": True,
            }

        new_loras[int(target_index)] = updated_entry
        spec["loras"] = new_loras
        doc["spec"] = spec
        _save_manifest_yaml(manifest_path, doc)
        logger.info(
            f"Updated LoRA '{source}' verification state in manifest {manifest_path.name}"
        )
        return True
    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Failed to update LoRA verification state for '{source}': {e}")
        return None


@ray.remote(num_cpus=0.1)
def download_unified(
    item_type: str,
    source: Any,
    job_id: str,
    ws_bridge,
    save_path: Optional[str] = None,
    manifest_id: Optional[str] = None,
    lora_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified downloader for components, LoRAs, and preprocessors.

    Behavior:
    - If item_type == "preprocessor" and source is a known preprocessor id, we initialize it via `.from_pretrained()`
      and mark it as downloaded, reporting progress over websocket.
    - Otherwise, we use DownloadMixin to download one or multiple paths directly into the appropriate default folder
      based on item_type (component, lora, preprocessor) or an explicit save_path override.

    Args:
        item_type: One of {"component", "lora", "preprocessor"}.
        source: A preprocessor id (string) OR a path/url/hf-repo (string) OR a list of such strings.
        job_id: Job ID for progress tracking.
        ws_bridge: Ray actor bridge to forward websocket updates.
        save_path: Optional override directory to save into; otherwise inferred from item_type.
        manifest_id: Optional manifest ID to use for LoRA verification.
        lora_name: Optional name to use for the LoRA.
    """
    # Helper to send progress
    logger.info(
        f"Downloading {item_type} {source} for job {job_id} with manifest {manifest_id} and lora name {lora_name}"
    )

    def send_progress(
        progress: Optional[float],
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            if progress is not None:
                logger.debug(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
            else:
                logger.debug(f"[{job_id}] {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update to websocket: {e}")

    try:
        norm_type = (item_type or "").strip().lower()
        if norm_type not in {"component", "lora", "preprocessor"}:
            raise ValueError(
                f"Unknown item_type '{item_type}'. Expected one of: component, lora, preprocessor."
            )

        # Determine default directory if not explicitly provided
        base_save_dir = save_path
        if base_save_dir is None:
            if norm_type == "component":
                base_save_dir = get_components_path()
            elif norm_type == "lora":
                base_save_dir = get_lora_path()
            else:
                base_save_dir = get_preprocessor_path()
        os.makedirs(base_save_dir, exist_ok=True)

        # Case 1: Preprocessor-id based download and initialization
        if norm_type == "preprocessor" and isinstance(source, str):
            try:
                preprocessor_info = get_preprocessor_info(source)

                # Force CPU in worker to avoid MPS/CUDA fork issues
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                if hasattr(torch, "set_default_device"):
                    torch.set_default_device("cpu")

                send_progress(0.0, f"Starting download of preprocessor '{source}'")
                send_progress(0.1, "Loading preprocessor module")

                module = importlib.import_module(preprocessor_info["module"])
                preprocessor_class = getattr(module, preprocessor_info["class"])

                # Wire download progress into util
                from src.preprocess.download_tracker import DownloadProgressTracker
                from src.preprocess import util as util_module

                tracker = DownloadProgressTracker(
                    job_id, lambda p, m, md=None: send_progress(p, m, md)
                )
                util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
                try:
                    preprocessor_class.from_pretrained()
                    from src.preprocess.base_preprocessor import BasePreprocessor

                    BasePreprocessor._mark_as_downloaded(source)
                finally:
                    util_module.DOWNLOAD_PROGRESS_CALLBACK = None

                send_progress(1.0, "Download complete")
                send_progress(1.0, "Complete", {"status": "complete"})
                return {
                    "job_id": job_id,
                    "status": "complete",
                    "type": "preprocessor",
                    "id": source,
                    "message": "Preprocessor downloaded and initialized",
                }
            except Exception as maybe_not_preproc:
                logger.error(traceback.format_exc())
                # Not a registered preprocessor id; fall through to generic downloader
                logger.info(
                    f"'{source}' is not a registered preprocessor id or failed to init. Falling back to generic download. Reason: {maybe_not_preproc}"
                )
                # continue to generic path-based downloading below

        elif norm_type == "lora" and isinstance(source, str):
            try:
                from src.preprocess.download_tracker import DownloadProgressTracker

                lora_manager = LoraManager()

                # 1) Start the actual download using the LoRA manager with progress tracking
                def _lora_download_progress_adapter(
                    p: Optional[float],
                    message: str,
                    metadata: Optional[Dict[str, Any]] = None,
                ):
                    """
                    Map LoRA download progress into the 0..0.75 range of the unified
                    job progress so the frontend can visualize the entire download.

                    We prefer byte-level aggregation from metadata when available,
                    falling back to the fractional progress `p` from the tracker.
                    """
                    # Prefer explicit byte counts when provided by DownloadProgressTracker
                    frac: float
                    try:
                        downloaded = None
                        total = None
                        if isinstance(metadata, dict):
                            downloaded = (
                                metadata.get("downloaded")
                                or metadata.get("bytes_downloaded")
                                or metadata.get("current_bytes")
                            )
                            total = (
                                metadata.get("total")
                                or metadata.get("bytes_total")
                                or metadata.get("total_bytes")
                            )
                        if (
                            isinstance(downloaded, (int, float))
                            and isinstance(total, (int, float))
                            and total > 0
                        ):
                            frac = max(0.0, min(1.0, float(downloaded) / float(total)))
                        elif p is not None:
                            frac = max(0.0, min(1.0, float(p)))
                        else:
                            # If we truly have no signal, just don't touch progress
                            send_progress(None, message, metadata)
                            return
                    except Exception:
                        if p is None:
                            send_progress(None, message, metadata)
                            return
                        frac = max(0.0, min(1.0, float(p)))

                    scaled = 0.75 * frac
                    send_progress(scaled, message, metadata)

                tracker = DownloadProgressTracker(
                    job_id, _lora_download_progress_adapter
                )
                lora_item = lora_manager.resolve(
                    source,
                    prefer_name=lora_name,
                    progress_callback=tracker.update_progress,
                )

                # Guardrail: sometimes a "successful" download is an HTML error page.
                # If that happens, delete the downloaded file(s) immediately.
                try:
                    local_paths = getattr(lora_item, "local_paths", None)
                    if isinstance(local_paths, list) and local_paths:
                        html_like = [p for p in local_paths if isinstance(p, str) and _file_looks_like_html(p)]
                        invalid_safetensors = [
                            p
                            for p in local_paths
                            if isinstance(p, str)
                            and str(p).lower().endswith(".safetensors")
                            and (not _is_valid_safetensors_file(p))
                        ]
                        if html_like or invalid_safetensors:
                            _cleanup_downloaded_lora_paths(
                                [p for p in local_paths if isinstance(p, str)]
                            )
                            # Best-effort: clear any persisted resolution mapping
                            try:
                                from src.utils.lora_resolution import delete_lora_resolution

                                delete_lora_resolution(getattr(lora_item, "source", None) or source)
                            except Exception:
                                pass
                            # Ensure manifest doesn't retain an entry (best-effort; should be empty at this point)
                            removed = _remove_lora_from_manifest(lora_name or source, manifest_id)
                            if not removed:
                                _remove_lora_from_manifest(source, manifest_id)
                            reason = (
                                "Downloaded file looks like HTML (auth/rate-limit/error page)"
                                if html_like
                                else "Downloaded file is not a valid safetensors archive"
                            )
                            send_progress(
                                0.0,
                                f"LoRA download failed for '{source}' ({reason})",
                                {
                                    "status": "error",
                                    "bucket": norm_type,
                                    "stage": "lora_download",
                                    "error": reason,
                                },
                            )
                            return {
                                "job_id": job_id,
                                "status": "error",
                                "type": "lora",
                                "id": source,
                                "message": f"LoRA download failed: {reason}. Downloaded artifacts were removed.",
                            }
                except Exception:
                    # Keep the existing flow if our guardrail logic fails for any reason.
                    pass

                if (
                    not getattr(lora_item, "local_paths", None)
                    or len(lora_item.local_paths) == 0
                ):
                    # Resolved but no actual files -> treat as failure and (best-effort) clean up any manifest traces
                    logger.error(
                        f"No LoRA files were found for source '{source}'. "
                        "Ensuring LoRA is not present in manifest."
                    )
                    # Best-effort: remove by name first, then by source
                    removed = _remove_lora_from_manifest(lora_name or source, manifest_id)
                    if not removed:
                        _remove_lora_from_manifest(source, manifest_id)
                    # Best-effort: clear any persisted resolution mapping
                    try:
                        from src.utils.lora_resolution import delete_lora_resolution

                        delete_lora_resolution(getattr(lora_item, "source", None) or source)
                    except Exception:
                        pass
                    send_progress(
                        0.0,
                        f"LoRA download failed for '{source}' (no files found)",
                        {
                            "status": "error",
                            "bucket": norm_type,
                            "stage": "lora_download",
                            "error": "No LoRA files found for source",
                        },
                    )
                    return {
                        "job_id": job_id,
                        "status": "error",
                        "type": "lora",
                        "id": source,
                        "message": "LoRA download failed: no files found for source; removed from manifest",
                    }

                # 2) Only after a successful download do we register/update the LoRA in the manifest.
                source = lora_item.source
                try:
                    entry = _ensure_lora_registered_in_manifests(
                        source, manifest_id, lora_name
                    )
                    logger.info(
                        f"Registered LoRA '{source}' in manifests after download: {entry}"
                    )
                except Exception as register_err:
                    traceback.print_exc()
                    logger.warning(
                        f"Failed to register LoRA '{source}' in manifest after download: {register_err}"
                    )

                # Persist the source->local_paths mapping for UI resolution (cross-process).
                try:
                    from src.utils.lora_resolution import set_lora_resolution

                    set_lora_resolution(source, list(getattr(lora_item, "local_paths", []) or []))
                except Exception:
                    pass
                # Explicitly mark the end of the download phase at 75%
                try:
                    send_progress(
                        0.75,
                        "LoRA download complete",
                        {
                            "status": "processing",
                            "bucket": norm_type,
                            "stage": "lora_download",
                        },
                    )
                except Exception:
                    pass

                # 3) Verification phase: use the remaining 25% of progress
                try:
                    send_progress(
                        0.80,
                        "Verifying LoRA",
                        {
                            "status": "processing",
                            "bucket": norm_type,
                            "stage": "lora_verification",
                        },
                    )
                except Exception:
                    pass

                verified = _mark_lora_verified_in_manifests(
                    source,
                    manifest_id,
                    lora_name,
                    local_paths=getattr(lora_item, "local_paths", None),
                )

                # If verification explicitly failed, surface an error and remove the LoRA
                if verified is False:
                    removed = _remove_lora_from_manifest(lora_name or source, manifest_id)
                    if not removed:
                        _remove_lora_from_manifest(source, manifest_id)
                    _cleanup_lora_artifacts_if_remote(lora_item)
                    # Best-effort: clear any persisted resolution mapping
                    try:
                        from src.utils.lora_resolution import delete_lora_resolution

                        delete_lora_resolution(source)
                    except Exception:
                        pass
                    send_progress(
                        0.0,
                        "LoRA verification failed; removed from manifest",
                        {
                            "status": "error",
                            "bucket": norm_type,
                            "stage": "lora_verification",
                            "verified": False,
                        },
                    )
                    return {
                        "job_id": job_id,
                        "status": "error",
                        "type": "lora",
                        "id": source,
                        "message": "LoRA verification failed; removed from manifest",
                    }

                try:
                    send_progress(
                        1.0,
                        "LoRA verification complete",
                        {
                            "status": "complete",
                            "bucket": norm_type,
                            "stage": "lora_verification",
                            "verified": (
                                bool(verified) if verified is not None else False
                            ),
                        },
                    )
                except Exception:
                    pass

                result: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": "complete",
                    "type": "lora",
                    "id": source,
                    "message": "LoRA downloaded and initialized",
                    "local_paths": lora_item.local_paths,
                }
                if verified is not None:
                    result["verified"] = bool(verified)
                return result
            except Exception as maybe_not_lora:
                # Not a valid LoRA source or failed to download; clean up manifest and surface error
                logger.warning(
                    f"LoRA download/resolve failed for '{source}'. "
                    f"Removing LoRA from manifest. Reason: {maybe_not_lora}"
                )
                _remove_lora_from_manifest(source, manifest_id)
                # Best-effort: clear any persisted resolution mapping
                try:
                    from src.utils.lora_resolution import delete_lora_resolution

                    delete_lora_resolution(source)
                except Exception:
                    pass
                send_progress(
                    0.0,
                    f"LoRA download failed for '{source}'",
                    {
                        "status": "error",
                        "bucket": norm_type,
                        "stage": "lora_download",
                        "error": str(maybe_not_lora),
                    },
                )
                return {
                    "job_id": job_id,
                    "status": "error",
                    "type": "lora",
                    "id": source,
                    "message": f"LoRA download failed; removed from manifest: {maybe_not_lora}",
                }

        # Case 2: Generic path/url/hf download(s) using DownloadMixin
        # Normalize sources to a list
        paths: List[str] = []
        if isinstance(source, list):
            paths = [str(p) for p in source]
        elif isinstance(source, str):
            paths = [source]
        else:
            raise ValueError(
                "source must be a string or list of strings representing paths/urls/hf repos."
            )

        @ray.remote
        class ProgressAggregator:
            def __init__(self, total_items: int):
                self.total_items = max(1, int(total_items))
                # Per-item fractional progress (0..1) for fallback aggregation
                self.per_index_progress: Dict[int, float] = {}
                # Per-item byte accounting for more accurate aggregation on multi-path/folder downloads
                self.bytes_downloaded: Dict[int, int] = {}
                self.bytes_total: Dict[int, int] = {}
                self.last_overall: float = 0.0

            def update(
                self,
                index: int,
                frac: float,
                label: str,
                downloaded: Optional[int] = None,
                total: Optional[int] = None,
                filename: Optional[str] = None,
                message: Optional[str] = None,
            ):
                # Clamp per-item fraction
                frac = max(0.0, min(1.0, float(frac)))
                self.per_index_progress[index] = frac

                # Track byte-level progress when we know the total size
                if downloaded is not None and total is not None and total > 0:
                    try:
                        d = max(0, int(downloaded))
                        t = max(1, int(total))
                        # Ensure we never exceed the total for safety
                        self.bytes_downloaded[index] = min(d, t)
                        self.bytes_total[index] = t
                    except Exception:
                        # Best-effort only; fall back to fractional aggregation if anything goes wrong
                        pass

                # Prefer aggregate byte-based progress when we have enough information,
                # otherwise fall back to simple average of per-item fractions.
                overall_progress: float
                try:
                    total_bytes = sum(self.bytes_total.values())
                    if total_bytes > 0:
                        done_bytes = 0
                        for idx, t in self.bytes_total.items():
                            d = self.bytes_downloaded.get(idx, 0)
                            if d < 0:
                                d = 0
                            if d > t:
                                d = t
                            done_bytes += d
                        overall_progress = max(
                            0.0, min(1.0, done_bytes / float(total_bytes))
                        )
                    else:
                        # Fallback: average of known item fractions
                        overall_progress = sum(
                            self.per_index_progress.values()
                        ) / float(self.total_items)
                except Exception:
                    overall_progress = sum(self.per_index_progress.values()) / float(
                        self.total_items
                    )

                # Clamp to [0, 1]; allow progress to move down slightly when a new file
                # in a multi-file/folder download starts (so we reflect aggregate progress
                # across all bytes, not just the current file).
                overall_progress = max(0.0, min(1.0, overall_progress))
                self.last_overall = overall_progress

                if filename:
                    filename_parts = filename.split("_")
                    if len(filename_parts) > 1:
                        filename_parts = filename_parts[1:]
                    filename = "_".join(filename_parts)

                meta = {"label": label, "bucket": norm_type}
                if downloaded is not None:
                    meta["downloaded"] = int(downloaded)
                if total is not None:
                    meta["total"] = int(total)
                if filename is not None:
                    meta["filename"] = filename
                msg = message or f"Downloading {label}"
                try:
                    # Report aggregated overall progress to the websocket, not per-file fraction
                    return ray.get(
                        ws_bridge.send_update.remote(
                            job_id, overall_progress, msg, meta
                        )
                    )
                except Exception:
                    return False

            def complete(self, index: int, label: str):
                return self.update(index, 1.0, label, message=f"Completed {label}")

            def error(self, index: int, label: str, error_msg: str):
                try:
                    return ray.get(
                        ws_bridge.send_update.remote(
                            job_id,
                            self.last_overall,
                            error_msg,
                            {"label": label, "status": "error", "bucket": norm_type},
                        )
                    )
                except Exception:
                    return False

        @ray.remote
        def download_single(
            path: str, dest_dir: str, index: int, aggregator
        ) -> Dict[str, Any]:
            label = os.path.basename(path.rstrip("/")) or path
            try:

                def _cb(
                    downloaded: int,
                    total: Optional[int],
                    filename: Optional[str] = None,
                ):
                    frac = 0.0
                    if total and total > 0:
                        frac = max(0.0, min(1.0, downloaded / total))
                    ray.get(
                        aggregator.update.remote(
                            index, frac, label, downloaded, total, filename
                        )
                    )

                mixin = DownloadMixin()
                os.makedirs(dest_dir, exist_ok=True)
                mixin.logger.info(f"[{job_id}] Downloading {path} into {dest_dir}")
                result_path = mixin.download(path, dest_dir, progress_callback=_cb)
                ray.get(aggregator.complete.remote(index, label))
                return {"path": path, "status": "complete", "result_path": result_path}
            except Exception as e:
                ray.get(aggregator.error.remote(index, label, str(e)))
                return {"path": path, "status": "error", "error": str(e)}

        total = len(paths)
        aggregator = ProgressAggregator.remote(total)
        refs = [
            download_single.remote(p, base_save_dir, i, aggregator)
            for i, p in enumerate(paths, start=1)
        ]
        results = ray.get(refs)
        try:
            ray.get(
                ws_bridge.send_update.remote(
                    job_id,
                    1.0,
                    "All downloads complete",
                    {"status": "complete", "bucket": norm_type},
                )
            )
        except Exception:
            pass
        has_error = any(r.get("status") == "error" for r in results)
        return {
            "job_id": job_id,
            "status": "error" if has_error else "complete",
            "bucket": norm_type,
            "save_dir": base_save_dir,
            "results": results,
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        try:
            send_progress(0.0, str(e), {"status": "error", "error": str(e)})
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": str(e), "traceback": tb}


def _execute_preprocessor(
    preprocessor_name: str,
    input_path: str,
    job_id: str,
    send_progress: Callable[[Optional[float], str, Optional[Dict[str, Any]]], None],
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    preprocessor_info = get_preprocessor_info(preprocessor_name)
    cache = AuxillaryCache(
        input_path,
        preprocessor_name,
        start_frame,
        end_frame,
        kwargs,
        supports_alpha_channel=preprocessor_info.get("supports_alpha_channel", False),
    )

    media_type = cache.type
    send_progress(0.05, "Checking cache")

    if cache.is_cached():
        send_progress(1.0, "Cache found and returning")
        send_progress(1.0, "Complete", {"status": "complete"})
        return {
            "job_id": job_id,
            "status": "complete",
            "result_path": cache.get_result_path(),
            "type": media_type,
        }

    send_progress(0.1, "Loading preprocessor module")
    module = importlib.import_module(preprocessor_info["module"])
    preprocessor_class = getattr(module, preprocessor_info["class"])

    from src.preprocess.download_tracker import DownloadProgressTracker
    from src.preprocess import util as util_module

    tracker = DownloadProgressTracker(
        job_id,
        lambda p, m, md=None: send_progress(
            0.05 + (max(0.0, min(1.0, float(p))) * 0.15), m, md
        ),
    )
    util_module.DOWNLOAD_PROGRESS_CALLBACK = tracker.update_progress
    try:
        preprocessor = preprocessor_class.from_pretrained()
    finally:
        util_module.DOWNLOAD_PROGRESS_CALLBACK = None

    send_progress(0.2, "Preprocessor loaded")

    from src.preprocess.base_preprocessor import BasePreprocessor

    BasePreprocessor._mark_as_downloaded(preprocessor_name)

    def progress_callback(idx: int, total: int, message: str = None):
        total = max(1, int(total))
        frac = idx / float(total)
        scaled_progress = 0.2 + (max(0.0, min(1.0, frac)) * 0.6)
        send_progress(scaled_progress, message or f"Processing frame {idx} of {total}")

    try:
        if media_type == "video":
            frame_range = cache._get_video_frame_range()
            total_frames = len([f for f in frame_range if f not in cache.cached_frames])
            frames = cache.video_frames(batch_size=1)
            result = preprocessor(
                frames,
                job_id=job_id,
                progress_callback=progress_callback,
                total_frames=total_frames,
                **kwargs,
            )
        else:
            result = preprocessor(cache.image, job_id=job_id, **kwargs)

        result_path = cache.save_result(result)
        send_progress(1.0, "Result saved")
        send_progress(1.0, "Complete", {"status": "complete"})

        return {
            "status": "complete",
            "result_path": result_path,
            "type": cache.type,
        }

    except Exception as e:
        error_msg = f"Error processing {preprocessor_name}: {str(e)}"
        error_traceback = traceback.format_exc()
        logger.error(f"[{job_id}] Processing failed: {error_traceback}")
        try:
            send_progress(0.0, error_msg, {"status": "error", "error": error_msg})
        except Exception as ws_error:
            logger.error(
                f"[{job_id}] Processing failed AND websocket notification failed: {error_msg}, WS Error: {ws_error}"
            )
        return {
            "job_id": job_id,
            "status": "error",
            "error": error_msg,
            "traceback": error_traceback,
        }

    finally:
        empty_cache()


@ray.remote
def run_preprocessor(
    preprocessor_name: str,
    input_path: str,
    job_id: str,
    ws_bridge,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a preprocessor on input media
    """

    def send_progress(progress: float, message: str, metadata: Optional[Dict] = None):
        """Local send_progress that uses the passed ws_bridge"""
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            logger.debug(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update to websocket: {e}")

    return _execute_preprocessor(
        preprocessor_name,
        input_path,
        job_id,
        send_progress,
        start_frame=start_frame,
        end_frame=end_frame,
        **kwargs,
    )


@ray.remote
def run_engine_from_manifest(
    manifest_path: str,
    job_id: str,
    ws_bridge,
    inputs: Dict[str, Any],
    selected_components: Optional[Dict[str, Any]] = None,
    folder_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a manifest YAML with provided inputs and persist result to disk."""

    def send_progress(
        progress: float | None, message: str, metadata: Optional[Dict] = None
    ):
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
            if progress is not None:
                logger.info(f"[{job_id}] Progress: {progress*100:.1f}% - {message}")
            else:
                logger.info(f"[{job_id}] Progress: {message}")
        except Exception as e:
            logger.error(f"Failed to send progress update: {e}")

    # Track large objects so we can explicitly drop references in a finally block
    engine = None
    engine_pool_key = None
    engine_pooled = False
    raw = None
    config = None
    prepared_inputs: Dict[str, Any] = {}
    preprocessor_jobs: List[Dict[str, Any]] = []
    output = None

    try:
        from src.utils.yaml import load_yaml as load_manifest_yaml
        from src.manifest.loader import validate_and_normalize
        from src.engine.registry import UniversalEngine
        from src.utils.defaults import DEFAULT_CACHE_PATH
        import numpy as np
        from PIL import Image

        logger.info(manifest_path, "manifest_path")

        # Normalize manifest (handles v1 -> engine shape)
        raw = load_manifest_yaml(manifest_path)
        config = validate_and_normalize(raw)
        inputs = inputs or {}
        selected_components = selected_components or {}

        # Extract any audio inputs that should be persisted alongside the final video
        spec_block = (raw.get("spec") or {}) if isinstance(raw, dict) else {}
        audio_inputs_to_save = spec_block.get("audio_inputs_to_save") or []
        if not isinstance(audio_inputs_to_save, list):
            audio_inputs_to_save = []
        audio_inputs_to_save = [
            str(x) for x in audio_inputs_to_save if isinstance(x, (str, int))
        ]
        # Resolved file system paths for those audio inputs (filled after preprocessing)
        audio_input_paths: List[str] = []

        def _extract_ui_inputs() -> List[Dict[str, Any]]:
            spec_ui = (raw.get("spec") or {}).get("ui") or {}
            raw_inputs = spec_ui.get("inputs")
            if isinstance(raw_inputs, list):
                return raw_inputs
            normalized_ui = (config.get("ui") or {}).get("inputs")
            if isinstance(normalized_ui, list):
                return normalized_ui
            return []

        preprocessor_map: Dict[str, str] = {}
        for item in _extract_ui_inputs():
            if not isinstance(item, dict):
                continue
            input_id = item.get("id")
            preproc_ref = item.get("preprocessor_ref")
            if input_id and preproc_ref:
                preprocessor_map[input_id] = preproc_ref

        # Resolve engine settings
        engine_type = config.get("engine") or (config.get("spec") or {}).get("engine")
        model_type = config.get("type") or (config.get("spec") or {}).get("model_type")
        if isinstance(model_type, list):
            model_type = model_type[0] if model_type else None

        attention_type = selected_components.pop("attention", {}).get("name", None)

        input_kwargs = {
            "engine_type": engine_type,
            "yaml_path": manifest_path,
            "model_type": model_type,
            "selected_components": selected_components,
            "auto_memory_management": os.environ.get("AUTO_MEMORY_MANAGEMENT", False),
            **(config.get("engine_kwargs", {}) or {}),
        }

        if attention_type:
            input_kwargs["attention_type"] = attention_type

        engine_pool_key = _engine_pool_key(
            manifest_path=manifest_path,
            engine_type=engine_type,
            model_type=model_type,
            selected_components=selected_components or {},
            engine_kwargs=(config.get("engine_kwargs", {}) or {}),
            attention_type=attention_type,
            auto_memory_management=os.environ.get("AUTO_MEMORY_MANAGEMENT", False),
        )

        def _factory():
            return UniversalEngine(**input_kwargs)

        engine, engine_pooled = _get_warm_pool().acquire(engine_pool_key, _factory, allow_pool=True)

        # Compute FPS once so we don't capture the full engine/config inside callbacks.
        fps_for_video: int = 16
        has_fps = False
        try:
            fps_candidate = config.get("fps", None)
            if fps_candidate:
                fps_for_video = int(fps_candidate)
                has_fps = True
            else:
                impl = getattr(engine.engine, "implementation_engine", None)
                if impl is not None:
                    sig = inspect.signature(impl.run)
                    param = sig.parameters.get("fps")
                    if (
                        param is not None
                        and param.default is not inspect._empty
                        and param.default is not None
                    ):
                        fps_for_video = int(param.default)
        except Exception:
            fps_for_video = 16

        def _coerce_media_input(value: Any) -> tuple[Optional[str], Optional[bool]]:
            if isinstance(value, dict):
                path_candidate = (
                    value.get("input_path") or value.get("src") or value.get("path")
                )
                apply_flag = value.get("apply_preprocessor")
                path_str = path_candidate if isinstance(path_candidate, str) else None
                apply_bool = apply_flag if isinstance(apply_flag, bool) else None
                return path_str, apply_bool
            if isinstance(value, str):
                return value, None
            return None, None

        def _looks_like_video_file(path_str: str) -> bool:
            lower = (path_str or "").lower()
            return any(
                lower.endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm")
            )

        def _extract_input_video_path_for_audio_mux(
            raw_inputs: Dict[str, Any],
        ) -> Optional[str]:
            """
            Best-effort: find the main input video path so we can carry its audio
            into the final output for upscalers.
            """
            try:
                # Prefer the canonical id used by upscaler manifests.
                val = (raw_inputs or {}).get("video")
                p, _ = _coerce_media_input(val)
                if (
                    isinstance(p, str)
                    and _looks_like_video_file(p)
                    and os.path.isfile(p)
                ):
                    return p

                # Fallback: scan other inputs for a plausible video path.
                for _, v in (raw_inputs or {}).items():
                    p2, _ = _coerce_media_input(v)
                    if (
                        isinstance(p2, str)
                        and _looks_like_video_file(p2)
                        and os.path.isfile(p2)
                    ):
                        return p2
            except Exception:
                return None
            return None

        # Detect our upscale engines (SeedVR / FlashVSR) so we can preserve input audio.
        spec_engine = None
        try:
            spec_engine = (
                spec_block.get("engine") if isinstance(spec_block, dict) else None
            ) or engine_type
        except Exception:
            spec_engine = engine_type
        engine_name_lc = (
            str(spec_engine).strip().lower() if spec_engine is not None else ""
        )
        model_type_lc = (
            str(model_type).strip().lower() if model_type is not None else ""
        )
        is_upscaler_engine = (
            engine_name_lc in {"seedvr", "flashvsr"} or model_type_lc == "upscale"
        )
        input_video_for_audio_mux = (
            _extract_input_video_path_for_audio_mux(inputs or {})
            if is_upscaler_engine
            else None
        )

        prepared_inputs: Dict[str, Any] = {}
        preprocessor_jobs: List[Dict[str, Any]] = []
        for input_key, raw_value in inputs.items():
            if input_key in preprocessor_map:
                media_path, apply_flag = _coerce_media_input(raw_value)
                if media_path and isinstance(raw_value, dict):
                    prepared_inputs[input_key] = media_path
                    should_apply = apply_flag if isinstance(apply_flag, bool) else True
                    preprocessor_kwargs = raw_value.get("preprocessor_kwargs", {})
                    if should_apply:
                        preprocessor_jobs.append(
                            {
                                "input_id": input_key,
                                "preprocessor_name": preprocessor_map[input_key],
                                "input_path": media_path,
                                "preprocessor_kwargs": preprocessor_kwargs,
                            }
                        )
                else:
                    prepared_inputs[input_key] = raw_value
            else:
                prepared_inputs[input_key] = raw_value

        # Prepare job directory early (needed for previews)
        safe_job_id = _safe_fs_component(job_id, fallback="job")
        if folder_uuid:
            safe_folder_uuid = _safe_fs_component(folder_uuid, fallback="folder")
            job_dir = Path(DEFAULT_CACHE_PATH) / "engine_results" / safe_folder_uuid / safe_job_id
        else:
            job_dir = Path(DEFAULT_CACHE_PATH) / "engine_results" / safe_job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Unified saver usable for previews and final outputs
        def _mux_audio_into_video(
            video_path: str, audio_paths: List[str]
        ) -> Optional[str]:
            """
            Best-effort helper to mux one or more audio files into a video using ffmpeg.
            Returns the new video path on success, or None on failure.
            """
            import subprocess
            import os
            from src.utils.ffmpeg import get_ffmpeg_path

            try:
                valid_audio_paths = [
                    p for p in audio_paths if isinstance(p, str) and os.path.isfile(p)
                ]
                if not valid_audio_paths:
                    return None

                base = Path(video_path)
                # We first mux into a temporary file, then overwrite the original
                # video so that the final saved filename remains unchanged.
                temp_out_path = base.with_name(f"{base.stem}_with_audio{base.suffix}")

                cmd: List[str] = [get_ffmpeg_path(), "-y", "-i", video_path]
                for ap in valid_audio_paths:
                    cmd.extend(["-i", ap])

                if len(valid_audio_paths) == 1:
                    # Single audio track: simple stream copy, similar to test_humo.py
                    cmd.extend(
                        [
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0?",
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-shortest",
                            str(temp_out_path),
                        ]
                    )
                else:
                    # Multiple audio inputs: mix them down into a single track with amix.
                    # Audio inputs start from index 1 (0 is the video).
                    inputs_count = len(valid_audio_paths)
                    filter_inputs = "".join(f"[{i+1}:a]" for i in range(inputs_count))
                    filter_spec = f"{filter_inputs}amix=inputs={inputs_count}:dropout_transition=0[aout]"
                    cmd.extend(
                        [
                            "-filter_complex",
                            filter_spec,
                            "-map",
                            "0:v:0",
                            "-map",
                            "[aout]",
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-shortest",
                            "-movflags",
                            "+faststart",
                            str(temp_out_path),
                        ]
                    )

                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode != 0 or not temp_out_path.is_file():
                    logger.warning(
                        f"ffmpeg audio mux failed with code {proc.returncode}"
                    )
                    return None
                try:
                    # Overwrite the original file so callers keep the same filename.
                    temp_out_path.replace(base)
                except Exception as move_err:
                    logger.warning(
                        f"ffmpeg audio mux succeeded but failed to move into place: {move_err}"
                    )
                    return None
                return str(base)
            except Exception as e:
                logger.warning(f"Failed to mux audio into video: {e}")
                return None

        def _mux_audio_from_source_video(
            video_path: str, source_video_path: str
        ) -> Optional[str]:
            """
            Best-effort helper to mux the first audio track from `source_video_path`
            into `video_path` using ffmpeg.
            """
            import subprocess
            import os
            from src.utils.ffmpeg import get_ffmpeg_path

            try:
                if not (
                    isinstance(video_path, str)
                    and isinstance(source_video_path, str)
                    and os.path.isfile(video_path)
                    and os.path.isfile(source_video_path)
                ):
                    return None

                base = Path(video_path)
                # Mux into a temporary file, then overwrite the original.
                temp_out_path = base.with_name(f"{base.stem}_with_audio{base.suffix}")

                # Video from generated output (0), audio from source input (1).
                # Encode audio to AAC for MP4 compatibility; copy video stream.
                cmd: List[str] = [
                    get_ffmpeg_path(),
                    "-y",
                    "-i",
                    video_path,
                    "-i",
                    source_video_path,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(temp_out_path),
                ]
                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode != 0 or not temp_out_path.is_file():
                    return None
                try:
                    temp_out_path.replace(base)
                except Exception as move_err:
                    logger.warning(
                        f"ffmpeg audio mux succeeded but failed to move into place: {move_err}"
                    )
                    return None
                return str(base)
            except Exception as e:
                logger.warning(f"Failed to mux audio from source video: {e}")
                return None

        def save_output(
            output_obj,
            filename_prefix: str = "result",
            final: bool = False,
            audio_inputs: Optional[List[str]] = None,
        ):
            result_path: Optional[str] = None
            media_type: Optional[str] = None
            try:
                # String path passthrough
                if isinstance(output_obj, str):
                    result_path = output_obj
                    media_type = "path"
                # Single image
                elif isinstance(output_obj, Image.Image):
                    ext = f"png" if final else "jpg"
                    result_path = str(job_dir / f"{filename_prefix}.{ext}")
                    output_obj.save(result_path)
                    media_type = "image"
                # Sequence of frames
                elif isinstance(output_obj, list) and len(output_obj) > 0:
                    fps = fps_for_video or 16
                    result_path = str(job_dir / f"{filename_prefix}.mp4")

                    export_to_video(
                        output_obj,
                        result_path,
                        fps=int(fps),
                        quality=8.0 if final else 5.0,
                    )
                    media_type = "video"

                    # If this is the final video and we have audio inputs to save, try to mux them in.
                    if final and media_type == "video" and result_path and audio_inputs:
                        try:
                            muxed = _mux_audio_into_video(result_path, audio_inputs)
                            if muxed:
                                result_path = muxed
                        except Exception as mux_err:
                            logger.warning(
                                "Audio muxing failed; returning video-only output. "
                                f"Error: {mux_err}"
                            )

                    # Upscalers (SeedVR/FlashVSR): preserve input video audio if present.
                    if (
                        final
                        and media_type == "video"
                        and result_path
                        and is_upscaler_engine
                        and input_video_for_audio_mux
                    ):
                        try:
                            muxed = _mux_audio_from_source_video(
                                result_path, input_video_for_audio_mux
                            )
                            if muxed:
                                result_path = muxed
                        except Exception as mux_err:
                            logger.warning(
                                "Upscaler input audio muxing failed; returning video-only output. "
                                f"Error: {mux_err}"
                            )
                else:
                    # Fallback best-effort serialization
                    try:
                        arr = np.asarray(output_obj)  # type: ignore[arg-type]
                        result_path = str(job_dir / f"{filename_prefix}.png")
                        Image.fromarray(arr).save(result_path)
                        media_type = "image"
                    except Exception as e:
                        logger.error(f"Failed to save output: {e}")
                        result_path = str(job_dir / f"{filename_prefix}.txt")
                        with open(result_path, "w") as f:
                            f.write(str(type(output_obj)))
                        media_type = "unknown"
            except Exception as save_err:
                traceback.print_exc()
                logger.error(f"Failed to save output: {save_err}")
                raise
            return result_path, media_type


        total_steps = max(1, len(preprocessor_jobs) + 1)

        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Preprocessor jobs: {preprocessor_jobs}")

        for idx, job in enumerate(preprocessor_jobs):
            stage_start = idx / total_steps
            stage_span = 1.0 / total_steps

            def stage_send_progress(
                local_progress: Optional[float],
                message: str,
                metadata: Optional[Dict] = None,
            ):
                merged_meta = dict(metadata or {})
                merged_meta.setdefault("stage", "preprocessor")
                merged_meta.setdefault("input_id", job["input_id"])
                if merged_meta.get("status") == "complete":
                    merged_meta["status"] = "processing"
                if local_progress is None:
                    send_progress(None, message, merged_meta)
                    return
                bounded = max(0.0, min(1.0, float(local_progress)))
                send_progress(stage_start + bounded * stage_span, message, merged_meta)

            stage_send_progress(
                0.0,
                f"Running {job['preprocessor_name']} preprocessor for {job['input_id']}",
            )
            result = _execute_preprocessor(
                job["preprocessor_name"],
                job["input_path"],
                f"{job_id}:{job['input_id']}",
                stage_send_progress,
                **job.get("preprocessor_kwargs", {}),
            )
            if result.get("status") != "complete":
                raise RuntimeError(
                    result.get("error")
                    or f"Preprocessor {job['preprocessor_name']} failed"
                )

            prepared_inputs[job["input_id"]] = result.get("result_path")

        # Resolve concrete file paths for any audio inputs that should be saved with the final video
        if audio_inputs_to_save:
            for audio_key in audio_inputs_to_save:
                try:
                    key_str = str(audio_key)
                    val = prepared_inputs.get(key_str)
                    media_path, _ = (
                        _coerce_media_input(val)
                        if isinstance(val, dict)
                        else (val, None)
                    )
                    if (
                        isinstance(media_path, str)
                        and media_path not in audio_input_paths
                    ):
                        audio_input_paths.append(media_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve audio input '{audio_key}' for saving: {e}"
                    )

        engine_stage_start = len(preprocessor_jobs) / total_steps
        engine_stage_span = 1.0 / total_steps

        # Render-on-step callback that writes previews
        step_counter = {"i": 0}
        # Optional: limit how many *preview* saves we do per run (final result always saves).
        # Useful for test-suite runs to avoid saving a frame on every denoising step.
        try:
            _max_preview_saves = int(os.environ.get("APEX_RENDER_STEP_MAX_SAVES", "0") or "0")
        except Exception:
            _max_preview_saves = 0

        def render_on_step_callback(
            frames,
            is_result: bool = False,
            audio_inputs: Optional[List[str]] = None,
        ):
            try:
                idx = step_counter["i"]
                step_counter["i"] = idx + 1
                # Enforce preview-save limit (do not block final outputs).
                if not is_result and _max_preview_saves > 0 and idx >= _max_preview_saves:
                    return None, None
                # Persist preview to cache and notify over websocket with metadata only
                result_path, media_type = save_output(
                    frames,
                    filename_prefix=f"preview_{idx:04d}" if not is_result else "result",
                    final=is_result,
                    audio_inputs=audio_inputs if is_result else None,
                )
                logger.info(
                    f"Preview saved to {result_path} with media type {media_type}"
                )
                try:
                    # Send an update that does not overwrite progress (progress=None)
                    logger.info(
                        f"Sending preview websocket update at step {idx} with result path {result_path} and media type {media_type}"
                    )
                    send_progress(
                        1.0 if is_result else None,
                        f"Preview frame {idx}",
                        {
                            "status": "complete" if is_result else "preview",
                            "preview_path": result_path,
                            "type": media_type,
                            "index": idx,
                        },
                    )
                except Exception as se:
                    logger.warning(
                        f"Failed sending preview websocket update at step {idx}: {se}"
                    )
                return result_path, media_type
            except Exception as e:
                logger.warning(f"Preview save failed at step {step_counter['i']}: {e}")

        def render_on_step_callback_audio_video(
            output_obj: Tuple[np.ndarray, np.ndarray],
            is_result: bool = False,
        ):
            idx = step_counter["i"]
            step_counter["i"] = idx + 1
            # Enforce preview-save limit (do not block final outputs).
            if not is_result and _max_preview_saves > 0 and idx >= _max_preview_saves:
                return None, None
            logger.info(
                f"Saving audio video output at step {idx} with output object {output_obj[0].shape} and {output_obj[1].shape}"
            )
            if model_type.lower() == "ovi":
                save_func = save_video_ovi
            else:
                save_func = save_video_ltx2
            result_path, media_type = save_func(
                output_obj[0],
                output_obj[1],
                filename_prefix=f"preview_{idx:04d}" if not is_result else "result",
                job_dir=job_dir,
            )
            logger.info(f"Preview saved to {result_path} with media type {media_type}")
            try:
                logger.info(
                    f"Sending preview websocket update at step {idx} with result path {result_path} and media type {media_type}"
                )
                send_progress(
                    1.0 if is_result else None,
                    f"Preview frame {idx}",
                    {
                        "status": "complete" if is_result else "preview",
                        "preview_path": result_path,
                        "type": media_type,
                        "index": idx,
                    },
                )
            except Exception as se:
                logger.warning(
                    f"Failed sending preview websocket update at step {idx}: {se}"
                )
            return result_path, media_type

        # Progress callback forwarded into the engine
        def progress_callback(
            progress: float, message: str, metadata: Optional[Dict] = None
        ):
            logger.info(f"Progress callback: {progress}, {message}, {metadata}")
            if progress is None:
                send_progress(None, message, metadata)
                return
            bounded = max(0.0, min(1.0, progress))
            send_progress(
                engine_stage_start + bounded * engine_stage_span, message, metadata
            )

        
        # Persist a snapshot of the invocation into the structured `runs` directory

        render_func = (
            render_on_step_callback_audio_video
            if model_type.lower() == "ovi" or engine_type.lower() == "ltx2"
            else render_on_step_callback
        )
    
        if os.environ.get("ENABLE_PERSIST_RUN_CONFIG", "true") == "true":
            _persist_run_config(manifest_path, input_kwargs, prepared_inputs)

        # get if the model is video or image
        if has_fps:
            render_on_step = (
                os.environ.get("ENABLE_VIDEO_RENDER_STEP", "true") == "true"
            )
        else:
            render_on_step = (
                os.environ.get("ENABLE_IMAGE_RENDER_STEP", "true") == "true"
            )

        output = engine.run(
            **(prepared_inputs or {}),
            progress_callback=progress_callback,
            render_on_step=render_on_step,
            render_on_step_callback=render_func,
        )
        # Avoid logging giant tensors/lists (can be very large and can amplify RSS).
        try:
            if isinstance(output, tuple):
                logger.info(
                    "Engine output: tuple("
                    + ", ".join(type(x).__name__ for x in output)
                    + ")"
                )
            else:
                logger.info(f"Engine output type: {type(output).__name__}")
        except Exception:
            pass

        # OVI models return a tuple of (video_tensor, audio_numpy). Use a dedicated
        # saver so we correctly embed the generated audio into the MP4 output.
        if isinstance(model_type, str) and model_type.lower() == "ovi" or model_type.lower() == "ti2v":
            result_path, media_type = render_func(
                output,
                is_result=True,
            )
        else:
            result_path, media_type = render_func(
                output[0],
                is_result=True,
                audio_inputs=audio_input_paths or None,
            )
        try:
            logger.info(f"Result path: {result_path}")
        except Exception:
            pass

        # Post-warm: keep engine warm after a successful run when pooled.
        if engine_pooled and engine_pool_key:
            try:
                _get_warm_pool().release(engine_pool_key)
            except Exception:
                pass
        else:
            # Legacy behavior when not pooled: offload and clear caches.
            try:
                engine.offload_engine()
            except Exception as e:
                logger.warning(f"Failed to offload engine: {e}")
        send_progress(1.0, "Complete", {"status": "complete"})
        return {"status": "complete", "result_path": result_path, "type": media_type}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb, "traceback")
        try:
            send_progress(0.0, str(e), {"status": "error", "error": str(e)})
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": str(e), "traceback": tb}
    finally:
        # Ensure we aggressively release references.
        # If the engine is pooled, do NOT clear torch caches (keeps it warm).
        # If not pooled, do best-effort offload + cache clearing.
        try:
            if (not engine_pooled) and engine is not None:
                try:
                    engine.offload_engine()
                except Exception:
                    pass
            engine = None
            raw = None
            config = None
            prepared_inputs = {}
            preprocessor_jobs = []
            output = None
        except Exception as cleanup_err:
            logger.warning(f"run_engine_from_manifest cleanup failed: {cleanup_err}")
        _aggressive_ram_cleanup(clear_torch_cache=not bool(engine_pooled))


@ray.remote
def run_frame_interpolation(
    input_path: str,
    target_fps: float,
    job_id: str,
    ws_bridge,
    exp: Optional[int] = None,
    scale: float = 1.0,
) -> Dict[str, Any]:
    """Run RIFE frame interpolation on a video and save an output video.

    Args:
        input_path: Path to input video file
        target_fps: Desired output frames per second
        job_id: Job id for websocket/job tracking
        ws_bridge: Ray actor bridge for websocket updates
        exp: Optional exponent for 2**exp interpolation (overrides target_fps if provided)
        scale: RIFE scale (for UHD set 0.5)

    Returns:
        Dict with status, result_path and type
    """

    def send_update(
        progress: float | None, message: str, metadata: Optional[Dict[str, Any]] = None
    ):
        try:
            ray.get(ws_bridge.send_update.remote(job_id, progress, message, metadata))
        except Exception:
            pass

    from pathlib import Path
    from src.utils.defaults import DEFAULT_CACHE_PATH

    try:
        from src.postprocess.rife.rife import RifePostprocessor

        send_update(0.05, "Initializing RIFE")
        pp = RifePostprocessor(target_fps=target_fps, exp=exp, scale=scale)

        send_update(0.15, "Running frame interpolation")

        # Wire progress from postprocessor (scale 0.2 -> 0.95)
        def frame_progress(idx: int, total: int, message: Optional[str] = None):
            try:
                total = max(1, int(total))
                frac = max(0.0, min(1.0, float(idx) / float(total)))
                scaled = 0.20 + frac * 0.75
                send_update(scaled, message or f"Interpolating {idx}/{total}")
            except Exception:
                pass

        frames = pp(
            input_path,
            target_fps=target_fps,
            exp=exp,
            scale=scale,
            progress_callback=frame_progress,
        )

        # Save output video (video-only first), then mux original audio if present
        import subprocess
        import shutil

        job_dir = Path(DEFAULT_CACHE_PATH) / "postprocessor_results" / _safe_fs_component(
            job_id, fallback="job"
        )
        job_dir.mkdir(parents=True, exist_ok=True)

        video_only_path = str(job_dir / "result_video.mp4")
        final_out_path = str(job_dir / "result.mp4")

        fps_to_write = int(max(1, round(target_fps)))
        export_to_video(frames, video_only_path, fps=fps_to_write, quality=8.0)

        # Try to mux audio from input_path into the final output without changing rate/tempo
        # If no audio is present, fall back to the video-only file
        try:
            from src.utils.ffmpeg import get_ffmpeg_path

            # Use ffmpeg with stream copy to preserve original audio rate/tempo
            # -map 0:v:0 takes video from the first input (our generated video)
            # -map 1:a:0? takes the first audio track from the second input if it exists
            ffmpeg_cmd = [
                get_ffmpeg_path(),
                "-y",
                "-i",
                video_only_path,
                "-i",
                input_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-shortest",
                "-movflags",
                "+faststart",
                final_out_path,
            ]
            proc = subprocess.run(ffmpeg_cmd, capture_output=True)
            if proc.returncode != 0:
                # If muxing failed (e.g., no audio stream), just use the video-only output
                shutil.move(video_only_path, final_out_path)
        except Exception as e:
            logger.error(f"Failed to mux audio: {e}")
            # On any unexpected error, fall back to video-only output
            try:
                shutil.move(video_only_path, final_out_path)
            except Exception:
                # If move also fails, keep path consistent
                final_out_path = video_only_path

        send_update(
            1.0,
            "Complete",
            {"status": "complete", "result_path": final_out_path, "type": "video"},
        )
        return {"status": "complete", "result_path": final_out_path, "type": "video"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        try:
            send_update(0.0, str(e), {"status": "error", "error": str(e)})
        except Exception:
            pass
        return {"job_id": job_id, "status": "error", "error": str(e), "traceback": tb}
