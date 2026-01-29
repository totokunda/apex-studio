import os
from functools import lru_cache, partial
from loguru import logger
import anyio
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path, get_config_path, get_lora_path
from src.utils.compute import (
    get_compute_capability,
    validate_compute_requirements,
    ComputeCapability,
)
from src.engine import UniversalEngine
import traceback

router = APIRouter(prefix="/manifest", tags=["manifest"])

# Base path to manifest directory
MANIFEST_BASE_PATH = Path(__file__).parent.parent.parent / "manifest"




# Cache the system's compute capability (it doesn't change during runtime)
_SYSTEM_COMPUTE_CAPABILITY: Optional[ComputeCapability] = None


async def _run_blocking(func, *args, **kwargs):
    """
    Run blocking (sync) work in a worker thread so async request handlers don't
    block the event loop.
    """
    return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))


def _get_system_compute_capability() -> ComputeCapability:
    """Get the system's compute capability (cached)."""
    global _SYSTEM_COMPUTE_CAPABILITY
    if _SYSTEM_COMPUTE_CAPABILITY is None:
        _SYSTEM_COMPUTE_CAPABILITY = get_compute_capability()
    return _SYSTEM_COMPUTE_CAPABILITY


class ModelTypeInfo(BaseModel):
    key: str
    label: str
    description: str


class ModelCategoryInfo(ModelTypeInfo):
    pass


def _normalize_subengine_relative_path(yaml_path: str) -> Optional[str]:
    """
    Normalise a sub-engine YAML reference into a path relative to MANIFEST_BASE_PATH.

    Expected common forms:
    - 'manifest/engine/longcat/longcat-13b-text-to-video-1.0.0.v1.yml'
    - 'longcat/longcat-13b-text-to-video-1.0.0.v1.yml'

    Returns a string suitable to pass as `relative_path` into `_load_and_enrich_manifest`,
    or None if the path cannot be interpreted.
    """
    if not isinstance(yaml_path, str) or not yaml_path.strip():
        return None

    yaml_path = yaml_path.strip()
    prefix = "manifest/engine/"
    if yaml_path.startswith(prefix):
        return yaml_path[len(prefix) :]

    # If it already looks like a path relative to manifest/engine, just return it
    return yaml_path


def _load_and_enrich_manifest(relative_path: str) -> Dict[Any, Any]:
    """Load a manifest by relative path and enrich it with runtime info."""
    file_path = MANIFEST_BASE_PATH / relative_path
    content = load_yaml_content(file_path)

    # ----------------- Attention backends enrichment (name/label/desc) ----------------- #
    spec = content.get("spec", {}) if isinstance(content, dict) else {}
    metadata = content.get("metadata", {}) if isinstance(content, dict) else {}

    # ----------------- Sub-engine components merge (for composite manifests) ----------------- #
    # Some manifests (e.g. interactive variants) define `spec.sub_engines` pointing at other
    # engine YAMLs. For those, we want the enriched document to expose a merged view of all
    # components from the sub-engines (plus any locally defined ones), with duplicates removed.
    sub_engines = spec.get("sub_engines")
    if isinstance(sub_engines, list):
        merged_components: list[Dict[str, Any]] = []
        seen_keys: set[tuple] = set()
        merged_loras: list[Dict[str, Any]] = []
        seen_loras: set[tuple] = set()

        def _component_key(component: Dict[str, Any]) -> tuple:
            """
            Build a lightweight identity key for de-duplicating components.
            This intentionally only uses a subset of fields that should be stable
            across manifests for the same logical component.
            """
            if not isinstance(component, dict):
                return ("__invalid__", id(component))
            return (
                component.get("type"),
                component.get("name"),
                component.get("base"),
                component.get("label"),
            )

        def _add_components(components: Any) -> None:
            if not isinstance(components, list):
                return
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                key = _component_key(comp)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_components.append(comp)

        def _lora_key(lora: Dict[str, Any]) -> tuple:
            return (
                lora.get("source"),
                lora.get("name"),
                lora.get("label"),
            )

        def _add_loras(loras: Any) -> None:
            if not isinstance(loras, list):
                return

            for lora in loras:
                if not isinstance(lora, dict):
                    continue
                key = _lora_key(lora)
                if key in seen_loras:
                    continue
                seen_loras.add(key)
                merged_loras.append(lora)

        # Start with any components already defined on the parent manifest
        _add_components(spec.get("components") or [])
        _add_loras(spec.get("loras") or [])
        # Then merge in components from each sub-engine manifest
        for sub in sub_engines:
            if not isinstance(sub, dict):
                continue
            sub_yaml = sub.get("yaml")
            rel = _normalize_subengine_relative_path(sub_yaml)
            if not rel:
                continue
            try:
                sub_doc = _load_and_enrich_manifest(rel)
            except HTTPException:
                # If a sub-engine cannot be loaded, skip it instead of failing the parent
                continue
            if not isinstance(sub_doc, dict):
                continue
            sub_spec = sub_doc.get("spec", {})
            if not isinstance(sub_spec, dict):
                continue
            _add_components(sub_spec.get("components") or [])
            _add_loras(sub_spec.get("loras") or [])
        # If we collected anything, overwrite components on the parent spec
        if merged_components:
            if "spec" not in content or not isinstance(content["spec"], dict):
                content["spec"] = {}
            content["spec"]["components"] = merged_components
        if merged_loras:
            if "spec" not in content or not isinstance(content["spec"], dict):
                content["spec"] = {}
            content["spec"]["loras"] = merged_loras

    configured_attention = spec.get("support_attention")
    if configured_attention is None:
        configured_attention = spec.get("attention_types")
    if isinstance(configured_attention, list):
        attention_allowed: Optional[List[str]] = [
            x for x in configured_attention if isinstance(x, str)
        ]
    else:
        attention_allowed = None  # fall back to all available backends

    attention_options = _build_attention_options(attention_allowed)
    if "spec" not in content:
        content["spec"] = {}
    content["spec"]["attention_types_detail"] = attention_options

    # Enrich LoRA entries
    #
    # Important: keep the manifest YAML `spec.loras[*].source` as the user-provided
    # identifier (URN/spec/url). For UI convenience we expose the resolved local
    # downloaded file path(s) via `source` (backwards-compat) and `local_paths`,
    # while preserving the original identifier in `remote_source`.
    required_loras_downloaded = True
    for lora_index, lora in enumerate(content.get("spec", {}).get("loras", [])):
        if isinstance(lora, str):
            remote_source = lora
            try:
                from src.utils.lora_resolution import resolve_lora_local_paths

                local_paths = resolve_lora_local_paths(remote_source)
            except Exception:
                local_paths = []

            # Name/label best-effort (avoid showing entire URN in UI)
            lora_basename = os.path.basename(remote_source)
            lora_name = (
                lora_basename.split(".")[0] if "." in lora_basename else lora_basename
            )
            try:
                if (
                    remote_source.startswith("urn:air:")
                    or remote_source.startswith("civitai:")
                    or remote_source.startswith("civitai-file:")
                ):
                    tail = remote_source.rsplit(":", 1)[-1]
                    tail = tail.split("@", 1)[0]
                    tail = tail.split(".", 1)[0]
                    if tail:
                        lora_name = tail
            except Exception:
                pass

            out_lora = {
                "label": lora_name,
                "name": lora_name,
                "scale": 1.0,
                "remote_source": remote_source,
            }
            if local_paths:
                out_lora["is_downloaded"] = True
                out_lora["source"] = local_paths[0]
                out_lora["local_paths"] = local_paths
            else:
                out_lora["is_downloaded"] = False
                out_lora["source"] = remote_source
                out_lora["local_paths"] = []
                required_loras_downloaded = False
            content["spec"]["loras"][lora_index] = out_lora
        elif isinstance(lora, dict):
            out_lora = dict(lora)
            # If previous enrichments ran, `source` may already be a local path. Prefer `remote_source` if present.
            remote_source = (
                out_lora.get("remote_source")
                or out_lora.get("source")
                or out_lora.get("path")
                or out_lora.get("url")
                or out_lora.get("remote")
            )
            remote_source = remote_source if isinstance(remote_source, str) else ""
            try:
                from src.utils.lora_resolution import resolve_lora_local_paths

                local_paths = (
                    resolve_lora_local_paths(remote_source) if remote_source else []
                )
            except Exception:
                local_paths = []

            out_lora["remote_source"] = (
                remote_source or out_lora.get("remote_source") or out_lora.get("source")
            )
            if local_paths:
                out_lora["is_downloaded"] = True
                out_lora["source"] = local_paths[0]
                out_lora["local_paths"] = local_paths
            else:
                out_lora["is_downloaded"] = False
                out_lora["source"] = remote_source or out_lora.get("source")
                out_lora["local_paths"] = []
                if out_lora.get("required", False):
                    required_loras_downloaded = False
            content["spec"]["loras"][lora_index] = out_lora

    # Enrich components entries
    for component_index, component in enumerate(
        content.get("spec", {}).get("components", [])
    ):
        is_component_downloaded = True
        if config_path := component.get("config_path"):
            is_downloaded = DownloadMixin.is_downloaded(
                config_path, get_components_path()
            )
            if is_downloaded:
                component["config_path"] = is_downloaded

        if component.get("type") == "scheduler":
            options = component.get("scheduler_options", []) or []
            is_scheduler_downloaded = False
            has_downloadable_config = False
            for idx, option in enumerate(options):
                if option.get("config_path"):
                    has_downloadable_config = True
                    is_downloaded = DownloadMixin.is_downloaded(
                        option.get("config_path"), get_config_path()
                    )
                    if is_downloaded is not None:
                        is_scheduler_downloaded = True
                        options[idx]["config_path"] = is_downloaded
                    is_downloaded = DownloadMixin.is_downloaded(
                        option.get("config_path"), get_components_path()
                    )
                    if is_downloaded is not None:
                        is_scheduler_downloaded = True
                        options[idx]["config_path"] = is_downloaded

            if not is_scheduler_downloaded and has_downloadable_config:
                is_component_downloaded = False

            component["scheduler_options"] = options

        any_path_downloaded = False
        for index, model_path in enumerate(component.get("model_path", [])):
            is_downloaded = DownloadMixin.is_downloaded(
                model_path.get("path"), get_components_path()
            )
            if is_downloaded is not None:
                model_path["is_downloaded"] = True
                model_path["path"] = is_downloaded
                any_path_downloaded = True
            else:
                model_path["is_downloaded"] = False

            component["model_path"][index] = model_path

        any_extra_path_downloaded = False
        for index, model_path in enumerate(component.get("extra_model_paths", [])):
            path = (
                model_path.get("path") if isinstance(model_path, dict) else model_path
            )
            out_model_path = {}
            is_downloaded = DownloadMixin.is_downloaded(path, get_components_path())
            if is_downloaded is not None:
                out_model_path["is_downloaded"] = True
                out_model_path["path"] = is_downloaded
                any_extra_path_downloaded = True
            else:
                out_model_path["is_downloaded"] = False
                out_model_path["path"] = path

            component["extra_model_paths"][index] = out_model_path

        if (not any_path_downloaded and len(component.get("model_path", [])) > 0) or (
            not any_extra_path_downloaded
            and len(component.get("extra_model_paths", [])) > 0
        ):
            is_component_downloaded = False

        component["is_downloaded"] = is_component_downloaded
        content["spec"]["components"][component_index] = component

    # Convenience fields for filtering and compatibility with previous ManifestInfo
    # Normalize and expose common metadata at the top level
    # ID, name, model
    content["id"] = metadata.get("id", "")
    content["name"] = metadata.get("name", "")
    content["model"] = metadata.get("model", "")
    content["model_type"] = spec.get("model_type", [])
    content["categories"] = metadata.get("categories", [])
    # Other top-level convenience fields
    content["version"] = str(metadata.get("version", ""))
    content["description"] = metadata.get("description", "")
    content["tags"] = [str(t) for t in metadata.get("tags", [])]
    content["author"] = metadata.get("author", "")
    content["license"] = metadata.get("license", "")
    content["demo_path"] = metadata.get("demo_path", "")
    # Keep relative path for downstream use
    content["full_path"] = relative_path
    # Manifest-level downloaded flag: true if there are components and all are downloaded
    components_list = content.get("spec", {}).get("components", []) or []
    content["downloaded"] = (
        bool(components_list)
        and all(
            isinstance(c, dict) and c.get("is_downloaded", False)
            for c in components_list
        )
        and required_loras_downloaded
    )

    # Compute compatibility check
    compute_requirements = spec.get("compute_requirements")
    if compute_requirements:
        system_capability = _get_system_compute_capability()
        is_compatible, compatibility_error = validate_compute_requirements(
            compute_requirements, system_capability
        )
        content["compute_compatible"] = is_compatible
        content["compute_compatibility_error"] = compatibility_error
        content["compute_requirements_present"] = True
    else:
        # No compute requirements means it's compatible with all systems
        content["compute_compatible"] = True
        content["compute_compatibility_error"] = None
        content["compute_requirements_present"] = False

    return content


def get_manifest(manifest_id: str):
    """Get the actual YAML content of a specific manifest by name."""
    # Resolve manifest path via cached id->path index to avoid full list load
    id_index = _get_manifest_id_index()
    relative_path = id_index.get(manifest_id)
    if not relative_path:
        raise HTTPException(
            status_code=404, detail=f"Manifest not found: {manifest_id}"
        )
    # Load and enrich only the requested manifest
    return _load_and_enrich_manifest(relative_path)


def _get_all_manifest_files_uncached() -> List[Dict[str, Any]]:
    """Scan manifest directory and return all enriched manifest contents (no cache).

    Filters out manifests that are not compatible with the current system's compute capabilities.
    """
    manifests: List[Dict[str, Any]] = []

    for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if file.endswith(".yml") and not file.startswith("shared"):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(MANIFEST_BASE_PATH)

                # Load and enrich each manifest directly
                enriched = _load_and_enrich_manifest(str(relative_path))

                # Only include manifests that are compatible with the current system
                if (
                    enriched.get("compute_compatible", True)
                    and enriched.get("spec", {}).get("ui", None) is not None
                ):
                    manifests.append(enriched)

    return manifests


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _get_all_manifest_files_cached(cache_key: str) -> List[Dict[str, Any]]:
    # cache_key is only used to differentiate cache entries; actual logic ignores it
    return _get_all_manifest_files_uncached()


def get_all_manifest_files() -> List[Dict[str, Any]]:
    """Return all enriched manifests, optionally cached based on environment variables.

    Controls:
    - APEX_MANIFEST_CACHE or APEX_MANIFEST_CACHE_ENABLED: enable caching when truthy (default disabled)
    - APEX_MANIFEST_CACHE_BUSTER: changing this string invalidates the cache (e.g., set to a timestamp)
    """
    enabled = _env_truthy(
        os.getenv("APEX_MANIFEST_CACHE", os.getenv("APEX_MANIFEST_CACHE_ENABLED", "0"))
    )
    if not enabled:
        return _get_all_manifest_files_uncached()
    buster = os.getenv("APEX_MANIFEST_CACHE_BUSTER", "")
    cache_key = f"v1:{buster}"
    return _get_all_manifest_files_cached(cache_key)


def _build_manifest_id_index_uncached() -> Dict[str, str]:
    """
    Build a mapping of manifest_id -> relative_path (str) without enriching all manifests.
    """
    index: Dict[str, str] = {}
    for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if not file.endswith(".yml") or file.startswith("shared"):
                continue
            file_path = Path(root) / file
            relative_path = str(file_path.relative_to(MANIFEST_BASE_PATH))
            try:
                data = load_yaml_content(file_path)
                manifest_id = ""
                if isinstance(data, dict):
                    meta = data.get("metadata", {})
                    if isinstance(meta, dict):
                        manifest_id = str(meta.get("id", "")).strip()
                if manifest_id:
                    index.setdefault(manifest_id, relative_path)
            except HTTPException:
                continue
    return index


@lru_cache(maxsize=1)
def _get_manifest_id_index_cached(cache_key: str) -> Dict[str, str]:
    return _build_manifest_id_index_uncached()


def _get_manifest_id_index() -> Dict[str, str]:
    """
    Get the manifest id index, optionally cached controlled by the same env flags
    used for manifest list caching.
    """
    enabled = _env_truthy(
        os.getenv("APEX_MANIFEST_CACHE", os.getenv("APEX_MANIFEST_CACHE_ENABLED", "0"))
    )
    if not enabled:
        return _build_manifest_id_index_uncached()
    buster = os.getenv("APEX_MANIFEST_CACHE_BUSTER", "")
    cache_key = f"v1:{buster}"
    return _get_manifest_id_index_cached(cache_key)


def load_yaml_content(file_path: Path) -> Dict[Any, Any]:
    """Load and return YAML file content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Manifest file not found")
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=500, detail=f"Error parsing YAML file: {str(e)}"
        )


def _sizeof_local_path(path_str: str) -> int:
    """
    Compute the total size in bytes of a local file or directory.
    """
    p = Path(path_str)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        total = 0
        for root, _dirs, files in os.walk(p):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except Exception:
                    # Best-effort; ignore unreadable files
                    pass
        return total
    return 0


# ----------------------------- Attention Helpers ----------------------------- #
def _attention_label_description_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Centralised mapping for attention backend labels and short descriptions.
    """
    label_map = {
        "sdpa": "PyTorch SDPA",
        "sdpa_varlen": "PyTorch SDPA (VarLen)",
        "sdpa_streaming": "SDPA Streaming",
        "flash": "FlashAttention-2",
        "flash3": "FlashAttention-3",
        "sage": "SageAttention",
        "xformers": "xFormers",
        "flex": "Flex Attention",
        "xla_flash": "XLA Flash Attention",
        "flex-block-attn": "Flex Block Attention",
        # Some stacks use "efficient_sdpa" naming; others register this as
        # "efficient_dot_product_attention". We support both labels.
        "efficient_sdpa": "Efficient SDPA",
        "efficient_dot_product_attention": "Efficient Dot Product Attention",
        "block-sparse-attn": "Block Sparse Attention",
        "metal_flash": "Metal Flash Attention",
        "metal_flash-varlen": "Metal Flash Attention (VarLen)",
        # The attention registry uses an underscore variant; support it too.
        "metal_flash_varlen": "Metal Flash Attention (VarLen)",
    }

    description_map = {
        "sdpa": "Built-in torch scaled_dot_product_attention backend.",
        "sdpa_varlen": "VarLen wrapper using SDPA compatible with flash-attn varlen APIs.",
        "sdpa_streaming": "Streaming softmax SDPA variant for long sequences.",
        "flash": "NVIDIA FlashAttention-2 kernel (fast, memory-efficient).",
        "flash3": "FlashAttention-3 kernel via flash_attn_interface.",
        "metal_flash": "Metal Flash Attention kernel for MPS.",
        "metal_flash-varlen": "Variable Length kernel for MPS.",
        "metal_flash_varlen": "Variable Length kernel for MPS.",
        "sage": "SageAttention kernel backend.",
        "xformers": "xFormers memory-efficient attention implementation.",
        "flex": "PyTorch Flex Attention (experimental flexible masks).",
        "xla_flash": "XLA/TPU Flash Attention kernel.",
        "flex-block-attn": "Flex Block Attention kernel.",
        # Slowest fallbacks (kept last in the UI by default)
        "efficient_sdpa": "Fallback 'efficient SDPA' attention implementation (slowest).",
        "efficient_dot_product_attention": "Efficient Dot Product Attention kernel (slowest).",
        "block-sparse-attn": "Block Sparse Attention kernel.",
    }

    return label_map, description_map


def _attention_backend_sort_key(name: str) -> tuple[int, str]:
    """
    Sort attention backends from fastest to slowest for UI display.

    Notes:
    - This is a heuristic ordering for a better default UX; actual performance
      depends on hardware, shapes, and installed kernels.
    - We intentionally force the "efficient" SDPA fallback(s) to the end.
    """
    # Heuristic fastest -> slowest.
    # Keep this list small and opinionated; unknown backends fall after known
    # ones, but before the explicit slowest fallbacks.
    fastest_to_slowest = [
        # Modern fused kernels
        "flash3",
        "flash",
        "sage",
        # Platform-specific fused kernels
        "metal_flash",
        "metal_flash_varlen",
        "metal_flash-varlen",
        "xla_flash",
        # Built-in / portable paths
        "sdpa_varlen",
        "sdpa_streaming",
        "sdpa",
        "xformers",
        "flex",
        "flex-block-attn",
        "block-sparse-attn",
    ]

    # Explicit slowest fallbacks: always last (regardless of any unknowns).
    slowest_fallbacks = {"efficient_sdpa", "efficient_dot_product_attention"}

    if name in slowest_fallbacks:
        return (9_000, name)

    rank_map = {k: i for i, k in enumerate(fastest_to_slowest)}
    rank = rank_map.get(name, 5_000)
    return (rank, name)


def _build_attention_options(
    allowed: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Build a list of attention backend options from the attention registry, each
    containing name, label and description. If `allowed` is provided, the list is
    filtered to those names; otherwise all installed/available backends are used.
    """
    # Local import to avoid import cycles at startup
    try:
        from src.attention.functions import (
            verify_attention_backends,
        )
    except Exception as e:
        # If attention stack cannot be imported, return an empty list gracefully
        return []

    label_map, description_map = _attention_label_description_maps()

    # "Installed" or runtime-available attention backends
    available_keys = set(verify_attention_backends())

    if allowed is not None:
        allowed_set = {a for a in allowed if isinstance(a, str)}
        keys = list(available_keys.intersection(allowed_set))
    else:
        # If the manifest does not specify support/attention types, expose all
        # available implementations.
        keys = list(available_keys)

    # Default ordering: fastest -> slowest for a better UX.
    final_keys = sorted(keys, key=_attention_backend_sort_key)

    results: List[Dict[str, str]] = []
    for key in final_keys:
        label = label_map.get(key, key.replace("_", " ").title())
        desc = description_map.get(key, f"{key} attention backend.")
        results.append(
            {
                "name": key,
                "label": label,
                "description": desc,
            }
        )

    return results


def _list_model_types_sync() -> List[ModelTypeInfo]:
    """Blocking implementation for list_model_types()."""
    if not MANIFEST_BASE_PATH.exists():
        return []

    # Friendly labels and short descriptions per known category.
    # This covers all categories currently present under MANIFEST_BASE_PATH
    # (see the YAML manifests), but the code below still gracefully handles
    # any new categories that may be added in the future.
    label_map = {
        "text-to-video": "Text to Video",
        "image-to-video": "Image to Video",
        "video-to-video": "Video to Video",
        "text-to-image": "Text to Image",
        "audio-to-video": "Audio to Video",
        "reference-image-to-video": "Reference Image to Video",
        "video-edit": "Video Edit",
        "image-edit": "Image Edit",
        "image-to-image": "Image to Image",
        "inpaint": "Inpaint",
        "control": "Control / Animation",
        "edit": "Edit",
    }

    description_map = {
        "text-to-video": "Generate videos from text prompts.",
        "image-to-video": "Animate or transform images into videos.",
        "video-to-video": "Transform an input video with a new style or prompt.",
        "text-to-image": "Generate images from text prompts.",
        "audio-to-video": "Generate videos from audio (and optionally images or video).",
        "reference-image-to-video": "Generate videos guided by a reference image.",
        "video-edit": "Edit or transform existing videos using prompts and tools.",
        "image-edit": "Edit or transform existing images using prompts and tools.",
        "image-to-image": "Transform an input image into a new one while preserving structure.",
        "inpaint": "Fill in or replace regions of an image using prompts and masks.",
        "control": "Control or animate content using additional inputs such as pose, masks, or reference videos.",
        "edit": "General-purpose editing of images or videos.",
    }

    discovered_categories = set()

    for root, _, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if not file.endswith(".yml") or file.startswith("shared"):
                continue
            file_path = Path(root) / file
            try:
                data = load_yaml_content(file_path)
            except HTTPException:
                # Skip invalid YAMLs
                continue

            if not isinstance(data, dict):
                continue

            metadata = data.get("metadata", {}) or {}
            categories_field = metadata.get("categories")

            if isinstance(categories_field, list):
                for c in categories_field:
                    if isinstance(c, str) and c.strip():
                        discovered_categories.add(c.strip())
            elif isinstance(categories_field, str) and categories_field.strip():
                discovered_categories.add(categories_field.strip())

    results: List[ModelTypeInfo] = []
    for key in sorted(discovered_categories):
        # Ensure every discovered category has a label and description, even if not pre-defined above
        label = label_map.get(key)
        if not label:
            # Generate a human-friendly label from the key
            label = key.replace("_", " ").replace("-", " ").title()

        description = description_map.get(key)
        if not description:
            description = f"Models in the '{label}' category."

        results.append(ModelTypeInfo(key=key, label=label, description=description))

    return results


@router.get("/types", response_model=List[ModelTypeInfo])
async def list_model_types() -> List[ModelTypeInfo]:
    """Async wrapper for list_model_types; runs blocking work off the event loop."""
    return await _run_blocking(_list_model_types_sync)


@router.get("/categories", response_model=List[ModelCategoryInfo])
async def list_model_categories() -> List[ModelCategoryInfo]:
    """List distinct metadata.categories values across manifests with label and description."""
    return await list_model_types()


def _get_system_compute_info_sync():
    capability = _get_system_compute_capability()
    return capability.to_dict()


@router.get("/system/compute")
async def get_system_compute_info():
    """Get information about the current system's compute capabilities."""
    return await _run_blocking(_get_system_compute_info_sync)


def _list_all_manifests_sync(include_incompatible: bool = False):
    """Blocking implementation for list_all_manifests()."""
    try:
        if include_incompatible:
            # Load all manifests without filtering
            manifests: List[Dict[str, Any]] = []
            for root, _dirs, files in os.walk(MANIFEST_BASE_PATH):
                for file in files:
                    if file.endswith(".yml") and not file.startswith("shared"):
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(MANIFEST_BASE_PATH)
                        enriched = _load_and_enrich_manifest(str(relative_path))
                        manifests.append(enriched)
            return manifests
        # Use the normal filtered list
        return get_all_manifest_files()
    except Exception as e:
        logger.error(f"Error listing manifests: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing manifests: {e}")


@router.get("/list")
async def list_all_manifests(include_incompatible: bool = False):
    """List all available manifests (async; runs blocking work off the event loop)."""
    return await _run_blocking(_list_all_manifests_sync, include_incompatible)


def _list_manifests_by_model_sync(model: str, include_incompatible: bool = False):
    if include_incompatible:
        manifests = _list_all_manifests_sync(include_incompatible=True)
    else:
        manifests = get_all_manifest_files()
    filtered = [m for m in manifests if m.get("model") == model]
    if not filtered:
        raise HTTPException(
            status_code=404, detail=f"No manifests found for model: {model}"
        )
    return filtered


@router.get("/list/model/{model}")
async def list_manifests_by_model(model: str, include_incompatible: bool = False):
    """List all manifest names for a specific model (async)."""
    return await _run_blocking(
        _list_manifests_by_model_sync, model, include_incompatible
    )


def _list_manifests_by_model_type_sync(
    model_type: str, include_incompatible: bool = False
):
    if include_incompatible:
        manifests = _list_all_manifests_sync(include_incompatible=True)
    else:
        manifests = get_all_manifest_files()
    filtered: List[Dict[str, Any]] = []
    for m in manifests:
        mt = m.get("model_type")
        if isinstance(mt, list):
            if model_type in mt:
                filtered.append(m)
        else:
            if mt == model_type:
                filtered.append(m)
    if not filtered:
        raise HTTPException(
            status_code=404, detail=f"No manifests found for model_type: {model_type}"
        )
    return filtered


@router.get("/list/type/{model_type}")
async def list_manifests_by_model_type(
    model_type: str, include_incompatible: bool = False
):
    """List all manifest names for a specific model type (async)."""
    return await _run_blocking(
        _list_manifests_by_model_type_sync, model_type, include_incompatible
    )


def _list_manifests_by_model_and_type_sync(
    model: str, model_type: str, include_incompatible: bool = False
):
    if include_incompatible:
        manifests = _list_all_manifests_sync(include_incompatible=True)
    else:
        manifests = get_all_manifest_files()
    filtered: List[Dict[str, Any]] = []
    for m in manifests:
        model_match = m.get("model") == model
        mt = m.get("model_type")
        if isinstance(mt, list):
            type_match = model_type in mt
        else:
            type_match = mt == model_type
        if model_match and type_match:
            filtered.append(m)
    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"No manifests found for model: {model} and model_type: {model_type}",
        )
    return filtered


@router.get("/list/model/{model}/model_type/{model_type}")
async def list_manifests_by_model_and_type(
    model: str, model_type: str, include_incompatible: bool = False
):
    """List all manifest names for a specific model and model type combination (async)."""
    return await _run_blocking(
        _list_manifests_by_model_and_type_sync, model, model_type, include_incompatible
    )


@router.get("/{manifest_id}")
async def get_manifest_by_id(manifest_id: str) -> Dict[Any, Any]:
    return await _run_blocking(get_manifest, manifest_id)


@router.get("/{manifest_id}/part")
async def get_manifest_part(manifest_id: str, path: Optional[str] = None):
    """
    Return a specific part of the enriched manifest given a dot-separated path.
    Examples:
      - path=spec.loras
      - path=spec.components
      - path=spec.components.0.model_path
    Supports numeric tokens to index into lists.
    """
    doc = await _run_blocking(get_manifest, manifest_id)
    if not path:
        return doc
    value: Any = doc
    for token in path.split("."):
        if isinstance(value, dict):
            if token in value:
                value = value[token]
            else:
                raise HTTPException(
                    status_code=404, detail=f"Key not found at segment '{token}'"
                )
        elif isinstance(value, list):
            try:
                idx = int(token)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Expected list index at segment '{token}'"
                )
            if idx < 0 or idx >= len(value):
                raise HTTPException(
                    status_code=404,
                    detail=f"List index out of range at segment '{token}'",
                )
            value = value[idx]
        else:
            raise HTTPException(
                status_code=404, detail=f"Path not traversable at segment '{token}'"
            )

    return value


class CustomModelPathRequest(BaseModel):
    manifest_id: str
    component_index: int
    path: str
    name: Optional[str] = None


class DeleteCustomModelPathRequest(BaseModel):
    manifest_id: str
    component_index: int
    path: str


class UpdateLoraScaleRequest(BaseModel):
    """
    Request body for updating a single LoRA entry's scale inside a manifest YAML.

    LoRAs are stored under spec.loras as either strings or mappings. This endpoint
    normalizes the targeted entry to a mapping (dict) and writes a `scale` field.
    """

    manifest_id: str
    lora_index: int
    scale: float


@router.post("/lora/scale")
async def update_lora_scale(req: UpdateLoraScaleRequest) -> Dict[str, Any]:
    return await _run_blocking(_update_lora_scale_sync, req)


def _update_lora_scale_sync(req: UpdateLoraScaleRequest) -> Dict[str, Any]:
    """
    Update the `scale` value for a LoRA entry inside a manifest's YAML file.

    The LoRA is addressed by its index in ``spec.loras`` for the given manifest.
    If the entry is a plain string, it will be normalized into a mapping with a
    ``source`` field and the provided ``scale``.
    """
    if not req.manifest_id:
        raise HTTPException(status_code=400, detail="manifest_id is required")
    if req.lora_index < 0:
        raise HTTPException(status_code=400, detail="lora_index must be non-negative")

    # Clamp scale to a reasonable range [0, 1] but allow minor overshoot before clamping
    try:
        scale = float(req.scale)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="scale must be a number")
    if not (scale >= 0.0 and scale <= 1.0):
        # Clamp into [0, 1] instead of rejecting outright so UI sliders can't break it
        scale = max(0.0, min(1.0, scale))

    # Load enriched manifest to locate backing YAML path
    manifest = get_manifest(req.manifest_id)
    if not manifest:
        raise HTTPException(
            status_code=404, detail=f"Manifest not found: {req.manifest_id}"
        )

    relative_path = manifest.get("full_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(
            status_code=500,
            detail="Manifest missing full_path metadata; cannot locate YAML.",
        )
    yaml_path = MANIFEST_BASE_PATH / relative_path
    if not yaml_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest YAML not found on disk: {yaml_path}",
        )

    # Load the raw YAML document for mutation
    doc = load_yaml_content(yaml_path)
    if not isinstance(doc, dict):
        raise HTTPException(
            status_code=500,
            detail="Manifest YAML is not a mapping; cannot update.",
        )

    spec = doc.get("spec") or {}
    loras = spec.get("loras") or []
    if not isinstance(loras, list) or req.lora_index >= len(loras):
        raise HTTPException(
            status_code=400,
            detail=f"LoRA entry not found at index {req.lora_index}",
        )

    entry = loras[req.lora_index]
    if isinstance(entry, dict):
        entry["scale"] = float(scale)
    elif isinstance(entry, str):
        # Normalize legacy string entry into a mapping with a source and scale
        entry = {
            "source": entry,
            "scale": float(scale),
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported LoRA entry type at index {req.lora_index}; expected string or mapping.",
        )

    loras[req.lora_index] = entry
    doc.setdefault("spec", {})["loras"] = loras

    try:
        yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write updated manifest: {e}"
        )

    return {
        "success": True,
        "manifest_id": req.manifest_id,
        "lora_index": req.lora_index,
        "scale": float(scale),
    }


class UpdateLoraNameRequest(BaseModel):
    """
    Request body for updating a single LoRA entry's name/label inside a manifest YAML.

    This normalizes the target entry to a mapping (dict) and writes both `name` and `label`.
    """

    manifest_id: str
    lora_index: int
    name: str


@router.post("/lora/name")
async def update_lora_name(req: UpdateLoraNameRequest) -> Dict[str, Any]:
    return await _run_blocking(_update_lora_name_sync, req)


def _update_lora_name_sync(req: UpdateLoraNameRequest) -> Dict[str, Any]:
    """
    Update the `name` / `label` fields for a LoRA entry inside a manifest's YAML file.

    Behaviour:
      - Normalizes legacy string entries into dicts with a `source` field.
      - Writes `name` and `label` to the entry.
    """
    if not req.manifest_id:
        raise HTTPException(status_code=400, detail="manifest_id is required")
    if req.lora_index < 0:
        raise HTTPException(status_code=400, detail="lora_index must be non-negative")
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    manifest = get_manifest(req.manifest_id)
    if not manifest:
        raise HTTPException(
            status_code=404, detail=f"Manifest not found: {req.manifest_id}"
        )

    relative_path = manifest.get("full_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(
            status_code=500,
            detail="Manifest missing full_path metadata; cannot locate YAML.",
        )
    yaml_path = MANIFEST_BASE_PATH / relative_path
    if not yaml_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest YAML not found on disk: {yaml_path}",
        )

    doc = load_yaml_content(yaml_path)
    if not isinstance(doc, dict):
        raise HTTPException(
            status_code=500,
            detail="Manifest YAML is not a mapping; cannot update.",
        )

    spec = doc.get("spec") or {}
    loras = spec.get("loras") or []
    if not isinstance(loras, list) or req.lora_index >= len(loras):
        raise HTTPException(
            status_code=400,
            detail=f"LoRA entry not found at index {req.lora_index}",
        )

    entry = loras[req.lora_index]
    if isinstance(entry, str):
        entry = {"source": entry}
    if not isinstance(entry, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported LoRA entry type at index {req.lora_index}; expected string or mapping.",
        )

    entry["name"] = name
    entry["label"] = name
    loras[req.lora_index] = entry
    doc.setdefault("spec", {})["loras"] = loras

    try:
        yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write updated manifest: {e}"
        )

    return {
        "success": True,
        "manifest_id": req.manifest_id,
        "lora_index": req.lora_index,
        "name": name,
    }


class DeleteLoraRequest(BaseModel):
    """
    Request body for deleting a LoRA entry from a manifest YAML.

    If the entry contains a `source` that points into the local LoRA directory and/or a
    `remote_source` value, the associated local path may also be removed depending on how
    it was resolved.
    """

    manifest_id: str
    lora_index: int


@router.delete("/lora")
async def delete_lora(req: DeleteLoraRequest) -> Dict[str, Any]:
    return await _run_blocking(_delete_lora_sync, req)


def _delete_lora_sync(req: DeleteLoraRequest) -> Dict[str, Any]:
    """
    Delete a LoRA entry from a manifest's YAML file.

    Behaviour:
      - Removes the entry at `spec.loras[req.lora_index]`.
      - Attempts to delete any associated local LoRA file/folder if the entry has a
        concrete `source` path inside the lora directory.
    """
    if not req.manifest_id:
        raise HTTPException(status_code=400, detail="manifest_id is required")
    if req.lora_index < 0:
        raise HTTPException(status_code=400, detail="lora_index must be non-negative")

    manifest = get_manifest(req.manifest_id)
    if not manifest:
        raise HTTPException(
            status_code=404, detail=f"Manifest not found: {req.manifest_id}"
        )

    relative_path = manifest.get("full_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(
            status_code=500,
            detail="Manifest missing full_path metadata; cannot locate YAML.",
        )
    yaml_path = MANIFEST_BASE_PATH / relative_path
    if not yaml_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest YAML not found on disk: {yaml_path}",
        )

    doc = load_yaml_content(yaml_path)
    if not isinstance(doc, dict):
        raise HTTPException(
            status_code=500,
            detail="Manifest YAML is not a mapping; cannot update.",
        )

    spec = doc.get("spec") or {}
    loras = spec.get("loras") or []
    if not isinstance(loras, list) or req.lora_index >= len(loras):
        raise HTTPException(
            status_code=400,
            detail=f"LoRA entry not found at index {req.lora_index}",
        )

    entry = loras[req.lora_index]
    local_path: Optional[str] = None

    if isinstance(entry, str):
        local_path = entry
    elif isinstance(entry, dict):
        # Prefer concrete local source path, but if only a remote_source is present,
        # resolve it into a local path using the LoRA base directory.
        src = entry.get("source")
        remote_src = entry.get("remote_source")
        if isinstance(src, str) and src:
            local_path = src
        elif isinstance(remote_src, str) and remote_src:
            try:
                resolved = DownloadMixin.is_downloaded(remote_src, get_lora_path())
            except Exception:
                resolved = None
            if isinstance(resolved, str) and resolved:
                local_path = resolved

    # Remove entry from list and write back to YAML
    del loras[req.lora_index]
    doc.setdefault("spec", {})["loras"] = loras

    try:
        yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write updated manifest: {e}"
        )

    # Best-effort local file/folder removal if path is under the LoRA base directory
    removed_local = False
    if isinstance(local_path, str) and local_path:
        try:
            base = Path(get_lora_path()).resolve()
            target = Path(local_path).expanduser()
            if not target.is_absolute():
                target = base / target
            target = target.resolve()
            if str(target).startswith(str(base)):
                if target.is_file() or target.is_symlink():
                    target.unlink(missing_ok=True)
                    removed_local = True
                elif target.is_dir():
                    import shutil

                    shutil.rmtree(target, ignore_errors=True)
                    removed_local = True
        except Exception:
            # Best-effort; ignore filesystem errors
            removed_local = False

    return {
        "success": True,
        "manifest_id": req.manifest_id,
        "lora_index": req.lora_index,
        "removed_local": removed_local,
    }


@router.post("/custom-model-path")
async def validate_and_register_custom_model_path(
    req: CustomModelPathRequest,
) -> Dict[str, Any]:
    return await _run_blocking(_validate_and_register_custom_model_path_sync, req)


def _validate_and_register_custom_model_path_sync(
    req: CustomModelPathRequest,
) -> Dict[str, Any]:
    """
    Validate a local model path and, if valid, register it in the manifest as a
    custom component model_path entry.

    Behaviour:
      - validates that `req.path` exists (file or directory),
      - attempts to load the model via `BaseEngine.validate_model_path`,
      - and, if successful, appends a new entry to the component's `model_path`
        list in the underlying YAML manifest, including a computed `file_size`
        and a `custom: true` flag.
    """

    if not req.manifest_id:
        raise HTTPException(status_code=400, detail="manifest_id is required")
    if req.component_index < 0:
        raise HTTPException(
            status_code=400, detail="component_index must be non-negative"
        )
    if not req.path:
        raise HTTPException(status_code=400, detail="path is required")

    # Basic filesystem validation – local file or directory
    if not os.path.exists(req.path):
        raise HTTPException(
            status_code=400, detail=f"Model path does not exist: {req.path}"
        )
    if not os.path.isfile(req.path) and not os.path.isdir(req.path):
        raise HTTPException(
            status_code=400,
            detail=f"Model path is not a file or directory: {req.path}",
        )

    # Load enriched manifest to locate the backing YAML path
    manifest = get_manifest(req.manifest_id)
    if not manifest:
        raise HTTPException(
            status_code=404, detail=f"Manifest not found: {req.manifest_id}"
        )

    relative_path = manifest.get("full_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(
            status_code=500,
            detail="Manifest missing full_path metadata; cannot locate YAML.",
        )
    yaml_path = MANIFEST_BASE_PATH / relative_path
    if not yaml_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest YAML not found on disk: {yaml_path}",
        )

    # Load the raw YAML document for mutation
    doc = load_yaml_content(yaml_path)
    if not isinstance(doc, dict):
        raise HTTPException(
            status_code=500,
            detail="Manifest YAML is not a mapping; cannot update.",
        )

    spec = doc.get("spec") or {}
    components = spec.get("components") or []
    if not isinstance(components, list) or req.component_index >= len(components):
        raise HTTPException(
            status_code=400,
            detail=f"Component not found at index {req.component_index}",
        )

    # Ensure this path (normalized via DownloadMixin.is_downloaded) is not already
    # registered on any component in the manifest. We compare against the "real"
    # on-disk path if the model has already been downloaded, not just the raw
    # manifest string, since that may be a remote URL or outdated location.
    components_base = get_components_path()

    def _canonical_model_path(path_str: str) -> str:
        try:
            resolved = DownloadMixin.is_downloaded(path_str, components_base)
            return resolved or path_str
        except Exception:
            return path_str

    requested_path_canonical = _canonical_model_path(req.path)
    for idx, comp in enumerate(components):
        if not isinstance(comp, dict):
            continue
        existing_mp = comp.get("model_path")

        # Normalize and iterate over existing paths for this component
        candidate_paths: List[str] = []
        if isinstance(existing_mp, str):
            candidate_paths.append(existing_mp)
        elif isinstance(existing_mp, list):
            for item in existing_mp:
                if isinstance(item, str):
                    candidate_paths.append(item)
                elif isinstance(item, dict):
                    p = item.get("path")
                    if isinstance(p, str):
                        candidate_paths.append(p)
        elif isinstance(existing_mp, dict):
            p = existing_mp.get("path")
            if isinstance(p, str):
                candidate_paths.append(p)

        for raw_path in candidate_paths:
            canonical = _canonical_model_path(raw_path)
            if canonical == requested_path_canonical:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Model path is already registered for this manifest "
                        f"(component index {idx}): {req.path}"
                    ),
                )

    component_doc = components[req.component_index]
    if not isinstance(component_doc, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Component at index {req.component_index} is not a mapping",
        )

    # First, validate that the engine can load this component with the new path
    # without mutating the on-disk YAML.
    component_for_validation: Dict[str, Any] = dict(component_doc)
    component_for_validation["model_path"] = req.path
    # check if config_path is in the component_for_validation and if

    engine = UniversalEngine(yaml_path=str(yaml_path), should_download=False).engine

    try:
        engine.validate_model_path(component_for_validation)
    except Exception as e:

        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Failed to validate model path: {e}"
        )

    # Compute file size for the new path
    size_bytes = _sizeof_local_path(req.path)

    # Build new model_path entry for YAML
    variant_name = req.name or "custom"
    new_item: Dict[str, Any] = {
        "path": req.path,
        "variant": variant_name,
        "custom": True,
    }
    if size_bytes > 0:
        new_item["file_size"] = int(size_bytes)

    # Normalize existing model_path into list[dict]
    mp = component_doc.get("model_path")
    if mp is None:
        mp_items: List[Any] = []
    elif isinstance(mp, str):
        mp_items = [{"path": mp}]
    elif isinstance(mp, list):
        mp_items = mp
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported model_path format on component; expected string or list.",
        )

    mp_items.append(new_item)
    component_doc["model_path"] = mp_items
    components[req.component_index] = component_doc
    doc.setdefault("spec", {})["components"] = components

    # Persist updated YAML back to disk
    try:
        yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write updated manifest: {e}"
        )

    return {
        "success": True,
        "message": "Model path validated and registered successfully.",
    }


@router.delete("/custom-model-path")
async def delete_custom_model_path(req: DeleteCustomModelPathRequest) -> Dict[str, Any]:
    return await _run_blocking(_delete_custom_model_path_sync, req)


def _delete_custom_model_path_sync(req: DeleteCustomModelPathRequest) -> Dict[str, Any]:
    """
    Remove a *custom* model_path entry from a component in the manifest YAML,
    without deleting any underlying model files from disk.
    """
    if not req.manifest_id:
        raise HTTPException(status_code=400, detail="manifest_id is required")
    if req.component_index < 0:
        raise HTTPException(
            status_code=400, detail="component_index must be non-negative"
        )
    if not req.path:
        raise HTTPException(status_code=400, detail="path is required")

    manifest = get_manifest(req.manifest_id)
    if not manifest:
        raise HTTPException(
            status_code=404, detail=f"Manifest not found: {req.manifest_id}"
        )

    relative_path = manifest.get("full_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(
            status_code=500,
            detail="Manifest missing full_path metadata; cannot locate YAML.",
        )
    yaml_path = MANIFEST_BASE_PATH / relative_path
    if not yaml_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest YAML not found on disk: {yaml_path}",
        )

    doc = load_yaml_content(yaml_path)
    if not isinstance(doc, dict):
        raise HTTPException(
            status_code=500,
            detail="Manifest YAML is not a mapping; cannot update.",
        )

    spec = doc.get("spec") or {}
    components = spec.get("components") or []
    if not isinstance(components, list) or req.component_index >= len(components):
        raise HTTPException(
            status_code=400,
            detail=f"Component not found at index {req.component_index}",
        )

    component_doc = components[req.component_index]
    if not isinstance(component_doc, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Component at index {req.component_index} is not a mapping",
        )

    mp = component_doc.get("model_path")
    removed = False

    # Only remove entries that are explicitly marked as custom and match the path.
    if isinstance(mp, list):
        new_items: List[Any] = []
        for item in mp:
            if isinstance(item, dict):
                p = item.get("path")
                is_custom = item.get("custom") is True
                if is_custom and isinstance(p, str) and p == req.path:
                    removed = True
                    continue
            new_items.append(item)
        if removed:
            if new_items:
                component_doc["model_path"] = new_items
            else:
                component_doc.pop("model_path", None)
    elif isinstance(mp, dict):
        p = mp.get("path")
        is_custom = mp.get("custom") is True
        if is_custom and isinstance(p, str) and p == req.path:
            removed = True
            component_doc.pop("model_path", None)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail="Custom model path not found on component; nothing to delete.",
        )

    components[req.component_index] = component_doc
    doc.setdefault("spec", {})["components"] = components

    try:
        yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write updated manifest: {e}"
        )

    return {
        "success": True,
        "message": "Custom model path removed successfully.",
    }
