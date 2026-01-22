from __future__ import annotations

from typing import Any, Dict

from loguru import logger

try:
    import jsonschema

    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover - fail-soft if dependency missing
    jsonschema = None  # type: ignore
    _HAS_JSONSCHEMA = False

from src.manifest.schema_v1 import MANIFEST_SCHEMA_V1


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def _normalize_ui(ui_like: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if ui_like is None:
        return None
    # Accept both "UI" and "ui" and normalize component names/helpers
    ui = dict(ui_like)
    if "mode" in ui and isinstance(ui["mode"], str):
        ui["mode"] = ui["mode"].lower()
    # Normalize input components to known set
    inputs = ui.get("simple", {}).get("inputs", [])
    for item in inputs:
        comp = item.get("component") or item.get("type")
        if isinstance(comp, str):
            comp_l = comp.lower()
            # canonical component mapping
            comp_map = {
                "text": "text",
                "string": "text",
                "number": "number",
                "int": "number",
                "integer": "number",
                "float": "float",
                "double": "float",
                "bool": "bool",
                "boolean": "bool",
                "list": "list",
                "array": "list",
                "file": "file",
                "path": "file",
                "select": "select",
                "slider": "slider",
            }
            item["component"] = comp_map.get(comp_l, comp_l)
    return ui


def validate_and_normalize(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a raw YAML document into the engine's expected shape.

    - If the document is already in legacy shape (no apiVersion), return as-is.
    - If it is a v1 manifest, validate then map fields to the legacy engine shape,
      while preserving metadata and ui under dedicated keys.
    """

    if not isinstance(doc, dict):
        return doc

    # Accept both 'ui' and 'UI' regardless of location
    possible_ui = doc.get("ui") or doc.get("UI")

    if "api_version" not in doc and "apiVersion" not in doc:
        # Legacy document. If UI present at top-level, keep as-is.
        if possible_ui is not None and "ui" not in doc:
            doc["ui"] = _normalize_ui(possible_ui)
        return doc

    # v1 schema
    if _HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=doc, schema=MANIFEST_SCHEMA_V1)
        except Exception as e:  # validation error
            raise ValueError(f"Manifest validation failed: {e}")
    else:
        logger.warning(
            "jsonschema not installed; skipping strict manifest validation. Add 'jsonschema' to requirements to enable validation."
        )

    metadata = doc.get("metadata", {}) or {}
    spec = doc.get("spec", {}) or {}

    # Allow UI under top-level UI, top-level ui, or spec.ui
    ui_spec = doc.get("ui") or doc.get("UI") or spec.get("ui") or spec.get("UI")

    # Build normalized config expected by engines
    normalized: Dict[str, Any] = {}

    # core identifiers
    normalized["name"] = metadata.get("name")
    if metadata.get("description"):
        normalized["description"] = metadata.get("description")
    # keep version and meta for downstream usage
    if metadata.get("version"):
        normalized["version"] = metadata.get("version")
    normalized["metadata"] = metadata

    # engine specs
    if spec.get("engine"):
        normalized["engine"] = spec["engine"]
    # Allow multiple model types; engines typically take a single enum at init
    if spec.get("model_type") is not None:
        normalized["type"] = spec["model_type"]
    elif spec.get("model_types") is not None:
        normalized["type"] = spec["model_types"]
    # Backward compatibility for camelCase
    elif spec.get("modelType") is not None:
        normalized["type"] = spec["modelType"]
    elif spec.get("modelTypes") is not None:
        normalized["type"] = spec["modelTypes"]
    if spec.get("engine_type"):
        normalized["engine_type"] = spec["engine_type"]
    elif spec.get("engineType"):  # backward compatibility
        normalized["engine_type"] = spec["engineType"]
    if spec.get("denoise_type"):
        normalized["denoise_type"] = spec["denoise_type"]
    elif spec.get("denoiseType"):  # backward compatibility
        normalized["denoise_type"] = spec["denoiseType"]
    if spec.get("engine_kwargs"):
        normalized["engine_kwargs"] = spec["engine_kwargs"]

    if spec.get("sub_engines"):
        normalized["sub_engines"] = spec["sub_engines"]
    elif spec.get("subEngines"):
        normalized["sub_engines"] = spec["subEngines"]
    elif spec.get("subengines"):
        normalized["sub_engines"] = spec["subengines"]

    # components and stages
    for key in (
        "components",
        "preprocessors",
        "postprocessors",
        "shared",
        "helpers",
        "loras",
        "attention_types",
        "compute_requirements",
    ):
        if key in spec:
            normalized[key] = spec[key]

    # defaults and save options
    if "defaults" in spec:
        normalized["defaults"] = spec["defaults"]
    if "save" in spec:
        normalized["save_kwargs"] = spec["save"]

    # Normalize components list: ensure every component has a stable name
    comps = normalized.get("components", []) or []
    for comp in comps:
        if "name" not in comp:
            comp_type = comp.get("type")
            comp["name"] = comp_type

    # UI spec normalized under 'ui'
    if ui_spec is not None:
        normalized["ui"] = _normalize_ui(ui_spec)

    # Pass through any remaining top-level keys (except nested blocks we already handled)
    for top_key, top_val in doc.items():
        if top_key in ("metadata", "spec", "ui", "UI"):
            continue
        if top_key not in normalized:
            normalized[top_key] = top_val

    # Pass through any remaining spec keys that were not explicitly normalized above.
    # This keeps the legacy engine shape while still exposing the full v1 spec surface.
    passthrough_exclude = {
        # Core engine wiring already mapped
        "engine",
        "model_type",
        "model_types",
        "modelType",
        "modelTypes",
        "engine_type",
        "engineType",
        "denoise_type",
        "denoiseType",
        "engine_kwargs",
        "sub_engines",
        "subEngines",
        "subengines",
        # Stages/components we already copied verbatim
        "components",
        "preprocessors",
        "postprocessors",
        "shared",
        "helpers",
        "loras",
        "attention_types",
        "compute_requirements",
        # Defaults/save handled explicitly
        "defaults",
        "save",
        # UI handled via ui_spec above
        "ui",
        "UI",
    }
    for key, value in spec.items():
        if key in passthrough_exclude:
            continue
        if key not in normalized:
            normalized[key] = value

    return normalized
