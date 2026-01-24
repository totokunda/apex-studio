from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from loguru import logger
import yaml

try:
    import jsonschema

    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore
    _HAS_JSONSCHEMA = False

from src.manifest.shared_schema_v1 import SHARED_MANIFEST_SCHEMA_V1


def load_shared_manifest(file_path: str | Path) -> Dict[str, Any]:
    file_path = Path(file_path)
    text = file_path.read_text()
    doc = yaml.load(text, Loader=yaml.FullLoader)

    # Legacy: pass-through
    if not isinstance(doc, dict) or "api_version" not in doc:
        return doc

    if _HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=doc, schema=SHARED_MANIFEST_SCHEMA_V1)
        except Exception as e:
            raise ValueError(f"Shared manifest validation failed: {e}")
    else:
        logger.warning("jsonschema not installed; skipping shared manifest validation.")

    spec = doc.get("spec", {}) or {}
    # Normalize to legacy-like structure with top-level lists for lookup
    normalized: Dict[str, Any] = {}
    for key in ("components", "preprocessors", "postprocessors"):
        if key in spec:
            normalized[key] = spec[key]
    # Also keep metadata for tooling
    normalized["metadata"] = doc.get("metadata", {}) or {}
    return normalized
