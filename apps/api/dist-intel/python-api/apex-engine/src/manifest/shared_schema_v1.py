from __future__ import annotations

# JSON Schema for Apex Shared Components Manifest v1

SHARED_MANIFEST_SCHEMA_V1: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Apex Shared Components v1",
    "type": "object",
    "required": ["api_version", "kind", "metadata", "spec"],
    "properties": {
        "api_version": {"type": "string", "pattern": r"^apex(/ai)?/v1$|^apex/v1$"},
        "kind": {
            "type": "string",
            "enum": ["SharedComponents", "ComponentLibrary"],
        },
        "metadata": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "min_length": 1},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "additional_properties": True,
        },
        "spec": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {"type": "object", "additional_properties": True},
                },
                "preprocessors": {
                    "type": "array",
                    "items": {"type": "object", "additional_properties": True},
                },
                "postprocessors": {
                    "type": "array",
                    "items": {"type": "object", "additional_properties": True},
                },
            },
            "additional_properties": True,
        },
    },
    "additional_properties": True,
}
