from __future__ import annotations

# JSON Schema for Apex Manifest v1
# Keep permissive defaults to avoid breaking existing flows while providing
# strong guidance and validation for new manifests.

MANIFEST_SCHEMA_V1: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Apex Manifest v1",
    "type": "object",
    "required": ["api_version", "kind", "metadata", "spec"],
    "properties": {
        "api_version": {"type": "string", "pattern": r"^apex(/ai)?/v1$|^apex/v1$"},
        "kind": {
            "type": "string",
            "enum": [
                "Model",
                "Pipeline",
            ],
        },
        "metadata": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "id": {"type": "string"},
                "model": {"type": "string"},
                "name": {"type": "string", "min_length": 1},
                "version": {
                    "type": "string",
                    "pattern": r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+].*)?$",
                },
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "string"},
                "license": {"type": "string"},
                "homepage": {"type": "string"},
                "registry": {"type": "string"},
                "demo_path": {"type": "string"},
                "annotations": {"type": "object", "additional_properties": True},
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {
                                "type": "object",
                                "additional_properties": True,
                            },
                        },
                        "additional_properties": True,
                    },
                },
            },
            "additional_properties": True,
        },
        "spec": {
            "type": "object",
            "required": ["engine", "model_type"],
            "properties": {
                "engine": {"type": "string"},
                "model_type": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "model_types": {"type": "array", "items": {"type": "string"}},
                "engine_type": {"type": "string", "enum": ["torch", "mlx"]},
                "denoise_type": {"type": "string"},
                "shared": {"type": "array", "items": {"type": "string"}},
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "scheduler",
                                    "vae",
                                    "text_encoder",
                                    "transformer",
                                    "helper",
                                    # Pseudo-component used to attach downloadable auxiliary
                                    # model paths onto a real component (handled in BaseEngine).
                                    "extra_model_path",
                                ],
                            },
                            "name": {"type": "string"},
                            "label": {"type": "string"},
                            "base": {"type": "string"},
                            "model_path": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["path"],
                                            "properties": {
                                                "path": {"type": "string"},
                                                "variant": {"type": "string"},
                                                "precision": {"type": "string"},
                                                "type": {"type": "string"},
                                                "file_size": {"type": "number"},
                                                "is_downloaded": {"type": "boolean"},
                                                "resource_requirements": {
                                                    "type": "object",
                                                    "properties": {
                                                        "min_vram_gb": {
                                                            "type": "number"
                                                        },
                                                        "recommended_vram_gb": {
                                                            "type": "number"
                                                        },
                                                        "compute_capability": {
                                                            "type": "string"
                                                        },
                                                    },
                                                    "additional_properties": True,
                                                },
                                            },
                                            "additional_properties": True,
                                        },
                                    },
                                ]
                            },
                            "config_path": {"type": "string"},
                            "file_pattern": {"type": "string"},
                            "tag": {"type": "string"},
                            "key_map": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "extra_kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "save_path": {"type": "string"},
                            "converter_kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "model_key": {"type": "string"},
                            "extra_model_paths": {
                                "type": "array",
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["path"],
                                            "properties": {
                                                "path": {"type": "string"},
                                                "variant": {"type": "string"},
                                                "precision": {"type": "string"},
                                            },
                                            "additional_properties": True,
                                        },
                                    },
                                ],
                            },
                            "converted_model_path": {"type": "string"},
                            "scheduler_options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                        "base": {"type": "string"},
                                        "config_path": {"type": "string"},
                                    },
                                    "additional_properties": True,
                                },
                            },
                            # Optional reference to a scheduler catalog YAML (kept outside
                            # individual model manifests). When present, runtime loaders
                            # expand it into `scheduler_options`.
                            "scheduler_manifest": {"type": "string"},
                            "scheduler_config_defaults": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "scheduler_config_overrides": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "gguf_files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["type", "path"],
                                    "properties": {
                                        "type": {"type": "string"},
                                        "path": {"type": "string"},
                                    },
                                    "additional_properties": True,
                                },
                            },
                            "deprecated": {"type": "boolean"},
                        },
                        "additional_properties": True,
                    },
                },
                "preprocessors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "name": {"type": "string"},
                            "model_path": {"type": "string"},
                            "config_path": {"type": "string"},
                            "save_path": {"type": "string"},
                            "kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "deprecated": {"type": "boolean"},
                        },
                        "additional_properties": True,
                    },
                },
                "postprocessors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "name": {"type": "string"},
                            "model_path": {"type": "string"},
                            "config_path": {"type": "string"},
                            "kwargs": {
                                "type": "object",
                                "additional_properties": True,
                            },
                            "deprecated": {"type": "boolean"},
                        },
                        "additional_properties": True,
                    },
                },
                "defaults": {
                    "type": "object",
                    "additional_properties": True,
                },
                "loras": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "path": {"type": "string"},
                                    "url": {"type": "string"},
                                    "scale": {"type": "number"},
                                    "name": {"type": "string"},
                                },
                                "additional_properties": True,
                            },
                        ]
                    },
                },
                "save": {
                    "type": "object",
                    "additional_properties": True,
                },
                "resource_requirements": {
                    "type": "object",
                    "properties": {
                        "min_vram_gb": {"type": "number"},
                        "recommended_vram_gb": {"type": "number"},
                        "compute_capability": {"type": "string"},
                    },
                    "additional_properties": True,
                },
                "ui": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["simple", "advanced", "complex"],
                        },
                        "timeline_inputs": {
                            "type": "object",
                            "properties": {
                                "inputs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["id", "type"],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "label": {"type": "string"},
                                            "type": {
                                                "type": "string",
                                                "enum": [
                                                    "text",
                                                    "audio",
                                                    "video",
                                                    "image",
                                                    "video_with_mask",
                                                    "image_with_mask",
                                                    "video_with_preprocessor",
                                                    "image_with_preprocessor",
                                                ],
                                            },
                                            "preprocessor_ref": {"type": "string"},
                                            "required": {"type": "boolean"},
                                            "default": {},
                                            "deprecated": {"type": "boolean"},
                                        },
                                        "additional_properties": True,
                                    },
                                },
                                "shortcuts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["key", "label"],
                                        "properties": {
                                            "key": {"type": "string"},
                                            "label": {"type": "string"},
                                            "type": {
                                                "type": "string",
                                                "enum": [
                                                    "number",
                                                    "text",
                                                    "boolean",
                                                    "select",
                                                ],
                                            },
                                            "default": {},
                                            "icon": {"type": "string"},
                                            "options": {
                                                "type": "array",
                                                "items": {},
                                            },
                                        },
                                        "additional_properties": True,
                                    },
                                },
                            },
                            "additional_properties": True,
                        },
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "type"],
                                "properties": {
                                    "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "number",
                                            "random",
                                            "text",
                                            "textarea",
                                            "categories",
                                            "boolean",
                                            "number_list",
                                        ],
                                    },
                                    "default": {},
                                    "category": {"type": "string"},
                                    "required": {"type": "boolean"},
                                    "min": {"type": "number"},
                                    "max": {"type": "number"},
                                    "step": {"type": "number"},
                                    "value_type": {
                                        "type": "string",
                                        "enum": ["integer", "float"],
                                    },
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "value": {},
                                            },
                                            "additional_properties": True,
                                        },
                                    },
                                    "order": {"type": "integer"},
                                    "deprecated": {"type": "boolean"},
                                },
                                "additional_properties": True,
                            },
                        },
                        "simple": {
                            "type": "object",
                            "properties": {
                                "inputs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["id"],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "label": {"type": "string"},
                                            "description": {"type": "string"},
                                            "type": {
                                                "type": "string",
                                                "enum": [
                                                    "text",
                                                    "number",
                                                    "float",
                                                    "bool",
                                                    "list",
                                                    "file",
                                                    "select",
                                                    "slider",
                                                ],
                                            },
                                            "default": {},
                                            "required": {"type": "boolean"},
                                            "options": {
                                                "type": "array",
                                                "items": {"type": ["string", "number"]},
                                            },
                                            "min": {"type": ["number", "integer"]},
                                            "max": {"type": ["number", "integer"]},
                                            "step": {"type": ["number", "integer"]},
                                            "group": {"type": "string"},
                                            "order": {"type": "integer"},
                                            "component": {"type": "string"},
                                            "mapping": {
                                                "type": "object",
                                                "properties": {
                                                    "target": {"type": "string"},
                                                    "param": {"type": "string"},
                                                    "path": {"type": "string"},
                                                },
                                                "additional_properties": True,
                                            },
                                        },
                                        "additional_properties": True,
                                    },
                                }
                            },
                            "additional_properties": True,
                        },
                        "advanced": {
                            "type": "object",
                            "properties": {
                                "expose": {
                                    "oneOf": [
                                        {"type": "string", "enum": ["all"]},
                                        {"type": "array", "items": {"type": "string"}},
                                    ]
                                },
                                "inputs": {
                                    "$ref": "#/properties/spec/properties/ui/properties/simple/properties/inputs"
                                },
                            },
                            "additional_properties": True,
                        },
                    },
                    "additional_properties": True,
                },
            },
            "additional_properties": True,
        },
        # Back-compat: allow uppercase UI at top-level or under spec
        "UI": {"$ref": "#/properties/spec/properties/ui"},
    },
    "additional_properties": True,
}
