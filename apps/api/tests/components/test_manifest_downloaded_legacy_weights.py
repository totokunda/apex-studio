from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import yaml


def _import_manifest_api():
    if "src.api.manifest" in sys.modules:
        return sys.modules["src.api.manifest"]

    engine_stub = types.ModuleType("src.engine")

    class _UniversalEngine:
        def __init__(self, *args, **kwargs):
            pass

    engine_stub.UniversalEngine = _UniversalEngine
    sys.modules["src.engine"] = engine_stub

    return importlib.import_module("src.api.manifest")


def _write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def test_manifest_downloaded_true_for_top_level_legacy_weights(tmp_path, monkeypatch):
    manifest_api = _import_manifest_api()
    root = tmp_path
    rel = "v0.1.2/legacy/top-level-legacy-weights.yml"
    weight_path = "legacy/repo/top-level.safetensors"
    _write_yaml(
        root / rel,
        {
            "api_version": "apex/v1",
            "kind": "Model",
            "metadata": {
                "id": "top-level-legacy-weights",
                "name": "Top Level Legacy Weights",
                "model": "legacy-model",
                "version": "1.0.0",
                "categories": ["legacy"],
            },
            "spec": {"engine": "legacy", "model_type": "legacy", "ui": {"inputs": []}},
            "weights": [{"path": weight_path}],
        },
    )

    monkeypatch.setattr(
        manifest_api,
        "resolve_manifest_path_for_read",
        lambda relative_path: root / relative_path,
    )

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/top-level.safetensors" if model_path == weight_path else None

    monkeypatch.setattr(
        manifest_api.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded)
    )

    enriched = manifest_api._load_and_enrich_manifest(rel)
    assert enriched["downloaded"] is True
    assert enriched["weights"][0]["is_downloaded"] is True


def test_manifest_downloaded_true_for_component_legacy_weights(tmp_path, monkeypatch):
    manifest_api = _import_manifest_api()
    root = tmp_path
    rel = "v0.1.2/legacy/component-legacy-weights.yml"
    weight_path = "legacy/repo/component.safetensors"
    _write_yaml(
        root / rel,
        {
            "api_version": "apex/v1",
            "kind": "Model",
            "metadata": {
                "id": "component-legacy-weights",
                "name": "Component Legacy Weights",
                "model": "legacy-model",
                "version": "1.0.0",
                "categories": ["legacy"],
            },
            "spec": {
                "engine": "legacy",
                "model_type": "legacy",
                "ui": {"inputs": []},
                "components": [
                    {
                        "type": "transformer",
                        "name": "transformer",
                        "weights": [{"path": weight_path}],
                    }
                ],
            },
        },
    )

    monkeypatch.setattr(
        manifest_api,
        "resolve_manifest_path_for_read",
        lambda relative_path: root / relative_path,
    )

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/component.safetensors" if model_path == weight_path else None

    monkeypatch.setattr(
        manifest_api.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded)
    )

    enriched = manifest_api._load_and_enrich_manifest(rel)
    assert enriched["downloaded"] is True
    assert enriched["spec"]["components"][0]["is_downloaded"] is True
    assert enriched["spec"]["components"][0]["weights"][0]["is_downloaded"] is True

