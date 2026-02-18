from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import yaml


def _write_manifest(path: Path, manifest_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "api_version": "apex/v1",
        "kind": "Model",
        "metadata": {
            "id": manifest_id,
            "name": manifest_id,
            "model": "wan",
            "version": "0.1.2",
            "categories": ["control"],
        },
        "spec": {
            "engine": "wan",
            "model_type": "animate",
            "components": [],
            "ui": {"inputs": []},
        },
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _import_manifest_api():
    if "src.api.manifest" in sys.modules:
        return sys.modules["src.api.manifest"]

    # Keep this test lightweight: manifest.py imports src.engine at module import
    # time, but these tests only exercise manifest id/path selection helpers.
    engine_stub = types.ModuleType("src.engine")

    class _UniversalEngine:
        def __init__(self, *args, **kwargs):
            pass

    engine_stub.UniversalEngine = _UniversalEngine
    sys.modules["src.engine"] = engine_stub

    return importlib.import_module("src.api.manifest")


def test_manifest_id_index_prefers_non_legacy_path(tmp_path, monkeypatch):
    manifest_api = _import_manifest_api()
    root = tmp_path
    old_rel = "v0.1.0/video/wan-2.2-14b-animate-1.0.0.v1.yml"
    legacy_rel = "v0.1.2/legacy/wan-2.2-14b-animate-1.0.0.v1.yml"
    latest_rel = "v0.1.2/video/wan-2.2-14b-animate.yml"

    _write_manifest(root / old_rel, "wan-2-2-14b-animate")
    _write_manifest(root / legacy_rel, "wan-2-2-14b-animate")
    _write_manifest(root / latest_rel, "wan-2-2-14b-animate")

    def _iter_manifest_relative_paths(*, suffixes=(".yml", ".yaml")):
        yield old_rel
        yield legacy_rel
        yield latest_rel

    monkeypatch.setattr(
        manifest_api, "iter_manifest_relative_paths", _iter_manifest_relative_paths
    )
    monkeypatch.setattr(
        manifest_api, "resolve_manifest_path_for_read", lambda relative_path: root / relative_path
    )

    index = manifest_api._build_manifest_id_index_uncached()
    assert index["wan-2-2-14b-animate"] == latest_rel


def test_get_manifest_resolves_latest_when_legacy_id_collides(tmp_path, monkeypatch):
    manifest_api = _import_manifest_api()
    root = tmp_path
    old_rel = "v0.1.0/video/wan-2.2-14b-animate-1.0.0.v1.yml"
    legacy_rel = "v0.1.2/legacy/wan-2.2-14b-animate-1.0.0.v1.yml"
    latest_rel = "v0.1.2/video/wan-2.2-14b-animate.yml"

    _write_manifest(root / old_rel, "wan-2-2-14b-animate")
    _write_manifest(root / legacy_rel, "wan-2-2-14b-animate")
    _write_manifest(root / latest_rel, "wan-2-2-14b-animate")

    def _iter_manifest_relative_paths(*, suffixes=(".yml", ".yaml")):
        yield old_rel
        yield legacy_rel
        yield latest_rel

    monkeypatch.setattr(
        manifest_api, "iter_manifest_relative_paths", _iter_manifest_relative_paths
    )
    monkeypatch.setattr(
        manifest_api, "resolve_manifest_path_for_read", lambda relative_path: root / relative_path
    )
    monkeypatch.setattr(
        manifest_api,
        "_load_and_enrich_manifest",
        lambda relative_path: {"id": "wan-2-2-14b-animate", "full_path": relative_path},
    )

    resolved = manifest_api.get_manifest("wan-2-2-14b-animate")
    assert resolved["full_path"] == latest_rel
