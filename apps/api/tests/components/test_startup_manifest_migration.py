from __future__ import annotations

from pathlib import Path

import yaml

from src.manifest import startup_migration as sm


def _write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _base_model_doc(
    *,
    manifest_id: str,
    model: str,
    engine: str,
    model_type: str,
    component_type: str,
    component_name: str,
    component_base: str,
    model_path: str,
) -> dict:
    return {
        "api_version": "apex/v1",
        "kind": "Model",
        "metadata": {
            "id": manifest_id,
            "model": model,
            "name": manifest_id,
            "version": "1.0.0",
            "categories": ["text-to-video"],
        },
        "spec": {
            "engine": engine,
            "model_type": model_type,
            "components": [
                {
                    "type": component_type,
                    "name": component_name,
                    "base": component_base,
                    "model_path": [{"path": model_path, "variant": "default"}],
                }
            ],
            "ui": {"inputs": []},
        },
    }


def _patch_roots(monkeypatch, source_root: Path, local_root: Path) -> None:
    monkeypatch.setattr(sm, "source_manifest_base_path", lambda: source_root)
    monkeypatch.setattr(sm, "local_manifest_base_path", lambda: local_root)
    monkeypatch.setattr(sm, "get_components_path", lambda: str(source_root / "_components"))


def test_startup_migration_maps_downloaded_legacy_path_into_v012(monkeypatch, tmp_path):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_roots(monkeypatch, source_root, local_root)

    old_rel = Path("v0.1.0/video/model-a.yml")
    new_rel = Path("v0.1.2/video/model-a.yml")
    old_path = "legacy/repo/model-a.safetensors"
    new_path = "new/repo/model-a.safetensors"

    _write_yaml(
        source_root / old_rel,
        _base_model_doc(
            manifest_id="model-a",
            model="model-a",
            engine="wan",
            model_type="t2v",
            component_type="transformer",
            component_name="transformer",
            component_base="wan.base",
            model_path=old_path,
        ),
    )
    _write_yaml(
        source_root / new_rel,
        _base_model_doc(
            manifest_id="model-a",
            model="model-a",
            engine="wan",
            model_type="t2v",
            component_type="transformer",
            component_name="transformer",
            component_base="wan.base",
            model_path=new_path,
        ),
    )

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/model-a.safetensors" if model_path == old_path else None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded))

    first_stats = sm.run_startup_manifest_migration()
    assert first_stats["mapped_manifests"] >= 1
    assert first_stats["mapped_paths"] >= 1

    local_new_path = local_root / new_rel
    assert local_new_path.exists()
    local_doc = yaml.safe_load(local_new_path.read_text(encoding="utf-8"))
    entries = local_doc["spec"]["components"][0]["model_path"]
    # Existing latest entry must remain present and unchanged at the front.
    assert isinstance(entries, list) and entries
    assert entries[0]["path"] == new_path
    assert entries[0].get("variant") == "default"
    # Legacy entry should be appended, never replacing the latest one.
    legacy_paths = [e["path"] for e in entries if isinstance(e, dict) and e.get("legacy")]
    assert old_path in legacy_paths
    assert any(isinstance(e, dict) and e.get("path") == new_path for e in entries)

    # Idempotent re-run should not duplicate the same legacy path entry.
    sm.run_startup_manifest_migration()
    local_doc_again = yaml.safe_load(local_new_path.read_text(encoding="utf-8"))
    entries_again = local_doc_again["spec"]["components"][0]["model_path"]
    assert sum(1 for e in entries_again if isinstance(e, dict) and e.get("path") == old_path) == 1

    # If legacy weight disappears from disk, the previously-added legacy entry
    # must be pruned while keeping the latest path.
    def _fake_none(_cls, _model_path: str, _save_path: str):
        return None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_none))
    sm.run_startup_manifest_migration()
    local_doc_pruned = yaml.safe_load(local_new_path.read_text(encoding="utf-8"))
    entries_pruned = local_doc_pruned["spec"]["components"][0]["model_path"]
    assert any(isinstance(e, dict) and e.get("path") == new_path for e in entries_pruned)
    assert not any(isinstance(e, dict) and e.get("path") == old_path for e in entries_pruned)


def test_startup_migration_applies_ltx_transformer_base_exception(monkeypatch, tmp_path):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_roots(monkeypatch, source_root, local_root)

    old_rel = Path("v0.1.0/video/ltx2-special.yml")
    new_rel = Path("v0.1.2/video/ltx2-special.yml")
    old_path = "legacy/ltx/transformer.safetensors"

    _write_yaml(
        source_root / old_rel,
        _base_model_doc(
            manifest_id="ltx2-special",
            model="ltx2",
            engine="ltx22",
            model_type="ti2v",
            component_type="transformer",
            component_name="legacy_transformer",
            component_base="ltx2.base",
            model_path=old_path,
        ),
    )
    _write_yaml(
        source_root / new_rel,
        _base_model_doc(
            manifest_id="ltx2-special",
            model="ltx2",
            engine="ltx22",
            model_type="ti2v",
            component_type="transformer",
            component_name="new_transformer",
            component_base="ltx2.base2",
            model_path="new/ltx/transformer.safetensors",
        ),
    )

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/ltx2.safetensors" if model_path == old_path else None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded))

    sm.run_startup_manifest_migration()
    local_new_path = local_root / new_rel
    assert local_new_path.exists()
    local_doc = yaml.safe_load(local_new_path.read_text(encoding="utf-8"))
    entries = local_doc["spec"]["components"][0]["model_path"]
    assert any(isinstance(e, dict) and e.get("path") == old_path for e in entries)


def test_startup_migration_materializes_unmapped_downloaded_legacy(monkeypatch, tmp_path):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_roots(monkeypatch, source_root, local_root)

    mapped_new = Path("v0.1.2/video/current.yml")
    _write_yaml(
        source_root / mapped_new,
        _base_model_doc(
            manifest_id="current",
            model="current-model",
            engine="wan",
            model_type="t2v",
            component_type="transformer",
            component_name="transformer",
            component_base="wan.base",
            model_path="new/current.safetensors",
        ),
    )

    old_downloaded_rel = Path("v0.1.0/video/old-download.yml")
    old_not_downloaded_rel = Path("v0.1.0/video/old-not-downloaded.yml")
    old_downloaded_path = "legacy/old-download.safetensors"
    old_not_downloaded_path = "legacy/old-not-downloaded.safetensors"
    _write_yaml(
        source_root / old_downloaded_rel,
        _base_model_doc(
            manifest_id="old-download",
            model="orphan",
            engine="orphan",
            model_type="t2v",
            component_type="transformer",
            component_name="transformer",
            component_base="orphan.base",
            model_path=old_downloaded_path,
        ),
    )
    _write_yaml(
        source_root / old_not_downloaded_rel,
        _base_model_doc(
            manifest_id="old-not-downloaded",
            model="orphan",
            engine="orphan",
            model_type="t2v",
            component_type="transformer",
            component_name="transformer",
            component_base="orphan.base",
            model_path=old_not_downloaded_path,
        ),
    )

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/legacy.safetensors" if model_path == old_downloaded_path else None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded))

    stats = sm.run_startup_manifest_migration()
    assert stats["legacy_materialized"] >= 1

    legacy_copy = local_root / "v0.1.2" / "legacy" / "old-download.yml"
    assert legacy_copy.exists()
    legacy_doc = yaml.safe_load(legacy_copy.read_text(encoding="utf-8"))
    tags = legacy_doc.get("metadata", {}).get("tags", [])
    assert any(str(t).lower() == "legacy" for t in tags)

    skipped_copy = local_root / "v0.1.2" / "legacy" / "old-not-downloaded.yml"
    assert not skipped_copy.exists()

    # If previously-downloaded legacy files are removed, the legacy fallback
    # manifest should be removed on next startup.
    def _fake_none(_cls, _model_path: str, _save_path: str):
        return None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_none))
    sm.run_startup_manifest_migration()
    assert not legacy_copy.exists()


def test_startup_migration_maps_exact_match_into_equivalent_sibling_manifest(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_roots(monkeypatch, source_root, local_root)

    old_rel = Path("v0.1.0/image/model-a-control.yml")
    exact_new_rel = Path("v0.1.2/image/model-a-control.yml")
    sibling_new_rel = Path("v0.1.2/image/model-a-inpaint.yml")
    old_path = "legacy/repo/model-a-control.safetensors"
    shared_new_path = "new/repo/model-a-shared.safetensors"

    old_doc = _base_model_doc(
        manifest_id="model-a-control",
        model="model-a",
        engine="zimage",
        model_type="control",
        component_type="transformer",
        component_name="transformer",
        component_base="zimage.control",
        model_path=old_path,
    )
    exact_new_doc = _base_model_doc(
        manifest_id="model-a-control",
        model="model-a",
        engine="zimage",
        model_type="control",
        component_type="transformer",
        component_name="transformer",
        component_base="zimage.control",
        model_path=shared_new_path,
    )
    sibling_new_doc = _base_model_doc(
        manifest_id="model-a-inpaint",
        model="model-a",
        engine="zimage",
        model_type="control",
        component_type="transformer",
        component_name="transformer",
        component_base="zimage.control",
        model_path=shared_new_path,
    )
    sibling_new_doc["spec"]["ui"] = {
        "inputs": [{"id": "mask_image"}, {"id": "inpaint_image"}]
    }

    _write_yaml(source_root / old_rel, old_doc)
    _write_yaml(source_root / exact_new_rel, exact_new_doc)
    _write_yaml(source_root / sibling_new_rel, sibling_new_doc)

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/model-a-control.safetensors" if model_path == old_path else None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded))

    stats = sm.run_startup_manifest_migration()
    assert stats["mapped_paths"] >= 2

    for rel in (exact_new_rel, sibling_new_rel):
        local_doc = yaml.safe_load((local_root / rel).read_text(encoding="utf-8"))
        entries = local_doc["spec"]["components"][0]["model_path"]
        assert any(isinstance(e, dict) and e.get("path") == old_path and e.get("legacy") is True for e in entries)


def test_startup_migration_shares_zimage_family_components_across_variants(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_roots(monkeypatch, source_root, local_root)

    old_rel = Path("v0.1.0/image/zimage-turbo-control.yml")
    new_control_rel = Path("v0.1.2/image/zimage-turbo-control.yml")
    new_turbo_rel = Path("v0.1.2/image/zimage-turbo.yml")
    new_base_rel = Path("v0.1.2/image/zimage.yml")
    old_text_encoder_path = "Tongyi-MAI/Z-Image-Turbo/text_encoder"
    new_text_encoder_path = "totoku/apex-models/FLUX.2-klein-4B/text_encoder/text_encoder-bf16.safetensors"

    _write_yaml(
        source_root / old_rel,
        _base_model_doc(
            manifest_id="zimage-turbo-control",
            model="zimage",
            engine="zimage",
            model_type="control",
            component_type="text_encoder",
            component_name="text_encoder",
            component_base="Qwen3ForCausalLM",
            model_path=old_text_encoder_path,
        ),
    )
    for rel, manifest_id, model_type in (
        (new_control_rel, "zimage-turbo-control", "control"),
        (new_turbo_rel, "zimage-turbo", "t2i"),
        (new_base_rel, "zimage", "t2i"),
    ):
        _write_yaml(
            source_root / rel,
            _base_model_doc(
                manifest_id=manifest_id,
                model="zimage",
                engine="zimage",
                model_type=model_type,
                component_type="text_encoder",
                component_name="text_encoder",
                component_base="Qwen3ForCausalLM",
                model_path=new_text_encoder_path,
            ),
        )

    def _fake_is_downloaded(_cls, model_path: str, _save_path: str):
        return "/tmp/downloaded/zimage-text-encoder.safetensors" if model_path == old_text_encoder_path else None

    monkeypatch.setattr(sm.DownloadMixin, "is_downloaded", classmethod(_fake_is_downloaded))

    stats = sm.run_startup_manifest_migration()
    assert stats["mapped_paths"] >= 3

    for rel in (new_control_rel, new_turbo_rel, new_base_rel):
        local_doc = yaml.safe_load((local_root / rel).read_text(encoding="utf-8"))
        entries = local_doc["spec"]["components"][0]["model_path"]
        assert any(
            isinstance(e, dict)
            and e.get("path") == old_text_encoder_path
            and e.get("legacy") is True
            and e.get("legacy_source_manifest_id") == "zimage-turbo-control"
            for e in entries
        )
