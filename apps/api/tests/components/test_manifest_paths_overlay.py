from __future__ import annotations

from pathlib import Path

from src.manifest import paths as manifest_paths


def _patch_manifest_roots(monkeypatch, source_root: Path, local_root: Path) -> None:
    monkeypatch.setattr(
        manifest_paths, "source_manifest_base_path", lambda: source_root
    )
    monkeypatch.setattr(manifest_paths, "local_manifest_base_path", lambda: local_root)


def test_resolve_manifest_path_prefers_local_overlay(tmp_path, monkeypatch):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_manifest_roots(monkeypatch, source_root, local_root)

    rel = Path("v0.1.2") / "video" / "demo.yml"
    source_file = source_root / rel
    local_file = local_root / rel
    source_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("metadata:\n  id: source\n", encoding="utf-8")
    local_file.write_text("metadata:\n  id: local\n", encoding="utf-8")

    resolved = manifest_paths.resolve_manifest_path(rel)
    assert resolved == local_file


def test_ensure_local_manifest_path_copies_from_source(tmp_path, monkeypatch):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_manifest_roots(monkeypatch, source_root, local_root)

    rel = Path("v0.1.2") / "video" / "copyme.yml"
    source_file = source_root / rel
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("metadata:\n  id: copied\n", encoding="utf-8")

    local_file = manifest_paths.ensure_local_manifest_path(rel)
    assert local_file == local_root / rel
    assert local_file.exists()
    assert local_file.read_text(encoding="utf-8") == source_file.read_text(
        encoding="utf-8"
    )


def test_iter_manifest_relative_paths_deduplicates_overlay(tmp_path, monkeypatch):
    source_root = tmp_path / "manifest"
    local_root = tmp_path / ".local_manifest"
    _patch_manifest_roots(monkeypatch, source_root, local_root)

    shared_rel = Path("v0.1.2") / "video" / "same.yml"
    source_only_rel = Path("v0.1.2") / "video" / "source_only.yml"
    (source_root / shared_rel).parent.mkdir(parents=True, exist_ok=True)
    (local_root / shared_rel).parent.mkdir(parents=True, exist_ok=True)

    (source_root / shared_rel).write_text("metadata:\n  id: source\n", encoding="utf-8")
    (local_root / shared_rel).write_text("metadata:\n  id: local\n", encoding="utf-8")
    (source_root / source_only_rel).write_text(
        "metadata:\n  id: source-only\n", encoding="utf-8"
    )

    rels = list(manifest_paths.iter_manifest_relative_paths())
    assert rels.count(shared_rel.as_posix()) == 1
    assert source_only_rel.as_posix() in rels
