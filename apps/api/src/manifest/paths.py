from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

MANIFEST_DIR_NAME = "manifest"
LOCAL_MANIFEST_DIR_NAME = ".local_manifest"
_MANIFEST_ROOT_NAMES = {MANIFEST_DIR_NAME, LOCAL_MANIFEST_DIR_NAME}


def project_root() -> Path:
    # src/manifest/paths.py -> src/manifest -> src -> <project root>
    return Path(__file__).resolve().parents[2]


def source_manifest_base_path() -> Path:
    return project_root() / MANIFEST_DIR_NAME


def local_manifest_base_path() -> Path:
    return project_root() / LOCAL_MANIFEST_DIR_NAME


def manifest_base_paths_precedence() -> tuple[Path, Path]:
    # Local manifests override source manifests.
    return (local_manifest_base_path(), source_manifest_base_path())


def normalize_manifest_relative_path(relative_path: str | Path) -> Path:
    p = Path(relative_path)
    if p.is_absolute():
        raise ValueError(f"Manifest path must be relative: {relative_path}")

    cleaned_parts: list[str] = []
    for part in p.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"Manifest path cannot traverse parents: {relative_path}")
        cleaned_parts.append(part)

    if not cleaned_parts:
        raise ValueError("Manifest path is empty")

    return Path(*cleaned_parts)


def resolve_manifest_path(relative_path: str | Path) -> Path:
    """
    Resolve a manifest relative path using overlay precedence:
      1. .local_manifest/<relative_path>
      2. manifest/<relative_path>
    """
    rel = normalize_manifest_relative_path(relative_path)
    local_path = local_manifest_base_path() / rel
    if local_path.exists():
        return local_path
    return source_manifest_base_path() / rel


def ensure_local_manifest_path(
    relative_path: str | Path, *, copy_if_missing: bool = True
) -> Path:
    """
    Return a writable path under .local_manifest for a given manifest relative path.

    If copy_if_missing is True and the local file doesn't exist, copy it from the
    resolved read path (typically manifest/<relative_path>).
    """
    rel = normalize_manifest_relative_path(relative_path)
    local_path = local_manifest_base_path() / rel
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not copy_if_missing:
        return local_path

    source_path = resolve_manifest_path(rel)
    if not source_path.exists():
        raise FileNotFoundError(f"Manifest not found for copy: {rel.as_posix()}")

    # Only copy when source and destination differ.
    if source_path.resolve() != local_path.resolve():
        shutil.copy2(source_path, local_path)

    return local_path


def iter_manifest_relative_paths(
    *, suffixes: tuple[str, ...] = (".yml", ".yaml")
) -> Iterator[str]:
    """
    Iterate manifest file relative paths using overlay precedence.

    If the same relative path exists in both roots, the local version wins.
    """
    seen: set[str] = set()
    for root in manifest_base_paths_precedence():
        if not root.exists():
            continue
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if suffixes and file_path.suffix.lower() not in suffixes:
                continue
            rel = file_path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            yield rel


def find_manifest_root(path: str | Path) -> Path | None:
    """
    Find nearest parent manifest root ('manifest' or '.local_manifest').
    """
    p = Path(path)
    for parent in (p, *p.parents):
        if parent.name in _MANIFEST_ROOT_NAMES:
            return parent
    return None

