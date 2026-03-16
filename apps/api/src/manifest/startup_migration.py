from __future__ import annotations

import re
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

import yaml
from loguru import logger

from src.manifest.paths import (
    local_manifest_base_path,
    source_manifest_base_path,
)
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path
from src.manifest.db import  setup_manifest_db


SOURCE_VERSION = "v0.1.0"
TARGET_VERSION = "v0.1.2"
STATE_FILE_NAME = ".startup_manifest_migration_state.json"
LEGACY_DIR_NAME = "legacy"
COMPONENT_SHARING_FAMILIES = {
    # ZImage family variants (base/turbo/control/inpaint) intentionally share
    # component weights (notably text encoder + VAE) across manifest ids.
    ("zimage", "zimage"),  # (engine, model)
}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _slug(value: str) -> str:
    s = _norm(value)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "legacy"


def _tokenize(value: str) -> set[str]:
    stopwords = {"to", "image", "video", "text", "audio", "model"}
    tokens = set(re.findall(r"[a-z0-9]+", _norm(value)))
    return {t for t in tokens if t not in stopwords}


def _looks_like_ltx(*values: str) -> bool:
    joined = " ".join(_norm(v) for v in values)
    return "ltx" in joined


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _safe_write_yaml(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def _normalize_relative_path(relative_path: str | Path) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError(f"Manifest path must be relative: {relative_path}")
    parts: List[str] = []
    for part in rel.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"Manifest path cannot traverse parents: {relative_path}")
        parts.append(part)
    if not parts:
        raise ValueError("Manifest path is empty")
    return Path(*parts)


def _ensure_local_manifest_path(relative_path: str | Path, *, copy_if_missing: bool) -> Path:
    rel = _normalize_relative_path(relative_path)
    local_path = local_manifest_base_path() / rel
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not copy_if_missing:
        return local_path

    source_path = source_manifest_base_path() / rel
    if not source_path.exists():
        raise FileNotFoundError(f"Manifest not found for copy: {rel.as_posix()}")

    shutil.copy2(source_path, local_path)
    return local_path


def _iter_components(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = doc.get("spec") or {}
    if not isinstance(spec, dict):
        return []
    components = spec.get("components") or []
    if not isinstance(components, list):
        return []
    return [c for c in components if isinstance(c, dict)]


def _iter_model_path_items(component: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return model_path entries as dicts with at minimum {"path": <str>}.
    """
    raw = component.get("model_path")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"path": raw}]
    if isinstance(raw, dict):
        p = raw.get("path")
        return [raw] if isinstance(p, str) and p else []
    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                out.append({"path": item})
            elif isinstance(item, dict):
                p = item.get("path")
                if isinstance(p, str) and p:
                    out.append(item)
        return out
    return []


def _ensure_component_model_path_list(component: Dict[str, Any]) -> List[Dict[str, Any]]:
    current = _iter_model_path_items(component)
    component["model_path"] = [dict(item) for item in current]
    return component["model_path"]


def _component_key_pairs(component: Dict[str, Any]) -> set[Tuple[str, str]]:
    comp_type = _norm(component.get("type"))
    if not comp_type:
        return set()
    keys: set[Tuple[str, str]] = set()
    for field in ("name", "base", "label"):
        v = _norm(component.get(field))
        if v:
            keys.add((comp_type, v))
    return keys


def _manifest_component_signature(doc: Dict[str, Any]) -> set[Tuple[str, str]]:
    sig: set[Tuple[str, str]] = set()
    for comp in _iter_components(doc):
        sig.update(_component_key_pairs(comp))
    return sig


def _manifest_component_types(doc: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for comp in _iter_components(doc):
        comp_type = _norm(comp.get("type"))
        if comp_type:
            out.add(comp_type)
    return out


def _manifest_component_path_signature(doc: Dict[str, Any]) -> set[Tuple[str, str, str]]:
    """
    Signature of model paths grouped by component identity hints.
    Used to detect manifests that are effectively the same model graph and weights.
    """
    out: set[Tuple[str, str, str]] = set()
    for comp in _iter_components(doc):
        comp_type = _norm(comp.get("type"))
        if not comp_type:
            continue
        comp_key = (
            _norm(comp.get("name"))
            or _norm(comp.get("base"))
            or _norm(comp.get("label"))
        )
        for item in _iter_model_path_items(comp):
            path = _norm(item.get("path"))
            if path:
                out.add((comp_type, comp_key, path))
    return out


def _overlap_ratio(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right))) / float(max(len(left), len(right)))


def _downloaded_model_paths_by_component(
    doc: Dict[str, Any],
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Returns tuples of (component_index, model_path_entry_dict) for downloaded paths.
    """
    downloaded: List[Tuple[int, Dict[str, Any]]] = []
    components = _iter_components(doc)
    components_root = get_components_path()

    for idx, comp in enumerate(components):
        for item in _iter_model_path_items(comp):
            p = item.get("path")
            if not isinstance(p, str) or not p.strip():
                continue
            local = DownloadMixin.is_downloaded(p, components_root)
            if local:
                downloaded.append((idx, dict(item)))
    return downloaded


@dataclass
class ManifestRecord:
    relative_path: str
    path: Path
    doc: Dict[str, Any]
    manifest_id: str
    model: str
    engine: str
    model_type: str
    categories: set[str]
    component_sig: set[Tuple[str, str]]
    component_types: set[str]
    component_path_sig: set[Tuple[str, str, str]]
    is_ltx: bool


def _build_manifest_record(path: Path, relative_path: str) -> Optional[ManifestRecord]:
    doc = _safe_load_yaml(path)
    if not isinstance(doc, dict):
        return None

    if _norm(doc.get("kind")) == "modelgroup":
        return None

    md = doc.get("metadata") or {}
    spec = doc.get("spec") or {}
    if not isinstance(md, dict) or not isinstance(spec, dict):
        return None

    manifest_id = _norm(md.get("id"))
    if not manifest_id:
        return None

    categories_raw = md.get("categories") or []
    if isinstance(categories_raw, str):
        categories = {_norm(categories_raw)}
    elif isinstance(categories_raw, list):
        categories = {_norm(c) for c in categories_raw if _norm(c)}
    else:
        categories = set()

    model = _norm(md.get("model"))
    engine = _norm(spec.get("engine"))
    model_type = _norm(spec.get("model_type"))
    if not model_type:
        model_type = _norm(spec.get("modelType"))

    is_ltx = _looks_like_ltx(manifest_id, model, engine, md.get("name", ""))
    return ManifestRecord(
        relative_path=relative_path,
        path=path,
        doc=doc,
        manifest_id=manifest_id,
        model=model,
        engine=engine,
        model_type=model_type,
        categories=categories,
        component_sig=_manifest_component_signature(doc),
        component_types=_manifest_component_types(doc),
        component_path_sig=_manifest_component_path_signature(doc),
        is_ltx=is_ltx,
    )


def _collect_records(version_dir: Path) -> List[ManifestRecord]:
    out: List[ManifestRecord] = []
    if not version_dir.exists():
        return out
    for path in sorted(version_dir.rglob("*.yml")):
        rel = path.relative_to(version_dir.parent).as_posix()
        rec = _build_manifest_record(path, rel)
        if rec is not None:
            out.append(rec)
    return out


def _manifest_similarity(old: ManifestRecord, new: ManifestRecord) -> Tuple[float, int]:
    score = 0.0

    if old.engine and new.engine and old.engine == new.engine:
        score += 3.0
    if old.model and new.model and old.model == new.model:
        score += 5.0
    if old.model_type and new.model_type and old.model_type == new.model_type:
        score += 3.0

    if old.categories and new.categories:
        overlap = old.categories.intersection(new.categories)
        score += min(4.0, float(len(overlap)) * 1.5)

    old_id_tokens = _tokenize(old.manifest_id)
    new_id_tokens = _tokenize(new.manifest_id)
    if old_id_tokens and new_id_tokens:
        overlap = old_id_tokens.intersection(new_id_tokens)
        union = old_id_tokens.union(new_id_tokens)
        if union:
            score += 6.0 * (len(overlap) / len(union))

    comp_overlap = old.component_sig.intersection(new.component_sig)
    comp_match_count = len(comp_overlap)
    score += float(comp_match_count) * 1.6

    type_overlap = old.component_types.intersection(new.component_types)
    score += float(len(type_overlap)) * 1.5

    if old.is_ltx and new.is_ltx:
        score += 2.0

    return score, comp_match_count


def _component_match_score(
    old_comp: Dict[str, Any], new_comp: Dict[str, Any], *, allow_ltx_transformer_base_change: bool
) -> float:
    old_type = _norm(old_comp.get("type"))
    new_type = _norm(new_comp.get("type"))
    if not old_type or old_type != new_type:
        return -1.0

    old_name = _norm(old_comp.get("name"))
    new_name = _norm(new_comp.get("name"))
    old_base = _norm(old_comp.get("base"))
    new_base = _norm(new_comp.get("base"))
    old_label = _norm(old_comp.get("label"))
    new_label = _norm(new_comp.get("label"))

    score = 0.0
    name_match = bool(old_name and new_name and old_name == new_name)
    base_match = bool(old_base and new_base and old_base == new_base)
    ltx_base_exception = False

    if name_match:
        score += 5.0
    if base_match:
        score += 4.0

    # LTX exception: transformer base changed from ltx2.base to ltx2.base2, but old
    # checkpoints should still be accepted as legacy entries.
    if (
        allow_ltx_transformer_base_change
        and old_type == "transformer"
        and old_base.startswith("ltx2.base")
        and new_base.startswith("ltx2.base")
    ):
        ltx_base_exception = True
        score += 4.0

    if old_label and new_label and old_label == new_label and (name_match or base_match or ltx_base_exception):
        score += 1.0

    if score <= 0.0:
        return -1.0
    return score


def _find_matching_component_index(
    old_component: Dict[str, Any],
    new_doc: Dict[str, Any],
    *,
    allow_ltx_transformer_base_change: bool,
) -> Optional[int]:
    components = _iter_components(new_doc)
    best_idx: Optional[int] = None
    best_score = -1.0
    for idx, comp in enumerate(components):
        score = _component_match_score(
            old_component,
            comp,
            allow_ltx_transformer_base_change=allow_ltx_transformer_base_change,
        )
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_score < 0:
        return None
    return best_idx


def _pick_target_records(old: ManifestRecord, candidates: Sequence[ManifestRecord]) -> List[ManifestRecord]:
    """
    Returns best target records in v0.1.2 for an old v0.1.0 manifest.

    - Exact id match is preferred.
    - If exact id exists, also include equivalent sibling manifests in target
      version (same model graph/weights, different input surface).
    - Otherwise choose high-similarity peers; allow ties to support split manifests
      (e.g., old single manifest split into 360p/720p).
    """
    def _are_equivalent_targets(primary: ManifestRecord, candidate: ManifestRecord) -> bool:
        if primary.engine and candidate.engine and primary.engine != candidate.engine:
            return False
        if primary.model and candidate.model and primary.model != candidate.model:
            return False
        if (
            primary.model_type
            and candidate.model_type
            and primary.model_type != candidate.model_type
        ):
            return False

        component_ratio = _overlap_ratio(primary.component_sig, candidate.component_sig)
        type_ratio = _overlap_ratio(primary.component_types, candidate.component_types)
        path_ratio = _overlap_ratio(primary.component_path_sig, candidate.component_path_sig)
        return (
            component_ratio >= 0.9
            and type_ratio >= 0.9
            and path_ratio >= 0.95
        )

    def _is_component_sharing_family(record: ManifestRecord) -> bool:
        return (record.engine, record.model) in COMPONENT_SHARING_FAMILIES

    exact = [c for c in candidates if c.manifest_id == old.manifest_id]
    if exact:
        selected: List[ManifestRecord] = list(exact)
        seen_paths = {rec.relative_path for rec in selected}
        for candidate in candidates:
            if candidate.relative_path in seen_paths:
                continue
            if any(_are_equivalent_targets(primary, candidate) for primary in exact):
                selected.append(candidate)
                seen_paths.add(candidate.relative_path)

        # Family-level component sharing for models whose variants intentionally
        # reuse major components but differ in input surface/conditioning.
        # Component-level merge matching still decides what is actually copied.
        if _is_component_sharing_family(old):
            for candidate in candidates:
                if candidate.relative_path in seen_paths:
                    continue
                if candidate.engine != old.engine or candidate.model != old.model:
                    continue
                selected.append(candidate)
                seen_paths.add(candidate.relative_path)
        return selected

    scored: List[Tuple[float, int, ManifestRecord]] = []
    for c in candidates:
        if old.engine and c.engine and old.engine != c.engine:
            continue
        if old.model and c.model and old.model != c.model:
            continue
        old_tokens = _tokenize(old.manifest_id)
        new_tokens = _tokenize(c.manifest_id)
        token_overlap = old_tokens.intersection(new_tokens) if old_tokens else set()
        model_type_matches = bool(
            old.model_type and c.model_type and old.model_type == c.model_type
        )
        if not token_overlap and not model_type_matches:
            continue
        score, comp_matches = _manifest_similarity(old, c)
        if comp_matches <= 0:
            continue
        if score < 7.5:
            continue
        scored.append((score, comp_matches, c))

    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_score = scored[0][0]
    return [rec for score, _matches, rec in scored if score >= (top_score - 1.2)]


def _merge_legacy_paths_into_target(
    old: ManifestRecord,
    target: ManifestRecord,
    downloaded_entries: Sequence[Tuple[int, Dict[str, Any]]],
) -> Tuple[int, int]:
    """
    Synchronize legacy entries for one old manifest into one target manifest.
    Returns (added_count, removed_count).
    """
    local_target_path = _ensure_local_manifest_path(target.relative_path, copy_if_missing=True)
    local_doc = _safe_load_yaml(local_target_path)
    if not local_doc:
        local_doc = _safe_load_yaml(target.path)
    if not local_doc:
        return (0, 0)

    old_components = _iter_components(old.doc)
    new_components = _iter_components(local_doc)
    if not old_components or not new_components:
        return (0, 0)

    added_count = 0
    removed_count = 0
    allow_ltx_transformer_base_change = old.is_ltx and target.is_ltx
    active_paths = {
        _norm(entry.get("path"))
        for _old_idx, entry in downloaded_entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }

    # Remove stale legacy entries from this source old manifest.
    for component in new_components:
        items = _iter_model_path_items(component)
        if not items:
            continue
        retained: List[Dict[str, Any]] = []
        for item in items:
            source_id = _norm(item.get("legacy_source_manifest_id"))
            is_legacy = bool(item.get("legacy") is True or source_id)
            if is_legacy and source_id == old.manifest_id:
                p = _norm(item.get("path"))
                if p not in active_paths:
                    removed_count += 1
                    continue
            retained.append(item)
        component["model_path"] = retained

    for old_component_index, old_path_item in downloaded_entries:
        if old_component_index < 0 or old_component_index >= len(old_components):
            continue
        old_component = old_components[old_component_index]
        old_path = old_path_item.get("path")
        if not isinstance(old_path, str) or not old_path.strip():
            continue

        match_idx = _find_matching_component_index(
            old_component,
            local_doc,
            allow_ltx_transformer_base_change=allow_ltx_transformer_base_change,
        )
        if match_idx is None:
            continue

        target_component = new_components[match_idx]
        target_items = _ensure_component_model_path_list(target_component)
        if any(_norm(item.get("path")) == _norm(old_path) for item in target_items):
            continue

        new_item = dict(old_path_item)
        new_item.setdefault("variant", f"LEGACY_{_slug(old.manifest_id).upper()}")
        new_item["custom"] = True
        new_item["legacy"] = True
        new_item["legacy_source_manifest_id"] = old.manifest_id
        new_item["legacy_source_manifest_path"] = old.relative_path
        target_items.append(new_item)
        added_count += 1

    if added_count <= 0 and removed_count <= 0:
        return (0, 0)

    spec = local_doc.setdefault("spec", {})
    if isinstance(spec, dict):
        spec["components"] = new_components
    _safe_write_yaml(local_target_path, local_doc)
    return (added_count, removed_count)


def _prune_stale_legacy_entries_from_local_manifests(
    *,
    active_paths_by_manifest_id: Dict[str, set[str]],
) -> int:
    """
    Final cleanup across local v0.1.2 manifests so stale legacy entries don't
    reappear after refresh.
    """
    local_version_root = local_manifest_base_path() / TARGET_VERSION
    if not local_version_root.exists():
        return 0

    removed_total = 0
    for manifest_path in sorted(local_version_root.rglob("*.yml")):
        doc = _safe_load_yaml(manifest_path)
        if not doc:
            continue
        components = _iter_components(doc)
        if not components:
            continue

        changed = False
        for component in components:
            items = _iter_model_path_items(component)
            if not items:
                continue

            retained: List[Dict[str, Any]] = []
            for item in items:
                source_id = _norm(item.get("legacy_source_manifest_id"))
                is_legacy = bool(item.get("legacy") is True or source_id)
                if not is_legacy:
                    retained.append(item)
                    continue

                if not source_id:
                    removed_total += 1
                    changed = True
                    continue

                active_paths = active_paths_by_manifest_id.get(source_id, set())
                if _norm(item.get("path")) not in active_paths:
                    removed_total += 1
                    changed = True
                    continue

                retained.append(item)

            component["model_path"] = retained

        if changed:
            spec = doc.setdefault("spec", {})
            if isinstance(spec, dict):
                spec["components"] = components
            _safe_write_yaml(manifest_path, doc)

    return removed_total


def _materialize_legacy_manifest(old: ManifestRecord) -> Path:
    """
    Create a local fallback manifest under v0.1.2/legacy for removed models.
    """
    old_name = Path(old.relative_path).name
    target_rel = Path(TARGET_VERSION) / LEGACY_DIR_NAME / old_name
    local_path = _ensure_local_manifest_path(target_rel, copy_if_missing=False)

    legacy_doc = _safe_load_yaml(local_path) or dict(old.doc)
    if not legacy_doc:
        legacy_doc = dict(old.doc)

    md = legacy_doc.get("metadata")
    if not isinstance(md, dict):
        md = {}
        legacy_doc["metadata"] = md
    tags = md.get("tags")
    if isinstance(tags, list):
        if "legacy" not in [str(t).strip().lower() for t in tags]:
            tags.append("legacy")
    else:
        md["tags"] = ["legacy"]

    _safe_write_yaml(local_path, legacy_doc)
    return local_path


def _write_state_file(stats: Dict[str, Any]) -> None:
    state_path = local_manifest_base_path() / STATE_FILE_NAME
    payload = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": SOURCE_VERSION,
        "target_version": TARGET_VERSION,
        **stats,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_startup_manifest_migration() -> Dict[str, Any]:
    """
    Startup-only migration:
    - discover downloaded legacy (v0.1.0) model paths,
    - map compatible legacy paths into v0.1.2 manifests in .local_manifest,
    - for legacy manifests with downloaded weights and no v0.1.2 target, copy them
      into .local_manifest/v0.1.2/legacy.
    """
    source_root = source_manifest_base_path()
    old_dir = source_root / SOURCE_VERSION
    new_dir = source_root / TARGET_VERSION
    if not old_dir.exists() or not new_dir.exists():
        return {
            "scanned_old_manifests": 0,
            "mapped_manifests": 0,
            "mapped_paths": 0,
            "legacy_materialized": 0,
            "skipped": "missing_source_dirs",
        }

    old_records = _collect_records(old_dir)
    new_records = _collect_records(new_dir)
    if not old_records or not new_records:
        return {
            "scanned_old_manifests": len(old_records),
            "mapped_manifests": 0,
            "mapped_paths": 0,
            "legacy_materialized": 0,
            "skipped": "no_records",
        }

    mapped_manifests = 0
    mapped_paths = 0
    removed_stale_legacy_paths = 0
    legacy_materialized = 0
    new_ids = {r.manifest_id for r in new_records}
    active_paths_by_old_id: Dict[str, set[str]] = {}

    for old in old_records:
        downloaded_entries = _downloaded_model_paths_by_component(old.doc)
        targets = _pick_target_records(old, new_records)
        active_paths_by_old_id[old.manifest_id] = {
            _norm(entry.get("path"))
            for _old_idx, entry in downloaded_entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        any_added = False
        for target in targets:
            added, removed = _merge_legacy_paths_into_target(
                old, target, downloaded_entries
            )
            if added > 0:
                mapped_manifests += 1
                mapped_paths += added
                any_added = True
            removed_stale_legacy_paths += removed

        # Only create legacy fallback manifests for models that truly no longer
        # exist as first-class manifests in the target version.
        if old.manifest_id not in new_ids:
            legacy_path = (
                local_manifest_base_path()
                / TARGET_VERSION
                / LEGACY_DIR_NAME
                / Path(old.relative_path).name
            )
            if downloaded_entries and not any_added:
                _materialize_legacy_manifest(old)
                legacy_materialized += 1
            elif not downloaded_entries and legacy_path.exists():
                try:
                    legacy_path.unlink()
                except Exception:
                    pass

    # Defensive cleanup in case target-selection heuristics change between runs.
    removed_stale_legacy_paths += _prune_stale_legacy_entries_from_local_manifests(
        active_paths_by_manifest_id=active_paths_by_old_id
    )

    stats = {
        "scanned_old_manifests": len(old_records),
        "mapped_manifests": mapped_manifests,
        "mapped_paths": mapped_paths,
        "removed_stale_legacy_paths": removed_stale_legacy_paths,
        "legacy_materialized": legacy_materialized,
    }
    _write_state_file(stats)
    
    return stats


def run_startup_manifest_migration_safe() -> Dict[str, Any]:
    try:
        stats = run_startup_manifest_migration()
        logger.info(f"Startup manifest migration completed: {stats}")
        return stats
    except Exception as e:
        logger.warning(f"Startup manifest migration failed: {e}")
        return {"error": str(e)}

    finally:
        setup_manifest_db()
        