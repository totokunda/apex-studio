from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_LEGACY_FLOW_SCHEDULER_NAME_MAP: Dict[str, str] = {
    "FlowMatchEulerDiscreteScheduler": "EulerFlowScheduler",
    "FlowMatchDiscreteScheduler": "EulerFlowScheduler",
    "FlowMatchDiscreteScheduler720p": "EulerFlowScheduler",
    "FlowMatchScheduler": "EulerFlowScheduler",
    "UniPCMultistepScheduler": "UniPCMultistepSchedulerBH2",
    "DPMSolverMultistepScheduler": "DPMpp2MFlowScheduler",
}

_LEGACY_FLOW_SCHEDULER_BASE_MARKERS = (
    "diffusers.flowmatcheulerdiscretescheduler",
    "diffusers.schedulers.flowmatcheulerdiscretescheduler",
    "diffusers.unipcmultistepscheduler",
    "diffusers.dpmsolvermultistepscheduler",
    "src.scheduler.unipc.unipcmultistepscheduler",
)


def _map_legacy_flow_scheduler_name(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    clean = name.strip()
    if not clean:
        return None
    return _LEGACY_FLOW_SCHEDULER_NAME_MAP.get(clean, clean)


def _looks_like_legacy_flow_scheduler_component(component: Dict[str, Any]) -> bool:
    local_options = component.get("scheduler_options") or []
    local_options = [x for x in local_options if isinstance(x, dict)]

    default_name = component.get("default")
    mapped_default = _map_legacy_flow_scheduler_name(default_name)
    if (
        isinstance(default_name, str)
        and default_name.strip()
        and mapped_default is not None
        and mapped_default != default_name.strip()
    ):
        return True

    for opt in local_options:
        mapped_name = _map_legacy_flow_scheduler_name(opt.get("name"))
        if mapped_name is None:
            continue
        original_name = str(opt.get("name") or "").strip()
        if original_name and mapped_name != original_name:
            return True

        base = str(opt.get("base") or "").strip().lower()
        if any(marker in base for marker in _LEGACY_FLOW_SCHEDULER_BASE_MARKERS):
            return True

    return False


def _resolve_scheduler_manifest_ref_for_component(
    component: Dict[str, Any],
) -> tuple[Optional[str], bool]:
    """
    Return (manifest_ref, inferred_legacy_flow_catalog).

    Legacy manifests (v0.1.0 / .local_manifest legacy copies) often embed old
    diffusers scheduler options directly without `scheduler_manifest`. For those,
    auto-upgrade to the shared flow catalog so UI always shows the modern list.
    """
    scheduler_manifest_ref = component.get("scheduler_manifest")
    if isinstance(scheduler_manifest_ref, str) and scheduler_manifest_ref.strip():
        return scheduler_manifest_ref.strip(), False

    if _looks_like_legacy_flow_scheduler_component(component):
        return "schedulers/flow_matching.yml", True

    return None, False


def _extract_scheduler_options(doc: Any) -> List[Dict[str, Any]]:
    """
    Accept a few shapes to keep the scheduler catalog flexible:

    - { "scheduler_options": [ ... ] }
    - { "spec": { "scheduler_options": [ ... ] } }
    - { "schedulers": [ ... ] }
    - { "spec": { "schedulers": [ ... ] } }
    """
    if not isinstance(doc, dict):
        return []

    candidates = [
        doc.get("scheduler_options"),
        (doc.get("spec") or {}).get("scheduler_options") if isinstance(doc.get("spec"), dict) else None,
        doc.get("schedulers"),
        (doc.get("spec") or {}).get("schedulers") if isinstance(doc.get("spec"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _extract_scheduler_fields(doc: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract optional scheduler field metadata map.

    Accepted shapes:
    - { "fields": { ... } }
    - { "spec": { "fields": { ... } } }
    """
    if not isinstance(doc, dict):
        return {}

    candidates = [
        doc.get("fields"),
        (doc.get("spec") or {}).get("fields")
        if isinstance(doc.get("spec"), dict)
        else None,
    ]
    for value in candidates:
        if isinstance(value, dict):
            return {
                str(k): v
                for k, v in value.items()
                if isinstance(k, (str, int)) and isinstance(v, dict)
            }
    return {}


def _resolve_manifest_ref(
    *,
    ref: str,
    base_path: Path,
    manifest_root: Optional[Path],
) -> Optional[Path]:
    """
    Resolve a scheduler manifest reference.

    - First try relative to the current manifest file.
    - Then try relative to the manifest root (e.g. .../manifest/v0.1.2).
    """
    ref = (ref or "").strip()
    if not ref:
        return None

    p = Path(ref)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((base_path.parent / p).resolve())
        if manifest_root is not None:
            candidates.append((manifest_root / p).resolve())
            # Common convention: shared catalogs live under manifest/schedulers/.
            # If the ref is a bare filename (e.g. "flow_matching.yml"), try that folder too.
            try:
                if len(p.parts) == 1:
                    candidates.append((manifest_root / "schedulers" / p).resolve())
            except Exception:
                pass

            # Overlay-aware fallback:
            # If manifest_root is ".local_manifest", shared scheduler catalogs may
            # still live in the sibling "manifest" tree (and vice versa).
            try:
                sibling_root: Optional[Path] = None
                if manifest_root.name == ".local_manifest":
                    sibling_root = manifest_root.parent / "manifest"
                elif manifest_root.name == "manifest":
                    sibling_root = manifest_root.parent / ".local_manifest"
                if sibling_root is not None:
                    candidates.append((sibling_root / p).resolve())
                    if len(p.parts) == 1:
                        candidates.append((sibling_root / "schedulers" / p).resolve())
            except Exception:
                pass

    return next((c for c in candidates if c.exists()), None)


def expand_scheduler_manifests(
    doc: Dict[str, Any],
    *,
    base_path: str | Path,
    manifest_root: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Expand any component with `scheduler_manifest` into `scheduler_options`.

    This keeps model manifests small while allowing schedulers to be curated
    in a dedicated folder and edited without touching model YAMLs.
    """
    if not isinstance(doc, dict):
        return doc

    base_path = Path(base_path)
    manifest_root_path = Path(manifest_root) if manifest_root is not None else None

    # Support both v1 raw shape (doc.spec.components) and normalized/legacy
    # engine shape (doc.components).
    spec = doc.get("spec")
    if isinstance(spec, dict) and isinstance(spec.get("components"), list):
        components = spec.get("components") or []
        components_container: str = "spec"
    else:
        components = doc.get("components")
        components_container = "root"
        if not isinstance(components, list):
            return doc

    for i, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        if component.get("type") != "scheduler":
            continue

        # Optional model-specific scheduler config tuning applied to all options.
        # - defaults: only fill missing keys
        # - overrides: always overwrite
        scheduler_config_defaults = component.get("scheduler_config_defaults") or {}
        scheduler_config_overrides = component.get("scheduler_config_overrides") or {}
        scheduler_config_defaults = (
            scheduler_config_defaults if isinstance(scheduler_config_defaults, dict) else {}
        )
        scheduler_config_overrides = (
            scheduler_config_overrides if isinstance(scheduler_config_overrides, dict) else {}
        )

        scheduler_manifest_ref, inferred_legacy_flow_catalog = (
            _resolve_scheduler_manifest_ref_for_component(component)
        )
        if not scheduler_manifest_ref:
            continue

        resolved = _resolve_manifest_ref(
            ref=scheduler_manifest_ref,
            base_path=base_path,
            manifest_root=manifest_root_path,
        )
        if resolved is None:
            # Best-effort: if it's missing, leave the manifest untouched.
            continue
        


        try:
            scheduler_doc = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except Exception:
            continue

        catalog_options = _extract_scheduler_options(scheduler_doc)
        catalog_fields = _extract_scheduler_fields(scheduler_doc)
        local_options = component.get("scheduler_options") or []
        local_options = [x for x in local_options if isinstance(x, dict)]

        # For legacy manifests auto-upgraded to a shared catalog, keep only
        # local options that map to known catalog entries and only as config/label
        # overrides. This avoids leaking legacy `base` / config path values.
        if inferred_legacy_flow_catalog and catalog_options:
            catalog_names = {
                str(opt.get("name")).strip()
                for opt in catalog_options
                if isinstance(opt, dict) and isinstance(opt.get("name"), str)
            }
            filtered_local_options: List[Dict[str, Any]] = []
            for opt in local_options:
                mapped_name = _map_legacy_flow_scheduler_name(opt.get("name"))
                if not mapped_name or mapped_name not in catalog_names:
                    continue
                normalized = dict(opt)
                normalized["name"] = mapped_name
                normalized.pop("base", None)
                normalized.pop("config_path", None)
                normalized.pop("config_id", None)
                filtered_local_options.append(normalized)
            local_options = filtered_local_options

            default_name = component.get("default")
            mapped_default = _map_legacy_flow_scheduler_name(default_name)
            if mapped_default and mapped_default in catalog_names:
                component["default"] = mapped_default
            elif "EulerFlowScheduler" in catalog_names:
                component["default"] = "EulerFlowScheduler"

        if not catalog_options and not local_options:
            continue

        # Merge by scheduler `name`. Local overrides catalog on conflicts.
        merged_by_name: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        def _upsert(option: Dict[str, Any], *, prefer_existing_order: bool) -> None:
            name = option.get("name")
            if not isinstance(name, str) or not name.strip():
                return
            name = name.strip()

            if name not in order:
                order.append(name)
            elif not prefer_existing_order:
                # no re-ordering currently, but keep hook for future
                pass

            existing = merged_by_name.get(name, {})
            merged = dict(existing)
            for k, v in option.items():
                if k == "config" and isinstance(v, dict) and isinstance(existing.get("config"), dict):
                    merged_config = dict(existing.get("config") or {})
                    merged_config.update(v)
                    merged["config"] = merged_config
                else:
                    merged[k] = v
            merged_by_name[name] = merged

        for opt in catalog_options:
            _upsert(opt, prefer_existing_order=True)
        for opt in local_options:
            _upsert(opt, prefer_existing_order=True)

        # Apply model-specific defaults/overrides to each scheduler option's config.
        if scheduler_config_defaults or scheduler_config_overrides:
            for name, opt in list(merged_by_name.items()):
                if not isinstance(opt, dict):
                    continue
                cfg = opt.get("config") or {}
                cfg = cfg if isinstance(cfg, dict) else {}
                if scheduler_config_defaults:
                    for k, v in scheduler_config_defaults.items():
                        if k not in cfg:
                            cfg[k] = v
                if scheduler_config_overrides:
                    cfg.update(scheduler_config_overrides)
                if cfg:
                    opt["config"] = cfg
                merged_by_name[name] = opt

        component["scheduler_options"] = [
            merged_by_name[n] for n in order if n in merged_by_name
        ]
        if catalog_fields:
            component["scheduler_fields"] = catalog_fields
        components[i] = component

    if components_container == "spec":
        # spec is guaranteed to be a dict here
        spec = doc.get("spec") or {}
        if isinstance(spec, dict):
            spec["components"] = components
            doc["spec"] = spec
    else:
        doc["components"] = components
    return doc
