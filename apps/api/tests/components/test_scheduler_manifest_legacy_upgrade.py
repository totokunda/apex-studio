from __future__ import annotations

from pathlib import Path

import yaml

from src.utils.scheduler_manifest import expand_scheduler_manifests


def _write_catalog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "api_version": "apex/v1",
        "kind": "SchedulerCatalog",
        "spec": {
            "fields": {
                "shift": {
                    "label": "Shift",
                    "type": "number+slider",
                }
            },
            "scheduler_options": [
                {
                    "name": "EulerFlowScheduler",
                    "base": "src.scheduler.flow_deterministic.EulerFlowScheduler",
                    "config": {"shift": 3.0},
                },
                {
                    "name": "UniPCMultistepSchedulerBH2",
                    "base": "src.scheduler.unipc.UniPCMultistepScheduler",
                    "config": {"solver_type": "bh2"},
                },
            ],
        },
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def test_legacy_flow_scheduler_component_gets_catalog_options(tmp_path: Path):
    manifest_root = tmp_path / "manifest"
    _write_catalog(manifest_root / "schedulers" / "flow_matching.yml")

    doc = {
        "spec": {
            "components": [
                {
                    "type": "scheduler",
                    "default": "FlowMatchEulerDiscreteScheduler",
                    "scheduler_options": [
                        {
                            "name": "FlowMatchEulerDiscreteScheduler",
                            "base": "diffusers.FlowMatchEulerDiscreteScheduler",
                            "config_path": "legacy/scheduler.json",
                        }
                    ],
                    "scheduler_config_defaults": {"shift": 2.5},
                }
            ]
        }
    }

    out = expand_scheduler_manifests(
        doc,
        base_path=manifest_root / "v0.1.0" / "image" / "legacy.yml",
        manifest_root=manifest_root,
    )

    component = out["spec"]["components"][0]
    option_names = [str(opt.get("name")) for opt in component.get("scheduler_options", [])]
    assert option_names == ["EulerFlowScheduler", "UniPCMultistepSchedulerBH2"]
    assert component.get("default") == "EulerFlowScheduler"
    assert component.get("scheduler_fields", {}).get("shift", {}).get("label") == "Shift"
    # Ensure the legacy base path is not kept after upgrade.
    euler = next(opt for opt in component["scheduler_options"] if opt["name"] == "EulerFlowScheduler")
    assert euler.get("base") == "src.scheduler.flow_deterministic.EulerFlowScheduler"
    assert euler.get("config", {}).get("shift") == 3.0


def test_non_flow_legacy_scheduler_component_not_auto_rewritten(tmp_path: Path):
    manifest_root = tmp_path / "manifest"
    _write_catalog(manifest_root / "schedulers" / "flow_matching.yml")

    doc = {
        "spec": {
            "components": [
                {
                    "type": "scheduler",
                    "default": "FlowMatchPairScheduler",
                    "scheduler_options": [
                        {
                            "name": "FlowMatchPairScheduler",
                            "base": "src.scheduler.flow_match_pair.FlowMatchPairScheduler",
                        }
                    ],
                }
            ]
        }
    }

    out = expand_scheduler_manifests(
        doc,
        base_path=manifest_root / "v0.1.0" / "video" / "legacy.yml",
        manifest_root=manifest_root,
    )

    component = out["spec"]["components"][0]
    option_names = [str(opt.get("name")) for opt in component.get("scheduler_options", [])]
    assert option_names == ["FlowMatchPairScheduler"]
    assert component.get("default") == "FlowMatchPairScheduler"
    assert "scheduler_fields" not in component


def test_local_manifest_can_resolve_catalog_from_source_manifest(tmp_path: Path):
    source_manifest_root = tmp_path / "manifest"
    local_manifest_root = tmp_path / ".local_manifest"
    _write_catalog(source_manifest_root / "schedulers" / "flow_matching.yml")

    doc = {
        "spec": {
            "components": [
                {
                    "type": "scheduler",
                    "default": "FlowMatchEulerDiscreteScheduler",
                    "scheduler_options": [
                        {
                            "name": "FlowMatchEulerDiscreteScheduler",
                            "base": "diffusers.FlowMatchEulerDiscreteScheduler",
                        }
                    ],
                }
            ]
        }
    }

    out = expand_scheduler_manifests(
        doc,
        base_path=local_manifest_root / "v0.1.2" / "legacy" / "zimage-turbo.yml",
        manifest_root=local_manifest_root,
    )

    component = out["spec"]["components"][0]
    option_names = [str(opt.get("name")) for opt in component.get("scheduler_options", [])]
    assert "EulerFlowScheduler" in option_names
    assert "UniPCMultistepSchedulerBH2" in option_names
