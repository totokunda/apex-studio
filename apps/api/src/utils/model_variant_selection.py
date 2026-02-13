from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional
import os

import psutil
import torch

from src.utils.config_store import config_store_lock, read_json_dict
from src.utils.defaults import get_config_store_path

ModelDownloadProfile = Literal["auto", "maximum_performance"]

MODEL_DOWNLOAD_PROFILE_ENV_KEY = "APEX_MODEL_DOWNLOAD_PROFILE"
MODEL_DOWNLOAD_PROFILE_DEFAULT: ModelDownloadProfile = "auto"


def normalize_model_download_profile(value: Any) -> ModelDownloadProfile:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"maximum_performance", "max_performance", "max", "performance"}:
        return "maximum_performance"
    # Aliases from earlier wording in product discussions.
    if raw in {"full", "full_models", "full_model"}:
        return "maximum_performance"
    return "auto"


def get_effective_model_download_profile(
    default: ModelDownloadProfile = MODEL_DOWNLOAD_PROFILE_DEFAULT,
) -> ModelDownloadProfile:
    env_value = os.environ.get(MODEL_DOWNLOAD_PROFILE_ENV_KEY)
    if env_value is not None and str(env_value).strip() != "":
        return normalize_model_download_profile(env_value)

    try:
        store_path = Path(get_config_store_path())
        with config_store_lock(store_path):
            persisted = read_json_dict(store_path)
        if MODEL_DOWNLOAD_PROFILE_ENV_KEY in persisted:
            return normalize_model_download_profile(
                persisted.get(MODEL_DOWNLOAD_PROFILE_ENV_KEY),
            )
    except Exception:
        pass

    return normalize_model_download_profile(default)


@dataclass(frozen=True)
class HardwareMemoryProfile:
    total_ram_gb: float
    max_vram_gb: float
    unified_memory: bool
    has_gpu: bool


def detect_hardware_memory_profile() -> HardwareMemoryProfile:
    total_ram_gb = 0.0
    try:
        total_ram_gb = float(psutil.virtual_memory().total) / float(1024**3)
    except Exception:
        total_ram_gb = 0.0

    has_mps = False
    try:
        has_mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        has_mps = False

    if has_mps:
        # Apple Silicon uses unified memory. Treat RAM as the practical VRAM ceiling.
        return HardwareMemoryProfile(
            total_ram_gb=total_ram_gb,
            max_vram_gb=total_ram_gb,
            unified_memory=True,
            has_gpu=True,
        )

    max_vram_gb = 0.0
    try:
        if torch.cuda.is_available():
            totals = []
            for idx in range(int(torch.cuda.device_count())):
                prop = torch.cuda.get_device_properties(idx)
                totals.append(float(getattr(prop, "total_memory", 0.0)) / float(1024**3))
            if totals:
                max_vram_gb = max(totals)
    except Exception:
        max_vram_gb = 0.0

    return HardwareMemoryProfile(
        total_ram_gb=total_ram_gb,
        max_vram_gb=max_vram_gb,
        unified_memory=False,
        has_gpu=max_vram_gb > 0.0,
    )


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_model_path_item(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, str):
        path = raw.strip()
        if not path:
            return None
        return {"path": path}
    if isinstance(raw, dict):
        path = raw.get("path")
        if isinstance(path, str) and path.strip():
            return dict(raw)
    return None


def _tier_for_model_path_item(item: Dict[str, Any]) -> str:
    variant = str(item.get("variant") or "").lower()
    precision = str(item.get("precision") or "").lower()
    model_type = str(item.get("type") or "").lower()
    path = str(item.get("path") or "").lower()
    joined = " ".join([variant, precision, model_type, path])

    if "fp8" in joined:
        return "fp8"
    if "q8" in joined:
        return "q8"
    if "q6" in joined or "q5" in joined:
        return "q6"
    if "q4" in joined or "q3" in joined or "q2" in joined:
        return "q4"

    if (
        variant in {"default", "full"}
        or "bf16" in joined
        or "fp16" in joined
        or "float16" in joined
        or "float32" in joined
        or "fp32" in joined
    ):
        return "full"

    return "other"


def _estimated_ram_gb(file_size_gb: Optional[float], tier: str) -> Optional[float]:
    if file_size_gb is None:
        return None
    factors = {
        "full": 2.8,
        "fp8": 1.9,
        "q8": 1.8,
        "q6": 1.45,
        "q4": 1.2,
        "other": 2.0,
    }
    return max(0.0, file_size_gb * factors.get(tier, 2.0))


def _estimated_vram_gb(file_size_gb: Optional[float], tier: str) -> Optional[float]:
    if file_size_gb is None:
        return None
    factors = {
        "full": 1.35,
        "fp8": 0.95,
        "q8": 0.95,
        "q6": 0.8,
        "q4": 0.65,
        "other": 1.0,
    }
    overhead = {
        "full": 1.0,
        "fp8": 0.5,
        "q8": 0.5,
        "q6": 0.35,
        "q4": 0.25,
        "other": 0.4,
    }
    return max(0.0, file_size_gb * factors.get(tier, 1.0) + overhead.get(tier, 0.4))


def _is_flux2_dev_text_encoder(
    component_type: Optional[str],
    manifest_metadata: Optional[Dict[str, Any]],
) -> bool:
    if str(component_type or "").strip().lower() != "text_encoder":
        return False

    md = manifest_metadata or {}
    model = str(md.get("model") or "").strip().lower()
    manifest_id = str(md.get("id") or "").strip().lower()
    manifest_name = str(md.get("name") or "").strip().lower()
    joined = " ".join([model, manifest_id, manifest_name]).replace("_", "-")

    if "flux2-dev" in joined:
        return True
    if "flux dev 2" in joined:
        return True
    if model == "flux2" and "dev" in joined:
        return True
    return False


def _tier_order(
    profile: ModelDownloadProfile,
    component_type: Optional[str],
    manifest_metadata: Optional[Dict[str, Any]],
    hardware: HardwareMemoryProfile,
) -> list[str]:
    if profile == "maximum_performance":
        return ["full", "fp8", "q8", "q6", "q4", "other"]

    # Auto profile defaults.
    if _is_flux2_dev_text_encoder(component_type, manifest_metadata):
        if not hardware.has_gpu and not hardware.unified_memory:
            return ["q6", "q4", "q8", "fp8", "full", "other"]
        return ["q6", "fp8", "q8", "q4", "full", "other"]

    if not hardware.has_gpu and not hardware.unified_memory:
        # CPU-only fallback: still prefer q8 by default, then downshift when needed.
        return ["q8", "q6", "q4", "fp8", "full", "other"]

    return ["fp8", "q8", "q6", "q4", "full", "other"]


def _fits_hardware(
    item: Dict[str, Any],
    tier: str,
    hardware: HardwareMemoryProfile,
) -> bool:
    if hardware.total_ram_gb <= 0:
        return True

    requirements = (
        item.get("resource_requirements")
        if isinstance(item.get("resource_requirements"), dict)
        else {}
    )
    req_vram_gb = _as_float(requirements.get("recommended_vram_gb"))
    if req_vram_gb is None:
        req_vram_gb = _as_float(requirements.get("min_vram_gb"))

    file_size_bytes = _as_float(item.get("file_size"))
    file_size_gb = (
        (file_size_bytes / float(1024**3))
        if isinstance(file_size_bytes, float) and file_size_bytes > 0
        else None
    )

    req_ram_gb = _estimated_ram_gb(file_size_gb, tier)
    if req_ram_gb is not None and req_ram_gb > (hardware.total_ram_gb * 0.82):
        return False

    if hardware.unified_memory:
        unified_need = max(
            req_vram_gb or 0.0,
            req_ram_gb or 0.0,
            _estimated_vram_gb(file_size_gb, tier) or 0.0,
        )
        if unified_need > 0 and unified_need > (hardware.total_ram_gb * 0.78):
            return False
        return True

    if hardware.max_vram_gb > 0:
        effective_req_vram = req_vram_gb
        if effective_req_vram is None:
            effective_req_vram = _estimated_vram_gb(file_size_gb, tier)
        if effective_req_vram is not None and effective_req_vram > (
            hardware.max_vram_gb * 0.92
        ):
            return False

    return True


def _match_selected_item(
    candidates: list[Dict[str, Any]],
    selected_model_spec: Any,
) -> Optional[Dict[str, Any]]:
    if selected_model_spec is None:
        return None

    desired_path: Optional[str] = None
    desired_variant: Optional[str] = None

    if isinstance(selected_model_spec, str):
        desired_path = selected_model_spec
    elif isinstance(selected_model_spec, dict):
        p = selected_model_spec.get("path")
        v = selected_model_spec.get("variant")
        if isinstance(p, str) and p.strip():
            desired_path = p
        if isinstance(v, str) and v.strip():
            desired_variant = v

    if desired_path:
        for item in candidates:
            if str(item.get("path")) == desired_path:
                return item

    if desired_variant:
        for item in candidates:
            if str(item.get("variant") or "") == desired_variant:
                return item

    return None


def select_model_path_item(
    model_paths: Iterable[Any],
    *,
    selected_model_spec: Any = None,
    component_type: Optional[str] = None,
    manifest_metadata: Optional[Dict[str, Any]] = None,
    model_download_profile: Optional[str] = None,
    hardware_profile: Optional[HardwareMemoryProfile] = None,
) -> Optional[Dict[str, Any]]:
    candidates = []
    for raw in model_paths:
        normalized = _normalize_model_path_item(raw)
        if normalized is not None:
            candidates.append(normalized)
    if not candidates:
        return None

    selected = _match_selected_item(candidates, selected_model_spec)
    if selected is not None:
        return selected

    auto_candidates = [c for c in candidates if not bool(c.get("custom"))]
    if not auto_candidates:
        auto_candidates = list(candidates)

    profile = normalize_model_download_profile(
        model_download_profile or get_effective_model_download_profile(),
    )
    hardware = hardware_profile or detect_hardware_memory_profile()
    order = _tier_order(profile, component_type, manifest_metadata, hardware)
    order_map = {tier: idx for idx, tier in enumerate(order)}

    scored: list[Dict[str, Any]] = []
    for item in auto_candidates:
        tier = _tier_for_model_path_item(item)
        file_size = _as_float(item.get("file_size"))
        file_size_gb = (
            (float(file_size) / float(1024**3))
            if isinstance(file_size, float) and file_size > 0
            else None
        )
        scored.append(
            {
                "item": item,
                "tier": tier,
                "fits": _fits_hardware(item, tier, hardware),
                "order": order_map.get(tier, len(order_map) + 1),
                "file_size_gb": file_size_gb,
                "is_default": str(item.get("variant") or "").strip().lower()
                == "default",
            }
        )

    def _sort_key(entry: Dict[str, Any]):
        size = entry["file_size_gb"]
        if profile == "maximum_performance":
            # Prefer known larger artifacts; unknown sizes should not outrank known ones.
            size_key = -float(size) if size is not None else float("inf")
            default_bias = 0 if entry["is_default"] else 1
            return (entry["order"], default_bias, size_key)
        size_key = float(size) if size is not None else float("inf")
        return (entry["order"], size_key, 0 if entry["is_default"] else 1)

    if profile == "maximum_performance":
        scored.sort(key=_sort_key)
        return scored[0]["item"] if scored else None

    fitting = [s for s in scored if s["fits"]]
    if fitting:
        fitting.sort(key=_sort_key)
        return fitting[0]["item"]

    # If nothing fits estimates, choose a conservative fallback for auto.
    if profile == "auto":
        fallback_order = {"q4": 0, "q6": 1, "q8": 2, "fp8": 3, "full": 4, "other": 5}
        scored.sort(
            key=lambda s: (
                fallback_order.get(s["tier"], 99),
                float(s["file_size_gb"]) if s["file_size_gb"] is not None else float("inf"),
            ),
        )
    else:
        scored.sort(key=_sort_key)

    return scored[0]["item"] if scored else None
