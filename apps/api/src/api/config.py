import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
import socket

from src.utils.defaults import (
    set_torch_device,
    get_torch_device,
    HOME_DIR,
    set_cache_path,
    get_cache_path as get_cache_path_default,
    set_components_path,
    get_components_path as get_components_path_default,
    get_config_path,
    get_lora_path,
    get_preprocessor_path,
    set_preprocessor_path,
    get_postprocessor_path,
    set_postprocessor_path,
    get_config_store_path,
)
from src.utils.config_store import config_store_lock, read_json_dict, write_json_dict_atomic

router = APIRouter(prefix="/config", tags=["config"])


class HomeDirectoryRequest(BaseModel):
    home_dir: str


class HomeDirectoryResponse(BaseModel):
    home_dir: str


class TorchDeviceRequest(BaseModel):
    device: str


class TorchDeviceResponse(BaseModel):
    device: str


class CachePathRequest(BaseModel):
    cache_path: str


class CachePathResponse(BaseModel):
    cache_path: str


class ComponentsPathRequest(BaseModel):
    components_path: str


class ComponentsPathResponse(BaseModel):
    components_path: str


class ConfigPathRequest(BaseModel):
    config_path: str


class ConfigPathResponse(BaseModel):
    config_path: str


class LoraPathRequest(BaseModel):
    lora_path: str


class LoraPathResponse(BaseModel):
    lora_path: str


class PreprocessorPathRequest(BaseModel):
    preprocessor_path: str


class PreprocessorPathResponse(BaseModel):
    preprocessor_path: str


class PostprocessorPathRequest(BaseModel):
    postprocessor_path: str


class PostprocessorPathResponse(BaseModel):
    postprocessor_path: str


class PathSizesResponse(BaseModel):
    cache_path_bytes: Optional[int] = None
    components_path_bytes: Optional[int] = None
    config_path_bytes: Optional[int] = None
    lora_path_bytes: Optional[int] = None
    preprocessor_path_bytes: Optional[int] = None
    postprocessor_path_bytes: Optional[int] = None


def _compute_path_size_bytes(path_value: str) -> Optional[int]:
    """
    Best-effort recursive size calculation for a directory path on the backend machine.
    Returns None if the path cannot be accessed.
    """
    try:
        p = Path(str(path_value)).expanduser().resolve()
        if not p.exists():
            return 0
        if p.is_file():
            return int(p.stat().st_size)
        total = 0
        # Do not follow symlinks to avoid surprises/loops.
        for root, _dirs, files in os.walk(str(p), followlinks=False):
            for f in files:
                fp = Path(root) / f
                try:
                    total += int(fp.stat().st_size)
                except Exception:
                    # Skip unreadable files
                    continue
        return int(total)
    except Exception:
        return None


class HuggingFaceTokenRequest(BaseModel):
    token: str


class HuggingFaceTokenResponse(BaseModel):
    is_set: bool
    masked_token: Optional[str] = None


class CivitaiApiKeyRequest(BaseModel):
    token: str


class CivitaiApiKeyResponse(BaseModel):
    is_set: bool
    masked_token: Optional[str] = None


class MaskModelRequest(BaseModel):
    mask_model: str


class MaskModelResponse(BaseModel):
    mask_model: str


class RenderStepEnabledRequest(BaseModel):
    enabled: bool


class RenderStepEnabledResponse(BaseModel):
    enabled: bool


class MemorySettingsRequest(BaseModel):
    APEX_LOAD_MODEL_VRAM_MULT: Optional[float] = None
    APEX_LOAD_MODEL_VRAM_EXTRA_BYTES: Optional[int] = None
    APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES: Optional[int] = None
    APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION: Optional[float] = None
    APEX_WEIGHT_TARGET_FREE_RAM_FRACTION: Optional[float] = None


class MemorySettingsResponse(BaseModel):
    settings: Dict[str, Any]


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() == "true"


def _refresh_hf_token_runtime(token: Optional[str]) -> None:
    """
    Ensure Hugging Face token updates take effect immediately in a long-running process.

    Why: Some code paths/libraries read tokens from token files (HfFolder) or cache get_token().
    We update env, clear any caches we can find, and (best-effort) sync the token file too.
    """
    try:
        # Optional dependency; only present if huggingface_hub is installed.
        import huggingface_hub  # noqa: F401

        # Clear any lru_cache on get_token (varies by huggingface_hub version).
        try:
            from huggingface_hub import get_token as hf_get_token

            if hasattr(hf_get_token, "cache_clear"):
                hf_get_token.cache_clear()
        except Exception:
            pass

        try:
            from huggingface_hub.utils import _auth as hf_auth

            if hasattr(hf_auth, "get_token") and hasattr(
                hf_auth.get_token, "cache_clear"
            ):
                hf_auth.get_token.cache_clear()
        except Exception:
            pass

        # Keep legacy HfFolder readers in sync (best-effort; may write to HF_HOME).
        try:
            from huggingface_hub import HfFolder

            if token is None:
                if hasattr(HfFolder, "delete_token"):
                    HfFolder.delete_token()
            else:
                if hasattr(HfFolder, "save_token"):
                    HfFolder.save_token(token)
        except Exception:
            pass
    except Exception:
        # If huggingface_hub isn't installed (or any unexpected error), do nothing.
        pass


@router.get("/home-dir", response_model=HomeDirectoryResponse)
def get_home_directory():
    """Get the current apex home directory"""
    return HomeDirectoryResponse(home_dir=str(HOME_DIR))


@router.post("/home-dir", response_model=HomeDirectoryResponse)
def set_home_directory(request: HomeDirectoryRequest):
    """Set the apex home directory. Requires restart to take full effect."""
    try:
        home_path = Path(request.home_dir).expanduser().resolve()
        if not home_path.exists():
            home_path.mkdir(parents=True, exist_ok=True)

        os.environ["APEX_HOME_DIR"] = str(home_path)
        return HomeDirectoryResponse(home_dir=str(home_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set home directory: {str(e)}"
        )


@router.get("/torch-device", response_model=TorchDeviceResponse)
def get_device():
    """Get the current torch device"""
    device = get_torch_device()
    return TorchDeviceResponse(device=str(device))


@router.post("/torch-device", response_model=TorchDeviceResponse)
def set_device(request: TorchDeviceRequest):
    """Set the torch device (cpu, cuda, mps, cuda:0, etc.)"""
    import torch
    try:
        valid_devices = ["cpu", "cuda", "mps"]
        device_str = request.device.lower()

        if device_str.startswith("cuda:"):
            device_index = int(device_str.split(":")[1])
            if not torch.cuda.is_available():
                raise HTTPException(status_code=400, detail="CUDA is not available")
            if device_index >= torch.cuda.device_count():
                raise HTTPException(
                    status_code=400, detail=f"CUDA device {device_index} not found"
                )
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise HTTPException(status_code=400, detail="CUDA is not available")
        elif device_str == "mps":
            if not torch.backends.mps.is_available():
                raise HTTPException(status_code=400, detail="MPS is not available")
        elif device_str != "cpu":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid device: {device_str}. Must be one of {valid_devices} or cuda:N",
            )

        set_torch_device(device_str)
        return TorchDeviceResponse(device=device_str)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set device: {str(e)}")


@router.get("/cache-path", response_model=CachePathResponse)
def get_cache_path():
    """Get the current cache path for media-related cache items"""
    return CachePathResponse(cache_path=str(get_cache_path_default()))


@router.post("/cache-path", response_model=CachePathResponse)
def set_cache_path(request: CachePathRequest):
    """Set the cache path for media-related cache items"""
    try:
        cache_path = Path(request.cache_path).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        set_cache_path(str(cache_path))
        _update_persisted_config(cache_path=str(cache_path))
        return CachePathResponse(cache_path=str(cache_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set cache path: {str(e)}"
        )


@router.get("/components-path", response_model=ComponentsPathResponse)
def get_components_path():
    """Get the current components path"""
    return ComponentsPathResponse(components_path=str(get_components_path_default()))


@router.post("/components-path", response_model=ComponentsPathResponse)
def set_components_path(request: ComponentsPathRequest):
    """Set the components path"""
    try:
        components_path = Path(request.components_path).expanduser().resolve()
        components_path.mkdir(parents=True, exist_ok=True)
        set_components_path(str(components_path))
        _update_persisted_config(components_path=str(components_path))
        return ComponentsPathResponse(components_path=str(components_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set components path: {str(e)}"
        )


@router.get("/config-path", response_model=ConfigPathResponse)
def get_config_path_api():
    """Get the current config path"""
    return ConfigPathResponse(config_path=str(get_config_path()))


@router.post("/config-path", response_model=ConfigPathResponse)
def set_config_path_api(request: ConfigPathRequest):
    """Set the config path"""
    try:
        config_path = Path(request.config_path).expanduser().resolve()
        config_path.mkdir(parents=True, exist_ok=True)
        os.environ["APEX_CONFIG_SAVE_PATH"] = str(config_path)
        _update_persisted_config(config_path=str(config_path))
        return ConfigPathResponse(config_path=str(config_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set config path: {str(e)}"
        )


@router.get("/lora-path", response_model=LoraPathResponse)
def get_lora_path_api():
    """Get the current LoRA path"""
    return LoraPathResponse(lora_path=str(get_lora_path()))


@router.post("/lora-path", response_model=LoraPathResponse)
def set_lora_path_api(request: LoraPathRequest):
    """Set the LoRA path"""
    try:
        lora_path = Path(request.lora_path).expanduser().resolve()
        lora_path.mkdir(parents=True, exist_ok=True)
        os.environ["APEX_LORA_SAVE_PATH"] = str(lora_path)
        _update_persisted_config(lora_path=str(lora_path))
        return LoraPathResponse(lora_path=str(lora_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set LoRA path: {str(e)}"
        )


@router.get("/preprocessor-path", response_model=PreprocessorPathResponse)
def get_preprocessor_path_api():
    """Get the current preprocessor path"""
    return PreprocessorPathResponse(preprocessor_path=str(get_preprocessor_path()))


@router.post("/preprocessor-path", response_model=PreprocessorPathResponse)
def set_preprocessor_path_api(request: PreprocessorPathRequest):
    """Set the preprocessor path"""
    try:
        pre_path = Path(request.preprocessor_path).expanduser().resolve()
        pre_path.mkdir(parents=True, exist_ok=True)
        set_preprocessor_path(str(pre_path))
        _update_persisted_config(preprocessor_path=str(pre_path))
        return PreprocessorPathResponse(preprocessor_path=str(pre_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set preprocessor path: {str(e)}"
        )


@router.get("/postprocessor-path", response_model=PostprocessorPathResponse)
def get_postprocessor_path_api():
    """Get the current postprocessor path"""
    return PostprocessorPathResponse(postprocessor_path=str(get_postprocessor_path()))


@router.post("/postprocessor-path", response_model=PostprocessorPathResponse)
def set_postprocessor_path_api(request: PostprocessorPathRequest):
    """Set the postprocessor path"""
    try:
        post_path = Path(request.postprocessor_path).expanduser().resolve()
        post_path.mkdir(parents=True, exist_ok=True)
        set_postprocessor_path(str(post_path))
        _update_persisted_config(postprocessor_path=str(post_path))
        return PostprocessorPathResponse(postprocessor_path=str(post_path))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set postprocessor path: {str(e)}"
        )


@router.get("/path-sizes", response_model=PathSizesResponse)
def get_path_sizes_api():
    """
    Get the sizes (bytes) for the configured save-path folders on the backend machine.
    """
    return PathSizesResponse(
        cache_path_bytes=_compute_path_size_bytes(str(get_cache_path_default())),
        components_path_bytes=_compute_path_size_bytes(
            str(get_components_path_default())
        ),
        config_path_bytes=_compute_path_size_bytes(str(get_config_path())),
        lora_path_bytes=_compute_path_size_bytes(str(get_lora_path())),
        preprocessor_path_bytes=_compute_path_size_bytes(str(get_preprocessor_path())),
        postprocessor_path_bytes=_compute_path_size_bytes(str(get_postprocessor_path())),
    )


@router.get("/enable-image-render-steps", response_model=RenderStepEnabledResponse)
def get_enable_image_render_steps():
    """Get whether per-step image rendering is enabled during inference."""
    return RenderStepEnabledResponse(
        enabled=_get_env_bool("ENABLE_IMAGE_RENDER_STEP", default=True)
    )


@router.post("/enable-image-render-steps", response_model=RenderStepEnabledResponse)
def set_enable_image_render_steps(request: RenderStepEnabledRequest):
    """Enable/disable per-step image rendering during inference."""
    try:
        value = "true" if bool(request.enabled) else "false"
        os.environ["ENABLE_IMAGE_RENDER_STEP"] = value
        _update_persisted_config(ENABLE_IMAGE_RENDER_STEP=value)
        return RenderStepEnabledResponse(enabled=request.enabled)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set ENABLE_IMAGE_RENDER_STEP: {str(e)}",
        )


@router.get("/enable-video-render-steps", response_model=RenderStepEnabledResponse)
def get_enable_video_render_steps():
    """Get whether per-step video rendering is enabled during inference."""
    return RenderStepEnabledResponse(
        enabled=_get_env_bool("ENABLE_VIDEO_RENDER_STEP", default=True)
    )


@router.post("/enable-video-render-steps", response_model=RenderStepEnabledResponse)
def set_enable_video_render_steps(request: RenderStepEnabledRequest):
    """Enable/disable per-step video rendering during inference."""
    try:
        value = "true" if bool(request.enabled) else "false"
        os.environ["ENABLE_VIDEO_RENDER_STEP"] = value
        _update_persisted_config(ENABLE_VIDEO_RENDER_STEP=value)
        return RenderStepEnabledResponse(enabled=request.enabled)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set ENABLE_VIDEO_RENDER_STEP: {str(e)}",
        )


@router.get("/hf-token", response_model=HuggingFaceTokenResponse)
def get_huggingface_token():
    """Check if HUGGING_FACE_HUB_TOKEN is set; returns masked token if available"""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        masked = (token[:4] + "..." + token[-4:]) if len(token) > 8 else "***"
        return HuggingFaceTokenResponse(is_set=True, masked_token=masked)
    return HuggingFaceTokenResponse(is_set=False, masked_token=None)


@router.post("/hf-token", response_model=HuggingFaceTokenResponse)
def set_huggingface_token(request: HuggingFaceTokenRequest):
    """Set HUGGING_FACE_HUB_TOKEN for the running process"""
    try:
        token = (request.token or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="Token cannot be empty")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        _refresh_hf_token_runtime(token)
        _update_persisted_config(hf_token=token)
        masked = (token[:4] + "..." + token[-4:]) if len(token) > 8 else "***"
        return HuggingFaceTokenResponse(is_set=True, masked_token=masked)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set HUGGING_FACE_HUB_TOKEN: {str(e)}"
        )


@router.get("/civitai-api-key", response_model=CivitaiApiKeyResponse)
def get_civitai_api_key():
    """Check if CIVITAI_API_KEY is set; returns masked key if available"""
    token = os.environ.get("CIVITAI_API_KEY")
    if token:
        masked = (token[:4] + "..." + token[-4:]) if len(token) > 8 else "***"
        return CivitaiApiKeyResponse(is_set=True, masked_token=masked)
    return CivitaiApiKeyResponse(is_set=False, masked_token=None)


@router.post("/civitai-api-key", response_model=CivitaiApiKeyResponse)
def set_civitai_api_key(request: CivitaiApiKeyRequest):
    """Set CIVITAI_API_KEY for the running process"""
    try:
        token = (request.token or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="Token cannot be empty")
        os.environ["CIVITAI_API_KEY"] = token
        _update_persisted_config(civitai_api_key=token)
        masked = (token[:4] + "..." + token[-4:]) if len(token) > 8 else "***"
        return CivitaiApiKeyResponse(is_set=True, masked_token=masked)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set CIVITAI_API_KEY: {str(e)}"
        )


@router.get("/mask-model", response_model=MaskModelResponse)
def get_mask_model():
    """Get the current mask model"""
    return MaskModelResponse(mask_model=os.environ.get("MASK_MODEL", "sam2_base_plus"))


@router.post("/mask-model", response_model=MaskModelResponse)
def set_mask_model(request: MaskModelRequest):
    """Set the mask model"""
    try:
        mask_model = request.mask_model.lower()
        if mask_model not in [
            "sam2_tiny",
            "sam2_small",
            "sam2_base_plus",
            "sam2_large",
        ]:
            raise HTTPException(
                status_code=400,
                detail="Invalid mask model. Must be one of: sam2_tiny, sam2_small, sam2_base_plus, sam2_large",
            )
        os.environ["MASK_MODEL"] = mask_model
        _update_persisted_config(mask_model=request.mask_model)
        return MaskModelResponse(mask_model=request.mask_model)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to set mask model: {str(e)}"
        )


def _update_persisted_config(**updates: Any) -> None:
    """
    Persist config-related values (paths, hf_token, etc.) so they survive backend restarts.
    """
    config_store_path = Path(get_config_store_path())
    try:
        with config_store_lock(config_store_path):
            data = read_json_dict(config_store_path)
            for key, value in updates.items():
                if value is not None:
                    data[key] = value
            write_json_dict_atomic(config_store_path, data, indent=2)
    except Exception as e:
        # Silently ignore persistence errors; API behavior should not depend on disk writes.
        # For debugging, you may want to log this error.
        print(f"Warning: failed to persist config settings: {e}")


class AutoUpdateConfigResponse(BaseModel):
    enabled: bool
    interval_hours: float
    repo_owner: str
    repo_name: str
    include_prerelease: bool
    status: dict[str, Any] = {}


class AutoUpdateConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    interval_hours: Optional[float] = None
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None
    include_prerelease: Optional[bool] = None


def _read_persisted_config_raw() -> dict:
    try:
        p = Path(get_config_store_path())
        with config_store_lock(p):
            return read_json_dict(p)
    except Exception:
        pass
    return {}


@router.get("/auto-update", response_model=AutoUpdateConfigResponse)
def get_auto_update_config_api():
    """
    Get automatic API code-update configuration.
    Defaults: enabled=true, interval_hours=4.
    """
    persisted = _read_persisted_config_raw()
    enabled = bool(persisted.get("auto_update_enabled", True))
    interval_raw = persisted.get("auto_update_interval_hours", 4)
    try:
        interval_hours = float(interval_raw)
    except Exception:
        interval_hours = 4.0

    repo_owner = str(
        persisted.get("auto_update_repo_owner")
        or os.environ.get("APEX_UPDATE_REPO_OWNER")
        or "totokunda"
    ).strip()
    repo_name = str(
        persisted.get("auto_update_repo_name")
        or os.environ.get("APEX_UPDATE_REPO_NAME")
        or "apex-studio"
    ).strip()
    include_prerelease = bool(
        persisted.get(
            "auto_update_include_prerelease",
            os.environ.get("APEX_UPDATE_INCLUDE_PRERELEASE", "").strip().lower()
            in {"1", "true", "yes"},
        )
    )

    # Include last-known status fields for UI/debugging.
    status_keys = [
        "auto_update_last_checked_at",
        "auto_update_current_version",
        "auto_update_current_gpu_support",
        "auto_update_available_version",
        "auto_update_available_asset",
        "auto_update_available_tag",
        "auto_update_download_url",
        "auto_update_last_apply_started_at",
        "auto_update_last_apply_finished_at",
        "auto_update_last_apply_status",
        "auto_update_last_apply_asset",
        "auto_update_last_apply_version",
        "auto_update_last_apply_output_tail",
        "auto_update_last_error",
        "auto_update_last_error_at",
    ]
    status: dict[str, Any] = {}
    for k in status_keys:
        if k in persisted:
            status[k] = persisted.get(k)

    return AutoUpdateConfigResponse(
        enabled=enabled,
        interval_hours=interval_hours,
        repo_owner=repo_owner,
        repo_name=repo_name,
        include_prerelease=include_prerelease,
            status=status,
        )


def _memory_env_defaults() -> dict:
    persisted = _read_persisted_config_raw()
    def _float(name: str, default: float) -> float:
        try:
            raw = persisted.get(name, None)
            if raw is None:
                raw = os.environ.get(name, None)
            if raw is None or str(raw).strip() == "":
                return default
            return float(raw)
        except Exception:
            return default

    def _int(name: str, default: int) -> int:
        try:
            raw = persisted.get(name, None)
            if raw is None:
                raw = os.environ.get(name, None)
            if raw is None or str(raw).strip() == "":
                return default
            return int(float(raw))
        except Exception:
            return default

    def _str(name: str, default: str) -> str:
        val = persisted.get(name)
        if val is None:
            val = os.environ.get(name)
        return default if val is None or val == "" else str(val)

    return {
        "APEX_LOAD_MODEL_VRAM_MULT": _float("APEX_LOAD_MODEL_VRAM_MULT", 1.20),
        "APEX_LOAD_MODEL_VRAM_EXTRA_BYTES": _int("APEX_LOAD_MODEL_VRAM_EXTRA_BYTES", 512 * 1024**2),
        "APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES": _int("APEX_VRAM_PRESSURE_CPU_SAFETY_BYTES", 2 * 1024**3),
        "APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION": _float("APEX_WEIGHT_TARGET_FREE_VRAM_FRACTION", 0.12),
        "APEX_WEIGHT_TARGET_FREE_RAM_FRACTION": _float("APEX_WEIGHT_TARGET_FREE_RAM_FRACTION", 0.10),
    }


@router.get("/memory", response_model=MemorySettingsResponse)
def get_memory_settings():
    """Get effective memory management settings (env + defaults)."""
    return MemorySettingsResponse(settings=_memory_env_defaults())


@router.post("/memory", response_model=MemorySettingsResponse)
def set_memory_settings(request: MemorySettingsRequest):
    """Update memory management settings via environment variables."""
    # Only apply keys explicitly sent by the client.
    # This preserves other persisted memory overrides while still allowing deletion
    # via explicit `null` or empty string.
    updates = request.dict(exclude_unset=True)
    try:
        # Load existing persisted config so we can add/remove keys.
        store_path = Path(get_config_store_path())
        persisted = _read_persisted_config_raw()

        for k, v in updates.items():
            # Treat None/empty string as a request to remove the override.
            if v is None or (isinstance(v, str) and v.strip() == ""):
                os.environ.pop(k, None)
                if k in persisted:
                    persisted.pop(k, None)
                continue
            try:
                os.environ[k] = str(v)
                persisted[k] = v
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to set {k}: {e}")

        # Persist the merged dict.
        try:
            with config_store_lock(store_path):
                write_json_dict_atomic(store_path, persisted, indent=2)
        except Exception as e:
            # Do not fail the API if persistence fails; log/return best-effort.
            print(f"Warning: failed to persist memory settings: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MemorySettingsResponse(settings=_memory_env_defaults())


@router.post("/auto-update", response_model=AutoUpdateConfigResponse)
def set_auto_update_config_api(request: AutoUpdateConfigRequest):
    """
    Update automatic code-update configuration and persist it.
    """
    try:
        updates: dict[str, Any] = {}
        if request.enabled is not None:
            updates["auto_update_enabled"] = bool(request.enabled)
        if request.interval_hours is not None:
            interval = float(request.interval_hours)
            if interval <= 0:
                raise HTTPException(status_code=400, detail="interval_hours must be > 0")
            updates["auto_update_interval_hours"] = interval
        if request.repo_owner is not None:
            updates["auto_update_repo_owner"] = str(request.repo_owner).strip()
        if request.repo_name is not None:
            updates["auto_update_repo_name"] = str(request.repo_name).strip()
        if request.include_prerelease is not None:
            updates["auto_update_include_prerelease"] = bool(request.include_prerelease)

        if updates:
            _update_persisted_config(**updates)
        return get_auto_update_config_api()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set auto-update config: {e}")


class HostnameResponse(BaseModel):
    hostname: str


@router.get("/hostname", response_model=HostnameResponse)
def get_hostname():
    """Get the hostname of the current server machine"""
    return HostnameResponse(hostname=socket.gethostname())


class UseFastDownloadRequest(BaseModel):
    enabled: bool


class UseFastDownloadResponse(BaseModel):
    enabled: bool


@router.get("/enable-fast-download", response_model=UseFastDownloadResponse)
def get_use_fast_download():
    """Get whether fast download is enabled"""
    return UseFastDownloadResponse(
        enabled=_get_env_bool("APEX_USE_FAST_DOWNLOAD", default=True)
    )


@router.post("/enable-fast-download", response_model=UseFastDownloadResponse)
def set_use_fast_download(request: UseFastDownloadRequest):
    """Enable/disable fast download"""
    try:
        value = "true" if bool(request.enabled) else "false"
        os.environ["APEX_USE_FAST_DOWNLOAD"] = value
        _update_persisted_config(APEX_USE_FAST_DOWNLOAD=value)
        return UseFastDownloadResponse(enabled=request.enabled)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set APEX_USE_FAST_DOWNLOAD: {str(e)}",
        )
