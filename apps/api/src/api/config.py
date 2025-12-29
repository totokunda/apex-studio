import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import socket
import torch
from src.utils.defaults import (
    set_torch_device,
    get_torch_device,
    HOME_DIR,
    CONFIG_STORE_PATH,
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
)

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

            if hasattr(hf_auth, "get_token") and hasattr(hf_auth.get_token, "cache_clear"):
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


    

def _update_persisted_config(**updates: str) -> None:
    """
    Persist config-related values (paths, hf_token, etc.) so they survive backend restarts.
    """
    try:
        data = {}
        if CONFIG_STORE_PATH.exists():
            try:
                with CONFIG_STORE_PATH.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        data = existing
            except Exception:
                data = {}

        for key, value in updates.items():
            if value is not None:
                data[key] = value

        CONFIG_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_STORE_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        # Silently ignore persistence errors; API behavior should not depend on disk writes.
        # For debugging, you may want to log this error.
        print(f"Warning: failed to persist config settings: {e}")


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
    return UseFastDownloadResponse(enabled=_get_env_bool("APEX_USE_FAST_DOWNLOAD", default=True))

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