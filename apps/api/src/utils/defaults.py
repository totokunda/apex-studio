import torch
import os
import json
from pathlib import Path

HOME_DIR = Path(os.getenv("APEX_HOME_DIR", Path.home()))

CONFIG_STORE_PATH = HOME_DIR / "apex-diffusion" / "apex-config.json"

DEFAULT_CONFIG_SAVE_PATH = os.getenv(
    "APEX_CONFIG_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "configs")
)
DEFAULT_SAVE_PATH = os.getenv("APEX_SAVE_PATH", str(HOME_DIR / "apex-diffusion"))

DEFAULT_COMPONENTS_PATH = os.getenv(
    "APEX_COMPONENTS_PATH", str(HOME_DIR / "apex-diffusion" / "components")
)

DEFAULT_PREPROCESSOR_SAVE_PATH = os.getenv(
    "APEX_PREPROCESSOR_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "preprocessors")
)

DEFAULT_POSTPROCESSOR_SAVE_PATH = os.getenv(
    "APEX_POSTPROCESSOR_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "postprocessors")
)

DEFAULT_CACHE_PATH = os.getenv(
    "APEX_CACHE_PATH", str(HOME_DIR / "apex-diffusion" / "cache")
)

# New default path to store LoRA adapters and related artifacts
DEFAULT_LORA_SAVE_PATH = os.getenv(
    "APEX_LORA_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "loras")
)

# Default path used for offloading (e.g. to disk)
DEFAULT_OFFLOAD_PATH = os.getenv(
    "APEX_OFFLOAD_PATH", str(HOME_DIR / "apex-diffusion" / "offload")
)


def _load_persisted_config() -> dict:
    try:
        if CONFIG_STORE_PATH.exists():
            with CONFIG_STORE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        # Swallow errors; fall back to env/defaults
        pass
    return {}


_persisted = _load_persisted_config()

if isinstance(_persisted, dict):
    DEFAULT_CACHE_PATH = _persisted.get("cache_path", DEFAULT_CACHE_PATH)
    DEFAULT_COMPONENTS_PATH = _persisted.get("components_path", DEFAULT_COMPONENTS_PATH)
    DEFAULT_CONFIG_SAVE_PATH = _persisted.get("config_path", DEFAULT_CONFIG_SAVE_PATH)
    DEFAULT_LORA_SAVE_PATH = _persisted.get("lora_path", DEFAULT_LORA_SAVE_PATH)
    DEFAULT_PREPROCESSOR_SAVE_PATH = _persisted.get(
        "preprocessor_path", DEFAULT_PREPROCESSOR_SAVE_PATH
    )
    DEFAULT_POSTPROCESSOR_SAVE_PATH = _persisted.get(
        "postprocessor_path", DEFAULT_POSTPROCESSOR_SAVE_PATH
    )
    # Persisted render-step toggles for inference progress callbacks
    # (stored as env-var-compatible "true"/"false" strings).
    _enable_image_render_step = _persisted.get("ENABLE_IMAGE_RENDER_STEP")
    if isinstance(_enable_image_render_step, str) and _enable_image_render_step.strip():
        os.environ["ENABLE_IMAGE_RENDER_STEP"] = _enable_image_render_step.strip().lower()
    _enable_video_render_step = _persisted.get("ENABLE_VIDEO_RENDER_STEP")
    if isinstance(_enable_video_render_step, str) and _enable_video_render_step.strip():
        os.environ["ENABLE_VIDEO_RENDER_STEP"] = _enable_video_render_step.strip().lower()
    # HF token persistence for backend process
    _hf_token = _persisted.get("hf_token")
    if isinstance(_hf_token, str) and _hf_token.strip():
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token.strip()
    # CivitAI API key persistence for backend process
    _civitai_key = _persisted.get("civitai_api_key")
    if isinstance(_civitai_key, str) and _civitai_key.strip():
        os.environ["CIVITAI_API_KEY"] = _civitai_key.strip()

os.makedirs(DEFAULT_CONFIG_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_COMPONENTS_PATH, exist_ok=True)
os.makedirs(DEFAULT_PREPROCESSOR_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_POSTPROCESSOR_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)
os.makedirs(DEFAULT_LORA_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_OFFLOAD_PATH, exist_ok=True)

os.environ["HF_HOME"] = os.getenv(
    "APEX_HF_HOME", str(HOME_DIR / "apex-diffusion" / "huggingface")
)

# Check if running in Ray worker (avoid MPS in forked processes)
_IN_RAY_WORKER = os.environ.get("RAY_WORKER_NAME") or "ray::" in os.environ.get("_", "")

if _IN_RAY_WORKER or os.environ.get("FORCE_CPU", ""):
    # Force CPU in Ray workers to avoid MPS/CUDA fork issues
    DEFAULT_DEVICE = torch.device("cpu")
else:
    DEFAULT_DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}


def set_torch_device(device: torch.device | str | None = None) -> None:
    global DEFAULT_DEVICE
    if device is None:
        DEFAULT_DEVICE = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )

    else:
        DEFAULT_DEVICE = torch.device(device)
    torch.set_default_device(DEFAULT_DEVICE)


def get_torch_device() -> torch.device:
    # check if cuda or mps from default device is available otherwise return cpu 
    if DEFAULT_DEVICE.type == "cuda" and torch.cuda.is_available():
        return DEFAULT_DEVICE
    elif DEFAULT_DEVICE.type == "mps" and torch.backends.mps.is_available():
        return DEFAULT_DEVICE
    else:
        return torch.device("cpu")


def get_cache_path() -> str:
    return DEFAULT_CACHE_PATH


def set_cache_path(path: str) -> None:
    global DEFAULT_CACHE_PATH
    DEFAULT_CACHE_PATH = path
    os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)


def get_offload_path() -> str:
    return DEFAULT_OFFLOAD_PATH


def set_offload_path(path: str) -> None:
    global DEFAULT_OFFLOAD_PATH
    DEFAULT_OFFLOAD_PATH = path
    os.makedirs(DEFAULT_OFFLOAD_PATH, exist_ok=True)


def get_components_path() -> str:
    return DEFAULT_COMPONENTS_PATH


def get_config_path() -> str:
    return DEFAULT_CONFIG_SAVE_PATH


def get_lora_path() -> str:
    return DEFAULT_LORA_SAVE_PATH


def set_components_path(path: str) -> None:
    global DEFAULT_COMPONENTS_PATH
    DEFAULT_COMPONENTS_PATH = path
    os.makedirs(DEFAULT_COMPONENTS_PATH, exist_ok=True)


def get_preprocessor_path() -> str:
    return DEFAULT_PREPROCESSOR_SAVE_PATH


def set_preprocessor_path(path: str | None = None) -> None:
    global DEFAULT_PREPROCESSOR_SAVE_PATH
    DEFAULT_PREPROCESSOR_SAVE_PATH = path
    os.makedirs(DEFAULT_PREPROCESSOR_SAVE_PATH, exist_ok=True)


def get_postprocessor_path() -> str:
    return DEFAULT_POSTPROCESSOR_SAVE_PATH


def set_postprocessor_path(path: str | None = None) -> None:
    global DEFAULT_POSTPROCESSOR_SAVE_PATH
    DEFAULT_POSTPROCESSOR_SAVE_PATH = path
    os.makedirs(DEFAULT_POSTPROCESSOR_SAVE_PATH, exist_ok=True)
