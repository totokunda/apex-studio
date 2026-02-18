import os
import json
from pathlib import Path
from urllib.parse import urlparse

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

DEFAULT_TORCH_COMPILE_PATH = os.getenv(
    "APEX_TORCH_COMPILE_PATH", str(HOME_DIR / "apex-diffusion" / "torch_compile")
)

# Hugging Face cache root.
#
# Important: respect a user-provided HF_HOME. Historically we always overwrote HF_HOME which
# caused downloads to land in the wrong cache (often ~/.cache) or a different location than
# the user configured.
#
# Also prefer keeping HF artifacts under apex-diffusion/.cache to avoid mixing with general
# system caches.
DEFAULT_HF_HOME = (
    os.getenv("APEX_HF_HOME")
    or os.getenv("HF_HOME")
    or str(HOME_DIR / "apex-diffusion" / ".cache" / "huggingface")
)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = DEFAULT_TORCH_COMPILE_PATH
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = DEFAULT_HF_HOME

# Keep Hub + Transformers caches pinned under HF_HOME unless explicitly overridden.
_DEFAULT_HF_HUB_CACHE = str(Path(DEFAULT_HF_HOME) / "hub")
_DEFAULT_TRANSFORMERS_CACHE = str(Path(DEFAULT_HF_HOME) / "transformers")
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_HUB_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _DEFAULT_HF_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_TRANSFORMERS_CACHE)


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
        os.environ["ENABLE_IMAGE_RENDER_STEP"] = (
            _enable_image_render_step.strip().lower()
        )
    _enable_video_render_step = _persisted.get("ENABLE_VIDEO_RENDER_STEP")
    if isinstance(_enable_video_render_step, str) and _enable_video_render_step.strip():
        os.environ["ENABLE_VIDEO_RENDER_STEP"] = (
            _enable_video_render_step.strip().lower()
        )
    # Auto memory manager toggle (persisted as env-var-compatible "true"/"false").
    _disable_auto_mem = _persisted.get("APEX_DISABLE_AUTO_MEMORY_MANAGEMENT")
    if isinstance(_disable_auto_mem, str) and _disable_auto_mem.strip():
        os.environ["APEX_DISABLE_AUTO_MEMORY_MANAGEMENT"] = (
            _disable_auto_mem.strip().lower()
        )
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
os.makedirs(DEFAULT_TORCH_COMPILE_PATH, exist_ok=True)
os.makedirs(DEFAULT_HF_HOME, exist_ok=True)

os.environ["HF_HOME"] = DEFAULT_HF_HOME

# Check if running in Ray worker (avoid MPS in forked processes)
_IN_RAY_WORKER = os.environ.get("RAY_WORKER_NAME") or "ray::" in os.environ.get("_", "")

# NOTE: Do not import torch at module import time.
# Many lightweight flows (setup/install, packaging) only need paths/config and should not
# pay the torch import cost. Device selection is therefore lazily computed.
DEFAULT_DEVICE = (
    None  # populated on first call to get_torch_device()/set_torch_device()
)
_DEFAULT_DEVICE_COMPUTED = False
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

_RUNTIME_SECRETS_LAST_MTIME: float | None = None


def _compute_config_store_path() -> Path:
    """
    Compute the config store path based on the *current* environment.
    Important for Ray workers: the module-level HOME_DIR/CONFIG_STORE_PATH are fixed at import
    time, but APEX_HOME_DIR can be set/changed later via the config API.
    """
    home_raw = os.getenv("APEX_HOME_DIR")
    home_dir = Path(home_raw).expanduser() if home_raw else Path.home()
    return home_dir / "apex-diffusion" / "apex-config.json"


def _refresh_runtime_secrets_from_persisted(force: bool = False) -> None:
    """
    Ensure runtime env vars (e.g. tokens) are available in the current process.

    This is especially relevant in Ray tasks: the API process may update/persist secrets, but
    Ray workers are separate processes and won't see updated `os.environ` unless they reload
    from the persisted config store.
    """
    global _RUNTIME_SECRETS_LAST_MTIME

    # If both are already set, no need to hit disk.
    if os.environ.get("CIVITAI_API_KEY") and os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return

    p = _compute_config_store_path()
    if not p.exists():
        # Fallback to the import-time constant path (in case APEX_HOME_DIR is unset here but
        # was set when the module loaded).
        p = Path(CONFIG_STORE_PATH)
        if not p.exists():
            return

    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = None

    if not force and mtime is not None and _RUNTIME_SECRETS_LAST_MTIME == mtime:
        return

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return

        civitai_key = data.get("civitai_api_key")
        if (
            not os.environ.get("CIVITAI_API_KEY")
            and isinstance(civitai_key, str)
            and civitai_key.strip()
        ):
            os.environ["CIVITAI_API_KEY"] = civitai_key.strip()

        hf_token = data.get("hf_token")
        if (
            not os.environ.get("HUGGING_FACE_HUB_TOKEN")
            and isinstance(hf_token, str)
            and hf_token.strip()
        ):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()

        _RUNTIME_SECRETS_LAST_MTIME = mtime
    except Exception:
        # Swallow errors; callers will fall back to whatever is already in env.
        return


def refresh_runtime_secrets(force: bool = False) -> None:
    """
    Public helper to refresh secrets (HF + CivitAI) from the persisted config store.
    Useful for Ray tasks that need tokens before any HTTP helper functions are called.
    """
    _refresh_runtime_secrets_from_persisted(force=force)


def get_default_headers(url: str) -> dict:
    # In Ray workers, secrets may be persisted by the API process but not present in this
    # worker's environment. Refresh lazily here since this is a common call site.
    _refresh_runtime_secrets_from_persisted(force=True)
    parsed_url = urlparse(url)
    if parsed_url.netloc.endswith("civitai.com"):
        token = os.environ.get("CIVITAI_API_KEY")
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        }
    return DEFAULT_HEADERS


def set_torch_device(device) -> None:
    global DEFAULT_DEVICE
    global _DEFAULT_DEVICE_COMPUTED
    import torch

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
    _DEFAULT_DEVICE_COMPUTED = True


def get_torch_device():
    import torch

    global DEFAULT_DEVICE
    global _DEFAULT_DEVICE_COMPUTED

    if not _DEFAULT_DEVICE_COMPUTED or DEFAULT_DEVICE is None:
        # Force CPU in Ray workers to avoid MPS/CUDA fork issues
        if _IN_RAY_WORKER or os.environ.get("FORCE_CPU", ""):
            DEFAULT_DEVICE = torch.device("cpu")
        else:
            DEFAULT_DEVICE = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else (
                    torch.device("mps")
                    if getattr(torch, "backends", None) is not None
                    and torch.backends.mps.is_available()
                    else torch.device("cpu")
                )
            )
        _DEFAULT_DEVICE_COMPUTED = True

    # check if cuda or mps from default device is available otherwise return cpu
    try:
        if DEFAULT_DEVICE.type == "cuda" and torch.cuda.is_available():
            return DEFAULT_DEVICE
        if DEFAULT_DEVICE.type == "mps" and torch.backends.mps.is_available():
            return DEFAULT_DEVICE
    except Exception:
        pass
    return torch.device("cpu")


def get_cache_path() -> str:
    return DEFAULT_CACHE_PATH


def get_engine_results_path() -> str:
    os.makedirs(os.path.join(DEFAULT_CACHE_PATH, "engine_results"), exist_ok=True)
    return os.path.join(DEFAULT_CACHE_PATH, "engine_results")


def get_preprocessor_results_path() -> str:
    os.makedirs(os.path.join(DEFAULT_CACHE_PATH, "preprocessor_results"), exist_ok=True)
    return os.path.join(DEFAULT_CACHE_PATH, "preprocessor_results")


def get_postprocessor_results_path() -> str:
    os.makedirs(
        os.path.join(DEFAULT_CACHE_PATH, "postprocessor_results"), exist_ok=True
    )
    return os.path.join(DEFAULT_CACHE_PATH, "postprocessor_results")


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


def get_config_store_path() -> str:
    return CONFIG_STORE_PATH


def get_hf_home() -> str:
    return DEFAULT_HF_HOME

def set_hf_home(path: str) -> None:
    global DEFAULT_HF_HOME
    DEFAULT_HF_HOME = path
    os.environ["HF_HOME"] = DEFAULT_HF_HOME
    os.makedirs(DEFAULT_HF_HOME, exist_ok=True)

def set_postprocessor_path(path: str | None = None) -> None:
    global DEFAULT_POSTPROCESSOR_SAVE_PATH
    DEFAULT_POSTPROCESSOR_SAVE_PATH = path
    os.makedirs(DEFAULT_POSTPROCESSOR_SAVE_PATH, exist_ok=True)
