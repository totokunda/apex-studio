import platform
import warnings
import torch
import torch.backends.mps

def _has_mps_bf16() -> bool:
    if not torch.backends.mps.is_available():
        return False
    
    try:
        # Try to create a dummy tensor and perform a simple operation
        device = torch.device("mps")
        test_tensor = torch.ones(1, 1, device=device, dtype=torch.bfloat16)
        _ = test_tensor * test_tensor
        return True
    except Exception:
        return False

def select_ideal_dtypes(
    *,
    prefer_bfloat16: bool = True,
    quiet: bool = False,
) -> dict[str, torch.dtype]:
    """
    Heuristically choose the best‐supported low-precision ``torch.dtype`` for the three
    major components of a diffusion pipeline – the *transformer diffusion model*,
    *VAE*, and *text-encoder* – **without ever falling back to ``float32``**.

    The rules are:

    1. **CUDA / HIP (NVIDIA or AMD GPUs)**
       *  If the runtime reports native BF16 support **and** ``prefer_bfloat16`` is
          ``True`` → use **``torch.bfloat16``**.
       *  Otherwise use **``torch.float16``**.

    2. **Apple Silicon (MPS backend)**
       *  Use **``torch.float16``** (Apple GPUs expose fast ``float16``; BF16 is
          emulated and slower).

    3. **CPU-only**
       *  If the CPU exposes AVX-512 BF16 (Intel Sapphire Rapids, AMD Zen 4, etc.)
          → use **``torch.bfloat16``**.
       *  Otherwise fall back to **``torch.float16``** (even though the speed-up on
          CPU will be modest, we respect the “no float32” requirement).

    Parameters
    ----------
    prefer_bfloat16 : bool, default ``True``
        When both BF16 *and* FP16 are supported on the active device, pick BF16 if
        ``True`` (recommended on Ampere+/Hopper/MI300 GPUs and AVX-512 machines).
    quiet : bool, default ``False``
        Suppress informational warnings.

    Returns
    -------
    dict[str, torch.dtype]
        ``{"diffusion_model": dtype, "vae": dtype, "text_encoder": dtype}``
    """

    # --------------------------- utility helpers ----------------------------
    def _warn(msg: str):
        if not quiet:
            warnings.warn(msg, stacklevel=2)

    def _device_type() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _has_gpu_bf16() -> bool:
        """
        Unified check for NVIDIA (CUDA) and AMD (HIP/ROCm).
        """
        if not torch.cuda.is_available():
            return False
        # PyTorch ≥2.3 provides torch.cuda.is_bf16_supported()
        is_bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_fn):
            return bool(is_bf16_fn())
        # Fallback: infer from compute capability (NVIDIA only)
        major, minor = torch.cuda.get_device_capability()
        # Ampere (8.x) and newer support BF16
        return major >= 8

    def _has_cpu_bf16() -> bool:
        is_bf16_cpu = getattr(torch.backends.cpu, "is_bf16_supported", None)
        return bool(is_bf16_cpu() if callable(is_bf16_cpu) else False)

    # ----------------------------- main logic -------------------------------
    device = _device_type()

    if device == "cuda":  # includes AMD ROCm (reported as "cuda" by PyTorch)
        if prefer_bfloat16 and _has_gpu_bf16():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        _warn(f"Using {dtype} on GPU: {gpu_name}")

    elif device == "mps":  # Apple Silicon
        # check if mps supports bfloat16
        if _has_mps_bf16():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        _warn(f"MPS backend detected (Apple Silicon) – using {dtype}")

    else:  # CPU only
        if prefer_bfloat16 and _has_cpu_bf16():
            dtype = torch.bfloat16
            _warn("CPU BF16 supported – using torch.bfloat16")
        else:
            dtype = torch.float16
            _warn(
                "CPU BF16 not detected – falling back to torch.float16. "
                "Performance may be limited."
            )

    return {
        "transformer": dtype,
        "vae": dtype,
        "text_encoder": dtype,
    }


def supports_double(device):
    device = torch.device(device)
    if device.type == "mps":
        # MPS backend has limited support for float64
        return False
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
        return True
    except RuntimeError:
        return False


def convert_str_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """
    Convert a string (with common aliases) to a ``torch.dtype``.

    Accepted inputs:
    - ``torch.dtype``: returned unchanged.
    - ``str``: case-insensitive, optional ``"torch."`` prefix, and tolerant of
      underscores, hyphens, and spaces. Supports many aliases:

      Floating point
      - "fp16", "f16", "half", "float16" → ``torch.float16``
      - "bf16", "bfloat16" → ``torch.bfloat16``
      - "fp32", "f32", "float32", "float", "single" → ``torch.float32``
      - "fp64", "f64", "float64", "double" → ``torch.float64``

      Integers
      - "i8", "int8", "char" → ``torch.int8``
      - "u8", "uint8", "byte" → ``torch.uint8``
      - "i16", "int16", "short" → ``torch.int16``
      - "i32", "int32", "int" → ``torch.int32``
      - "i64", "int64", "long" → ``torch.int64``

      Boolean
      - "bool", "boolean" → ``torch.bool``

    Raises ``TypeError`` if the input is not ``str`` or ``torch.dtype``.
    Raises ``ValueError`` for unknown dtype strings with a helpful message.
    """

    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(
            f"Expected dtype as str or torch.dtype, got {type(dtype).__name__}"
        )

    normalized = dtype.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized[len("torch.") :]
    # Remove common separators to be forgiving: "float_16", "float-16", etc.
    normalized = normalized.replace(" ", "").replace("_", "").replace("-", "")

    alias_map: dict[str, torch.dtype] = {
        # Floating point
        "fp16": torch.float16,
        "f16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "single": torch.float32,
        "fp64": torch.float64,
        "f64": torch.float64,
        "float64": torch.float64,
        "double": torch.float64,
        # Integers (signed)
        "i8": torch.int8,
        "int8": torch.int8,
        "char": torch.int8,
        "i16": torch.int16,
        "int16": torch.int16,
        "short": torch.int16,
        "i32": torch.int32,
        "int32": torch.int32,
        "int": torch.int32,
        "i64": torch.int64,
        "int64": torch.int64,
        "long": torch.int64,
        # Unsigned and boolean
        "u8": torch.uint8,
        "uint8": torch.uint8,
        "byte": torch.uint8,
        "bool": torch.bool,
        "boolean": torch.bool,
    }

    if normalized in alias_map:
        return alias_map[normalized]

    # As a last resort, try torch attribute names directly (e.g., "float16").
    maybe = getattr(torch, normalized, None)
    if isinstance(maybe, torch.dtype):
        return maybe

    known_keys = ", ".join(sorted(alias_map.keys()))
    raise ValueError(
        f"Unknown dtype string '{dtype}'. Try one of: {known_keys}, or a valid torch dtype name like 'float16'."
    )
