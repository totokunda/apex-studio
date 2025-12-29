"""
Utility functions for detecting and validating compute capabilities across different hardware platforms.
"""

import platform
import warnings
from typing import Dict, Optional, List, Any
import torch


class ComputeCapability:
    """Represents compute capabilities of the current system."""

    def __init__(
        self,
        compute_type: str,
        cuda_compute_capability: Optional[float] = None,
        device_name: Optional[str] = None,
        device_count: int = 0,
        metal_version: Optional[str] = None,
        cpu_info: Optional[Dict[str, Any]] = None,
    ):
        self.compute_type = compute_type
        self.cuda_compute_capability = cuda_compute_capability
        self.device_name = device_name
        self.device_count = device_count
        self.metal_version = metal_version
        self.cpu_info = cpu_info or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "compute_type": self.compute_type,
            "cuda_compute_capability": self.cuda_compute_capability,
            "device_name": self.device_name,
            "device_count": self.device_count,
            "metal_version": self.metal_version,
            "cpu_info": self.cpu_info,
        }

    def __repr__(self) -> str:
        if self.compute_type == "cuda":
            return f"ComputeCapability(type=cuda, capability={self.cuda_compute_capability}, device={self.device_name})"
        elif self.compute_type == "metal":
            return f"ComputeCapability(type=metal, version={self.metal_version})"
        else:
            return f"ComputeCapability(type=cpu, platform={self.cpu_info.get('platform', 'unknown')})"


def get_cuda_compute_capability() -> Optional[float]:
    """
    Get CUDA compute capability as a float (e.g., 7.0, 7.5, 8.0, 8.6, 9.0).

    Returns None if CUDA is not available.

    Common CUDA compute capabilities:
    - 7.0: Tesla V100
    - 7.5: RTX 2080, RTX 2080 Ti, Titan RTX
    - 8.0: A100
    - 8.6: RTX 3090, RTX 3080, RTX 4090
    - 8.9: RTX 4090, L4, L40
    - 9.0: H100
    """
    if not torch.cuda.is_available():
        return None

    try:
        major, minor = torch.cuda.get_device_capability()
        return float(f"{major}.{minor}")
    except Exception as e:
        warnings.warn(f"Failed to get CUDA compute capability: {e}")
        return None


def get_metal_version() -> Optional[str]:
    """
    Get Metal version for Apple Silicon devices.

    Returns None if Metal/MPS is not available.
    """
    if not hasattr(torch.backends, "mps"):
        return None

    if not torch.backends.mps.is_available():
        return None

    # Try to get Metal version from system
    try:
        system = platform.system()
        if system != "Darwin":
            return None

        mac_version = platform.mac_ver()[0]
        # Metal 3 requires macOS 13+, Metal 2 requires macOS 10.15+
        major_version = int(mac_version.split(".")[0]) if mac_version else 0

        if major_version >= 13:
            return "Metal 3"
        elif major_version >= 11:
            return "Metal 2"
        elif major_version >= 10:
            return "Metal 2"
        else:
            return "Metal"
    except Exception as e:
        warnings.warn(f"Failed to determine Metal version: {e}")
        return "Metal (version unknown)"


def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU information including platform, architecture, and capabilities.
    """
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Check for specific CPU features
    try:
        # Check for AVX512 support (useful for BF16)
        has_bf16 = getattr(torch.backends.cpu, "is_bf16_supported", None)
        info["bf16_supported"] = bool(has_bf16() if callable(has_bf16) else False)
    except Exception:
        info["bf16_supported"] = False

    return info


def get_compute_capability() -> ComputeCapability:
    """
    Detect and return the compute capability of the current system.

    Returns a ComputeCapability object with information about:
    - CUDA devices (with compute capability version)
    - Metal/Apple Silicon devices
    - CPU-only systems

    Example:
        >>> cap = get_compute_capability()
        >>> if cap.compute_type == "cuda":
        ...     print(f"CUDA {cap.cuda_compute_capability} on {cap.device_name}")
        >>> elif cap.compute_type == "metal":
        ...     print(f"Metal device: {cap.metal_version}")
        >>> else:
        ...     print(f"CPU: {cap.cpu_info['platform']}")
    """
    # Check CUDA first
    if torch.cuda.is_available():
        cuda_capability = get_cuda_compute_capability()
        device_name = None
        device_count = 0

        try:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
        except Exception as e:
            warnings.warn(f"Failed to get CUDA device info: {e}")

        return ComputeCapability(
            compute_type="cuda",
            cuda_compute_capability=cuda_capability,
            device_name=device_name,
            device_count=device_count,
        )

    # Check Metal/MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        metal_version = get_metal_version()
        return ComputeCapability(
            compute_type="metal",
            metal_version=metal_version,
            device_count=1,
        )

    # Fallback to CPU
    cpu_info = get_cpu_info()
    return ComputeCapability(
        compute_type="cpu",
        cpu_info=cpu_info,
    )


def get_cuda_architecture_from_capability(compute_capability: float) -> str:
    """
    Get the CUDA architecture name (SM version) from compute capability.

    Args:
        compute_capability: Float representing compute capability (e.g., 7.5, 8.6, 9.0)

    Returns:
        Architecture identifier (e.g., "sm_75", "sm_86", "sm_90")
    """
    major = int(compute_capability)
    minor = int((compute_capability - major) * 10)
    return f"sm_{major}{minor}"


def validate_compute_requirements(
    spec_requirements: Dict[str, Any],
    current_capability: Optional[ComputeCapability] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate that the current system meets the compute requirements specified in a model spec.

    Args:
        spec_requirements: Dictionary containing compute requirements from model spec.
                          Expected keys:
                          - min_cuda_compute_capability: float (e.g., 7.0)
                          - supported_compute_types: list of strings (e.g., ["cuda", "cpu", "metal"])
                          - allowed_cuda_architectures: list of SM versions (e.g., ["sm_75", "sm_80", "sm_86"])
                          - excluded_cuda_architectures: list of SM versions to exclude (e.g., ["sm_90"])
        current_capability: ComputeCapability object. If None, will auto-detect.

    Returns:
        Tuple of (is_valid, error_message).
        - is_valid: True if requirements are met, False otherwise
        - error_message: None if valid, otherwise a string describing the issue

    Example:
        >>> requirements = {
        ...     "min_cuda_compute_capability": 7.0,
        ...     "supported_compute_types": ["cuda", "cpu"],
        ...     "excluded_cuda_architectures": ["sm_90"]  # Skip Hopper
        ... }
        >>> is_valid, error = validate_compute_requirements(requirements)
        >>> if not is_valid:
        ...     raise RuntimeError(f"Compute requirements not met: {error}")
    """
    if current_capability is None:
        current_capability = get_compute_capability()

    # Check if compute type is supported
    supported_types = spec_requirements.get("supported_compute_types", [])
    if supported_types and current_capability.compute_type not in supported_types:
        return False, (
            f"Compute type '{current_capability.compute_type}' is not supported. "
            f"Supported types: {', '.join(supported_types)}"
        )

    # Check CUDA-specific requirements
    if current_capability.compute_type == "cuda":
        if current_capability.cuda_compute_capability is None:
            return False, "CUDA compute capability could not be determined"

        current_arch = get_cuda_architecture_from_capability(
            current_capability.cuda_compute_capability
        )
        current_arch_name = get_cuda_capability_name(
            current_capability.cuda_compute_capability
        )

        # Check minimum compute capability
        min_cuda_cc = spec_requirements.get("min_cuda_compute_capability")
        if min_cuda_cc is not None:
            if current_capability.cuda_compute_capability < min_cuda_cc:
                return False, (
                    f"CUDA compute capability {current_capability.cuda_compute_capability} "
                    f"is below minimum required {min_cuda_cc}. "
                    f"Device: {current_capability.device_name or 'unknown'}"
                )

        # Check if architecture is in allowed list
        allowed_archs = spec_requirements.get("allowed_cuda_architectures", [])
        if allowed_archs:
            if current_arch not in allowed_archs:
                return False, (
                    f"CUDA architecture '{current_arch}' ({current_arch_name}) is not in allowed list. "
                    f"Allowed: {', '.join(allowed_archs)}. "
                    f"Device: {current_capability.device_name or 'unknown'}"
                )

        # Check if architecture is in excluded list
        excluded_archs = spec_requirements.get("excluded_cuda_architectures", [])
        if excluded_archs:
            if current_arch in excluded_archs:
                return False, (
                    f"CUDA architecture '{current_arch}' ({current_arch_name}) is excluded. "
                    f"Excluded: {', '.join(excluded_archs)}. "
                    f"Device: {current_capability.device_name or 'unknown'}"
                )

    return True, None


def get_cuda_capability_name(compute_capability: float) -> str:
    """
    Get a human-readable name for a CUDA compute capability version.

    Args:
        compute_capability: Float representing compute capability (e.g., 7.5, 8.6)

    Returns:
        Architecture name and generation (e.g., "Turing", "Ampere")
    """
    capability_map = {
        5.0: "Maxwell",
        5.2: "Maxwell",
        5.3: "Maxwell",
        6.0: "Pascal",
        6.1: "Pascal",
        6.2: "Pascal",
        7.0: "Volta",
        7.2: "Volta",
        7.5: "Turing",
        8.0: "Ampere",
        8.6: "Ampere",
        8.7: "Ampere",
        8.9: "Ada Lovelace",
        9.0: "Hopper",
    }

    return capability_map.get(compute_capability, f"Unknown (CC {compute_capability})")
