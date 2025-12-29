# Compute Requirements Validation

This document describes how to specify and validate compute requirements for models in the Apex system.

## Overview

Models can specify minimum compute requirements in their YAML manifest files. When an engine is initialized, these requirements are automatically validated against the current system's capabilities. If the requirements are not met, the engine initialization will fail with a detailed error message.

## Supported Compute Types

- **cuda**: NVIDIA GPUs with CUDA support
- **metal**: Apple Silicon (M1/M2/M3/M4) with Metal support
- **cpu**: CPU-only execution

## YAML Configuration

Add a `compute_requirements` section to your model's `spec` in the YAML manifest:

```yaml
spec:
  engine: your_engine
  model_type: your_model_type
  compute_requirements:
    min_cuda_compute_capability: 7.0
    supported_compute_types:
    - cuda
    - cpu
    - metal
```

### Configuration Options

#### `min_cuda_compute_capability` (optional)

Minimum CUDA compute capability required (as a float). This is only checked when running on CUDA devices.

Common CUDA compute capabilities:
- **7.0**: Tesla V100
- **7.5**: RTX 2080, RTX 2080 Ti, Titan RTX (Turing)
- **8.0**: A100 (Ampere)
- **8.6**: RTX 3090, RTX 3080, RTX 4090 (Ampere)
- **8.9**: RTX 4090, L4, L40 (Ada Lovelace)
- **9.0**: H100 (Hopper)

#### `supported_compute_types` (optional)

List of compute types that this model supports. If the current system's compute type is not in this list, initialization will fail.

#### `allowed_cuda_architectures` (optional)

List of specific CUDA architectures (SM versions) that are allowed. Only use this if you need to whitelist specific architectures. If specified, the current GPU's architecture must be in this list.

Common SM versions:
- **sm_75**: Turing (RTX 2080, RTX 2080 Ti)
- **sm_80**: Ampere (A100)
- **sm_86**: Ampere (RTX 3090, RTX 3080)
- **sm_89**: Ada Lovelace (RTX 4090, L4, L40)
- **sm_90**: Hopper (H100)

#### `excluded_cuda_architectures` (optional)

List of specific CUDA architectures (SM versions) to exclude. Use this to skip problematic GPU generations. If the current GPU's architecture is in this list, initialization will fail.

## Examples

### CUDA-only Model with Minimum Capability

This configuration requires CUDA with compute capability 7.0 or higher:

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.0
    supported_compute_types:
    - cuda
```

### Multi-Platform Model

This configuration allows the model to run on CUDA, Metal, or CPU:

```yaml
spec:
  compute_requirements:
    supported_compute_types:
    - cuda
    - cpu
    - metal
```

### High-End CUDA Model

This configuration requires modern GPUs with Ampere architecture or newer:

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 8.0
    supported_compute_types:
    - cuda
```

### Excluding Specific Architectures

Skip Hopper GPUs due to a known issue:

```yaml
spec:
  compute_requirements:
    supported_compute_types:
    - cuda
    excluded_cuda_architectures:
    - sm_90  # Skip H100 (Hopper)
```

### Allowing Only Specific Architectures

Restrict to tested Ampere GPUs only:

```yaml
spec:
  compute_requirements:
    supported_compute_types:
    - cuda
    allowed_cuda_architectures:
    - sm_80  # A100
    - sm_86  # RTX 3090, RTX 3080
```

### Complex Requirements

Combine multiple constraints:

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.5
    supported_compute_types:
    - cuda
    - cpu
    excluded_cuda_architectures:
    - sm_90  # Skip Hopper
    - sm_89  # Skip Ada Lovelace
```

## Programmatic Usage

You can also use the compute capability utilities directly in your code:

```python
from src.utils.compute import (
    get_compute_capability,
    validate_compute_requirements,
    get_cuda_capability_name,
)

# Get current system's compute capability
cap = get_compute_capability()
print(f"Compute type: {cap.compute_type}")

if cap.compute_type == "cuda":
    print(f"CUDA capability: {cap.cuda_compute_capability}")
    print(f"Device: {cap.device_name}")
    print(f"Architecture: {get_cuda_capability_name(cap.cuda_compute_capability)}")

# Validate requirements
requirements = {
    "min_cuda_compute_capability": 7.0,
    "supported_compute_types": ["cuda"]
}

is_valid, error_message = validate_compute_requirements(requirements)
if not is_valid:
    print(f"Requirements not met: {error_message}")
```

## Error Messages

When compute requirements are not met, you'll receive a detailed error message like:

```
Compute Validation Failed:
  Compute type 'cpu' is not supported. Supported types: cuda

Current System:
  Compute Type: cpu
  Platform: Linux

Required:
  Supported Types: cuda
```

Or for CUDA capability mismatches:

```
Compute Validation Failed:
  CUDA compute capability 7.5 is below minimum required 8.0. Device: NVIDIA GeForce RTX 2080 Ti

Current System:
  Compute Type: cuda
  CUDA Capability: 7.5
  Device: NVIDIA GeForce RTX 2080 Ti

Required:
  Min CUDA Capability: 8.0
  Supported Types: cuda
```

## Disabling Validation

If you need to bypass validation during development (not recommended for production), you can temporarily remove or comment out the `compute_requirements` section from your YAML manifest.

## Testing

Run the compute capability tests to verify the system is working correctly:

```bash
pytest tests/test_compute_capability.py
```

