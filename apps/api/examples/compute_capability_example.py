"""
Example demonstrating compute capability detection and validation.

This script shows:
1. How to detect the current system's compute capabilities
2. How to validate requirements programmatically
3. How the engine initialization respects compute requirements
"""

from src.utils.compute import (
    get_compute_capability,
    validate_compute_requirements,
    get_cuda_capability_name,
    get_cuda_architecture_from_capability,
)


def main():
    print("=" * 70)
    print("Compute Capability Detection Example")
    print("=" * 70)

    # Get current system capabilities
    cap = get_compute_capability()

    print(f"\nCurrent System:")
    print(f"  Compute Type: {cap.compute_type}")

    if cap.compute_type == "cuda":
        print(f"  CUDA Capability: {cap.cuda_compute_capability}")
        cuda_arch = get_cuda_architecture_from_capability(cap.cuda_compute_capability)
        print(
            f"  Architecture: {get_cuda_capability_name(cap.cuda_compute_capability)}"
        )
        print(f"  SM Version: {cuda_arch}")
        print(f"  Device: {cap.device_name}")
        print(f"  Device Count: {cap.device_count}")
    elif cap.compute_type == "metal":
        print(f"  Metal Version: {cap.metal_version}")
    else:
        print(f"  Platform: {cap.cpu_info.get('platform', 'unknown')}")
        print(f"  Machine: {cap.cpu_info.get('machine', 'unknown')}")
        print(f"  BF16 Support: {cap.cpu_info.get('bf16_supported', False)}")

    print("\n" + "=" * 70)
    print("Validation Examples")
    print("=" * 70)

    # Example 1: Basic validation with supported types
    print("\n1. Checking if system supports CUDA or CPU:")
    requirements_1 = {"supported_compute_types": ["cuda", "cpu"]}
    is_valid, error = validate_compute_requirements(requirements_1)
    print(f"   Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"   Error: {error}")

    # Example 2: CUDA with minimum capability
    print("\n2. Checking CUDA capability >= 7.0:")
    requirements_2 = {
        "min_cuda_compute_capability": 7.0,
        "supported_compute_types": ["cuda"],
    }
    is_valid, error = validate_compute_requirements(requirements_2)
    print(f"   Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"   Error: {error}")

    # Example 3: High-end requirement (will likely fail unless on H100)
    print("\n3. Checking CUDA capability >= 9.0 (Hopper):")
    requirements_3 = {
        "min_cuda_compute_capability": 9.0,
        "supported_compute_types": ["cuda"],
    }
    is_valid, error = validate_compute_requirements(requirements_3)
    print(f"   Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"   Error: {error}")

    # Example 4: Metal-only requirement
    print("\n4. Checking Metal-only support:")
    requirements_4 = {"supported_compute_types": ["metal"]}
    is_valid, error = validate_compute_requirements(requirements_4)
    print(f"   Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"   Error: {error}")

    # Example 5: Exclude Hopper architecture
    print("\n5. Checking exclusion of Hopper (sm_90):")
    requirements_5 = {
        "supported_compute_types": ["cuda"],
        "excluded_cuda_architectures": ["sm_90"],
    }
    is_valid, error = validate_compute_requirements(requirements_5)
    print(f"   Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"   Error: {error}")

    # Example 6: Allow only Ampere GPUs
    print("\n6. Checking allowed architectures (Ampere only: sm_80, sm_86):")
    requirements_6 = {
        "supported_compute_types": ["cuda"],
        "allowed_cuda_architectures": ["sm_80", "sm_86"],
    }
    is_valid, error = validate_compute_requirements(requirements_6)
    print(f"   Result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if error:
        print(f"   Error: {error}")

    print("\n" + "=" * 70)
    print("Engine Initialization")
    print("=" * 70)
    print("\nWhen you initialize an engine with a YAML manifest that includes")
    print("compute_requirements, the validation happens automatically.")
    print("\nExample YAML config:")
    print(
        """
spec:
  engine: qwenimage
  model_type: t2i
  compute_requirements:
    min_cuda_compute_capability: 7.0
    supported_compute_types:
    - cuda
    - cpu
    - metal
    # Optional: Exclude specific architectures
    excluded_cuda_architectures:
    - sm_90  # Skip Hopper
    # OR: Allow only specific architectures
    # allowed_cuda_architectures:
    # - sm_80  # A100
    # - sm_86  # RTX 3090
"""
    )
    print("\nIf requirements aren't met, initialization will fail with a")
    print("detailed error message showing what's required vs. what's available.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
