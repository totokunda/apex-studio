# Compute requirements: excluding / allowing CUDA architectures

This guide explains how to **exclude** (denylist) or **restrict to** (allowlist) specific NVIDIA GPU architectures when using `compute_requirements` in an Apex model manifest.

## Terminology

- **CUDA compute capability**: a float like `8.6` or `9.0` (what you set via `min_cuda_compute_capability`).
- **CUDA architecture (SM version)**: a string like `sm_86` or `sm_90` (what you set via `allowed_cuda_architectures` / `excluded_cuda_architectures`).

In Apex, the SM string is derived from the compute capability as:

```text
sm_{major}{minor}
```

So \(8.6 \rightarrow \) `sm_86`, \(9.0 \rightarrow \) `sm_90`.

## What the validator checks (CUDA)

When the current system is CUDA, validation is applied in this order:

1. **`supported_compute_types`** (if set)
2. **`min_cuda_compute_capability`** (if set)
3. **`allowed_cuda_architectures`** (if set, current `sm_XX` must be in the list)
4. **`excluded_cuda_architectures`** (if set, current `sm_XX` must *not* be in the list)

If you set **both** allowlist and denylist, **both apply** (denylist wins if the current architecture is listed in both).

## Excluding an architecture (denylist)

Use `excluded_cuda_architectures` when a specific generation is known-bad (for example, a kernel/regression that only affects Hopper).

```yaml
spec:
  # ...
  compute_requirements:
    min_cuda_compute_capability: 7.5
    supported_compute_types:
    - cuda
    - cpu
    excluded_cuda_architectures:
    - sm_90  # Hopper (H100/H800)
```

This is the same pattern as `examples/compute_requirements_skip_hopper.yml`.

## Allowing only specific architectures (allowlist)

Use `allowed_cuda_architectures` when you’ve only tested on certain SMs and want to hard-block everything else (even if it would otherwise meet `min_cuda_compute_capability`).

```yaml
spec:
  # ...
  compute_requirements:
    min_cuda_compute_capability: 8.0
    supported_compute_types:
    - cuda
    allowed_cuda_architectures:
    - sm_80  # Ampere (A100/A30/A10)
    - sm_86  # Ampere (RTX 30xx / A40)
```

This is the same pattern as `examples/compute_requirements_ampere_only.yml`.

## Choosing the right `sm_XX`

To see what Apex thinks your current system is, you can run the same utilities the validator uses:

```python
from src.utils.compute import (
    get_compute_capability,
    get_cuda_architecture_from_capability,
    validate_compute_requirements,
)

cap = get_compute_capability()
print("compute_type:", cap.compute_type)

if cap.compute_type == "cuda":
    current_sm = get_cuda_architecture_from_capability(cap.cuda_compute_capability)
    print("cuda_compute_capability:", cap.cuda_compute_capability)
    print("sm:", current_sm)

    ok, err = validate_compute_requirements(
        {"supported_compute_types": ["cuda"], "excluded_cuda_architectures": ["sm_90"]}
    )
    print("exclude hopper valid?", ok, err)
```

The full, more verbose example is in `examples/compute_capability_example.py`.

