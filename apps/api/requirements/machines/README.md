# Machine-specific requirements entrypoints

These files are **entrypoints** you can install directly with pip:

```bash
pip install -r requirements/machines/<file>.txt
```

They are layered on top of:
- `requirements/requirements.txt` (**base**, mostly pure-python / core deps)
- `requirements/processors.requirements.txt` (**optional/heavy processors**)
- `requirements/device_requirements` (**legacy**, generic accelerator deps)

## Pick one

- **CPU-only**: `cpu.txt`
- **macOS / Apple Silicon (Metal/MPS)**: `mps.txt`
- **AMD ROCm (Linux)**: `rocm.txt`
- **NVIDIA CUDA (Ampere)**: `cuda-sm80-ampere.txt`
- **NVIDIA CUDA (Ada / L4/L40/4090)**: `cuda-sm89-ada.txt`
- **NVIDIA CUDA (Hopper / H100)**: `cuda-sm90-hopper.txt`
- **NVIDIA CUDA (Blackwell)**: `cuda-sm100-blackwell.txt`

## Notes on FlashAttention / SageAttention

Some performance libraries are best installed from source and may require a full
build toolchain + a matching CUDA toolkit.

## Triton

- **CUDA (NVIDIA)**:
  - Linux: installed as `triton` (pinned in the CUDA entrypoints)
  - Windows: installed as `triton-windows` (selected via `sys_platform == "win32"`)
- **ROCm (AMD)**: installed as `triton-rocm` (ROCm-specific distribution)

## Windows CUDA wheels

Windows-specific wheel installs are now included directly in the CUDA entrypoints:
- `cuda-sm80-ampere.txt`: includes the FlashAttention Windows wheel (cu126/torch2.6.0/cp312)
- `cuda-sm89-ada.txt`: includes the SageAttention Windows wheel (cu128/torch2.7.1/cp312)
- `cuda-sm100-blackwell.txt`: includes the SageAttention Windows wheel (cu128/torch2.7.1/cp312)

If you want the repo to handle this automatically, use:

```bash
bash scripts/install.sh
```


