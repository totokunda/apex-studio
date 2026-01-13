# Machine-specific requirements entrypoints

These files are **entrypoints** you can install directly with pip.
Choose the one matching your hardware.

## Structure

- `requirements/cpu/requirements.txt`: CPU-only
- `requirements/mps/requirements.txt`: macOS / Apple Silicon (Metal/MPS)
- `requirements/rocm/requirements.txt`: AMD ROCm (Linux)
- `requirements/cuda/*.txt`: NVIDIA CUDA variants
- `requirements/platform/*.txt`: OS/CPU entrypoints (requested “one file per target” wrappers)

## CUDA Variants

All CUDA entrypoints include performance libraries (FlashAttention, SageAttention, xformers).
Choose based on your GPU architecture:

- **Ampere (A100, RTX 30xx)**: `requirements/cuda/ampere.txt`
- **Ada Lovelace (RTX 4070, L4/L40)**: `requirements/cuda/ada.txt` (SageAttention v1)
- **Ada Lovelace (RTX 4090)**: `requirements/cuda/ada-rtx4090.txt` (SageAttention v2)
- **Hopper (H100)**: `requirements/cuda/hopper.txt` (Includes FlashAttention 3)
- **Blackwell (B200)**: `requirements/cuda/blackwell.txt` (Includes FlashAttention 3)

## Usage

```bash
# OS/CPU wrappers (simple defaults)
pip install -r requirements/platform/cpu.txt
pip install -r requirements/platform/mac.txt
pip install -r requirements/platform/linux.txt
pip install -r requirements/platform/windows.txt

# Example for CPU
pip install -r requirements/cpu/requirements.txt

# Example for CUDA Ampere
pip install -r requirements/cuda/ampere.txt
```

## Layering

They are layered on top of:
- `requirements/requirements.txt` (**base**, mostly pure-python / core deps)
- `requirements/processors.requirements.txt` (**optional/heavy processors**)

## Installation Helper

You can also use the helper script:

```bash
python scripts/dev/dev_pip_install.py --machine linux
python scripts/dev/dev_pip_install.py --machine windows
python scripts/dev/dev_pip_install.py --machine mac
python scripts/dev/dev_pip_install.py --machine cpu

# Or pick a specific CUDA arch stack:
python scripts/dev/dev_pip_install.py --machine cuda-ampere
```

### Nunchaku

Nunchaku is **installed by default** when you use `scripts/dev/dev_pip_install.py` on
supported CUDA platforms (Linux/Windows). This is done via a small selector that chooses
the correct wheel for your installed torch version.

## Windows CUDA wheels

Windows-specific wheel installs are included directly in the CUDA entrypoints.
Hopper/Blackwell variants include FlashAttention 3 wheels.

If you want the repo to handle setup automatically (including system deps), use:

```bash
bash scripts/install.sh
```
