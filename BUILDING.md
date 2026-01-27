# Building Apex Studio from Source

This guide is for developers or advanced users who want to run the latest development version of Apex Studio or contribute to the project.

**If you just want to use the application, we recommend using our [official releases](https://github.com/totokunda/apex-studio/releases) for the easiest and fastest installation.**

## Run locally (development)

### 0) Clone the repo (Git + Git LFS)

This repo uses **Git submodules** and **Git LFS** (Large File Storage) for some assets.

```bash
# Install Git LFS (if you don't already have it)
git lfs install

# Clone + init submodules
git clone --recurse-submodules https://github.com/totokunda/apex-studio.git

cd apex-studio

# Or git submodule update --init --recursive

# Fetch LFS files (recommended after cloning)
git lfs pull
```

### Prerequisites

- **Node.js 24.x** (required; see `apps/app/package.json` → `engines.node`)
- **Python 3.12** (recommended/tested for `apex-engine`; required for CUDA builds)
- **FFmpeg** (required for media processing; see **Install FFmpeg** below)

#### GPU acceleration notes (CUDA / ROCm)

- **NVIDIA / CUDA**: you must have **NVIDIA device drivers 12.8** installed before using the CUDA requirements / installer paths.
- **AMD / ROCm**: ROCm support currently **requires building from source** and is **not tested** (expect rough edges). The prebuilt app/release artifacts are not validated for ROCm.

Notes:
- **Windows**: use **PowerShell 7+**. If `python` isn’t on PATH, try `py -V` (Python Launcher).
  If venv activation is blocked, run (once) in an elevated PowerShell:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Linux**: ensure `python3.12` + `python3.12-venv` are installed (or use your distro equivalents).

### Install FFmpeg (required)

FFmpeg is used by the dev workflow for media processing. After installing, verify with `ffmpeg -version`.

#### Windows (PowerShell)

```powershell
# Recommended (Windows 10/11): winget
winget install --id Gyan.FFmpeg -e

# Verify
ffmpeg -version
```

Alternative package managers:
- Chocolatey: `choco install ffmpeg -y`
- Scoop: `scoop install ffmpeg`

#### macOS

```bash
brew install ffmpeg
ffmpeg -version
```

#### Linux

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
```

Arch:

```bash
sudo pacman -S --noconfirm ffmpeg
sudo pacman -S --noconfirm ffmpeg
ffmpeg -version
```

### 1) Start the desktop app (Electron)

From the app workspace:

macOS / Linux:

```bash
cd apps/app
npm i
npm start
```

Windows (PowerShell):

```powershell
cd apps\app
npm install
npm start
```

### 2) Install + run the engine (`apex-engine`)

From the API workspace (`apps/api`), install Python deps and the `apex-engine` CLI.

#### Option A (recommended): use the dev pip installer (venv or current env)

This path installs Torch, installs the correct requirements set for your machine, applies Apex-maintained third-party patches, and installs `apex-engine` as a CLI.

macOS (Apple Silicon / MPS):

```bash
cd apps/api
python3.12 -m venv .venv
source .venv/bin/activate
python3 scripts/dev/dev_pip_install.py --machine mac --venv .venv
```

Linux:

```bash
cd apps/api
python3.12 -m venv .venv
source .venv/bin/activate
# pick one:
python3 scripts/dev/dev_pip_install.py --machine linux --venv .venv
# python3 scripts/dev/dev_pip_install.py --machine cpu --venv .venv # CPU on Linux
# python3 scripts/dev/dev_pip_install.py --machine rocm --venv .venv  # AMD ROCm on Linux (source build; not tested)
```

Windows (PowerShell):

```powershell
cd apps\api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# pick one:
py scripts\dev\dev_pip_install.py --machine windows --venv .venv
# py scripts\dev\dev_pip_install.py --machine cpu --venv .venv # CPU on Windows
```

Note: **Option A** applies Apex-maintained third-party patches (diffusers + xformers) automatically.

Then run the engine:

```bash
cd apps/api
apex-engine start --procfile Procfile.dev
```

By default the API binds to `http://127.0.0.1:8765`.

Overrides:
- `apex-engine start --procfile Procfile.dev --host 0.0.0.0 --port 8765`
- Environment variables: `APEX_HOST`, `APEX_PORT`

Stop the engine:
- `apex-engine stop` (or `apex-engine stop --force`)

#### Option B: manual pip install (current environment)

This is mainly useful for debugging/bisects. If you just want to get running, use **Option A**.

```bash
cd apps/api
python3 -m pip install -U pip setuptools wheel

# Install torch (pick one)
python3 -m pip install torch torchvision torchaudio
# python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install a requirements set (pick one)
python3 -m pip install -r requirements/cpu/requirements.txt
# python3 -m pip install -r requirements/mps/requirements.txt          # macOS / MPS
# python3 -m pip install -r requirements/cuda/linux.txt                # Linux / CUDA
# python3 -m pip install -r requirements/cuda/windows.txt              # Windows / CUDA
# python3 -m pip install -r requirements/rocm/linux.txt                # Linux / ROCm (source build; not tested)
# python3 -m pip install -r requirements/rocm/windows.txt              # Windows / ROCm (experimental; source build; not tested)

# Install the CLI entrypoint
python3 -m pip install -e . --no-deps
```

Apply Apex-maintained third-party patches (recommended):

```bash
cd apps/api
python3 scripts/updates/apply_third_party_patches.py
```

This patches:
- `xformers` FA3 import resilience (see `apps/api/patches/xformers-ops-fmha-flash3.patch`)
- `diffusers` PEFT adapter-scale mapping strictness

Disable (for debugging/bisects):
- `APEX_PATCH_DIFFUSERS_PEFT=0`
- `APEX_PATCH_XFORMERS_FLASH3=0`

Then run:

```bash
cd apps/api
apex-engine start --procfile Procfile.dev
```

### Nunchaku
Optional CUDA-only acceleration wheels for some models. The installer probes the *current Python environment* (and installed Torch version) and installs a matching wheel if one exists.

From `apps/api` (with your venv activated), run:
```bash
python3 scripts/deps/maybe_install_nunchaku.py --install
```

#### Windows
```powershell
py scripts\deps\maybe_install_nunchaku.py --install
```

Notes:
- By default this installs with `--no-deps` to avoid dependency resolver churn.
- To allow deps installs, pass `--with-deps` or set `APEX_NUNCHAKU_WITH_DEPS=1`.

### Optional: Rust-accelerated downloader (for max download throughput)

`src/mixins/download_mixin.py` can use an optional Rust/PyO3 module (`apex_download_rs`) for faster URL downloads
(less Python overhead, better chance of saturating network/disk).

Build/install (requires Rust toolchain + cargo):

```bash
python -m pip install -U pip maturin
cd rust/apex_download_rs
maturin develop --release
```

Runtime toggles:
- `APEX_USE_RUST_DOWNLOAD=0` to disable (default is enabled if the module is importable)
- `APEX_DOWNLOAD_PROGRESS_INTERVAL` (seconds, default `0.2`) to throttle progress callbacks
- `APEX_DOWNLOAD_PROGRESS_MIN_BYTES` (bytes, default `1048576`) to throttle progress callbacks

Rust parallel range-download tuning (hf_transfer-style, signature unchanged; all optional):
- `APEX_RUST_DOWNLOAD_MAX_FILES` (default `8`): max concurrent range requests / file handles
- `APEX_RUST_DOWNLOAD_PARALLEL_FAILURES` (default `0`) + `APEX_RUST_DOWNLOAD_MAX_RETRIES` (default `0`): enable per-chunk retry with backoff
- `APEX_RUST_DOWNLOAD_RETRY_BASE_MS` (default `300`) and `APEX_RUST_DOWNLOAD_RETRY_MAX_MS` (default `10000`)
