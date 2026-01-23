# Building Apex Studio

This document describes how to build and package Apex Studio for distribution.

## Prerequisites

### All Platforms

- **Node.js** >= 24.0.0
- **npm** >= 10.0.0
- **Python** >= 3.12
- **Git**

### Platform-Specific

#### macOS
- **Xcode Command Line Tools**: `xcode-select --install`
- **Apple Developer Account** (for code signing and notarization)
- **Code Signing Certificate** (Developer ID Application)

#### Windows
- **Visual Studio Build Tools** (with C++ workload)
- **Windows SDK** (for signtool)
- **Code Signing Certificate** (.pfx file)

#### Linux
- **Build essentials**: `sudo apt-get install build-essential`
- **Additional dependencies**:
  ```bash
  sudo apt-get install libgtk-3-dev libnotify-dev libnss3-dev libxss1 libxtst6 libatspi2.0-dev
  ```

### GPU Support (Optional)

#### NVIDIA (Linux/Windows)
- **CUDA Toolkit** >= 12.1
- **cuDNN** >= 8.9
- **NVIDIA Drivers** >= 525

#### AMD ROCm (Linux)
- **ROCm** >= 6.0

#### Apple Silicon (macOS)
- Metal Performance Shaders (MPS) is automatically available on M1/M2/M3 Macs

## Quick Start

```bash
# Clone and enter the app directory
cd apps/app

# Install dependencies
npm install

# Build and package for current platform
npm run bundle
```

## Build Commands

## ffmpeg / ffprobe

Apex Studio uses native `ffmpeg` / `ffprobe` for previews, proxies, and export.

- **Packaged builds**: `electron-builder` bundles `ffmpeg` and `ffprobe` into the app at `resources/ffmpeg/`, so end users do **not** need them installed.
- **Dev**: if `ffmpeg` / `ffprobe` are not on your PATH, install dependencies normally (`npm install`) and the app will fall back to `ffmpeg-static` / `ffprobe-static`.

Overrides (useful for debugging):
- `APEX_FFMPEG_PATH` / `APEX_FFPROBE_PATH` (absolute paths)

### Full Application Bundle

These commands build both the Python API and Electron app:

```bash
# Build for current platform (auto-detects architecture)
npm run bundle

# macOS builds
npm run bundle:mac          # Universal (x64 + arm64)
npm run bundle:mac:arm64    # Apple Silicon only
npm run bundle:mac:x64      # Intel only

# Windows build
npm run bundle:win

# Linux build
npm run bundle:linux
```


### Electron Only (Using Existing Python Bundle)

```bash
npm run compile         # Current platform
npm run compile:mac     # macOS
npm run compile:win     # Windows
npm run compile:linux   # Linux
```

Note: `npm run compile*` now builds a **lean Electron-only** app by default (does not bundle Python).
To explicitly include the Python bundle (very large), set:

```bash
export APEX_INCLUDE_PYTHON_BUNDLE=1
```

## Code Signing

### macOS Code Signing & Notarization

Set these environment variables before building:

```bash
export APPLE_ID="your@email.com"
export APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"  # App-specific password
export APPLE_TEAM_ID="XXXXXXXXXX"
export APPLE_IDENTITY="Developer ID Application: Your Name (XXXXXXXXXX)"
```

To create an app-specific password:
1. Go to [appleid.apple.com](https://appleid.apple.com)
2. Sign in and go to Security > App-Specific Passwords
3. Generate a new password for "Apex Studio"

### Windows Code Signing

Set these environment variables:

```bash
export WINDOWS_CERT_FILE="/path/to/certificate.pfx"
export WINDOWS_CERT_PASSWORD="your-password"
export WINDOWS_PUBLISHER_NAME="Your Company Name"
```

For Azure Key Vault signing (CI/CD):

```bash
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export AZURE_KEY_VAULT_URI="https://your-vault.vault.azure.net"
export AZURE_KEY_VAULT_CERT_NAME="your-certificate-name"
```

## Publishing Releases

### GitHub Releases

Set your GitHub token:

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
export GITHUB_OWNER="your-org"
export GITHUB_REPO="apex-studio"
```

Then publish:

```bash
# Create release
npm run publish

# Create draft release
npm run publish:draft
```

### S3 Distribution (Optional)

For faster downloads, you can also distribute via S3:

```bash
export AWS_ACCESS_KEY_ID="AKIAXXXXXXXX"
export AWS_SECRET_ACCESS_KEY="xxxxxxxx"
export AWS_S3_BUCKET="your-releases-bucket"
export AWS_REGION="us-east-1"
```

## Build Output

After building, you'll find the distributables in `apps/app/dist/`:

| Platform | Format | File |
|----------|--------|------|
| macOS | DMG | `Apex Studio-{version}-mac-{arch}.dmg` |
| macOS | ZIP | `Apex Studio-{version}-mac-{arch}.zip` |
| Windows | Installer | `Apex Studio-{version}-win-x64.exe` |
| Windows | Portable | `Apex Studio-{version}-win-x64-portable.exe` |
| Linux | Debian | `Apex Studio-{version}-linux-x64.deb` |
| Linux | AppImage | `Apex Studio-{version}-x64.AppImage` |
| Linux | Tarball | `Apex Studio-{version}-linux-x64.tar.gz` |

## GPU Support Matrix

The bundled Python API automatically detects and uses available GPU acceleration:

| Platform | GPU | Backend | Notes |
|----------|-----|---------|-------|
| macOS (Apple Silicon) | M1/M2/M3 | MPS | Automatic |
| macOS (Intel) | None | CPU | Automatic |
| Windows/Linux | NVIDIA | CUDA 12.x | Requires NVIDIA drivers |
| Windows/Linux | AMD | ROCm 6.x | Requires ROCm installation |
| Windows/Linux | None | CPU | Fallback mode |

### Bundling with Specific GPU Support

To force a specific GPU backend in the bundle:

```bash
# Python bundle with CUDA (for NVIDIA GPUs)
node scripts/build-app.js --cuda cuda126

# Python bundle with CPU only (smallest size)
node scripts/build-app.js --cuda cpu
```

## Development Mode

In development mode, the Electron app connects to an external Python API:

```bash
# Terminal 1: Start Python API
cd apps/api
conda activate apex  # or your virtual environment
python -m src dev

# Terminal 2: Start Electron app
cd apps/app
npm run start
```

## Troubleshooting

### macOS: "App is damaged" or Gatekeeper Issues

This usually means the app isn't properly signed or notarized:

```bash
# Check signature
codesign -dv --verbose=4 "/path/to/Apex Studio.app"

# Check notarization
spctl -a -vv "/path/to/Apex Studio.app"
```

### Windows: Missing Dependencies

If the app fails to start, install the Visual C++ Redistributable:
- Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Linux: CUDA Not Detected

Ensure NVIDIA drivers and CUDA toolkit are properly installed:

```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA version
```

### Python Bundle Too Large

The full Python bundle with ML libraries can be 3-5GB. To reduce size:

1. Use CPU-only mode: `--cuda cpu`
2. Exclude unused models from the manifest
3. Use quantized model variants

### Build Fails with "Out of Memory"

Building the Python venv bundle (ML deps) is memory-intensive. Try:

1. Close other applications
2. Increase swap space
3. Build on a machine with more RAM (16GB+ recommended)

## Architecture Overview

```
Apex Studio
├── Electron App (apps/app/)
│   ├── Main Process
│   │   ├── Window Management
│   │   ├── Python Process Manager  ← Starts/stops bundled API
│   │   └── IPC Handlers
│   ├── Renderer Process
│   │   └── React UI
│   └── Preload Scripts
│       └── IPC Bridge
│
└── Python API (apps/api/)
    ├── FastAPI Server
    ├── Ray Distributed Processing
    ├── ML Models & Preprocessors
    └── Bundled as a self-contained venv + source/assets (no PyInstaller)
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        os: [macos-latest, windows-latest, ubuntu-latest]
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-node@v4
        with:
          node-version: '23'
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          cd apps/app
          npm install
      
      - name: Build
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          # Add signing credentials for each platform
        run: |
          cd apps/app
          npm run publish
```

## License

See the LICENSE file in the project root.

