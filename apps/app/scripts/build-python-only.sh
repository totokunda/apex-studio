#!/bin/bash
#
# Build Python API Bundle Only
# 
# This script builds just the Python API bundle for the current platform.
# Useful for development and testing the Python bundling separately.
#
# Usage:
#   ./scripts/build-python-only.sh [cuda|cpu|mps]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
API_DIR="$(dirname "$APP_DIR")/api"
OUTPUT_DIR="$APP_DIR/python-api-bundle"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
GPU_MODE="${1:-auto}"

# Detect platform
case "$(uname -s)" in
    Darwin*)    PLATFORM="darwin" ;;
    Linux*)     PLATFORM="linux" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="win32" ;;
    *)          PLATFORM="unknown" ;;
esac

log_info "Building Python API bundle"
log_info "Platform: $PLATFORM"
log_info "GPU Mode: $GPU_MODE"
log_info "Output: $OUTPUT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
log_info "Using $PYTHON_VERSION"

# Check if we're in a conda environment
if [ -n "$CONDA_PREFIX" ]; then
    log_info "Using conda environment: $CONDA_PREFIX"
fi

# Clean previous build
if [ -d "$OUTPUT_DIR" ]; then
    log_info "Removing previous bundle..."
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the bundle
log_info "Running Python bundler..."

cd "$API_DIR"

BUNDLER_ARGS=(
    "scripts/bundle_python.py"
    "--platform" "$PLATFORM"
    "--output" "$APP_DIR/temp-python-build"
)

if [ "$GPU_MODE" != "auto" ]; then
    BUNDLER_ARGS+=("--cuda" "$GPU_MODE")
fi

# Check for signing on macOS
if [ "$PLATFORM" = "darwin" ] && [ -n "$APPLE_IDENTITY" ]; then
    log_info "Code signing enabled with identity: $APPLE_IDENTITY"
    BUNDLER_ARGS+=("--sign")
fi

python3 "${BUNDLER_ARGS[@]}"

# Move bundle to final location
if [ -d "$APP_DIR/temp-python-build/python-api/apex-engine" ]; then
    cp -r "$APP_DIR/temp-python-build/python-api/apex-engine/." "$OUTPUT_DIR/"
    rm -rf "$APP_DIR/temp-python-build"
    log_success "Python bundle created at: $OUTPUT_DIR"
else
    log_error "Bundle was not created successfully"
    exit 1
fi

# Test the bundle
log_info "Testing bundle..."

if [ "$PLATFORM" = "win32" ]; then
    APEX_BIN="$OUTPUT_DIR/apex-engine.exe"
else
    APEX_BIN="$OUTPUT_DIR/apex-engine"
fi

if [ -f "$APEX_BIN" ]; then
    chmod +x "$APEX_BIN" 2>/dev/null || true
    
    # Quick version check
    if "$APEX_BIN" --help &> /dev/null; then
        log_success "Bundle executable is working"
    else
        log_warning "Bundle executable may have issues"
    fi
else
    log_warning "Bundle executable not found at expected path"
fi

# Print bundle size
BUNDLE_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
log_success "Bundle size: $BUNDLE_SIZE"

echo ""
log_success "Python API bundle complete!"
echo ""
echo "To test the bundle:"
echo "  cd $OUTPUT_DIR"
echo "  ./apex-engine start --port 8765"
echo ""
echo "To build the full Electron app with this bundle:"
echo "  cd $APP_DIR"
echo "  node scripts/build-app.js --skip-python"

