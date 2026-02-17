#!/bin/bash
# Run this from your project root to vendor and patch libyuv
set -e

LIBYUV_DIR="deps/libyuv"

if [ -d "$LIBYUV_DIR" ]; then
  echo "libyuv already present at $LIBYUV_DIR — delete it to re-setup"
  exit 0
fi

echo "Cloning libyuv..."
mkdir -p deps
git clone --depth 1 https://chromium.googlesource.com/libyuv/libyuv "$LIBYUV_DIR"

# --- macOS patch ---
# Apple Silicon (M1/M2/M3/M4) does NOT support SVE, SVE2, or SME.
# libyuv's own Chromium build (libyuv.gni) restricts SME to Linux/Android
# and SVE requires special compiler flags not available in Xcode clang.
#
# The problem: row.h unconditionally defines HAS_*_SVE2 and HAS_*_SME
# for __aarch64__, causing the core .cc files (convert.cc, convert_argb.cc,
# etc.) to reference SVE2/SME functions. But the _sve.cc/_sme.cc files
# that implement those functions can't compile without SVE compiler support.
#
# Fix: Comment out all HAS_*_SVE2 and HAS_*_SME #define lines in row.h.
# This leaves NEON (which Apple Silicon fully supports) as the fast path.
if [ "$(uname)" = "Darwin" ]; then
  echo "Patching libyuv for macOS (disabling SVE2/SME defines)..."
  ROW_H="$LIBYUV_DIR/include/libyuv/row.h"
  if [ -f "$ROW_H" ]; then
    # Comment out all lines that define HAS_*_SVE2 or HAS_*_SME
    sed -i '' 's/^#define HAS_\(.*\)_SVE2$/\/\/ #define HAS_\1_SVE2  \/\/ disabled for macOS/' "$ROW_H"
    sed -i '' 's/^#define HAS_\(.*\)_SVE$/\/\/ #define HAS_\1_SVE  \/\/ disabled for macOS/' "$ROW_H"
    sed -i '' 's/^#define HAS_\(.*\)_SME$/\/\/ #define HAS_\1_SME  \/\/ disabled for macOS/' "$ROW_H"

    # Also patch compare_row.h and scale_row.h if they have similar defines
    for header in "$LIBYUV_DIR/include/libyuv/compare_row.h" \
                  "$LIBYUV_DIR/include/libyuv/scale_row.h"; do
      if [ -f "$header" ]; then
        sed -i '' 's/^#define HAS_\(.*\)_SVE2$/\/\/ #define HAS_\1_SVE2/' "$header"
        sed -i '' 's/^#define HAS_\(.*\)_SVE$/\/\/ #define HAS_\1_SVE/' "$header"
        sed -i '' 's/^#define HAS_\(.*\)_SME$/\/\/ #define HAS_\1_SME/' "$header"
      fi
    done

    echo "Patched libyuv headers — SVE2/SME disabled"
  fi
fi

echo "Done. libyuv vendored at $LIBYUV_DIR"