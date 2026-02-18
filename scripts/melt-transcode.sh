#!/usr/bin/env bash
# Melt transcoding helper for macOS Homebrew
# Fixes: "Could not find the Qt platform plugin cocoa" by setting QT_PLUGIN_PATH
#
# Usage: ./scripts/melt-transcode.sh input.mp4 output.mp4
# Or:   ./scripts/melt-transcode.sh input.mp4 output.mp4 in=0 out=100

set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input> <output> [producer options...]"
  echo ""
  echo "Example: $0 video.mp4 output.mp4"
  echo "Example: $0 video.mp4 output.mp4 in=0 out=300"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

# Fix Qt plugin path for Homebrew MLT on macOS
export QT_PLUGIN_PATH="/opt/homebrew/opt/qtbase/share/qt/plugins"

melt "$INPUT" -consumer "avformat:${OUTPUT}" -progress "$@"
