#!/usr/bin/env bash
set -euo pipefail

# Sample MLT "hole player" using melt+sdl2 consumer.
# Plays a video in a separate window sized/positioned to match your Electron hole.

VIDEO_DEFAULT="/Users/tosinkuye/Downloads/YTDown.com_YouTube_DJ-Khaled-Wild-Thoughts-Official-Video-f_Media_fyaI4-5849w_001_1080p.mp4"
X_DEFAULT=571
Y_DEFAULT=246
W_DEFAULT=274
H_DEFAULT=365

VIDEO="$VIDEO_DEFAULT"
X="$X_DEFAULT"
Y="$Y_DEFAULT"
W="$W_DEFAULT"
H="$H_DEFAULT"
MUTE=0
LOOP=1
FOCUS_APEX=1
SYNC_TO_RECT_FILE=1
SYNC_INTERVAL_SEC=0.08
# Must match renderer export path in Preview.tsx.
RECT_FILE_DEFAULT="/tmp/apex-hole-rect.json"
RECT_FILE="$RECT_FILE_DEFAULT"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [options]

Options:
  --video PATH        Video file path (default: hardcoded sample video)
  --x N               Window x position in screen pixels (default: $X_DEFAULT)
  --y N               Window y position in screen pixels (default: $Y_DEFAULT)
  --width N           Window width in pixels (default: $W_DEFAULT)
  --height N          Window height in pixels (default: $H_DEFAULT)
  --mute              Disable audio output
  --no-loop           Do not loop playback
  --no-focus-apex     Do not refocus Apex Studio after launching melt
  --rect-file PATH    Path to hole-rect JSON file (default: $RECT_FILE_DEFAULT)
  --no-sync           Disable live window sync from rect file
  --sync-interval S   Live sync interval in seconds (default: $SYNC_INTERVAL_SEC)
  -h, --help          Show help

Notes:
  - Position/size should match your current hole rect.
  - Keep Apex Studio window in front so this window renders through the hole.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO="${2:-}"
      shift 2
      ;;
    --x)
      X="${2:-}"
      shift 2
      ;;
    --y)
      Y="${2:-}"
      shift 2
      ;;
    --width)
      W="${2:-}"
      shift 2
      ;;
    --height)
      H="${2:-}"
      shift 2
      ;;
    --mute)
      MUTE=1
      shift
      ;;
    --no-loop)
      LOOP=0
      shift
      ;;
    --no-focus-apex)
      FOCUS_APEX=0
      shift
      ;;
    --rect-file)
      RECT_FILE="${2:-}"
      shift 2
      ;;
    --no-sync)
      SYNC_TO_RECT_FILE=0
      shift
      ;;
    --sync-interval)
      SYNC_INTERVAL_SEC="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v melt >/dev/null 2>&1; then
  echo "Error: melt is not installed or not in PATH." >&2
  exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "Error: video file not found: $VIDEO" >&2
  exit 1
fi

# Homebrew's mlt may auto-load Qt modules (libmltqt6.so) even when using sdl2,
# which can fail with missing Qt platform plugin configuration. To avoid that,
# build a minimal no-Qt repository and point MLT_REPOSITORY at it.
MELT_BIN="$(command -v melt)"
MLT_CELLAR="$(cd "$(dirname "$MELT_BIN")/.." && pwd)"
MLT_MODULES_DIR="${MLT_CELLAR}/lib/mlt"
NOQT_REPO_DIR="${TMPDIR:-/tmp}/mlt-noqt-modules"

if [[ -d "$MLT_MODULES_DIR" ]]; then
  mkdir -p "$NOQT_REPO_DIR"
  for module in libmltcore.so libmltavformat.so libmltsdl2.so libmltxml.so; do
    if [[ -f "$MLT_MODULES_DIR/$module" && ! -e "$NOQT_REPO_DIR/$module" ]]; then
      ln -s "$MLT_MODULES_DIR/$module" "$NOQT_REPO_DIR/$module"
    fi
  done
  export MLT_REPOSITORY="$NOQT_REPO_DIR"
fi

# SDL2 respects this to place the output window.
export SDL_VIDEO_WINDOW_POS="${X},${Y}"

CONSUMER_ARGS=(
  -consumer sdl2
  "resolution=${W}x${H}"
  terminate_on_pause=1
)

if [[ "$MUTE" -eq 1 ]]; then
  CONSUMER_ARGS+=(audio_off=1)
fi

if [[ "$LOOP" -eq 1 ]]; then
  PRODUCER_ARGS=("$VIDEO" repeat=-1)
else
  PRODUCER_ARGS=("$VIDEO")
fi

melt "${PRODUCER_ARGS[@]}" "${CONSUMER_ARGS[@]}" &
MELT_PID=$!

echo "Started MLT hole player"
echo "  pid:    $MELT_PID"
echo "  video:  $VIDEO"
echo "  rect:   x=$X y=$Y w=$W h=$H"
if [[ "$SYNC_TO_RECT_FILE" -eq 1 ]]; then
  echo "  sync:   $RECT_FILE (interval ${SYNC_INTERVAL_SEC}s)"
fi

if [[ "$FOCUS_APEX" -eq 1 ]] && command -v osascript >/dev/null 2>&1; then
  sleep 0.25
  osascript -e 'tell application "Apex Studio" to activate' >/dev/null 2>&1 || true
fi

move_resize_melt_window() {
  local rx="$1"
  local ry="$2"
  local rw="$3"
  local rh="$4"

  if ! command -v osascript >/dev/null 2>&1; then
    return 1
  fi

  osascript - "$rx" "$ry" "$rw" "$rh" "$FOCUS_APEX" "$MELT_PID" <<'OSA' >/dev/null 2>&1
on run argv
  set rx to item 1 of argv as integer
  set ry to item 2 of argv as integer
  set rw to item 3 of argv as integer
  set rh to item 4 of argv as integer
  set shouldFocusApex to item 5 of argv as integer
  set meltPid to item 6 of argv as integer

  tell application "System Events"
    if (count of (every process whose unix id is meltPid)) > 0 then
      tell (first process whose unix id is meltPid)
        if (count of windows) > 0 then
          set position of window 1 to {rx, ry}
          set size of window 1 to {rw, rh}
        end if
      end tell
    end if
  end tell

  if shouldFocusApex is 1 then
    tell application "Apex Studio" to activate
  end if
end run
OSA
}

SYNC_PID=""
sync_melt_window_to_rect_file() {
  local last_rect=""
  local warned_missing_rect_file=0
  local warned_applescript_failure=0

  while kill -0 "$MELT_PID" >/dev/null 2>&1; do
    if [[ -f "$RECT_FILE" ]]; then
      warned_missing_rect_file=0
      local rect_line
      rect_line="$(
        python3 - "$RECT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("visible") is False:
        print("NONE")
        raise SystemExit(0)
    rect = data.get("rect")
    if not isinstance(rect, dict):
        print("NONE")
        raise SystemExit(0)
    x = int(round(float(rect.get("left", 0))))
    y = int(round(float(rect.get("top", 0))))
    w = int(round(float(rect.get("width", 0))))
    h = int(round(float(rect.get("height", 0))))
    if w <= 0 or h <= 0:
        print("NONE")
        raise SystemExit(0)
    print(f"{x} {y} {w} {h}")
except Exception:
    print("ERR")
PY
      )"

      if [[ "$rect_line" != "ERR" && "$rect_line" != "NONE" && -n "$rect_line" && "$rect_line" != "$last_rect" ]]; then
        local rx ry rw rh
        read -r rx ry rw rh <<<"$rect_line"
        if move_resize_melt_window "$rx" "$ry" "$rw" "$rh"; then
          warned_applescript_failure=0
          last_rect="$rect_line"
        else
          if [[ "$warned_applescript_failure" -eq 0 ]]; then
            echo "Warning: failed to move/resize melt window via AppleScript."
            echo "Grant Accessibility permission to your terminal app in macOS Settings."
            warned_applescript_failure=1
          fi
        fi
      fi
    else
      if [[ "$warned_missing_rect_file" -eq 0 ]]; then
        echo "Waiting for hole rect file: $RECT_FILE"
        warned_missing_rect_file=1
      fi
    fi

    sleep "$SYNC_INTERVAL_SEC"
  done
}

if [[ "$SYNC_TO_RECT_FILE" -eq 1 ]]; then
  sync_melt_window_to_rect_file &
  SYNC_PID=$!
fi

trap '[[ -n "${SYNC_PID:-}" ]] && kill "$SYNC_PID" >/dev/null 2>&1 || true' EXIT INT TERM

wait "$MELT_PID"
MLT_EXIT_CODE=$?
if [[ -n "$SYNC_PID" ]]; then
  kill "$SYNC_PID" >/dev/null 2>&1 || true
fi
exit "$MLT_EXIT_CODE"
