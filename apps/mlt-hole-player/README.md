# MLT Hole Player Sample

This sample runs a separate MLT (`melt`) process that plays a video in a window positioned to match your Electron "hole" region.

## File

- Script: `apps/mlt-hole-player/play-in-hole.sh`

## Quick start

```bash
/Users/tosinkuye/apex-workspace/apex-studio/apps/mlt-hole-player/play-in-hole.sh
```

It defaults to this video:

- `/Users/tosinkuye/Downloads/YTDown.com_YouTube_DJ-Khaled-Wild-Thoughts-Official-Video-f_Media_fyaI4-5849w_001_1080p.mp4`

## Set custom hole geometry

```bash
/Users/tosinkuye/apex-workspace/apex-studio/apps/mlt-hole-player/play-in-hole.sh \
  --x 571 --y 246 --width 274 --height 365
```

## Useful flags

- `--video /absolute/path/to/video.mp4`
- `--mute`
- `--no-loop`
- `--no-focus-apex`
- `--rect-file /tmp/apex-hole-rect.json`
- `--no-sync`
- `--sync-interval 0.08`

## Notes

- Keep Apex Studio in front so you see this through the transparent hole.
- By default, the script live-syncs to `/tmp/apex-hole-rect.json` (written by the renderer) so size and position track panel resizing.
- On macOS, window move/resize uses AppleScript UI automation (`System Events`). If this does not work, allow Accessibility permissions for Terminal (or your shell host app).
- The script auto-configures `MLT_REPOSITORY` to a Qt-free module subset (`core + avformat + sdl2 + xml`) to avoid macOS Qt plugin crashes like:
  `Could not find the Qt platform plugin "cocoa"`.
