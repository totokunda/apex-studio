export const MIN_DURATION = 5;
export const BASE_LONG_SIDE = 600; // world units
export const TIMELINE_DURATION_SECONDS = 60;
export const DEFAULT_FPS = 24;
export const MAX_DURATION = TIMELINE_DURATION_SECONDS * DEFAULT_FPS;
export const FRAMES_CACHE_MAX_BYTES = 512 * 1024 * 1024;
export const BASE_MEDIA_DIR = "media"; // subfolder inside AppLocalData
export const VIDEO_EXTS = [
  "mp4",
  "mov",
  "avi",
  "mkv",
  "webm",
  "flv",
  "wmv",
  "mpg",
  "mpeg",
  "m4v",
];
export const IMAGE_EXTS = [
  "jpg",
  "jpeg",
  "png",
  "gif",
  "bmp",
  "tiff",
  "ico",
  "webp",
];
export const AUDIO_EXTS = [
  "mp3",
  "wav",
  "ogg",
  "m4a",
  "aac",
  "flac",
  "wma",
  "m4b",
  "m4r",
  "m4p",
];
export const DECODER_STALE_MS = 30_000; // dispose if not used for 30s
export const PREFETCH_BACK = 32; // frames behind
export const PREFETCH_AHEAD = 32; // frames ahead
