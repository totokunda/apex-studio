export const VIDEO_EXTS = new Set([
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
]);

export const IMAGE_EXTS = new Set([
  "jpg",
  "jpeg",
  "png",
  "gif",
  "bmp",
  "tiff",
  "ico",
  "webp",
]);

export const AUDIO_EXTS = new Set([
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
]);

export function getLowercaseExtension(filename: string): string {
  const idx = filename.lastIndexOf(".");
  if (idx === -1) return "";
  // remove any query or hash params
  const filenameWithoutParams = filename.split("?")[0].split("#")[0];
  return filenameWithoutParams.slice(idx + 1).toLowerCase();
}

export function isVideoExt(ext: string): boolean {
  return VIDEO_EXTS.has(ext);
}

export function isImageExt(ext: string): boolean {
  return IMAGE_EXTS.has(ext);
}

export function isAudioExt(ext: string): boolean {
  return AUDIO_EXTS.has(ext);
}
