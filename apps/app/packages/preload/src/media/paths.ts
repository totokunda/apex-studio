import { join, dirname } from "node:path";

export const BASE_MEDIA_DIR = "media";

export function resolveMediaRootFromOriginalFile(
  originalFileAbs: string,
): string {
  // .../media/original/<file> => media root is parent of the original dir
  return dirname(dirname(originalFileAbs));
}

export function originalDir(mediaRootAbs: string): string {
  return join(mediaRootAbs, "original");
}

export function converted24Dir(mediaRootAbs: string): string {
  return join(mediaRootAbs, "24");
}

export function symlinksDir(mediaRootAbs: string): string {
  return join(mediaRootAbs, "symlinks");
}

export function proxyDir(mediaRootAbs: string): string {
  return join(mediaRootAbs, "proxy");
}

export function ensureDirSync(fs: typeof import("node:fs"), dirPath: string) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

export function fileNameFromPath(p: string): string {
  const parts = p.split(/[/\\]/g);
  return parts[parts.length - 1] ?? p;
}

export function pathDirName(p: string): string {
  return dirname(p);
}
