import { homedir } from "node:os";
import { join } from "node:path";
import fs from "node:fs";

const APP_FOLDER_CANDIDATES = ["apex-studio", "Apex Studio"];

function firstExisting(paths: string[]): string | undefined {
  for (const p of paths) {
    if (fs.existsSync(p)) return p;
  }
  return undefined;
}

export function guessUserDataDir(): string {
  const explicit = process.env.APEX_USER_DATA_DIR;
  if (explicit && explicit.length > 0) return explicit;

  const home = homedir();
  const platform = process.platform;
  if (platform === "win32") {
    const base = process.env.APPDATA || join(home, "AppData", "Roaming");
    const candidates = APP_FOLDER_CANDIDATES.map((n) => join(base, n));
    return firstExisting(candidates) || candidates[0];
  }
  if (platform === "darwin") {
    const base = join(home, "Library", "Application Support");
    const candidates = APP_FOLDER_CANDIDATES.map((n) => join(base, n));
    return firstExisting(candidates) || candidates[0];
  }
  // linux and others
  const base = join(home, ".config");
  const candidates = APP_FOLDER_CANDIDATES.map((n) => join(base, n));
  return firstExisting(candidates) || candidates[0];
}

export function getMediaRootAbsolute(): string {
  const explicit = process.env.APEX_MEDIA_ROOT;
  if (explicit && explicit.length > 0) return explicit;
  return join(guessUserDataDir(), "media");
}
