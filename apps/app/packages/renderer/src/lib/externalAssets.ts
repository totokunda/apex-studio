import { ensureExternalAsset, type ExternalAssetFolder } from "@app/preload";

export function fsPathToAppUserDataUrl(fsPath: string): string {
  if (fsPath.startsWith("app://")) return fsPath;
  const posix = String(fsPath || "").replace(/\\/g, "/");
  const pathPart = posix.startsWith("/") ? posix : `/${posix}`;
  return `app://user-data${encodeURI(pathPart)}`;
}

export function getLastPathSegment(pathOrUrl: string | undefined): string | null {
  const raw = String(pathOrUrl || "").trim();
  if (!raw) return null;
  try {
    const u = new URL(raw);
    const seg = u.pathname.split("/").filter(Boolean).pop() || "";
    return seg || null;
  } catch {
    const clean = raw.split("?")[0]!.split("#")[0]!;
    const seg = clean.split("/").filter(Boolean).pop() || "";
    return seg || null;
  }
}

export function inferExternalFolderFromPath(pathOrUrl: string | undefined): ExternalAssetFolder {
  const raw = String(pathOrUrl || "").toLowerCase();
  if (raw.includes("/preprocessors/")) return "preprocessors";
  return "models";
}

export async function ensureExternalAssetUrl(options: {
  folder: ExternalAssetFolder;
  filePath: string;
}): Promise<string | null> {
  const { folder, filePath } = options;
  const res = await ensureExternalAsset({ folder, filePath });
  if (!res.ok) return null;
  return fsPathToAppUserDataUrl(res.absPath);
}

