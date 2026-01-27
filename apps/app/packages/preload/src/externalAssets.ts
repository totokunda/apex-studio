import { ipcRenderer } from "electron";

export type ExternalAssetFolder = "models" | "preprocessors";

export type EnsureExternalAssetRequest = {
  folder: ExternalAssetFolder;
  /**
   * Relative path within the folder, e.g. "flux-dev.png" or "subdir/foo.mp4".
   * Must NOT be absolute and must NOT contain "..".
   */
  filePath: string;
};

export type EnsureExternalAssetResponse =
  | { ok: true; absPath: string }
  | { ok: false; error: string };

export async function ensureExternalAsset(
  request: EnsureExternalAssetRequest,
): Promise<EnsureExternalAssetResponse> {
  try {
    const res = (await ipcRenderer.invoke(
      "external-assets:ensure",
      request,
    )) as any;
    if (res && res.ok === true && typeof res.absPath === "string") {
      return { ok: true, absPath: res.absPath };
    }
    return {
      ok: false,
      error: typeof res?.error === "string" ? res.error : "Failed to ensure asset",
    };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

