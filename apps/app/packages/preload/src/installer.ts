import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

export type BundleSource =
  | { kind: "local"; path: string }
  | { kind: "remote"; url: string; assetName?: string };

export type InstallerProgressEvent =
  | {
      phase: "download";
      downloadedBytes: number;
      totalBytes?: number;
      percent?: number;
      message?: string;
    }
  | {
      phase: "extract";
      processedBytes?: number;
      totalBytes?: number;
      percent?: number;
      entryName?: string;
      entriesExtracted?: number;
      message?: string;
    }
  | { phase: "status"; message: string };

export async function extractServerBundle(request: {
  source: BundleSource;
  destinationDir: string;
  jobId?: string;
}): Promise<ConfigResponse<{ extractedTo: string }>> {
  return await ipcRenderer.invoke("installer:extract-server-bundle", request);
}

export function onInstallerProgress(
  jobId: string,
  callback: (ev: InstallerProgressEvent) => void,
): () => void {
  const channel = `installer:progress:${jobId}`;
  const handler = (_event: Electron.IpcRendererEvent, ev: InstallerProgressEvent) => {
    callback(ev);
  };
  ipcRenderer.on(channel, handler);
  return () => {
    ipcRenderer.off(channel, handler);
  };
}

export async function ensureFfmpegInstalled(): Promise<
  ConfigResponse<{
    installed: boolean;
    ffmpegPath: string;
    ffprobePath: string;
    method: "already-present" | "copied-from-ffmpeg-static";
  }>
> {
  return await ipcRenderer.invoke("installer:ensure-ffmpeg");
}


