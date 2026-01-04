import { ipcRenderer } from "electron";

export type LauncherStatus = {
  backendUrl: string | null;
  backendHealthy: boolean;
  pythonHealthy: boolean;
  pythonState: {
    status: "stopped" | "starting" | "running" | "stopping" | "error";
    pid?: number;
    port: number;
    host: string;
    error?: string;
    restartCount: number;
  };
  runtimeInfo: {
    available: boolean;
    mode: "dev" | "bundled" | "installed" | "missing";
    installedApiPath: string | null;
    bundleRoot: string | null;
    pythonExe: string | null;
    reason?: string;
  };
  runtimeVerified: { ok: boolean; reason?: string };
  canStartLocal: boolean;
  hasBackend: boolean;
  backendStarting: boolean;
  shouldShowInstaller: boolean;
  lastError?: string | null;
};

/**
 * Tell the main process to open the main app window and close the launcher window.
 */
export async function launchMainWindow(): Promise<{ ok: true } | { ok: false; error: string }> {
  try {
    await ipcRenderer.invoke("launcher:launch");
    return { ok: true };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "Failed to launch main window",
    };
  }
}

export async function getLauncherStatus(): Promise<
  { success: true; data: LauncherStatus } | { success: false; error: string }
> {
  try {
    const res = await ipcRenderer.invoke("launcher:status");
    if (res?.success && res.data) {
      return { success: true, data: res.data as LauncherStatus };
    }
    return {
      success: false,
      error: String(res?.error || "Failed to get launcher status"),
    };
  } catch (e) {
    return {
      success: false,
      error: e instanceof Error ? e.message : "Failed to get launcher status",
    };
  }
}

export function onLauncherStatusChange(
  callback: (status: LauncherStatus) => void,
): () => void {
  const handler = (_event: Electron.IpcRendererEvent, status: LauncherStatus) => {
    callback(status);
  };
  ipcRenderer.on("launcher:status-changed", handler);
  return () => {
    ipcRenderer.off("launcher:status-changed", handler);
  };
}

export async function startLauncherStatusWatch(opts?: {
  intervalMs?: number;
}): Promise<{ success: true } | { success: false; error: string }> {
  try {
    const res = await ipcRenderer.invoke(
      "launcher:watch-start",
      opts?.intervalMs,
    );
    if (res?.success) return { success: true };
    return {
      success: false,
      error: String(res?.error || "Failed to start launcher status watch"),
    };
  } catch (e) {
    return {
      success: false,
      error:
        e instanceof Error ? e.message : "Failed to start launcher status watch",
    };
  }
}

export async function stopLauncherStatusWatch(): Promise<
  { success: true } | { success: false; error: string }
> {
  try {
    const res = await ipcRenderer.invoke("launcher:watch-stop");
    if (res?.success) return { success: true };
    return {
      success: false,
      error: String(res?.error || "Failed to stop launcher status watch"),
    };
  } catch (e) {
    return {
      success: false,
      error:
        e instanceof Error ? e.message : "Failed to stop launcher status watch",
    };
  }
}


