import { ipcRenderer } from "electron";

export type AppUpdateEvent =
  | { type: "checking" }
  | { type: "available"; info: unknown }
  | { type: "not-available"; info: unknown }
  | { type: "progress"; progress: unknown }
  | { type: "downloaded"; info: unknown }
  | { type: "error"; message: string };

export type AppUpdateState = {
  status:
    | "idle"
    | "checking"
    | "available"
    | "not-available"
    | "downloading"
    | "downloaded"
    | "error";
  updateInfo?: unknown;
  progress?: unknown;
  errorMessage?: string;
  lastCheckedAt?: number;
};

const APP_UPDATE_EVENT_CHANNEL = "app-update:event";

export function onAppUpdateEvent(
  callback: (ev: AppUpdateEvent) => void,
): () => void {
  const handler = (_event: Electron.IpcRendererEvent, ev: AppUpdateEvent) => {
    callback(ev);
  };
  ipcRenderer.on(APP_UPDATE_EVENT_CHANNEL, handler);
  return () => {
    ipcRenderer.off(APP_UPDATE_EVENT_CHANNEL, handler);
  };
}

export async function getAppUpdateState(): Promise<AppUpdateState> {
  return await ipcRenderer.invoke("app-update:get-state");
}

export async function checkForAppUpdates(): Promise<unknown | null> {
  return await ipcRenderer.invoke("app-update:check");
}

export async function downloadAppUpdate(): Promise<{ ok: boolean }> {
  return await ipcRenderer.invoke("app-update:download");
}

export async function installAppUpdate(): Promise<{ ok: boolean }> {
  return await ipcRenderer.invoke("app-update:install");
}


