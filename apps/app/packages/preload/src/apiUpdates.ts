import { ipcRenderer } from "electron";

export type ApiUpdateEvent =
  | { type: "checking" }
  | { type: "available"; info: unknown }
  | { type: "not-available"; info: unknown }
  | { type: "updating" }
  | { type: "updated"; info?: unknown }
  | { type: "error"; message: string }
  | { type: "allow-nightly-changed"; allowNightly: boolean };

export type ApiUpdateState = {
  status:
    | "idle"
    | "checking"
    | "available"
    | "not-available"
    | "updating"
    | "updated"
    | "error";
  updateInfo?: unknown;
  errorMessage?: string;
  lastCheckedAt?: number;
  allowNightly: boolean;
};

const API_UPDATE_EVENT_CHANNEL = "api-update:event";

export function onApiUpdateEvent(
  callback: (ev: ApiUpdateEvent) => void,
): () => void {
  const handler = (_event: Electron.IpcRendererEvent, ev: ApiUpdateEvent) => {
    callback(ev);
  };
  ipcRenderer.on(API_UPDATE_EVENT_CHANNEL, handler);
  return () => {
    ipcRenderer.off(API_UPDATE_EVENT_CHANNEL, handler);
  };
}

export async function getApiUpdateState(): Promise<ApiUpdateState> {
  return await ipcRenderer.invoke("api-update:get-state");
}

export async function setApiAllowNightlyUpdates(
  allowNightly: boolean,
): Promise<{ ok: boolean; allowNightly: boolean }> {
  return await ipcRenderer.invoke("api-update:set-allow-nightly", allowNightly);
}

export async function checkForApiUpdates(): Promise<unknown | null> {
  return await ipcRenderer.invoke("api-update:check");
}

export async function applyApiUpdate(): Promise<{ ok: boolean; message?: string }> {
  return await ipcRenderer.invoke("api-update:apply");
}

