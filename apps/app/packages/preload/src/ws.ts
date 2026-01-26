import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

async function wsConnect(
  key: string,
  pathOrUrl: string,
): Promise<ConfigResponse<{ key: string }>> {
  return await ipcRenderer.invoke("ws:connect", { key, pathOrUrl });
}

async function wsDisconnect(
  key: string,
): Promise<ConfigResponse<{ message: string }>> {
  return await ipcRenderer.invoke("ws:disconnect", key);
}

async function wsStatus(
  key: string,
): Promise<ConfigResponse<{ key: string; connected: boolean }>> {
  return await ipcRenderer.invoke("ws:status", key);
}

function onWsUpdate(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-update:${key}`;
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function onWsStatus(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-status:${key}`;
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

function onWsError(key: string, callback: (data: any) => void): () => void {
  const channel = `ws-error:${key}`;
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

export { wsConnect, wsDisconnect, wsStatus, onWsUpdate, onWsStatus, onWsError };


