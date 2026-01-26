import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

async function downloadComponents(
  paths: string[],
  savePath?: string,
  jobId?: string,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return await ipcRenderer.invoke(
    "components:download",
    paths,
    savePath,
    jobId,
  );
}

async function deleteComponent(
  targetPath: string,
): Promise<ConfigResponse<{ status: string; path: string }>> {
  return await ipcRenderer.invoke("components:delete", targetPath);
}

async function getComponentsStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function cancelComponents(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

async function connectComponentsWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("components:connect-ws", jobId);
}

async function disconnectComponentsWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("components:disconnect-ws", jobId);
}

function onComponentsWebSocketUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`components:ws-update:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`components:ws-update:${jobId}`, listener);
}

function onComponentsWebSocketStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`components:ws-status:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`components:ws-status:${jobId}`, listener);
}

function onComponentsWebSocketError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`components:ws-error:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`components:ws-error:${jobId}`, listener);
}

export {
  downloadComponents,
  deleteComponent,
  getComponentsStatus,
  cancelComponents,
  connectComponentsWebSocket,
  disconnectComponentsWebSocket,
  onComponentsWebSocketUpdate,
  onComponentsWebSocketStatus,
  onComponentsWebSocketError,
};


