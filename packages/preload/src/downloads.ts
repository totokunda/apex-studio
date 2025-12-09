import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";
import { wsConnect, wsDisconnect, onWsUpdate, onWsStatus, onWsError } from "./ws.js";

async function startUnifiedDownload(request: {
  item_type: "component" | "lora" | "preprocessor";
  source: string | string[];
  save_path?: string;
  job_id?: string;
  manifest_id?: string;
  lora_name?: string;
  component?: string;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return await ipcRenderer.invoke("download:start", request);
}

async function resolveUnifiedDownload(request: {
  item_type: "component" | "lora" | "preprocessor";
  source: string | string[];
  save_path?: string;
}): Promise<
  ConfigResponse<{
    job_id: string;
    exists: boolean;
    running: boolean;
    downloaded: boolean;
    bucket: string;
    save_dir: string;
    source: string | string[];
  }>
> {
  return await ipcRenderer.invoke("download:resolve", request);
}

async function getUnifiedDownloadStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("download:status", jobId);
}

async function cancelUnifiedDownload(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("download:cancel", jobId);
}

function onUnifiedDownloadUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsUpdate(`download:${jobId}`, callback);
}

function onUnifiedDownloadStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsStatus(`download:${jobId}`, callback);
}

function onUnifiedDownloadError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsError(`download:${jobId}`, callback);
}

async function resolveUnifiedDownloadBatch(request: {
  item_type: "component" | "lora" | "preprocessor";
  sources: Array<string | string[]>;
  save_path?: string;
}): Promise<
  ConfigResponse<{
    results: Array<{
      job_id: string;
      exists: boolean;
      running: boolean;
      downloaded: boolean;
      bucket: string;
      save_dir: string;
      source: string | string[];
    }>;
  }>
> {
  return await ipcRenderer.invoke("download:resolve-batch", request);
}

async function connectUnifiedDownloadWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsConnect(`download:${jobId}`, `/ws/job/${jobId}`);
}

async function disconnectUnifiedDownloadWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsDisconnect(`download:${jobId}`);
}

async function deleteDownload(request: {
  path: string;
  item_type?: "component" | "lora" | "preprocessor";
  source?: string | string[];
  save_path?: string;
}): Promise<
  ConfigResponse<{
    path: string;
    status: string;
    removed_mapping?: boolean;
    unmarked?: boolean;
  }>
> {
  return await ipcRenderer.invoke("download:delete", request);
}

export {
  startUnifiedDownload,
  resolveUnifiedDownload,
  getUnifiedDownloadStatus,
  cancelUnifiedDownload,
  onUnifiedDownloadUpdate,
  onUnifiedDownloadStatus,
  onUnifiedDownloadError,
  resolveUnifiedDownloadBatch,
  connectUnifiedDownloadWebSocket,
  disconnectUnifiedDownloadWebSocket,
  deleteDownload,
};


