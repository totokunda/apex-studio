import { ipcRenderer } from "electron";
import { fileURLToPath } from "node:url";
import type { ConfigResponse } from "./types.js";

// Preprocessor registry/listing
async function listPreprocessors(
  checkDownloaded: boolean = true,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:list", checkDownloaded);
}

async function deletePreprocessor(name: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:delete", name);
}

async function getPreprocessor<T = any>(name: string): Promise<ConfigResponse<T>> {
  return await ipcRenderer.invoke("preprocessor:get", name);
}

async function downloadPreprocessor(
  name: string,
  jobId?: string,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return await ipcRenderer.invoke("preprocessor:download", name, jobId);
}

// Running preprocessors
async function runPreprocessor(request: {
  preprocessor_name: string;
  input_path: string;
  job_id?: string;
  download_if_needed?: boolean;
  params?: Record<string, any>;
  start_frame?: number;
  end_frame?: number;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  if (request.input_path.startsWith("file://")) {
    try {
      request.input_path = fileURLToPath(request.input_path);
    } catch {
      // fall through
    }
  }
  return await ipcRenderer.invoke("preprocessor:run", request);
}

async function getPreprocessorStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function getPreprocessorResult(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:result", jobId);
}

async function connectPreprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:connect-ws", jobId);
}

async function disconnectPreprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("preprocessor:disconnect-ws", jobId);
}

function onPreprocessorWebSocketUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-update:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`preprocessor:ws-update:${jobId}`, listener);
}

function onPreprocessorWebSocketStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-status:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`preprocessor:ws-status:${jobId}`, listener);
}

function onPreprocessorWebSocketError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  const listener = (_event: unknown, data: any) => callback(data);
  ipcRenderer.on(`preprocessor:ws-error:${jobId}`, listener);
  return () =>
    ipcRenderer.removeListener(`preprocessor:ws-error:${jobId}`, listener);
}

async function cancelPreprocessor(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

export {
  listPreprocessors,
  deletePreprocessor,
  getPreprocessor,
  downloadPreprocessor,
  runPreprocessor,
  getPreprocessorStatus,
  getPreprocessorResult,
  connectPreprocessorWebSocket,
  disconnectPreprocessorWebSocket,
  onPreprocessorWebSocketUpdate,
  onPreprocessorWebSocketStatus,
  onPreprocessorWebSocketError,
  cancelPreprocessor,
};


