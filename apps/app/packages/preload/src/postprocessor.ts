import { fileURLToPath } from "node:url";
import type { ConfigResponse } from "./types.js";
import { wsConnect, wsDisconnect, onWsUpdate, onWsStatus, onWsError } from "./ws.js";
import { ipcRenderer } from "electron";

type RunPostprocessorRequest = {
  method: "frame-interpolate";
  input_path: string;
  target_fps: number;
  job_id?: string;
  exp?: number;
  scale?: number;
};

async function runPostprocessor(
  request: RunPostprocessorRequest,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  if (request.input_path.startsWith("file://")) {
    try {
      request.input_path = fileURLToPath(request.input_path);
    } catch {
      // ignore
    }
  }
  return await ipcRenderer.invoke("postprocessor:run", request);
}

async function getPostprocessorStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function cancelPostprocessor(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

async function connectPostprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsConnect(`postprocessor:${jobId}`, `/ws/job/${jobId}`);
}

async function disconnectPostprocessorWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await wsDisconnect(`postprocessor:${jobId}`);
}

function onPostprocessorWebSocketUpdate(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsUpdate(`postprocessor:${jobId}`, callback);
}

function onPostprocessorWebSocketStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsStatus(`postprocessor:${jobId}`, callback);
}

function onPostprocessorWebSocketError(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return onWsError(`postprocessor:${jobId}`, callback);
}

export {
  runPostprocessor,
  getPostprocessorStatus,
  cancelPostprocessor,
  connectPostprocessorWebSocket,
  disconnectPostprocessorWebSocket,
  onPostprocessorWebSocketUpdate,
  onPostprocessorWebSocketStatus,
  onPostprocessorWebSocketError,
};

export type { RunPostprocessorRequest };


