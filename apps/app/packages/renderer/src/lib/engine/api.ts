import {
  runEngine as runEnginePreload,
  getEngineStatus as getEngineStatusPreload,
  getEngineResult as getEngineResultPreload,
  cancelEngine as cancelEnginePreload,
} from "@app/preload";
import { wsClient } from "../ws/client";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface JobResponse {
  job_id: string;
  status: string;
  message?: string;
}

export interface JobResult {
  job_id: string;
  status: string;
  result_path?: string;
  type?: string;
  error?: string;
  result?: any;
}

export interface RunEngineRequest {
  manifest_id?: string;
  yaml_path?: string;
  inputs: Record<string, any>;
  selected_components?: Record<string, any>;
  job_id?: string;
  folder_uuid?: string;
}

export async function runEngine(
  request: RunEngineRequest,
): Promise<ConfigResponse<JobResponse>> {
  return await runEnginePreload(request);
}

export async function getEngineStatus(
  jobId: string,
): Promise<ConfigResponse<JobResult>> {
  return await getEngineStatusPreload(jobId);
}

export async function getEngineResult(
  jobId: string,
): Promise<ConfigResponse<JobResult>> {
  return await getEngineResultPreload(jobId);
}

export async function cancelEngine(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await cancelEnginePreload(jobId);
}

// Websocket helpers (reuse unified ws bridge with engine namespace key)
export async function connectJobWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  try {
    await wsClient.connect(`engine:${jobId}`, `/ws/job/${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export async function disconnectJobWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  try {
    await wsClient.disconnect(`engine:${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export function subscribeToJobUpdates(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onUpdate(`engine:${jobId}`, callback);
}

export function subscribeToJobStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onStatus(`engine:${jobId}`, callback);
}

export function subscribeToJobErrors(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onError(`engine:${jobId}`, callback);
}

export class EngineJob {
  public jobId: string;
  private unsubscribers: Array<() => void> = [];

  constructor(jobId: string) {
    this.jobId = jobId;
  }

  async connect(): Promise<void> {
    const result = await connectJobWebSocket(this.jobId);
    if (!result.success) {
      throw new Error(result.error || "Failed to connect to WebSocket");
    }
  }

  onUpdate(callback: (data: any) => void): void {
    const off = subscribeToJobUpdates(this.jobId, callback);
    this.unsubscribers.push(off);
  }

  onStatus(callback: (data: any) => void): void {
    const off = subscribeToJobStatus(this.jobId, callback);
    this.unsubscribers.push(off);
  }

  onError(callback: (data: any) => void): void {
    const off = subscribeToJobErrors(this.jobId, callback);
    this.unsubscribers.push(off);
  }

  async getStatus(): Promise<ConfigResponse<JobResult>> {
    return await getEngineStatus(this.jobId);
  }

  async getResult(): Promise<ConfigResponse<JobResult>> {
    return await getEngineResult(this.jobId);
  }

  async cancel(): Promise<ConfigResponse<any>> {
    return await cancelEngine(this.jobId);
  }

  async disconnect(): Promise<void> {
    this.unsubscribers.forEach((fn) => fn());
    this.unsubscribers = [];
    await disconnectJobWebSocket(this.jobId);
  }
}

export {
  useEngineJob,
  useActiveJobs,
  useJobProgress,
  useEngineJobActions,
} from "./hooks";
export { useEngineJobStore, useLoraJobStore } from "./store";
export type { JobProgress, LoraJobProgress } from "./store";
