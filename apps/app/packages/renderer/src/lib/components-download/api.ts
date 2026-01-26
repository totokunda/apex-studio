export type ConfigResponse<T> = {
  success: boolean;
  data?: T;
  error?: string;
};

import {
  downloadComponents as downloadComponentsPreload,
  deleteComponent as deleteComponentPreload,
  getComponentsStatus as getComponentsStatusPreload,
  cancelComponents as cancelComponentsPreload,
} from "@app/preload";
import { wsClient } from "../ws/client";

export async function downloadComponents(
  paths: string[],
  savePath?: string,
  jobId?: string,
) {
  return await downloadComponentsPreload(paths, savePath, jobId);
}

export async function deleteComponent(path: string) {
  return await deleteComponentPreload(path);
}

export async function getComponentsStatus(jobId: string) {
  return await getComponentsStatusPreload(jobId);
}

export async function cancelComponents(jobId: string) {
  return await cancelComponentsPreload(jobId);
}

export async function connectComponentsWebSocket(jobId: string) {
  try {
    await wsClient.connect(`components:${jobId}`, `/ws/job/${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export async function disconnectComponentsWebSocket(jobId: string) {
  try {
    await wsClient.disconnect(`components:${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export function onComponentsWebSocketUpdate(
  jobId: string,
  cb: (data: any) => void,
) {
  return wsClient.onUpdate(`components:${jobId}`, cb);
}

export function onComponentsWebSocketStatus(
  jobId: string,
  cb: (data: any) => void,
) {
  return wsClient.onStatus(`components:${jobId}`, cb);
}

export function onComponentsWebSocketError(
  jobId: string,
  cb: (data: any) => void,
) {
  return wsClient.onError(`components:${jobId}`, cb);
}

// Job status payload returned from getComponentsStatus
export interface ComponentsJobStatus {
  status: string;
  latest?: any;
  message?: string;
}

/**
 * Helper class to manage a components download job with WebSocket tracking
 */
export class ComponentsDownloadJob {
  public jobId: string;
  private unsubscribers: Array<() => void> = [];

  constructor(jobId: string) {
    this.jobId = jobId;
  }

  async connect(): Promise<void> {
    const result = await connectComponentsWebSocket(this.jobId);
    if (!result.success) {
      throw new Error(result.error || "Failed to connect to WebSocket");
    }
  }

  onUpdate(callback: (data: any) => void): void {
    const unsubscribe = onComponentsWebSocketUpdate(this.jobId, callback);
    this.unsubscribers.push(unsubscribe);
  }

  onStatus(callback: (data: any) => void): void {
    const unsubscribe = onComponentsWebSocketStatus(this.jobId, callback);
    this.unsubscribers.push(unsubscribe);
  }

  onError(callback: (data: any) => void): void {
    const unsubscribe = onComponentsWebSocketError(this.jobId, callback);
    this.unsubscribers.push(unsubscribe);
  }

  async getStatus(): Promise<ConfigResponse<ComponentsJobStatus>> {
    return await getComponentsStatus(this.jobId);
  }

  async cancel(): Promise<ConfigResponse<any>> {
    return await cancelComponents(this.jobId);
  }

  async disconnect(): Promise<void> {
    this.unsubscribers.forEach((unsubscribe) => unsubscribe());
    this.unsubscribers = [];
    await disconnectComponentsWebSocket(this.jobId);
  }
}
