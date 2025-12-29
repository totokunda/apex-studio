import {
  runPostprocessor as runPostprocessorPreload,
  getPostprocessorStatus as getPostprocessorStatusPreload,
  cancelPostprocessor as cancelPostprocessorPreload,
} from "@app/preload";
import { wsClient } from "../ws/client";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export type PostprocessorMethod = "frame-interpolate";

export interface RunPostprocessorRequestBase {
  method: PostprocessorMethod;
  input_path: string;
  job_id?: string;
}

export interface RunFrameInterpolateRequest extends RunPostprocessorRequestBase {
  method: "frame-interpolate";
  target_fps: number;
  exp?: number;
  scale?: number;
}

export type RunPostprocessorRequest = RunFrameInterpolateRequest;

export interface JobResponse {
  job_id: string;
  status: string;
  message?: string;
}

export interface JobResult {
  job_id?: string;
  status?: string;
  result_path?: string;
  type?: string;
  error?: string;
  [key: string]: any;
}

export async function runPostprocessor(
  request: RunPostprocessorRequest,
): Promise<ConfigResponse<JobResponse>> {
  return await runPostprocessorPreload(request as any);
}

export async function getPostprocessorStatus(
  jobId: string,
): Promise<ConfigResponse<{ status: string; result?: JobResult }>> {
  return await getPostprocessorStatusPreload(jobId);
}

export async function cancelPostprocessor(
  jobId: string,
): Promise<ConfigResponse<any>> {
  // Unified cancel
  return await cancelPostprocessorPreload(jobId);
}

export async function connectJobWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  try {
    await wsClient.connect(`postprocessor:${jobId}`, `/ws/job/${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export async function disconnectJobWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  try {
    await wsClient.disconnect(`postprocessor:${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

export function subscribeToJobUpdates(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onUpdate(`postprocessor:${jobId}`, callback);
}

export function subscribeToJobStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onStatus(`postprocessor:${jobId}`, callback);
}

export function subscribeToJobErrors(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onError(`postprocessor:${jobId}`, callback);
}
