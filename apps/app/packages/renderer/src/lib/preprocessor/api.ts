import {
  listPreprocessors as listPreprocessorsPreload,
  getPreprocessor as getPreprocessorPreload,
  downloadPreprocessor as downloadPreprocessorPreload,
  runPreprocessor as runPreprocessorPreload,
  getPreprocessorStatus as getPreprocessorStatusPreload,
  getPreprocessorResult as getPreprocessorResultPreload,
  cancelPreprocessor as cancelPreprocessorPreload,
  deletePreprocessor as deletePreprocessorPreload,
} from "@app/preload";
import { wsClient } from "../ws/client";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

type PreprocessorParameterType = "int" | "float" | "bool" | "str" | "category";

interface ParameterOption {
  name: string;
  value: any;
}

export interface PreprocessorParameter {
  display_name?: string;
  name: string;
  type: PreprocessorParameterType;
  default?: any;
  required?: boolean;
  description?: string;
  min?: number;
  max?: number;
  options?: ParameterOption[];
}

export interface PreprocessorFile {
  path: string;
  size_bytes: number;
}

export interface Preprocessor {
  type: "preprocessor";
  name: string;
  id: string;
  description?: string;
  category: string;
  supports_video?: boolean;
  supports_image?: boolean;
  parameters?: PreprocessorParameter[];
  files?: PreprocessorFile[];
  is_downloaded?: boolean;
  download_size?: string;
  processor_url?: string;
}

export interface PreprocessorList {
  count: number;
  preprocessors: Preprocessor[];
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
  preprocessor?: string;
  error?: string;
  result?: any;
}

export interface RunPreprocessorRequest {
  preprocessor_name: string;
  input_path: string;
  job_id?: string;
  download_if_needed?: boolean;
  params?: Record<string, any>;
  start_frame?: number;
  end_frame?: number;
}

/**
 * List all available preprocessors
 */
export async function listPreprocessors(
  checkDownloaded: boolean = true,
): Promise<ConfigResponse<PreprocessorList>> {
  return await listPreprocessorsPreload(checkDownloaded);
}

/**
 * Get detailed information about a specific preprocessor
 */
export async function getPreprocessor(
  name: string,
): Promise<ConfigResponse<Preprocessor>> {
  return await getPreprocessorPreload(name);
}

/**
 * Download a preprocessor model
 * Returns a job_id that can be used to track download progress via WebSocket
 */
export async function downloadPreprocessor(
  name: string,
  jobId?: string,
): Promise<ConfigResponse<JobResponse>> {
  return await downloadPreprocessorPreload(name, jobId);
}

/**
 * Run a preprocessor on input media
 * Returns a job_id that can be used to track processing progress via WebSocket
 */
export async function runPreprocessor(
  request: RunPreprocessorRequest,
): Promise<ConfigResponse<JobResponse>> {
  return await runPreprocessorPreload(request);
}

/**
 * Get the current status of a preprocessing job
 */
export async function getPreprocessorStatus(
  jobId: string,
): Promise<ConfigResponse<JobResult>> {
  return await getPreprocessorStatusPreload(jobId);
}

/**
 * Get the result of a completed preprocessing job
 */
export async function getPreprocessorResult(
  jobId: string,
): Promise<ConfigResponse<JobResult>> {
  return await getPreprocessorResultPreload(jobId);
}

/**
 * Connect to WebSocket for real-time job updates
 */
export async function connectJobWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  try {
    await wsClient.connect(`preprocessor:${jobId}`, `/ws/job/${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

/**
 * Disconnect from WebSocket
 */
export async function disconnectJobWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  try {
    await wsClient.disconnect(`preprocessor:${jobId}`);
    return { success: true, data: { jobId } };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

/**
 * Subscribe to WebSocket updates for a job
 * Returns an unsubscribe function
 */
export function subscribeToJobUpdates(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onUpdate(`preprocessor:${jobId}`, callback);
}

/**
 * Subscribe to WebSocket connection status changes
 * Returns an unsubscribe function
 */
export function subscribeToJobStatus(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onStatus(`preprocessor:${jobId}`, callback);
}

/**
 * Subscribe to WebSocket errors
 * Returns an unsubscribe function
 */
export function subscribeToJobErrors(
  jobId: string,
  callback: (data: any) => void,
): () => void {
  return wsClient.onError(`preprocessor:${jobId}`, callback);
}

/**
 * Cancel a preprocessor job
/**
 * Cancel a preprocessor job
 */
export async function cancelPreprocessor(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await cancelPreprocessorPreload(jobId);
}

/**
 * Delete a preprocessor and its downloaded files
 */
export async function deletePreprocessor(
  name: string,
): Promise<ConfigResponse<any>> {
  return await deletePreprocessorPreload(name);
}

/**
 * Helper class to manage a preprocessing job with WebSocket tracking
 */
export class PreprocessorJob {
  public jobId: string;
  private unsubscribers: Array<() => void> = [];

  constructor(jobId: string) {
    this.jobId = jobId;
  }

  /**
   * Connect and start listening for updates
   */
  async connect(): Promise<void> {
    const result = await connectJobWebSocket(this.jobId);
    if (!result.success) {
      throw new Error(result.error || "Failed to connect to WebSocket");
    }
  }

  /**
   * Subscribe to job updates
   */
  onUpdate(callback: (data: any) => void): void {
    const unsubscribe = subscribeToJobUpdates(this.jobId, callback);
    this.unsubscribers.push(unsubscribe);
  }

  /**
   * Subscribe to connection status
   */
  onStatus(callback: (data: any) => void): void {
    const unsubscribe = subscribeToJobStatus(this.jobId, callback);
    this.unsubscribers.push(unsubscribe);
  }

  /**
   * Subscribe to errors
   */
  onError(callback: (data: any) => void): void {
    const unsubscribe = subscribeToJobErrors(this.jobId, callback);
    this.unsubscribers.push(unsubscribe);
  }

  /**
   * Get current status
   */
  async getStatus(): Promise<ConfigResponse<JobResult>> {
    return await getPreprocessorStatus(this.jobId);
  }

  /**
   * Get result (if complete)
   */
  async getResult(): Promise<ConfigResponse<JobResult>> {
    return await getPreprocessorResult(this.jobId);
  }

  /**
   * Cancel the job
   */
  async cancel(): Promise<ConfigResponse<any>> {
    return await cancelPreprocessor(this.jobId);
  }

  /**
   * Disconnect and cleanup
   */
  async disconnect(): Promise<void> {
    // Unsubscribe from all listeners
    this.unsubscribers.forEach((unsubscribe) => unsubscribe());
    this.unsubscribers = [];

    // Disconnect WebSocket
    await disconnectJobWebSocket(this.jobId);
  }
}

// Export hooks for convenience
export {
  usePreprocessorJob,
  useActiveJobs,
  useJobProgress,
  usePreprocessorJobActions,
} from "./hooks";
export { usePreprocessorJobStore } from "./store";
export type { JobProgress } from "./store";
