import {
  startUnifiedDownload as startUnifiedDownloadPreload,
  resolveUnifiedDownload as resolveUnifiedDownloadPreload,
  resolveUnifiedDownloadBatch as resolveUnifiedDownloadBatchPreload,
  getUnifiedDownloadStatus as getUnifiedDownloadStatusPreload,
  cancelUnifiedDownload as cancelUnifiedDownloadPreload,
  onUnifiedDownloadUpdate as onUnifiedDownloadUpdatePreload,
  onUnifiedDownloadStatus as onUnifiedDownloadStatusPreload,
  onUnifiedDownloadError as onUnifiedDownloadErrorPreload,
  connectUnifiedDownloadWebSocket as connectUnifiedDownloadWebSocketPreload,
  disconnectUnifiedDownloadWebSocket as disconnectUnifiedDownloadWebSocketPreload,
  deleteDownload as deleteDownloadPreload,
} from "@app/preload";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export type UnifiedBucket = "component" | "lora" | "preprocessor";

export interface UnifiedDownloadWsMetadata {
  bucket?: UnifiedBucket;
  label?: string;
  filename?: string;
  downloaded?: number;
  total?: number;
  // Allow future metadata without breaking the type
  [key: string]: any;
}

export interface UnifiedDownloadWsUpdate {
  progress?: number | null;
  message?: string;
  status: "processing" | "complete" | "error" | "canceled" | string;
  metadata?: UnifiedDownloadWsMetadata;
}

export interface UnifiedJobStatus {
  job_id: string;
  status: string;
  latest?: UnifiedDownloadWsUpdate;
  result?: any;
  error?: string;
  message?: string;
}

export interface WsConnectionStatus {
  status: "connected" | "disconnected" | string;
}

export async function startUnifiedDownload(request: {
  item_type: "component" | "lora" | "preprocessor";
  source: string | string[];
  save_path?: string;
  job_id?: string;
  manifest_id?: string;
  lora_name?: string;
  component?: string;
  index?: number;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  // remove index from request
  const { index, ...rest } = request;
  return await startUnifiedDownloadPreload(rest);
}

export async function resolveUnifiedDownload(request: {
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
  return await resolveUnifiedDownloadPreload(request);
}

export async function resolveUnifiedDownloadBatch(request: {
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
  return await resolveUnifiedDownloadBatchPreload(request);
}

export async function getUnifiedDownloadStatus(
  jobId: string,
): Promise<ConfigResponse<UnifiedJobStatus>> {
  return await getUnifiedDownloadStatusPreload(jobId);
}

export async function cancelUnifiedDownload(
  jobId: string,
): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  return (await cancelUnifiedDownloadPreload(jobId)) as ConfigResponse<{
    job_id: string;
    status: string;
    message?: string;
  }>;
}

export function onUnifiedDownloadUpdate(
  jobId: string,
  callback: (data: UnifiedDownloadWsUpdate) => void,
): () => void {
  return onUnifiedDownloadUpdatePreload(
    jobId,
    callback as unknown as (data: any) => void,
  );
}

export function onUnifiedDownloadStatus(
  jobId: string,
  callback: (data: WsConnectionStatus) => void,
): () => void {
  return onUnifiedDownloadStatusPreload(
    jobId,
    callback as unknown as (data: any) => void,
  );
}

export function onUnifiedDownloadError(
  jobId: string,
  callback: (data: { error: string; raw?: any }) => void,
): () => void {
  return onUnifiedDownloadErrorPreload(
    jobId,
    callback as unknown as (data: any) => void,
  );
}

export async function connectUnifiedDownloadWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await connectUnifiedDownloadWebSocketPreload(jobId);
}

export async function disconnectUnifiedDownloadWebSocket(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await disconnectUnifiedDownloadWebSocketPreload(jobId);
}

export async function deleteDownload(request: {
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
  return await deleteDownloadPreload(request);
}
