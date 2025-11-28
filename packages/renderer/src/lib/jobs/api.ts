import {
  listRayJobs as listRayJobsPreload,
  getRayJob as getRayJobPreload,
  cancelRayJob as cancelRayJobPreload,
  cancelAllRayJobs as cancelAllRayJobsPreload,
} from "@app/preload";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export type RayJobLatest = {
  progress?: number | null;
  message?: string | null;
  status?: string;
  metadata?: Record<string, any>;
};

export type RayJobStatus = {
  job_id: string;
  status: string;
  category?: "download" | "processor" | "engine" | "other" | string;
  message?: string;
  error?: string;
  result?: any;
  latest?: RayJobLatest | null;
};

export type ListRayJobsResponse = {
  jobs: RayJobStatus[];
};

export async function fetchRayJobs(): Promise<RayJobStatus[]> {
  const res = await listRayJobsPreload();
  if (!res.success || !res.data) return [];
  const data = res.data as ListRayJobsResponse;
  return Array.isArray(data.jobs) ? data.jobs : [];
}

export async function fetchRayJob(jobId: string): Promise<RayJobStatus | null> {
  const res = await getRayJobPreload(jobId);
  if (!res.success || !res.data) return null;
  return res.data as RayJobStatus;
}

export async function cancelRayJob(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await cancelRayJobPreload(jobId);
}

export async function cancelAllRayJobs(): Promise<ConfigResponse<any>> {
  return await cancelAllRayJobsPreload();
}
