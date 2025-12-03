import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

async function listRayJobs(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:list");
}

async function getRayJob(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:get", jobId);
}

async function cancelRayJob(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:cancel", jobId);
}

async function cancelAllRayJobs(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("ray:jobs:cancel-all");
}

export { listRayJobs, getRayJob, cancelRayJob, cancelAllRayJobs };


