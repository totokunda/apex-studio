import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

async function jobStatus(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:status", jobId);
}

async function jobCancel(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("jobs:cancel", jobId);
}

export { jobStatus, jobCancel };


