import { ipcRenderer } from "electron";
import { fileURLToPath } from "node:url";
import type { ConfigResponse } from "./types.js";

function normalizeInputsForIpc(obj: any): any {
  if (obj == null) return obj;
  if (typeof obj === "string") {
    if (obj.startsWith("file://")) {
      try {
        return fileURLToPath(obj);
      } catch {
        return obj;
      }
    }
    return obj;
  }
  if (Array.isArray(obj)) return obj.map(normalizeInputsForIpc);
  if (typeof obj === "object") {
    const out: any = {};
    for (const [k, v] of Object.entries(obj)) {
      out[k] = normalizeInputsForIpc(v);
    }
    return out;
  }
  return obj;
}

async function runEngine(request: {
  manifest_id?: string;
  yaml_path?: string;
  inputs: Record<string, any>;
  selected_components?: Record<string, any>;
  job_id?: string;
}): Promise<
  ConfigResponse<{ job_id: string; status: string; message?: string }>
> {
  const payload = {
    ...request,
    inputs: normalizeInputsForIpc(request.inputs || {}),
    selected_components: normalizeInputsForIpc(
      request.selected_components || {},
    ),
  };
  return await ipcRenderer.invoke("engine:run", payload);
}

async function getEngineStatus(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("engine:status", jobId);
}

async function getEngineResult(
  jobId: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("engine:result", jobId);
}

async function cancelEngine(jobId: string): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("engine:cancel", jobId);
}

export { runEngine, getEngineStatus, getEngineResult, cancelEngine };


