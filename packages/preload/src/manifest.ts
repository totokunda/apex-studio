import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

async function listManifestModelTypes(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:types");
}

async function listManifests(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:list");
}

async function listManifestsByModel(
  model: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:list-by-model", model);
}

async function listManifestsByType(
  modelType: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:list-by-type", modelType);
}

async function listManifestsByModelAndType(
  model: string,
  modelType: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke(
    "manifest:list-by-model-and-type",
    model,
    modelType,
  );
}

async function getManifest<T = any>(
  manifestId: string,
): Promise<ConfigResponse<T>> {
  return await ipcRenderer.invoke("manifest:get", manifestId);
}

async function getManifestPart<T = any>(
  manifestId: string,
  pathDot?: string,
): Promise<ConfigResponse<T>> {
  const qp =
    typeof pathDot === "string" && pathDot.length > 0 ? pathDot : undefined;
  return await ipcRenderer.invoke("manifest:get-part", manifestId, qp);
}

async function validateAndRegisterCustomModelPath(
  manifestId: string,
  componentIndex: number,
  name: string | undefined,
  path: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:validate-custom-model-path", {
    manifest_id: manifestId,
    component_index: componentIndex,
    name,
    path,
  });
}

async function deleteCustomModelPath(
  manifestId: string,
  componentIndex: number,
  path: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:delete-custom-model-path", {
    manifest_id: manifestId,
    component_index: componentIndex,
    path,
  });
}

async function updateManifestLoraScale(
  manifestId: string,
  loraIndex: number,
  scale: number,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:update-lora-scale", {
    manifest_id: manifestId,
    lora_index: loraIndex,
    scale,
  });
}

async function updateManifestLoraName(
  manifestId: string,
  loraIndex: number,
  name: string,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:update-lora-name", {
    manifest_id: manifestId,
    lora_index: loraIndex,
    name,
  });
}

async function deleteManifestLora(
  manifestId: string,
  loraIndex: number,
): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("manifest:delete-lora", {
    manifest_id: manifestId,
    lora_index: loraIndex,
  });
}

export {
  listManifestModelTypes,
  listManifests,
  listManifestsByModel,
  listManifestsByType,
  listManifestsByModelAndType,
  getManifest,
  getManifestPart,
  validateAndRegisterCustomModelPath,
  deleteCustomModelPath,
  updateManifestLoraScale,
  updateManifestLoraName,
  deleteManifestLora,
};


