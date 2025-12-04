import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

function getActiveProjectId(): Promise<string | number | undefined | null> {
  return ipcRenderer.invoke("settings:get-active-project-id");
}

function setActiveProjectId(projectId: string | number | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-active-project-id", projectId);
}

type PathsPayload = {
  cachePath?: string | null;
  componentsPath?: string | null;
  configPath?: string | null;
  loraPath?: string | null;
  preprocessorPath?: string | null;
  postprocessorPath?: string | null;
};

function getCachePathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-cache-path");
}

function setCachePathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-cache-path", path);
}

function getComponentsPathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-components-path");
}

function setComponentsPathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-components-path", path);
}

function getConfigPathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-config-path");
}

function setConfigPathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-config-path", path);
}

function getLoraPathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-lora-path");
}

function setLoraPathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-lora-path", path);
}

function getPreprocessorPathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-preprocessor-path");
}

function setPreprocessorPathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-preprocessor-path", path);
}

function getPostprocessorPathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-postprocessor-path");
}

function setPostprocessorPathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-postprocessor-path", path);
}

function getAllPathsSetting(): Promise<PathsPayload> {
  return ipcRenderer.invoke("settings:get-all-paths");
}

function setAllPathsSetting(
  payload: Partial<PathsPayload> & {
    hfToken?: string | null;
    civitaiApiKey?: string | null;
  },
): Promise<ConfigResponse<unknown>> {
  return ipcRenderer.invoke("settings:set-all-paths", payload);
}

function getHfTokenSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-hf-token");
}

function setHfTokenSetting(
  token: string | null,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-hf-token", token);
}

function getCivitaiApiKeySetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-civitai-api-key");
}

function setCivitaiApiKeySetting(
  token: string | null,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-civitai-api-key", token);
}

export {
  getActiveProjectId,
  setActiveProjectId,
  getCachePathSetting,
  setCachePathSetting,
  getComponentsPathSetting,
  setComponentsPathSetting,
  getConfigPathSetting,
  setConfigPathSetting,
  getLoraPathSetting,
  setLoraPathSetting,
  getPreprocessorPathSetting,
  setPreprocessorPathSetting,
  getPostprocessorPathSetting,
  setPostprocessorPathSetting,
  getAllPathsSetting,
  setAllPathsSetting,
  getHfTokenSetting,
  setHfTokenSetting,
  getCivitaiApiKeySetting,
  setCivitaiApiKeySetting,
};