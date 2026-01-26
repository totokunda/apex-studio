import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

function getActiveProjectId(): Promise<string | number | undefined | null> {
  return ipcRenderer.invoke("settings:get-active-project-id");
}

function setActiveProjectId(projectId: string | number | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-active-project-id", projectId);
}

type PathsPayload = {
  apiPath?: string | null;
  cachePath?: string | null;
  componentsPath?: string | null;
  configPath?: string | null;
  loraPath?: string | null;
  preprocessorPath?: string | null;
  postprocessorPath?: string | null;
  maskModel?: string | null;
};

export type BackendSyncedSettings = {
  cachePath: string | null;
  componentsPath: string | null;
  configPath: string | null;
  loraPath: string | null;
  preprocessorPath: string | null;
  postprocessorPath: string | null;
  maskModel: string | null;
  renderImageSteps: boolean;
  renderVideoSteps: boolean;
  useFastDownload: boolean;
  autoUpdateEnabled: boolean;
  disableAutoMemoryManagement: boolean;
};

export type BackendPathSizes = {
  cachePathBytes: number | null;
  componentsPathBytes: number | null;
  configPathBytes: number | null;
  loraPathBytes: number | null;
  preprocessorPathBytes: number | null;
  postprocessorPathBytes: number | null;
};

function getApiPathSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-api-path");
}

function setApiPathSetting(path: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-api-path", path);
}

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

function getMaskModelSetting(): Promise<string | null> {
  return ipcRenderer.invoke("settings:get-mask-model");
}

function setMaskModelSetting(model: string | null): Promise<void> {
  return ipcRenderer.invoke("settings:set-mask-model", model);
}

function getRenderImageStepsSetting(): Promise<boolean> {
  return ipcRenderer.invoke("settings:get-render-image-steps");
}

function setRenderImageStepsSetting(
  enabled: boolean,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-render-image-steps", enabled);
}

function getRenderVideoStepsSetting(): Promise<boolean> {
  return ipcRenderer.invoke("settings:get-render-video-steps");
}

function setRenderVideoStepsSetting(
  enabled: boolean,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-render-video-steps", enabled);
}

function getUseFastDownloadSetting(): Promise<boolean> {
  return ipcRenderer.invoke("settings:get-use-fast-download");
}

function setUseFastDownloadSetting(
  enabled: boolean,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-use-fast-download", enabled);
}

function getAutoUpdateEnabledSetting(): Promise<boolean> {
  return ipcRenderer.invoke("settings:get-auto-update-enabled");
}

function setAutoUpdateEnabledSetting(
  enabled: boolean,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-auto-update-enabled", enabled);
}

function getDisableAutoMemoryManagementSetting(): Promise<boolean> {
  return ipcRenderer.invoke("settings:get-disable-auto-memory-management");
}

function setDisableAutoMemoryManagementSetting(
  disabled: boolean,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("settings:set-disable-auto-memory-management", disabled);
}

function refreshSettingsFromBackend(): Promise<ConfigResponse<BackendSyncedSettings>> {
  return ipcRenderer.invoke("settings:refresh-from-backend");
}

function verifyBackendUrlAndFetchSettings(
  url: string,
): Promise<ConfigResponse<BackendSyncedSettings>> {
  return ipcRenderer.invoke("settings:verify-backend-url", url);
}

function getBackendPathSizes(
  url?: string | null,
): Promise<ConfigResponse<BackendPathSizes>> {
  return ipcRenderer.invoke("settings:get-backend-path-sizes", url ?? null);
}

export {
  getActiveProjectId,
  setActiveProjectId,
  getApiPathSetting,
  setApiPathSetting,
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
  getMaskModelSetting,
  setMaskModelSetting,
  getRenderImageStepsSetting,
  setRenderImageStepsSetting,
  getRenderVideoStepsSetting,
  setRenderVideoStepsSetting,
  getUseFastDownloadSetting,
  setUseFastDownloadSetting,
  getAutoUpdateEnabledSetting,
  setAutoUpdateEnabledSetting,
  getDisableAutoMemoryManagementSetting,
  setDisableAutoMemoryManagementSetting,
  refreshSettingsFromBackend,
  verifyBackendUrlAndFetchSettings,
  getBackendPathSizes,
};