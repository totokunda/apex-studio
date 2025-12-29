import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

async function getBackendUrl(): Promise<ConfigResponse<{ url: string }>> {
  return await ipcRenderer.invoke("backend:get-url");
}

async function setBackendUrl(
  url: string,
): Promise<ConfigResponse<{ url: string }>> {
  return await ipcRenderer.invoke("backend:set-url", url);
}

async function getBackendIsRemote(): Promise<
  ConfigResponse<{ isRemote: boolean }>
> {
  return await ipcRenderer.invoke("backend:is-remote");
}

async function getFileShouldUpload(
  inputPath: string,
): Promise<ConfigResponse<{ shouldUpload: boolean }>> {
  return await ipcRenderer.invoke("files:should-upload", inputPath);
}

async function getFileIsUploaded(
  inputPath: string,
): Promise<ConfigResponse<{ isUploaded: boolean }>> {
  return await ipcRenderer.invoke("files:is-uploaded", inputPath);
}

async function getHomeDir(): Promise<ConfigResponse<{ home_dir: string }>> {
  return await ipcRenderer.invoke("config:get-home-dir");
}

async function setHomeDir(
  homeDir: string,
): Promise<ConfigResponse<{ home_dir: string }>> {
  return await ipcRenderer.invoke("config:set-home-dir", homeDir);
}

async function getUserDataPath(): Promise<
  ConfigResponse<{ user_data: string }>
> {
  return await ipcRenderer.invoke("config:get-user-data-path");
}

async function getTorchDevice(): Promise<ConfigResponse<{ device: string }>> {
  return await ipcRenderer.invoke("config:get-torch-device");
}

async function setTorchDevice(
  device: string,
): Promise<ConfigResponse<{ device: string }>> {
  return await ipcRenderer.invoke("config:set-torch-device", device);
}

async function getCachePath(): Promise<
  ConfigResponse<{ cache_path: string }>
> {
  return await ipcRenderer.invoke("config:get-cache-path");
}

async function setCachePath(
  cachePath: string,
): Promise<ConfigResponse<{ cache_path: string }>> {
  return await ipcRenderer.invoke("config:set-cache-path", cachePath);
}

async function getSystemMemory(): Promise<ConfigResponse<any>> {
  return await ipcRenderer.invoke("system:memory");
}

async function getFolderSize(
  inputPath: string,
): Promise<ConfigResponse<{ size_bytes: number }>> {
  return await ipcRenderer.invoke("system:get-folder-size", inputPath);
}

async function getPathExists(
  pathOrUrl: string,
): Promise<ConfigResponse<{ exists: boolean }>> {
  return await ipcRenderer.invoke("apexapi:path-exists", pathOrUrl);
}

export {
  getBackendUrl,
  setBackendUrl,
  getBackendIsRemote,
  getFileShouldUpload,
  getFileIsUploaded,
  getHomeDir,
  setHomeDir,
  getUserDataPath,
  getTorchDevice,
  setTorchDevice,
  getCachePath,
  setCachePath,
  getSystemMemory,
  getFolderSize,
  getPathExists,
};


