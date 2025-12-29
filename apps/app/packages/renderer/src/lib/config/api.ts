import {
  getBackendUrl,
  setBackendUrl,
  getHomeDir,
  setHomeDir,
  getTorchDevice,
  setTorchDevice,
  getCachePath,
  setCachePath,
} from "@app/preload";

export interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface BackendUrlConfig {
  url: string;
}

export interface HomeDirectoryConfig {
  home_dir: string;
}

export interface TorchDeviceConfig {
  device: string;
}

export interface CachePathConfig {
  cache_path: string;
}

/**
 * Get the current backend API URL
 */
export async function getBackendApiUrl(): Promise<
  ConfigResponse<BackendUrlConfig>
> {
  return await getBackendUrl();
}

/**
 * Set the backend API URL. Changes take effect immediately.
 */
export async function setBackendApiUrl(
  url: string,
): Promise<ConfigResponse<BackendUrlConfig>> {
  return await setBackendUrl(url);
}

/**
 * Get the current apex home directory
 */
export async function getApexHomeDir(): Promise<
  ConfigResponse<HomeDirectoryConfig>
> {
  return await getHomeDir();
}

/**
 * Set the apex home directory. Requires restart to take full effect.
 */
export async function setApexHomeDir(
  homeDir: string,
): Promise<ConfigResponse<HomeDirectoryConfig>> {
  return await setHomeDir(homeDir);
}

/**
 * Get the current torch device (cpu, cuda, mps, etc.)
 */
export async function getApexTorchDevice(): Promise<
  ConfigResponse<TorchDeviceConfig>
> {
  return await getTorchDevice();
}

/**
 * Set the torch device (cpu, cuda, mps, cuda:0, etc.)
 */
export async function setApexTorchDevice(
  device: string,
): Promise<ConfigResponse<TorchDeviceConfig>> {
  return await setTorchDevice(device);
}

/**
 * Get the current cache path for media-related cache items
 */
export async function getApexCachePath(): Promise<
  ConfigResponse<CachePathConfig>
> {
  return await getCachePath();
}

/**
 * Set the cache path for media-related cache items
 */
export async function setApexCachePath(
  cachePath: string,
): Promise<ConfigResponse<CachePathConfig>> {
  return await setCachePath(cachePath);
}
