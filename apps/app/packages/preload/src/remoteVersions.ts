import { ipcRenderer } from "electron";
import type { ConfigResponse } from "./types.js";

export interface ServerBundleAsset {
  tag: string;
  tagVersion: string;
  releaseName?: string;
  publishedAt?: string;
  prerelease?: boolean;
  assetName: string;
  downloadUrl: string;
  size?: number;
  assetVersion: string;
  platform: string;
  arch: string;
  device: string;
  pythonTag: string;
}

export interface ListServerVersionsResponse {
  host: { platform: string; arch: string };
  repo: { owner: string; name: string };
  items: ServerBundleAsset[];
}

/**
 * Lists downloadable server bundles from GitHub releases named `v<<semver>>-server`,
 * filtered to the current host platform/arch and `.tar.zst` assets matching
 * `python-api-<version>-<platform>-<arch>-<device>-<python>.tar.zst`.
 */
export async function listServerReleaseBundles(): Promise<
  ConfigResponse<ListServerVersionsResponse>
> {
  return await ipcRenderer.invoke("versions:list-server-releases");
}


