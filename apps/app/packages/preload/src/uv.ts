import { ipcRenderer } from "electron";

export type BundledUvPaths = {
  uv: string;
  uvx: string | null;
};

export async function getUvPaths(): Promise<BundledUvPaths> {
  return ipcRenderer.invoke("uv:get-paths");
}

export async function getUvVersion(): Promise<
  { ok: true; version: string } | { ok: false; error: string }
> {
  return ipcRenderer.invoke("uv:get-version");
}


