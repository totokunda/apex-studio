import { ipcRenderer } from "electron";

/**
 * Global (project-independent) offload defaults persisted in the main process.
 * Defaults are keyed by the active backend URL (from SettingsModule) and manifest id.
 */

export type OffloadDefaults = Record<
  string,
  {
    enabled?: boolean;
    offload_mode?: "group" | "budget";
    level?: "leaf" | "block";
    num_blocks?: number;
    use_stream?: boolean;
    record_stream?: boolean;
    low_cpu_mem_usage?: boolean;
    budget_mb?: number | string | null;
    async_transfers?: boolean;
    prefetch?: boolean;
    pin_cpu_memory?: boolean;
    vram_safety_coefficient?: number;
    offload_after_forward?: boolean;
  }
>;

export function getOffloadDefaultsForManifest(
  manifestId: string,
): Promise<OffloadDefaults | null> {
  return ipcRenderer.invoke("offload:get-manifest-defaults", manifestId);
}

export function setOffloadDefaultsForManifest(
  manifestId: string,
  defaults: OffloadDefaults,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("offload:set-manifest-defaults", manifestId, defaults);
}

export function clearOffloadDefaultsForManifest(
  manifestId: string,
): Promise<{ success: boolean }> {
  return ipcRenderer.invoke("offload:clear-manifest-defaults", manifestId);
}

