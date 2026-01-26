import { AppModule } from "src/AppModule.js";
import { ModuleContext } from "src/ModuleContext.js";
import Store from "electron-store";
import { ipcMain } from "electron";
import { getSettingsModule } from "./SettingsModule.js";

/**
 * Persisted, global (project-independent) offload defaults.
 *
 * Storage is keyed by:
 * - backendUrl (normalized by SettingsModule)
 * - manifestId (model/manifest metadata id)
 *
 * This lets the same user have different defaults for different backends.
 */

type OffloadDefaults = Record<
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

type OffloadDefaultsByManifestId = Record<string, OffloadDefaults>;
type OffloadDefaultsByBackendUrl = Record<string, OffloadDefaultsByManifestId>;

interface OffloadStoreShape {
  /**
   * Global map of defaults:
   * { [backendUrl]: { [manifestId]: OffloadDefaults } }
   */
  offloadDefaultsByBackendUrl?: OffloadDefaultsByBackendUrl;
}

// Singleton instance
let offloadInstance: OffloadModule | null = null;

export class OffloadModule implements AppModule {
  private store: Store<OffloadStoreShape>;

  constructor() {
    // Use a dedicated electron-store file for global offload defaults.
    this.store = new Store<OffloadStoreShape>({
      name: "apex-offload-defaults",
    });

    if (!offloadInstance) {
      offloadInstance = this;
    }
    return offloadInstance;
  }

  enable(_ctx: ModuleContext): void {
    this.registerHandlers();
  }

  private getBackendUrlKey(): string {
    // SettingsModule normalizes/removes trailing slashes.
    try {
      const settings = getSettingsModule();
      return settings.getBackendUrl();
    } catch {
      // Fall back to a stable default key if settings is not available.
      return "http://127.0.0.1:8765";
    }
  }

  private readAll(): OffloadDefaultsByBackendUrl {
    const raw = this.store.get("offloadDefaultsByBackendUrl");
    if (!raw || typeof raw !== "object") return {};
    return raw as OffloadDefaultsByBackendUrl;
  }

  private writeAll(next: OffloadDefaultsByBackendUrl): void {
    this.store.set("offloadDefaultsByBackendUrl", next);
  }

  private getDefaultsForManifest(manifestId: string): OffloadDefaults | null {
    const id = String(manifestId ?? "").trim();
    if (!id) return null;
    const backendUrl = this.getBackendUrlKey();
    const all = this.readAll();
    const byManifest = all[backendUrl];
    if (!byManifest) return null;
    return (byManifest[id] as OffloadDefaults | undefined) ?? null;
  }

  private setDefaultsForManifest(manifestId: string, defaults: OffloadDefaults): void {
    const id = String(manifestId ?? "").trim();
    if (!id) return;
    const backendUrl = this.getBackendUrlKey();

    const all = this.readAll();
    const byManifest = { ...(all[backendUrl] || {}) };

    // Store as a plain JSON object (ensure no prototype surprises)
    const safeDefaults =
      defaults && typeof defaults === "object"
        ? (JSON.parse(JSON.stringify(defaults)) as OffloadDefaults)
        : ({} as OffloadDefaults);

    byManifest[id] = safeDefaults;
    all[backendUrl] = byManifest;
    this.writeAll(all);
  }

  private clearDefaultsForManifest(manifestId: string): void {
    const id = String(manifestId ?? "").trim();
    if (!id) return;
    const backendUrl = this.getBackendUrlKey();
    const all = this.readAll();
    if (!all[backendUrl]) return;
    const byManifest = { ...(all[backendUrl] || {}) };
    if (!Object.prototype.hasOwnProperty.call(byManifest, id)) return;
    delete byManifest[id];
    all[backendUrl] = byManifest;
    this.writeAll(all);
  }

  private registerHandlers(): void {
    ipcMain.handle("offload:get-manifest-defaults", (_event, manifestId: string) => {
      return this.getDefaultsForManifest(manifestId);
    });

    ipcMain.handle(
      "offload:set-manifest-defaults",
      (_event, manifestId: string, defaults: OffloadDefaults) => {
        this.setDefaultsForManifest(manifestId, defaults);
        return { success: true };
      },
    );

    ipcMain.handle("offload:clear-manifest-defaults", (_event, manifestId: string) => {
      this.clearDefaultsForManifest(manifestId);
      return { success: true };
    });
  }
}

export function offloadModule(...args: ConstructorParameters<typeof OffloadModule>) {
  if (!offloadInstance) {
    offloadInstance = new OffloadModule(...args);
  }
  return offloadInstance;
}

export function getOffloadModule() {
  if (!offloadInstance) {
    offloadInstance = new OffloadModule();
  }
  return offloadInstance;
}

