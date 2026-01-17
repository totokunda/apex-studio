import { AppModule } from "src/AppModule.js";
import { ModuleContext } from "src/ModuleContext.js";
import Store from "electron-store";
import { ipcMain } from "electron";
import fs from "node:fs";
import path from "node:path";
import { EventEmitter } from "node:events";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8765";

type PathKey =
  | "cachePath"
  | "componentsPath"
  | "configPath"
  | "loraPath"
  | "preprocessorPath"
  | "postprocessorPath"
  | "maskModel";

interface Settings {
  activeProjectId?: string | number | null;
  /**
   * Local path to the bundled/installed API runtime (e.g. python env, api package, etc).
   * Used by the launcher to decide whether the app can proceed even if the API server
   * isn't currently reachable.
   */
  apiPath?: string | null;
  cachePath?: string | null;
  componentsPath?: string | null;
  configPath?: string | null;
  loraPath?: string | null;
  preprocessorPath?: string | null;
  postprocessorPath?: string | null;
  hfToken?: string | null;
  civitaiApiKey?: string | null;
  maskModel?: string | null;
  renderImageSteps?: boolean;
  renderVideoSteps?: boolean;
  useFastDownload?: boolean;
  autoUpdateEnabled?: boolean;
}

interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

type BackendSyncedSettings = {
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
};

type BackendPathSizes = {
  cachePathBytes: number | null;
  componentsPathBytes: number | null;
  configPathBytes: number | null;
  loraPathBytes: number | null;
  preprocessorPathBytes: number | null;
  postprocessorPathBytes: number | null;
};

type PathsPayload = Pick<
  Settings,
  | "cachePath"
  | "componentsPath"
  | "configPath"
  | "loraPath"
  | "preprocessorPath"
  | "postprocessorPath"
  | "maskModel"
>;

// Singleton instance
let settingsInstance: SettingsModule | null = null;

export class SettingsModule extends EventEmitter implements AppModule {
  private store: Store<Settings>;
  private backendUrl: string = DEFAULT_BACKEND_URL;
  // The last backend URL that has been persisted to disk. This allows a "preview" backend URL
  // (used during Verify) without losing the ability to later commit the same URL.
  private persistedBackendUrl: string = DEFAULT_BACKEND_URL;
  private settingsPath: string | null = null;
  private refreshInFlight: Promise<ConfigResponse<BackendSyncedSettings>> | null =
    null;
  private verifyInFlight: Promise<ConfigResponse<BackendSyncedSettings>> | null =
    null;

  constructor() {
    super();
    this.store = new Store<Settings>();
    if (!settingsInstance) {
      settingsInstance = this;
    }
    return settingsInstance;
  }

  enable({ app }: ModuleContext): void {
    this.settingsPath = path.join(app.getPath("userData"), "apex-settings.json");
    const fromDisk = this.normalizeBackendUrl(
      this.readBackendUrlFromDisk(this.settingsPath),
    );
    this.persistedBackendUrl = fromDisk;
    this.backendUrl = fromDisk;
    this.registerHandlers();
  }

  getBackendUrl(): string {
    return this.backendUrl;
  }

  getApiPath(): string | null {
    const v = (this.store.get("apiPath") as string | null | undefined) ?? null;
    const trimmed = typeof v === "string" ? v.trim() : "";
    return trimmed ? trimmed : null;
  }

  async setBackendUrl(url: string): Promise<void> {
    const next = this.normalizeBackendUrl(url);
    const activeChanged = next !== this.backendUrl;
    this.backendUrl = next;
    // Persist even if this backend URL was already "previewed" as the active URL.
    if (next !== this.persistedBackendUrl) {
      this.persistedBackendUrl = next;
      this.saveBackendUrlToDisk();
    }
    if (activeChanged) {
      this.emit("backend-url-changed", next);
    }
  }

  /**
   * Temporarily switch the active backend URL in-memory (emits backend-url-changed),
   * without persisting it to disk. Intended for the Settings "Verify" flow.
   */
  previewBackendUrl(url: string): void {
    const next = this.normalizeBackendUrl(url);
    if (next === this.backendUrl) return;
    this.backendUrl = next;
    this.emit("backend-url-changed", next);
  }

  private normalizeBackendUrl(url: string): string {
    const trimmed = String(url || "").trim();
    if (!trimmed) return DEFAULT_BACKEND_URL;
    try {
      // Ensure it's a valid URL and normalize by removing trailing slash.
      const u = new URL(trimmed);
      return u.toString().replace(/\/$/, "");
    } catch {
      // Best-effort: keep user input but still remove trailing slash.
      return trimmed.replace(/\/$/, "");
    }
  }

  private normalizeCandidateBackendUrl(
    url: string,
  ): { ok: true; url: string } | { ok: false; error: string } {
    const trimmed = String(url ?? "").trim();
    if (!trimmed) return { ok: false, error: "Backend URL is required." };
    try {
      const u = new URL(trimmed);
      return { ok: true, url: u.toString().replace(/\/$/, "") };
    } catch (e) {
      return {
        ok: false,
        error: e instanceof Error ? e.message : "Invalid backend URL.",
      };
    }
  }

  private saveBackendUrlToDisk(): void {
    if (!this.settingsPath) return;
    try {
      // Create simple object matching ApexApi's previous structure
      const settings = {
        backendUrl: this.persistedBackendUrl,
      };
      fs.writeFileSync(
        this.settingsPath,
        JSON.stringify(settings, null, 2),
        "utf-8",
      );
    } catch (error) {
      console.error("Failed to save backend URL settings:", error);
    }
  }

  private readBackendUrlFromDisk(settingsPath: string | null): string {
    if (!settingsPath) return DEFAULT_BACKEND_URL;
    try {
      if (!fs.existsSync(settingsPath)) return DEFAULT_BACKEND_URL;
      const raw = fs.readFileSync(settingsPath, "utf-8");
      const parsed = JSON.parse(raw) as { backendUrl?: string };
      if (parsed.backendUrl && typeof parsed.backendUrl === "string") {
        return parsed.backendUrl;
      }
    } catch {
      // fall through to default
    }
    return DEFAULT_BACKEND_URL;
  }

  private async makeConfigRequest<T>(
    method: "GET" | "POST",
    endpoint: string,
    body?: any,
    timeoutMs: number = 6000,
  ): Promise<ConfigResponse<T>> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      const options: RequestInit = {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
      };

      if (body && method !== "GET") {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${this.backendUrl}${endpoint}`, options).finally(
        () => clearTimeout(timeoutId),
      );
      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ detail: "Unknown error" }));
        return {
          success: false,
          error:
            errorData.detail ||
            `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      const data = (await response.json()) as T;
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }

  private async makeConfigRequestAtUrl<T>(
    baseUrl: string,
    method: "GET" | "POST",
    endpoint: string,
    body?: any,
    timeoutMs: number = 6000,
  ): Promise<ConfigResponse<T>> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      const options: RequestInit = {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
      };

      if (body && method !== "GET") {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${baseUrl}${endpoint}`, options).finally(() =>
        clearTimeout(timeoutId),
      );
      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ detail: "Unknown error" }));
        return {
          success: false,
          error:
            errorData.detail ||
            `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      const data = (await response.json()) as T;
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }

  /**
   * Force-refresh backend-derived settings (paths, toggles, etc.) from the current backendUrl.
   * This is intended to be called ONLY when the backendUrl changes (or when renderer explicitly requests a refresh).
   *
   * Safety: if the backendUrl changes again while a refresh is in-flight, results are discarded.
   */
  async refreshFromBackend(): Promise<ConfigResponse<BackendSyncedSettings>> {
    if (this.refreshInFlight) return this.refreshInFlight;
    const urlAtStart = this.backendUrl;

    const run = (async (): Promise<ConfigResponse<BackendSyncedSettings>> => {
      const currentSnapshot = (): BackendSyncedSettings => ({
        cachePath: (this.store.get("cachePath") as string | null | undefined) ?? null,
        componentsPath:
          (this.store.get("componentsPath") as string | null | undefined) ?? null,
        configPath: (this.store.get("configPath") as string | null | undefined) ?? null,
        loraPath: (this.store.get("loraPath") as string | null | undefined) ?? null,
        preprocessorPath:
          (this.store.get("preprocessorPath") as string | null | undefined) ?? null,
        postprocessorPath:
          (this.store.get("postprocessorPath") as string | null | undefined) ?? null,
        maskModel: (this.store.get("maskModel") as string | null | undefined) ?? null,
        renderImageSteps: Boolean(this.store.get("renderImageSteps")),
        renderVideoSteps: Boolean(this.store.get("renderVideoSteps")),
        useFastDownload: (() => {
          const v = this.store.get("useFastDownload");
          return v === undefined ? true : Boolean(v);
        })(),
        autoUpdateEnabled: (() => {
          const v = this.store.get("autoUpdateEnabled");
          return v === undefined ? true : Boolean(v);
        })(),
      });

      let okCount = 0;
      const next: BackendSyncedSettings = currentSnapshot();

      const setIfStillCurrent = <K extends keyof BackendSyncedSettings>(
        key: K,
        value: BackendSyncedSettings[K],
      ) => {
        if (this.backendUrl !== urlAtStart) return;
        (next as any)[key] = value;
        // Mirror into electron-store keys where they exist.
        if (
          key === "cachePath" ||
          key === "componentsPath" ||
          key === "configPath" ||
          key === "loraPath" ||
          key === "preprocessorPath" ||
          key === "postprocessorPath" ||
          key === "maskModel"
        ) {
          this.store.set(key as any, value as any);
        } else if (
          key === "renderImageSteps" ||
          key === "renderVideoSteps" ||
          key === "useFastDownload" ||
          key === "autoUpdateEnabled"
        ) {
          // Settings interface includes these booleans
          this.store.set(key as any, Boolean(value));
        }
      };

      // Paths + mask model
      const pathKeys: PathKey[] = [
        "cachePath",
        "componentsPath",
        "configPath",
        "loraPath",
        "preprocessorPath",
        "postprocessorPath",
        "maskModel",
      ];
      for (const key of pathKeys) {
        const cfg = this.getPathConfig(key);
        if (!cfg) continue;
        const res = await this.makeConfigRequestAtUrl<any>(
          urlAtStart,
          "GET",
          cfg.getEndpoint,
        );
        if (res.success && res.data) {
          const v = res.data[cfg.responseField];
          if (typeof v === "string") {
            okCount++;
            setIfStillCurrent(key as any, v || null);
          }
        }
      }

      // Booleans
      const img = await this.makeConfigRequestAtUrl<any>(
        urlAtStart,
        "GET",
        "/config/enable-image-render-steps",
      );
      if (img.success && img.data && typeof img.data.enabled === "boolean") {
        okCount++;
        setIfStillCurrent("renderImageSteps", Boolean(img.data.enabled));
      }

      const vid = await this.makeConfigRequestAtUrl<any>(
        urlAtStart,
        "GET",
        "/config/enable-video-render-steps",
      );
      if (vid.success && vid.data && typeof vid.data.enabled === "boolean") {
        okCount++;
        setIfStillCurrent("renderVideoSteps", Boolean(vid.data.enabled));
      }

      const fast = await this.makeConfigRequestAtUrl<any>(
        urlAtStart,
        "GET",
        "/config/enable-fast-download",
      );
      if (fast.success && fast.data && typeof fast.data.enabled === "boolean") {
        okCount++;
        setIfStillCurrent("useFastDownload", Boolean(fast.data.enabled));
      }

      const au = await this.makeConfigRequestAtUrl<any>(
        urlAtStart,
        "GET",
        "/config/auto-update",
      );
      if (au.success && au.data && typeof au.data.enabled === "boolean") {
        okCount++;
        setIfStillCurrent("autoUpdateEnabled", Boolean(au.data.enabled));
      }

      // If backendUrl changed mid-refresh, discard to avoid stale writes.
      if (this.backendUrl !== urlAtStart) {
        return {
          success: false,
          error: "Backend URL changed while refreshing settings; discarded stale results.",
        };
      }

      if (okCount === 0) {
        return {
          success: false,
          error: "Failed to refresh settings: backend unreachable or missing config endpoints.",
        };
      }

      return { success: true, data: next };
    })();

    this.refreshInFlight = run;
    return await run.finally(() => {
      this.refreshInFlight = null;
    });
  }

  /**
   * Verify a candidate backend URL is reachable (via GET /config/hostname) and then
   * fetch backend-derived settings (paths + toggles) from that URL. This does NOT
   * persist the candidate URL or overwrite local settings store; it's meant for
   * "preview/sync" flows in the renderer before the user hits Save.
   */
  async verifyBackendUrlAndFetchSettings(
    candidateUrl: string,
  ): Promise<ConfigResponse<BackendSyncedSettings>> {
    if (this.verifyInFlight) return this.verifyInFlight;
    const urlRes = this.normalizeCandidateBackendUrl(candidateUrl);
    if (!urlRes.ok) return { success: false, error: urlRes.error };

    const run = (async (): Promise<ConfigResponse<BackendSyncedSettings>> => {
      const probe = await this.makeConfigRequestAtUrl<any>(
        urlRes.url,
        "GET",
        "/config/hostname",
        undefined,
        10_000,
      );
      if (!probe.success) {
        return {
          success: false,
          error: probe.error || "Backend verification failed.",
        };
      }

      return await this.peekFromBackendUrl(urlRes.url);
    })();

    this.verifyInFlight = run;
    return await run.finally(() => {
      this.verifyInFlight = null;
    });
  }

  async fetchBackendPathSizesAtUrl(
    candidateUrl?: string | null,
  ): Promise<ConfigResponse<BackendPathSizes>> {
    const baseUrl = (() => {
      if (typeof candidateUrl === "string" && candidateUrl.trim()) {
        const u = this.normalizeCandidateBackendUrl(candidateUrl);
        if (u.ok) return u.url;
      }
      return this.backendUrl;
    })();

    const res = await this.makeConfigRequestAtUrl<any>(
      baseUrl,
      "GET",
      "/config/path-sizes",
      undefined,
      15_000,
    );
    if (!res.success || !res.data) {
      return { success: false, error: res.error || "Failed to fetch path sizes." };
    }

    const numOrNull = (v: unknown): number | null =>
      typeof v === "number" && Number.isFinite(v) ? v : null;

    return {
      success: true,
      data: {
        cachePathBytes: numOrNull(res.data.cache_path_bytes),
        componentsPathBytes: numOrNull(res.data.components_path_bytes),
        configPathBytes: numOrNull(res.data.config_path_bytes),
        loraPathBytes: numOrNull(res.data.lora_path_bytes),
        preprocessorPathBytes: numOrNull(res.data.preprocessor_path_bytes),
        postprocessorPathBytes: numOrNull(res.data.postprocessor_path_bytes),
      },
    };
  }

  private async peekFromBackendUrl(
    backendUrl: string,
  ): Promise<ConfigResponse<BackendSyncedSettings>> {
    const currentSnapshot = (): BackendSyncedSettings => ({
      cachePath:
        (this.store.get("cachePath") as string | null | undefined) ?? null,
      componentsPath:
        (this.store.get("componentsPath") as string | null | undefined) ?? null,
      configPath:
        (this.store.get("configPath") as string | null | undefined) ?? null,
      loraPath: (this.store.get("loraPath") as string | null | undefined) ?? null,
      preprocessorPath:
        (this.store.get("preprocessorPath") as string | null | undefined) ?? null,
      postprocessorPath:
        (this.store.get("postprocessorPath") as string | null | undefined) ?? null,
      maskModel:
        (this.store.get("maskModel") as string | null | undefined) ?? null,
      renderImageSteps: Boolean(this.store.get("renderImageSteps")),
      renderVideoSteps: Boolean(this.store.get("renderVideoSteps")),
      useFastDownload: (() => {
        const v = this.store.get("useFastDownload");
        return v === undefined ? true : Boolean(v);
      })(),
      autoUpdateEnabled: (() => {
        const v = this.store.get("autoUpdateEnabled");
        return v === undefined ? true : Boolean(v);
      })(),
    });

    let okCount = 0;
    const next: BackendSyncedSettings = currentSnapshot();

    const setFetched = <K extends keyof BackendSyncedSettings>(
      key: K,
      value: BackendSyncedSettings[K],
    ) => {
      (next as any)[key] = value;
    };

    // Paths + mask model
    const pathKeys: PathKey[] = [
      "cachePath",
      "componentsPath",
      "configPath",
      "loraPath",
      "preprocessorPath",
      "postprocessorPath",
      "maskModel",
    ];
    for (const key of pathKeys) {
      const cfg = this.getPathConfig(key);
      if (!cfg) continue;
      const res = await this.makeConfigRequestAtUrl<any>(
        backendUrl,
        "GET",
        cfg.getEndpoint,
      );
      if (res.success && res.data) {
        const v = res.data[cfg.responseField];
        if (typeof v === "string") {
          okCount++;
          setFetched(key as any, v || null);
        }
      }
    }

    // Booleans
    const img = await this.makeConfigRequestAtUrl<any>(
      backendUrl,
      "GET",
      "/config/enable-image-render-steps",
    );
    if (img.success && img.data && typeof img.data.enabled === "boolean") {
      okCount++;
      setFetched("renderImageSteps", Boolean(img.data.enabled));
    }

    const vid = await this.makeConfigRequestAtUrl<any>(
      backendUrl,
      "GET",
      "/config/enable-video-render-steps",
    );
    if (vid.success && vid.data && typeof vid.data.enabled === "boolean") {
      okCount++;
      setFetched("renderVideoSteps", Boolean(vid.data.enabled));
    }

    const fast = await this.makeConfigRequestAtUrl<any>(
      backendUrl,
      "GET",
      "/config/enable-fast-download",
    );
    if (fast.success && fast.data && typeof fast.data.enabled === "boolean") {
      okCount++;
      setFetched("useFastDownload", Boolean(fast.data.enabled));
    }

    const au = await this.makeConfigRequestAtUrl<any>(
      backendUrl,
      "GET",
      "/config/auto-update",
    );
    if (au.success && au.data && typeof au.data.enabled === "boolean") {
      okCount++;
      setFetched("autoUpdateEnabled", Boolean(au.data.enabled));
    }

    if (okCount === 0) {
      return {
        success: false,
        error:
          "Failed to sync settings from backend: backend unreachable or missing config endpoints.",
      };
    }

    return { success: true, data: next };
  }

  private async setCivitaiApiKeyAndUpdateApi(
    token: string | null | undefined,
  ): Promise<void> {
    const trimmed = (token ?? "").trim();
    this.store.set("civitaiApiKey", trimmed || null);
    if (!trimmed) return;
    // Forward to backend /config/civitai-api-key
    void this.makeConfigRequest<any>("POST", "/config/civitai-api-key", {
      token: trimmed,
    });
  }

  private getPathConfig(key: PathKey): {
    getEndpoint: string;
    setEndpoint: string;
    responseField: string;
    requestField: string;
  } | null {
    switch (key) {
      case "cachePath":
        return {
          getEndpoint: "/config/cache-path",
          setEndpoint: "/config/cache-path",
          responseField: "cache_path",
          requestField: "cache_path",
        };
      case "componentsPath":
        return {
          getEndpoint: "/config/components-path",
          setEndpoint: "/config/components-path",
          responseField: "components_path",
          requestField: "components_path",
        };
      case "configPath":
        return {
          getEndpoint: "/config/config-path",
          setEndpoint: "/config/config-path",
          responseField: "config_path",
          requestField: "config_path",
        };
      case "loraPath":
        return {
          getEndpoint: "/config/lora-path",
          setEndpoint: "/config/lora-path",
          responseField: "lora_path",
          requestField: "lora_path",
        };
      case "preprocessorPath":
        return {
          getEndpoint: "/config/preprocessor-path",
          setEndpoint: "/config/preprocessor-path",
          responseField: "preprocessor_path",
          requestField: "preprocessor_path",
        };
      case "postprocessorPath":
        return {
          getEndpoint: "/config/postprocessor-path",
          setEndpoint: "/config/postprocessor-path",
          responseField: "postprocessor_path",
          requestField: "postprocessor_path",
        };
      case "maskModel":
        return {
          getEndpoint: "/config/mask-model",
          setEndpoint: "/config/mask-model",
          responseField: "mask_model",
          requestField: "mask_model",
        };
      default:
        return null;
    }
  }

  private async fetchPathFromApi(key: PathKey): Promise<string | null> {
    const cfg = this.getPathConfig(key);
    if (!cfg) return null;
    const res = await this.makeConfigRequest<any>("GET", cfg.getEndpoint);
    if (!res.success || !res.data) return null;
    const value = res.data[cfg.responseField];
    if (typeof value === "string" && value) {
      this.store.set(key as keyof Settings, value);
      return value;
    }
    return null;
  }

  private async setPathAndUpdateApi(
    key: PathKey,
    value: string | null | undefined,
  ): Promise<void> {
    const v = (value ?? "").trim();
    this.store.set(key as keyof Settings, v || null);
    const cfg = this.getPathConfig(key);
    if (!cfg || !v) return;
    // Fire and forget; caller does not depend on result
    void this.makeConfigRequest<any>("POST", cfg.setEndpoint, {
      [cfg.requestField]: v,
    });
  }

  private async setHfTokenAndUpdateApi(
    token: string | null | undefined,
  ): Promise<void> {
    const trimmed = (token ?? "").trim();
    this.store.set("hfToken", trimmed || null);
    if (!trimmed) return;
    // Forward to backend /config/hf-token
    void this.makeConfigRequest<any>("POST", "/config/hf-token", {
      token: trimmed,
    });
  }

  private setBooleanSettingAndUpdateApi(
    key: keyof Settings,
    enabled: boolean,
    endpoint: string,
    requestField: string,
  ): void {
    const v = Boolean(enabled);
    this.store.set(key, v);
    // Fire and forget; caller does not depend on result
    void this.makeConfigRequest<any>("POST", endpoint, {
      [requestField]: v,
    });
  }

  private async getAutoUpdateEnabledEnsuringFromApi(): Promise<boolean> {
    const current = this.store.get("autoUpdateEnabled");
    if (current !== undefined) return Boolean(current);

    // Prefer backend source-of-truth (persisted on API side).
    const res = await this.makeConfigRequest<any>("GET", "/config/auto-update").catch(
      () => ({ success: false } as any),
    );
    const enabled =
      res && res.success && res.data && typeof res.data.enabled === "boolean"
        ? Boolean(res.data.enabled)
        : true;
    this.store.set("autoUpdateEnabled", enabled);
    return enabled;
  }

  private setAutoUpdateEnabledAndUpdateApi(enabled: boolean): void {
    const v = Boolean(enabled);
    this.store.set("autoUpdateEnabled", v);
    void this.makeConfigRequest<any>("POST", "/config/auto-update", {
      enabled: v,
    }).catch(() => undefined);
  }

  private async getAllPathsEnsuringFromApi(): Promise<PathsPayload> {
    const keys: PathKey[] = [
      "cachePath",
      "componentsPath",
      "configPath",
      "loraPath",
      "preprocessorPath",
      "postprocessorPath",
      "maskModel",
    ];
    const result: PathsPayload = {};

    for (const key of keys) {
      let current = this.store.get(key as keyof Settings) as string | null;
      if (!current) {
        current = await this.fetchPathFromApi(key);
      }
      (result as any)[key] = current ?? null;
    }

    return result;
  }

  private registerHandlers() {
    // Active project id
    ipcMain.handle(
      "settings:get-active-project-id",
      (): string | number | undefined | null => {
        return this.store.get("activeProjectId") ?? null;
      },
    );

    ipcMain.handle(
      "settings:set-active-project-id",
      (_event, projectId: string | number | null): void => {
        this.store.set("activeProjectId", projectId ?? null);
      },
    );

    // API path storage (local-only)
    ipcMain.handle("settings:get-api-path", () => {
      return this.store.get("apiPath") ?? null;
    });

    ipcMain.handle("settings:set-api-path", (_event, apiPath: string | null) => {
      const v = (apiPath ?? "").trim();
      this.store.set("apiPath", v || null);
    });

    // Individual path getters (ensure from backend if missing)
    ipcMain.handle("settings:get-cache-path", async () => {
      const current = this.store.get("cachePath") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("cachePath");
    });

    ipcMain.handle("settings:get-components-path", async () => {
      const current = this.store.get("componentsPath") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("componentsPath");
    });

    ipcMain.handle("settings:get-config-path", async () => {
      const current = this.store.get("configPath") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("configPath");
    });

    ipcMain.handle("settings:get-lora-path", async () => {
      const current = this.store.get("loraPath") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("loraPath");
    });

    ipcMain.handle("settings:get-preprocessor-path", async () => {
      const current = this.store.get("preprocessorPath") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("preprocessorPath");
    });

    ipcMain.handle("settings:get-postprocessor-path", async () => {
      const current = this.store.get("postprocessorPath") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("postprocessorPath");
    });

    ipcMain.handle("settings:get-mask-model", async () => {
      const current = this.store.get("maskModel") as string | null;
      if (current) return current;
      return await this.fetchPathFromApi("maskModel");
    });

    // Individual path setters (persist + update backend)
    ipcMain.handle(
      "settings:set-cache-path",
      async (_event, cachePath: string | null) => {
        await this.setPathAndUpdateApi("cachePath", cachePath);
      },
    );

    ipcMain.handle(
      "settings:set-components-path",
      async (_event, componentsPath: string | null) => {
        await this.setPathAndUpdateApi("componentsPath", componentsPath);
      },
    );

    ipcMain.handle(
      "settings:set-config-path",
      async (_event, configPath: string | null) => {
        await this.setPathAndUpdateApi("configPath", configPath);
      },
    );

    ipcMain.handle(
      "settings:set-lora-path",
      async (_event, loraPath: string | null) => {
        await this.setPathAndUpdateApi("loraPath", loraPath);
      },
    );

    ipcMain.handle(
      "settings:set-preprocessor-path",
      async (_event, preprocessorPath: string | null) => {
        await this.setPathAndUpdateApi("preprocessorPath", preprocessorPath);
      },
    );

    ipcMain.handle(
      "settings:set-postprocessor-path",
      async (_event, postprocessorPath: string | null) => {
        await this.setPathAndUpdateApi("postprocessorPath", postprocessorPath);
      },
    );

    ipcMain.handle(
      "settings:set-mask-model",
      async (_event, maskModel: string | null) => {
        await this.setPathAndUpdateApi("maskModel", maskModel);
      },
    );

    // Bulk getters/setters for all paths
    ipcMain.handle("settings:get-all-paths", async () => {
      return await this.getAllPathsEnsuringFromApi();
    });

    ipcMain.handle(
      "settings:set-all-paths",
      async (
        _event,
        payload: Partial<PathsPayload> & {
          apiPath?: string | null;
          hfToken?: string | null;
          civitaiApiKey?: string | null;
        },
      ) => {
        const keys: PathKey[] = [
          "cachePath",
          "componentsPath",
          "configPath",
          "loraPath",
          "preprocessorPath",
          "postprocessorPath",
          "maskModel",
        ];

        for (const key of keys) {
          if (Object.prototype.hasOwnProperty.call(payload, key)) {
            const value = (payload as any)[key] as string | null | undefined;
            await this.setPathAndUpdateApi(key, value);
          }
        }

        if (Object.prototype.hasOwnProperty.call(payload, "hfToken")) {
          await this.setHfTokenAndUpdateApi(payload.hfToken ?? null);
        }

        if (Object.prototype.hasOwnProperty.call(payload, "civitaiApiKey")) {
          await this.setCivitaiApiKeyAndUpdateApi(
            payload.civitaiApiKey ?? null,
          );
        }

        if (Object.prototype.hasOwnProperty.call(payload, "apiPath")) {
          const v = ((payload.apiPath ?? "") as string).trim();
          this.store.set("apiPath", v || null);
        }

        return { success: true };
      },
    );

    // Hugging Face token storage
    ipcMain.handle("settings:get-hf-token", () => {
      return this.store.get("hfToken") ?? null;
    });

    ipcMain.handle(
      "settings:set-hf-token",
      async (_event, token: string | null) => {
        await this.setHfTokenAndUpdateApi(token);
        return { success: true };
      },
    );

    // CivitAI API key storage
    ipcMain.handle("settings:get-civitai-api-key", () => {
      return this.store.get("civitaiApiKey") ?? null;
    });

    ipcMain.handle(
      "settings:set-civitai-api-key",
      async (_event, token: string | null) => {
        await this.setCivitaiApiKeyAndUpdateApi(token);
        return { success: true };
      },
    );

    // Render intermediary steps toggles
    ipcMain.handle("settings:get-render-image-steps", () => {
      return Boolean(this.store.get("renderImageSteps"));
    });

    ipcMain.handle(
      "settings:set-render-image-steps",
      (_event, enabled: boolean) => {
        this.setBooleanSettingAndUpdateApi(
          "renderImageSteps",
          enabled,
          "/config/enable-image-render-steps",
          "enabled",
        );
        return { success: true };
      },
    );

    ipcMain.handle("settings:get-render-video-steps", () => {
      return Boolean(this.store.get("renderVideoSteps"));
    });

    ipcMain.handle(
      "settings:set-render-video-steps",
      (_event, enabled: boolean) => {
        this.setBooleanSettingAndUpdateApi(
          "renderVideoSteps",
          enabled,
          "/config/enable-video-render-steps",
          "enabled",
        );
        return { success: true };
      },
    );

    // Fast download toggle
    ipcMain.handle("settings:get-use-fast-download", () => {
      const current = this.store.get("useFastDownload");
      // Default to true when unset to match renderer's default.
      return current === undefined ? true : Boolean(current);
    });

    ipcMain.handle(
      "settings:set-use-fast-download",
      (_event, enabled: boolean) => {
        this.setBooleanSettingAndUpdateApi(
          "useFastDownload",
          enabled,
          "/config/enable-fast-download",
          "enabled",
        );
        return { success: true };
      },
    );

    // API auto-update toggle (backend-persisted)
    ipcMain.handle("settings:get-auto-update-enabled", async () => {
      return await this.getAutoUpdateEnabledEnsuringFromApi();
    });

    ipcMain.handle(
      "settings:set-auto-update-enabled",
      (_event, enabled: boolean) => {
        this.setAutoUpdateEnabledAndUpdateApi(enabled);
        return { success: true };
      },
    );

    // Force refresh settings from backend (intended to be used when backendUrl changes)
    ipcMain.handle("settings:refresh-from-backend", async () => {
      return await this.refreshFromBackend();
    });

    // Verify a candidate backend URL and fetch backend-derived settings from it
    ipcMain.handle("settings:verify-backend-url", async (_event, url: string) => {
      return await this.verifyBackendUrlAndFetchSettings(url);
    });

    // Live memory/env knobs (applied immediately on the backend process)
    ipcMain.handle("config:get-memory-settings", async () => {
      return await this.makeConfigRequest<any>("GET", "/config/memory");
    });

    ipcMain.handle(
      "config:set-memory-settings",
      async (_event, payload: Record<string, any>) => {
        return await this.makeConfigRequest<any>("POST", "/config/memory", payload);
      },
    );

    // Fetch backend save-path sizes (in bytes) from the current (or provided) backend URL
    ipcMain.handle(
      "settings:get-backend-path-sizes",
      async (_event, url?: string | null) => {
        return await this.fetchBackendPathSizesAtUrl(url ?? null);
      },
    );
  }
}

export function settingsModule(
  ...args: ConstructorParameters<typeof SettingsModule>
) {
  if (!settingsInstance) {
    settingsInstance = new SettingsModule(...args);
  }
  return settingsInstance;
}

export function getSettingsModule() {
  if (!settingsInstance) {
    settingsInstance = new SettingsModule();
  }
  return settingsInstance;
}
