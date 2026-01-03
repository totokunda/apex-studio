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
}

interface ConfigResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

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
  private settingsPath: string | null = null;

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
    this.backendUrl = this.readBackendUrlFromDisk(this.settingsPath);
    this.registerHandlers();
  }

  getBackendUrl(): string {
    return this.backendUrl;
  }

  async setBackendUrl(url: string): Promise<void> {
    this.backendUrl = url;
    this.saveBackendUrlToDisk();
    this.emit("backend-url-changed", url);
  }

  private saveBackendUrlToDisk(): void {
    if (!this.settingsPath) return;
    try {
      // Create simple object matching ApexApi's previous structure
      const settings = {
        backendUrl: this.backendUrl,
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
  ): Promise<ConfigResponse<T>> {
    try {
      const options: RequestInit = {
        method,
        headers: {
          "Content-Type": "application/json",
        },
      };

      if (body && method !== "GET") {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${this.backendUrl}${endpoint}`, options);
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
