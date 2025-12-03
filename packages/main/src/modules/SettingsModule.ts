import { AppModule } from "src/AppModule.js";
import { ModuleContext } from "src/ModuleContext.js";
import Store from "electron-store";
import { ipcMain } from "electron";
import fs from "node:fs";
import path from "node:path";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8765";

type PathKey =
  | "cachePath"
  | "componentsPath"
  | "configPath"
  | "loraPath"
  | "preprocessorPath"
  | "postprocessorPath";

interface Settings {
  activeProjectId?: string | number | null;
  cachePath?: string | null;
  componentsPath?: string | null;
  configPath?: string | null;
  loraPath?: string | null;
  preprocessorPath?: string | null;
  postprocessorPath?: string | null;
  hfToken?: string | null;
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
>;

export class SettingsModule implements AppModule {
  private store: Store<Settings>;
  private backendUrl: string = DEFAULT_BACKEND_URL;
  private settingsPath: string | null = null;

  constructor() {
    this.store = new Store<Settings>();
  }

  enable({ app }: ModuleContext): void {
    this.settingsPath = path.join(app.getPath("userData"), "apex-settings.json");
    this.backendUrl = this.readBackendUrlFromDisk(this.settingsPath);
    this.registerHandlers();
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

  private async getAllPathsEnsuringFromApi(): Promise<PathsPayload> {
    const keys: PathKey[] = [
      "cachePath",
      "componentsPath",
      "configPath",
      "loraPath",
      "preprocessorPath",
      "postprocessorPath",
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

    // Bulk getters/setters for all paths
    ipcMain.handle("settings:get-all-paths", async () => {
      return await this.getAllPathsEnsuringFromApi();
    });

    ipcMain.handle(
      "settings:set-all-paths",
      async (
        _event,
        payload: Partial<PathsPayload> & { hfToken?: string | null },
      ) => {
        const keys: PathKey[] = [
          "cachePath",
          "componentsPath",
          "configPath",
          "loraPath",
          "preprocessorPath",
          "postprocessorPath",
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
  }
}

export function settingsModule(
  ...args: ConstructorParameters<typeof SettingsModule>
) {
  return new SettingsModule(...args);
}