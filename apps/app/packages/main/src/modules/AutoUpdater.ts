import { AppModule } from "../AppModule.js";
import { BrowserWindow, ipcMain } from "electron";
import electronUpdater, {
  type AppUpdater,
  type Logger,
} from "electron-updater";

type DownloadNotification = Parameters<
  AppUpdater["checkForUpdatesAndNotify"]
>[0];

type AppUpdateEvent =
  | { type: "checking" }
  | { type: "available"; info: unknown }
  | { type: "not-available"; info: unknown }
  | { type: "progress"; progress: unknown }
  | { type: "downloaded"; info: unknown }
  | { type: "error"; message: string };

type AppUpdateState = {
  status:
    | "idle"
    | "checking"
    | "available"
    | "not-available"
    | "downloading"
    | "downloaded"
    | "error";
  updateInfo?: unknown;
  progress?: unknown;
  errorMessage?: string;
  lastCheckedAt?: number;
};

const APP_UPDATE_EVENT_CHANNEL = "app-update:event";

let lastKnownState: AppUpdateState = { status: "idle" };

function broadcastAppUpdateEvent(ev: AppUpdateEvent) {
  // Best-effort broadcast to all windows. We intentionally avoid hard dependencies on WindowManager.
  for (const win of BrowserWindow.getAllWindows()) {
    try {
      win.webContents.send(APP_UPDATE_EVENT_CHANNEL, ev);
    } catch {
      // ignore
    }
  }
}

export class AutoUpdater implements AppModule {
  readonly #logger: Logger | null;
  readonly #notification: DownloadNotification;

  constructor({
    logger = null,
    downloadNotification = undefined,
  }: {
    logger?: Logger | null | undefined;
    downloadNotification?: DownloadNotification;
  } = {}) {
    this.#logger = logger;
    this.#notification = downloadNotification;
  }

  async enable(): Promise<void> {
    // Skip auto-updater in development; it's only meaningful in production

    try {
      this.registerIpc();
      this.wireAutoUpdaterEvents();
      await this.runAutoUpdater();
    } catch (error) {
      // Auto-updater failures should never prevent the app from starting
      console.warn("[AutoUpdater] Failed to run auto updater:", error);
    }
  }

  getAutoUpdater(): AppUpdater {
    // Using destructuring to access autoUpdater due to the CommonJS module of 'electron-updater'.
    // It is a workaround for ESM compatibility issues, see https://github.com/electron-userland/electron-builder/issues/7976.
    const { autoUpdater } = electronUpdater;
    return autoUpdater;
  }

  registerIpc() {
    // Idempotent-ish registration
    if (ipcMain.listenerCount("app-update:check") > 0) return;

    ipcMain.handle("app-update:get-state", async () => {
      return lastKnownState;
    });

    ipcMain.handle("app-update:check", async () => {
      const updater = this.getAutoUpdater();
      lastKnownState = { ...lastKnownState, status: "checking" };
      try {
        const res = await updater.checkForUpdates();
        const now = Date.now();

        // `electron-updater` emits events like "update-available" / "update-not-available"
        // during `checkForUpdates()`. Our event handlers update `lastKnownState.status`
        // accordingly. We must not overwrite that richer status here.
        if (!res) {
          lastKnownState = {
            ...lastKnownState,
            status: lastKnownState.status === "checking" ? "idle" : lastKnownState.status,
            lastCheckedAt: now,
          };
          return null;
        }

        const updateInfo = (res as any)?.updateInfo ?? null;
        const isUpdateAvailable = Boolean((res as any)?.isUpdateAvailable);

        lastKnownState = {
          ...lastKnownState,
          status:
            lastKnownState.status === "checking"
              ? isUpdateAvailable
                ? "available"
                : "not-available"
              : lastKnownState.status,
          updateInfo: updateInfo ?? undefined,
          lastCheckedAt: now,
        };

        return updateInfo;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to check for updates";
        lastKnownState = { ...lastKnownState, status: "error", errorMessage: message };
        throw error;
      }
    });

    ipcMain.handle("app-update:download", async () => {
      const updater = this.getAutoUpdater();
      lastKnownState = { ...lastKnownState, status: "downloading" };
      await updater.downloadUpdate();
      return { ok: true };
    });

    ipcMain.handle("app-update:install", async () => {
      const updater = this.getAutoUpdater();
      // This restarts the app. Return value is mostly irrelevant, but keep it predictable.
      updater.quitAndInstall();

      // Explicitly close all windows to ensure they don't linger during
      // potential shutdown delays (e.g. Python process cleanup).
      // We assume WindowManager receives the 'before-quit' event from quitAndInstall()
      // and sets its isQuitting flag, preventing respawns.
      for (const win of BrowserWindow.getAllWindows()) {
        try {
          win.close();
        } catch {
          // ignore
        }
      }

      return { ok: true };
    });
  }

  wireAutoUpdaterEvents() {
    const updater = this.getAutoUpdater();
    // Avoid attaching listeners multiple times (in case enable() is called again unexpectedly).
    // `electron-updater` uses Node's EventEmitter.
    if ((updater as any).__apexWired) return;
    (updater as any).__apexWired = true;

    updater.on("checking-for-update", () => {
      lastKnownState = { ...lastKnownState, status: "checking", errorMessage: undefined };
      broadcastAppUpdateEvent({ type: "checking" });
    });

    updater.on("update-available", (info: unknown) => {
      lastKnownState = { ...lastKnownState, status: "available", updateInfo: info };
      broadcastAppUpdateEvent({ type: "available", info });
    });

    updater.on("update-not-available", (info: unknown) => {
      lastKnownState = { ...lastKnownState, status: "not-available", updateInfo: info };
      broadcastAppUpdateEvent({ type: "not-available", info });
    });

    updater.on("download-progress", (progress: unknown) => {
      lastKnownState = { ...lastKnownState, status: "downloading", progress };
      broadcastAppUpdateEvent({ type: "progress", progress });
    });

    updater.on("update-downloaded", (info: unknown) => {
      lastKnownState = { ...lastKnownState, status: "downloaded", updateInfo: info };
      broadcastAppUpdateEvent({ type: "downloaded", info });
    });

    updater.on("error", (error: unknown) => {
      const message =
        error instanceof Error ? error.message : "Auto-update error";
      lastKnownState = { ...lastKnownState, status: "error", errorMessage: message };
      broadcastAppUpdateEvent({ type: "error", message });
    });
  }

  async runAutoUpdater() {
    const updater = this.getAutoUpdater();
    try {
      updater.logger = this.#logger || null;
      updater.fullChangelog = true;
      // Keep the runtime UX in renderer-land; we disable implicit download and
      // let the UI trigger download/install. (Startup checks still happen.)
      updater.autoDownload = false;

      if (import.meta.env.VITE_DISTRIBUTION_CHANNEL) {
        updater.channel = import.meta.env.VITE_DISTRIBUTION_CHANNEL;
      }

      return await updater.checkForUpdatesAndNotify(this.#notification);
    } catch (error) {
      if (error instanceof Error) {
        if (error.message.includes("No published versions")) {
          return null;
        }
      }

      throw error;
    }
  }
}

export function autoUpdater(
  ...args: ConstructorParameters<typeof AutoUpdater>
) {
  return new AutoUpdater(...args);
}
