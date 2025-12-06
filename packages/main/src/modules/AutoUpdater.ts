import { AppModule } from "../AppModule.js";
import electronUpdater, {
  type AppUpdater,
  type Logger,
} from "electron-updater";

type DownloadNotification = Parameters<
  AppUpdater["checkForUpdatesAndNotify"]
>[0];

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
    if (!import.meta.env.PROD) {
      return;
    }

    try {
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

  async runAutoUpdater() {
    const updater = this.getAutoUpdater();
    try {
      updater.logger = this.#logger || null;
      updater.fullChangelog = true;

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
