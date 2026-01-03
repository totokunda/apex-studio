import type { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { BrowserWindow, ipcMain } from "electron";
import type { AppInitConfig } from "../AppInitConfig.js";

class WindowManager implements AppModule {
  readonly #preload: { path: string };
  readonly #renderer: { path: string } | URL;
  readonly #openDevTools: boolean;
  #launcherWindow: BrowserWindow | null = null;
  #mainWindow: BrowserWindow | null = null;
  #isQuitting = false;

  constructor({
    initConfig,
    openDevTools = false,
  }: {
    initConfig: AppInitConfig;
    openDevTools?: boolean;
  }) {
    this.#preload = initConfig.preload;
    this.#renderer = initConfig.renderer;
    this.#openDevTools = Boolean(openDevTools);
  }

  async enable({ app }: ModuleContext): Promise<void> {
    await app.whenReady();
    try {
      app.on("before-quit", () => {
        this.#isQuitting = true;
      });
      this.#registerLauncherIpc();
      await this.restoreOrCreateLauncherWindow(true);
    } catch (error) {
      console.error("[WindowManager] Failed to create/show window:", error);
    }
    app.on("second-instance", () => this.focusBestWindow());
    app.on("activate", () => this.focusBestWindow());
  }

  #registerLauncherIpc() {
    // idempotent-ish registration: ignore if already registered
    if (ipcMain.listenerCount("launcher:launch") > 0) return;
    ipcMain.handle("launcher:launch", async () => {
      await this.restoreOrCreateMainWindow(true);
      // Keep launcher alive (hidden) so that closing the main window returns to the launcher.
      try {
        this.#launcherWindow?.hide();
      } catch {
        // ignore
      }
      return { ok: true };
    });
  }

  #wireReadyToShow(browserWindow: BrowserWindow) {
    browserWindow.once("ready-to-show", () => {
      if (browserWindow.isDestroyed()) return;
      browserWindow.show();
      browserWindow.focus();
    });
  }

  async #loadRenderer(browserWindow: BrowserWindow, mode: "launcher" | "main") {
    try {
      if (this.#renderer instanceof URL) {
        const u = new URL(this.#renderer.href);
        u.hash = mode === "launcher" ? "launcher" : "";
        await browserWindow.loadURL(u.href);
      } else {
        await browserWindow.loadFile(
          this.#renderer.path,
          mode === "launcher" ? { hash: "launcher" } : undefined,
        );
      }
    } catch (error) {
      console.error("[WindowManager] Failed to load renderer:", error);
    }
  }

  async createLauncherWindow(): Promise<BrowserWindow> {
    const browserWindow = new BrowserWindow({
      show: false, // Use the 'ready-to-show' event to show the instantiated BrowserWindow.
      width: 980,
      height: 640,
      minWidth: 720,
      minHeight: 520,
      backgroundColor: "#000000",
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: false, // Sandbox disabled because the demo of preload script depend on the Node.js api
        webviewTag: false, // The webview tag is not recommended. Consider alternatives like an iframe or Electron's BrowserView. @see https://www.electronjs.org/docs/latest/api/webview-tag#warning
        preload: this.#preload.path,
      },
    });
    this.#wireReadyToShow(browserWindow);
    await this.#loadRenderer(browserWindow, "launcher");
    if (this.#openDevTools) {
      try {
        browserWindow.webContents.openDevTools({ mode: "detach" });
      } catch {
        // ignore
      }
    }

    browserWindow.on("closed", () => {
      if (this.#launcherWindow === browserWindow) {
        this.#launcherWindow = null;
      }
    });

    return browserWindow;
  }

  async createMainWindow(): Promise<BrowserWindow> {
    const browserWindow = new BrowserWindow({
      show: false,
      fullscreen: true,
      backgroundColor: "#000000",
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: false,
        webviewTag: false,
        preload: this.#preload.path,
      },
    });
    this.#wireReadyToShow(browserWindow);
    await this.#loadRenderer(browserWindow, "main");
    if (this.#openDevTools) {
      try {
        browserWindow.webContents.openDevTools({ mode: "detach" });
      } catch {
        // ignore
      }
    }

    // When the main window closes (user clicks X), return to the launcher.
    browserWindow.on("closed", () => {
      if (this.#mainWindow === browserWindow) {
        this.#mainWindow = null;
      }
      if (this.#isQuitting) return;
      void this.restoreOrCreateLauncherWindow(true);
    });

    return browserWindow;
  }

  async restoreOrCreateLauncherWindow(show = false) {
    let window = this.#launcherWindow;
    if (!window || window.isDestroyed()) {
      window = await this.createLauncherWindow();
      this.#launcherWindow = window;
    }

    if (!show) {
      return window;
    }

    if (window.isMinimized()) {
      window.restore();
    }

    window?.show();

    window.focus();

    return window;
  }

  async restoreOrCreateMainWindow(show = false) {
    let window = this.#mainWindow;
    if (!window || window.isDestroyed()) {
      window = await this.createMainWindow();
      this.#mainWindow = window;
    }

    if (!show) {
      return window;
    }

    if (window.isMinimized()) {
      window.restore();
    }

    window?.show();
    window.focus();
    return window;
  }

  focusBestWindow() {
    const main = this.#mainWindow;
    if (main && !main.isDestroyed()) {
      void this.restoreOrCreateMainWindow(true);
      return;
    }
    void this.restoreOrCreateLauncherWindow(true);
  }
}

export function createWindowManagerModule(
  ...args: ConstructorParameters<typeof WindowManager>
) {
  return new WindowManager(...args);
}
