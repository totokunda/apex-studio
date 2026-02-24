import type { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { BrowserWindow, ipcMain, nativeTheme } from "electron";
import type { AppInitConfig } from "../AppInitConfig.js";
import { getPythonProcess } from "./PythonProcess.js";

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
    if (ipcMain.listenerCount("launcher:launch") > 0) return;
    ipcMain.handle("launcher:launch", async () => {
      await this.restoreOrCreateMainWindow(true);
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

  #enableWindowsFullscreenEscapeHatch(browserWindow: BrowserWindow) {
    if (process.platform !== "win32") return;
    browserWindow.webContents.on("before-input-event", (_event, input) => {
      if (browserWindow.isDestroyed()) return;
      if (!browserWindow.isFullScreen()) return;

      const key = input.key;
      const isKeyDown = input.type === "keyDown";
      if (!isKeyDown) return;

      if (key === "Escape" || key === "F11") {
        browserWindow.setFullScreen(false);
      }
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
    if (process.platform === "win32") {
      nativeTheme.themeSource = "dark";
    }
    const browserWindow = new BrowserWindow({
      show: false,
      width: 980,
      height: 640,
      minWidth: 720,
      minHeight: 520,
      backgroundColor: "#000000",
      ...(process.platform === "win32"
        ? {
            darkTheme: true,
            autoHideMenuBar: true,
          }
        : {}),
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        nodeIntegrationInWorker: true,
        nodeIntegrationInSubFrames: true,
        sandbox: false,
        webviewTag: false,
        preload: this.#preload.path,
      },
    });
    if (process.platform === "win32") {
      try {
        browserWindow.setMenuBarVisibility(false);
      } catch {
        // ignore
      }
    }

    this.#wireReadyToShow(browserWindow);
    await this.#loadRenderer(browserWindow, "launcher");
    if (this.#openDevTools) {
      try {
        browserWindow.webContents.openDevTools({ mode: "detach" });
      } catch {
        // ignore
      }
    }

    browserWindow.on("close", () => {
      const py = getPythonProcess();
      if (!py) return;
      void py.stop().catch((error) => {
        console.warn("[WindowManager] Failed to stop Python API on launcher close:", error);
      });
    });

    browserWindow.on("closed", () => {
      const py = getPythonProcess();
      if (py) {
        void py.stop().catch((error) => {
          console.warn(
            "[WindowManager] Failed to stop Python API on launcher closed backstop:",
            error,
          );
        });
      }
      if (this.#launcherWindow === browserWindow) {
        this.#launcherWindow = null;
      }
    });

    return browserWindow;
  }

  async createMainWindow(): Promise<BrowserWindow> {
    const browserWindow = new BrowserWindow({
      show: false,
      transparent: true,
      backgroundColor: "#00000000",
      ...(process.platform === "win32"
        ? {
            fullscreen: false,
            autoHideMenuBar: true,
          }
        : { fullscreen: true }),
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: true,
        nodeIntegrationInWorker: true,
        nodeIntegrationInSubFrames: true,
        sandbox: false,
        webviewTag: false,
        backgroundThrottling: false,
        spellcheck: false,
        v8CacheOptions: "code",
        preload: this.#preload.path,
      },
    });

    this.#enableWindowsFullscreenEscapeHatch(browserWindow);
    if (process.platform === "win32") {
      try {
        browserWindow.setMenuBarVisibility(false);
      } catch {
        // ignore
      }
    }
    this.#wireReadyToShow(browserWindow);
    await this.#loadRenderer(browserWindow, "main");
    if (this.#openDevTools) {
      try {
        browserWindow.webContents.openDevTools({ mode: "detach" });
      } catch {
        // ignore
      }
    }

    try {
      browserWindow.webContents.setBackgroundThrottling(false);
    } catch {
      // ignore
    }

    if (process.platform === "win32") {
      try {
        browserWindow.maximize();
      } catch {
        // ignore
      }
    }

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
