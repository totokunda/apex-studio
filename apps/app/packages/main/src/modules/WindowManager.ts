import type { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";
import { BrowserWindow, ipcMain, nativeTheme } from "electron";
import type { AppInitConfig } from "../AppInitConfig.js";
import { fileURLToPath } from "node:url";
import { getPythonProcess } from "./PythonProcess.js";
import {
  createMediaNativeEngine,
  MediaNativeClipKind,
  MediaNativeCommandType,
  type MediaNativeEngine,
} from "@app/media-native";

type MediaNativePreviewClipSyncPayload = {
  clipId: string;
  clipType: string;
  assetId?: string;
  mediaPath?: string;
  timelineId?: string;
  startFrame?: number;
  endFrame?: number;
  trimStart?: number;
  trimEnd?: number;
  speed?: number;
  hidden?: boolean;
  transform?: {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    scaleX?: number;
    scaleY?: number;
    rotation?: number;
    opacity?: number;
    cornerRadius?: number;
    crop?: {
      x?: number;
      y?: number;
      width?: number;
      height?: number;
    };
  };
  adjustments?: {
    brightness?: number;
    contrast?: number;
    hue?: number;
    saturation?: number;
    blur?: number;
    sharpness?: number;
    noise?: number;
    vignette?: number;
    scanLines?: number;
    chromaticAberration?: number;
    interlace?: number;
    pixelate?: number;
    jitter?: number;
    colorTintColor?: string;
    colorTintIntensity?: number;
  };
  zIndex?: number;
};

const mapClipTypeToKind = (clipType: string): MediaNativeClipKind => {
  switch (clipType) {
    case "video":
      return MediaNativeClipKind.Video;
    case "image":
      return MediaNativeClipKind.Image;
    case "model":
      return MediaNativeClipKind.Model;
    case "shape":
      return MediaNativeClipKind.Shape;
    case "text":
      return MediaNativeClipKind.Text;
    case "draw":
      return MediaNativeClipKind.Drawing;
    case "audio":
      return MediaNativeClipKind.Audio;
    default:
      return MediaNativeClipKind.Unknown;
  }
};

const toFiniteNumber = (value: unknown, fallback: number): number => {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
};

const normalizeMediaPath = (mediaPath: unknown): string => {
  const raw = String(mediaPath ?? "").trim();
  if (!raw) return "";
  if (!raw.startsWith("file://")) return raw;
  try {
    return fileURLToPath(raw);
  } catch {
    return raw;
  }
};

const mapUpsertPayloadToNativeCommand = (
  payload: MediaNativePreviewClipSyncPayload,
) => {
  const transform = payload.transform ?? {};
  const crop = transform.crop ?? {};
  const adjustments = payload.adjustments ?? {};

  return {
    clip_id: String(payload.clipId ?? ""),
    clip_kind: mapClipTypeToKind(String(payload.clipType ?? "")),
    asset_id: String(payload.assetId ?? ""),
    media_path: normalizeMediaPath(payload.mediaPath),
    timeline: {
      start_frame: Math.max(0, Math.round(toFiniteNumber(payload.startFrame, 0))),
      end_frame: Math.max(0, Math.round(toFiniteNumber(payload.endFrame, 0))),
      trim_start: Math.max(0, Math.round(toFiniteNumber(payload.trimStart, 0))),
      trim_end: Math.max(0, Math.round(toFiniteNumber(payload.trimEnd, 0))),
      speed: Math.max(0.01, toFiniteNumber(payload.speed, 1)),
    },
    transform: {
      x: toFiniteNumber(transform.x, 0),
      y: toFiniteNumber(transform.y, 0),
      width: Math.max(1, toFiniteNumber(transform.width, 1)),
      height: Math.max(1, toFiniteNumber(transform.height, 1)),
      scale_x: toFiniteNumber(transform.scaleX, 1),
      scale_y: toFiniteNumber(transform.scaleY, 1),
      rotation_deg: toFiniteNumber(transform.rotation, 0),
      opacity: Math.max(0, Math.min(1, toFiniteNumber(transform.opacity, 1))),
      corner_radius: Math.max(0, toFiniteNumber(transform.cornerRadius, 0)),
      visible: !payload.hidden,
      has_crop: Boolean(transform.crop),
      crop: {
        x: Math.max(0, Math.min(1, toFiniteNumber(crop.x, 0))),
        y: Math.max(0, Math.min(1, toFiniteNumber(crop.y, 0))),
        width: Math.max(0, Math.min(1, toFiniteNumber(crop.width, 1))),
        height: Math.max(0, Math.min(1, toFiniteNumber(crop.height, 1))),
      },
    },
    filters: {
      brightness: toFiniteNumber(adjustments.brightness, 0),
      contrast: toFiniteNumber(adjustments.contrast, 0),
      hue: toFiniteNumber(adjustments.hue, 0),
      saturation: toFiniteNumber(adjustments.saturation, 0),
      blur: toFiniteNumber(adjustments.blur, 0),
      sharpness: toFiniteNumber(adjustments.sharpness, 0),
      noise: toFiniteNumber(adjustments.noise, 0),
      vignette: toFiniteNumber(adjustments.vignette, 0),
      scan_lines: toFiniteNumber(adjustments.scanLines, 0),
      chromatic_aberration: toFiniteNumber(adjustments.chromaticAberration, 0),
      interlace: toFiniteNumber(adjustments.interlace, 0),
      pixelate: toFiniteNumber(adjustments.pixelate, 0),
      jitter: toFiniteNumber(adjustments.jitter, 0),
      color_tint_color_hex: String(adjustments.colorTintColor ?? "#000000"),
      color_tint_intensity: toFiniteNumber(adjustments.colorTintIntensity, 0),
    },
    luts: [],
    masks: [],
    z_index: Math.max(0, Math.round(toFiniteNumber(payload.zIndex, 0))),
  };
};

const shouldLogMediaNativeDebug = true;


class WindowManager implements AppModule {
  readonly #preload: { path: string };
  readonly #renderer: { path: string } | URL;
  readonly #openDevTools: boolean;
  #launcherWindow: BrowserWindow | null = null;
  #mainWindow: BrowserWindow | null = null;
  #mediaNativeEngine: MediaNativeEngine | null = null;
  #isHolePlayerStarted = false;
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
      this.#registerMediaNativePreviewIpc();
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

  #registerMediaNativePreviewIpc() {
    if (ipcMain.listenerCount("media-native-preview:start") === 0) {
      ipcMain.handle(
        "media-native-preview:start",
        async (_event, _payload: { videoPath?: string }) => {
          const win = this.#mainWindow;
          if (!win || win.isDestroyed()) {
            return { ok: false, error: "main window is not available" };
          }

          try {
            const webContentsAny = win.webContents as any;
            const nativeHandle: Buffer =
              typeof webContentsAny.getNativeWindowHandle === "function"
                ? webContentsAny.getNativeWindowHandle()
                : win.getNativeWindowHandle();

            this.#attachMediaNativeEngine(nativeHandle, win);
            this.#isHolePlayerStarted = true;
            return { ok: true };
          } catch (error: any) {
            console.error("[media-native-preview:start] failed:", error);
            return {
              ok: false,
              error: error?.message ?? "failed to start media-native preview",
            };
          }
        },
      );
    }

    if (ipcMain.listenerCount("media-native-preview:set-rect") === 0) {
      ipcMain.handle(
        "media-native-preview:set-rect",
        async (
          _event,
          rect: {
            left?: number;
            top?: number;
            width?: number;
            height?: number;
            visible?: boolean;
          },
        ) => {
          if (!this.#isHolePlayerStarted) {
            return { ok: false, error: "hole player is not started" };
          }
          const win = this.#mainWindow;
          if (!win || win.isDestroyed()) {
            return { ok: false, error: "main window is not available" };
          }

          try {
            const contentSize = win.getContentSize();
            const incomingLeft = Math.round(Number(rect?.left) || 0);
            const incomingTop = Math.round(Number(rect?.top) || 0);
            const incomingWidth = Math.max(1, Math.round(Number(rect?.width) || 1));
            const incomingHeight = Math.max(1, Math.round(Number(rect?.height) || 1));

            const x = Math.max(0, incomingLeft);
            const y = Math.max(0, incomingTop);
            const width = Math.min(incomingWidth, Math.max(1, contentSize[0] - x));
            const height = Math.min(incomingHeight, Math.max(1, contentSize[1] - y));
            const visible = Boolean(rect?.visible);

            this.#setMediaNativeHoleRect({ x, y, width, height, visible });
            return { ok: true };
          } catch (error: any) {
            console.error("[media-native-preview:set-rect] failed:", error);
            return {
              ok: false,
              error: error?.message ?? "failed to set native hole rect",
            };
          }
        },
      );
    }

    if (ipcMain.listenerCount("media-native-preview:stop") === 0) {
      ipcMain.handle("media-native-preview:stop", async () => {
        this.#stopMediaNativeEngine();
        this.#isHolePlayerStarted = false;
        return { ok: true };
      });
    }

    if (ipcMain.listenerCount("media-native-preview:upsert-clip") === 0) {
      ipcMain.handle(
        "media-native-preview:upsert-clip",
        async (_event, payload: MediaNativePreviewClipSyncPayload) => {
          if (!this.#mediaNativeEngine) {
            return { ok: true, skipped: "engine-not-ready" };
          }
          try {
            const clipId = String(payload?.clipId ?? "").trim();
            if (!clipId) return { ok: false, error: "clipId is required" };
            const nativePayload = mapUpsertPayloadToNativeCommand(payload);
            if (shouldLogMediaNativeDebug) {
              console.log("[media-native-preview:upsert-clip]", {
                clipId: nativePayload.clip_id,
                kind: nativePayload.clip_kind,
                mediaPath: nativePayload.media_path,
                timeline: nativePayload.timeline,
                visible: nativePayload.transform.visible,
                zIndex: nativePayload.z_index,
              });
            }
            this.#mediaNativeEngine.submit(
              MediaNativeCommandType.UpsertClip,
              nativePayload,
            );
            return { ok: true };
          } catch (error: any) {
            console.error("[media-native-preview:upsert-clip] failed:", error);
            return {
              ok: false,
              error: error?.message ?? "failed to upsert media-native clip",
            };
          }
        },
      );
    }

    if (ipcMain.listenerCount("media-native-preview:remove-clip") === 0) {
      ipcMain.handle(
        "media-native-preview:remove-clip",
        async (_event, payload: { clipId?: string }) => {
          if (!this.#mediaNativeEngine) {
            return { ok: true, skipped: "engine-not-ready" };
          }
          try {
            const clipId = String(payload?.clipId ?? "").trim();
            if (!clipId) return { ok: false, error: "clipId is required" };
            this.#mediaNativeEngine.submit(MediaNativeCommandType.RemoveClip, {
              clip_id: clipId,
            });
            return { ok: true };
          } catch (error: any) {
            console.error("[media-native-preview:remove-clip] failed:", error);
            return {
              ok: false,
              error: error?.message ?? "failed to remove media-native clip",
            };
          }
        },
      );
    }

    if (ipcMain.listenerCount("media-native-preview:set-playhead") === 0) {
      ipcMain.handle(
        "media-native-preview:set-playhead",
        async (
          _event,
          payload: { focusFrame?: number; fps?: number; accurateSeek?: boolean },
        ) => {
          if (!this.#mediaNativeEngine) {
            return { ok: true, skipped: "engine-not-ready" };
          }
          try {
            const focusFrame = Math.max(
              0,
              Math.round(toFiniteNumber(payload?.focusFrame, 0)),
            );
            const fps = Math.max(1, toFiniteNumber(payload?.fps, 24));
            const accurateSeek = Boolean(payload?.accurateSeek);
            this.#mediaNativeEngine.submit(MediaNativeCommandType.SetPlayhead, {
              focus_frame: focusFrame,
              fps,
              accurate_seek: accurateSeek,
            });
            return { ok: true };
          } catch (error: any) {
            console.error("[media-native-preview:set-playhead] failed:", error);
            return {
              ok: false,
              error: error?.message ?? "failed to update media-native playhead",
            };
          }
        },
      );
    }

    if (ipcMain.listenerCount("media-native-preview:set-play-state") === 0) {
      ipcMain.handle(
        "media-native-preview:set-play-state",
        async (_event, payload: { isPlaying?: boolean }) => {
          if (!this.#mediaNativeEngine) {
            return { ok: true, skipped: "engine-not-ready" };
          }
          try {
            const isPlaying = Boolean(payload?.isPlaying);
            this.#mediaNativeEngine.submit(MediaNativeCommandType.SetPlayState, {
              is_playing: isPlaying,
            });
            return { ok: true };
          } catch (error: any) {
            console.error("[media-native-preview:set-play-state] failed:", error);
            return {
              ok: false,
              error:
                error?.message ?? "failed to update media-native play state",
            };
          }
        },
      );
    }

    if (ipcMain.listenerCount("media-native-preview:stats") === 0) {
      ipcMain.handle("media-native-preview:stats", async () => {
        if (!this.#mediaNativeEngine) {
          return { ok: true, skipped: "engine-not-ready" };
        }
        try {
          return {
            ok: true,
            stats: this.#mediaNativeEngine.stats(),
          };
        } catch (error: any) {
          console.error("[media-native-preview:stats] failed:", error);
          return {
            ok: false,
            error: error?.message ?? "failed to fetch media-native stats",
          };
        }
      });
    }
  }


  #wireReadyToShow(browserWindow: BrowserWindow) {
    browserWindow.once("ready-to-show", () => {
      if (browserWindow.isDestroyed()) return;
      browserWindow.show();
      browserWindow.focus();
    });
  }

  #applyWindowsTopbarFixes(browserWindow: BrowserWindow) {
    if (process.platform !== "win32") return;
    try {
      browserWindow.setTitleBarOverlay?.({
        color: "#000000",
        symbolColor: "#FFFFFF",
      });
    } catch {
      // ignore
    }
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
      this.#stopMediaNativeEngine();
      this.#isHolePlayerStarted = false;
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

  #attachMediaNativeEngine(
    nativeHandle: Buffer,
    win: BrowserWindow,
  ) {
    try {
      if (!this.#mediaNativeEngine) {
        const [width, height] = win.getContentSize();
        this.#mediaNativeEngine = createMediaNativeEngine({
          width,
          height,
          fps: 24,
        });
      }

      this.#mediaNativeEngine.attachSurface(nativeHandle);
      const [width, height] = win.getContentSize();
      this.#mediaNativeEngine.submit(MediaNativeCommandType.SetViewport, {
        width,
        height,
        scale: 1,
        stage_x: 0,
        stage_y: 0,
      });
    } catch (error) {
      console.warn("[media-native] attach failed:", error);
    }
  }

  #setMediaNativeHoleRect(rect: {
    x: number;
    y: number;
    width: number;
    height: number;
    visible: boolean;
  }) {
    if (!this.#mediaNativeEngine) return;
    try {
      this.#mediaNativeEngine.submit(MediaNativeCommandType.SetHoleRect, {
        rect: {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
        },
        visible: rect.visible,
      });
    } catch (error) {
      console.warn("[media-native] set hole rect failed:", error);
    }
  }

  #stopMediaNativeEngine() {
    if (!this.#mediaNativeEngine) return;
    try {
      this.#mediaNativeEngine.submit(MediaNativeCommandType.Shutdown, {});
    } catch {
      // ignore
    }
    try {
      this.#mediaNativeEngine.destroy();
    } catch {
      // ignore
    }
    this.#mediaNativeEngine = null;
  }
}

export function createWindowManagerModule(
  ...args: ConstructorParameters<typeof WindowManager>
) {
  return new WindowManager(...args);
}
