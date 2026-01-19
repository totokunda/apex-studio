import type { AppInitConfig } from "./AppInitConfig.js";
import { createModuleRunner } from "./ModuleRunner.js";
import { disallowMultipleAppInstance } from "./modules/SingleInstanceApp.js";
import { createWindowManagerModule } from "./modules/WindowManager.js";
import { terminateAppOnLastWindowClose } from "./modules/ApplicationTerminatorOnLastWindowClose.js";
import { hardwareAccelerationMode } from "./modules/HardwareAccelerationModule.js";
import { autoUpdater } from "./modules/AutoUpdater.js";
import { allowInternalOrigins } from "./modules/BlockNotAllowdOrigins.js";
import { allowExternalUrls } from "./modules/ExternalUrls.js";
import { chromeDevToolsExtension } from "./modules/ChromeDevToolsExtension.js";
import { createNativeFileDialogModule } from "./modules/NativeFileDialog.js";
import { apexApi } from "./modules/ApexApi.js";
import { settingsModule } from "./modules/SettingsModule.js";
import { jsonPersistenceModule } from "./modules/JSONPersistenceModule.js";
import { appDirProtocol } from "./modules/AppDirProtocol.js";
import { pythonProcess } from "./modules/PythonProcess.js";
import { remoteVersioningModule } from "./modules/RemoteVersioningModule.js";
import { installerModule } from "./modules/InstallerModule.js";
import { launcherStatusModule } from "./modules/LauncherStatusModule.js";
import { apexApiAutoUpdater } from "./modules/ApexApiAutoUpdater.js";
import { app } from "electron";

export async function initApp(initConfig: AppInitConfig) {
  // Set app name early, before any modules are initialized
  app.setName("Apex Studio");

  // Consider "dev mode" only when the renderer is served from an http(s) dev server.
  // In production we now load the renderer via `app://renderer/index.html`, which is also a URL,
  // so `instanceof URL` alone is not sufficient.
  const isDev =
    initConfig.renderer instanceof URL &&
    (initConfig.renderer.protocol === "http:" ||
      initConfig.renderer.protocol === "https:");

  let moduleRunner = createModuleRunner()
    // Ensure single instance lock before any window creation
    .init(disallowMultipleAppInstance())
    // Settings must be loaded early (before other modules read them)
    .init(settingsModule())
    // Register 'app://' protocol before app is ready and before creating the window
    .init(appDirProtocol())
    // Python process management - starts bundled API in production
    .init(pythonProcess({ devMode: isDev, autoStart: !isDev }))
    // Launcher readiness/status aggregation (installer gating + background polling)
    .init(launcherStatusModule())
    // Remote server bundle version discovery (GitHub releases)
    .init(remoteVersioningModule())
    // Local installer: extract server bundles + ensure ffmpeg is available
    .init(installerModule())
    // Core backend IPC and persistence should be ready before any renderer windows load
    .init(apexApi())
    .init(jsonPersistenceModule())
    .init(
      createWindowManagerModule({
        initConfig,
        // main/preload are built even in dev-mode, so import.meta.env.DEV isn't reliable here.
        // Auto-open devtools only for dev-server renderer URLs.
        // You can still force it for debugging by setting APEX_OPEN_DEVTOOLS=1.
        openDevTools: isDev || process.env.APEX_OPEN_DEVTOOLS === "1",
      }),
    )
    .init(terminateAppOnLastWindowClose())
    .init(hardwareAccelerationMode({ enable: true }))
    .init(autoUpdater())
    .init(apexApiAutoUpdater())
    .init(chromeDevToolsExtension({ extension: "REACT_DEVELOPER_TOOLS" }))
    // Security
    .init(
      allowInternalOrigins(
        new Set(
          initConfig.renderer instanceof URL
            ? [initConfig.renderer.origin]
            : [],
        ),
      ),
    )
    .init(
      allowExternalUrls(
        new Set(
          initConfig.renderer instanceof URL
            ? [
                "https://vite.dev",
                "https://developer.mozilla.org",
                "https://solidjs.com",
                "https://qwik.dev",
                "https://lit.dev",
                "https://react.dev",
                "https://preactjs.com",
                "https://www.typescriptlang.org",
                "https://vuejs.org",
              ]
            : [],
        ),
      ),
    );
  // Native file dialog for renderer via preload
  moduleRunner = moduleRunner.init(createNativeFileDialogModule());

  await moduleRunner;
}
