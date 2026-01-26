import { initApp } from "@app/main";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

// In CI and automated tests, fail fast on unhandled errors.
if (
  process.env.PLAYWRIGHT_TEST === "true" ||
  !!process.env.CI
) {
  function showAndExit(...args) {
    console.error(...args);
    process.exit(1);
  }

  process.on("uncaughtException", showAndExit);
  process.on("unhandledRejection", showAndExit);
} else if (process.env.NODE_ENV === "development") {
  // In interactive development, log unhandled errors but keep Electron running
  function logError(...args) {
    console.error("[unhandled]", ...args);
  }

  process.on("uncaughtException", logError);
  process.on("unhandledRejection", logError);
}

// noinspection JSIgnoredPromiseFromCall
/**
 * We resolve the built artifacts for '@app/renderer' and '@app/preload'
 * here and not in '@app/main'
 * to observe good practices of modular design.
 * This allows fewer dependencies and better separation of concerns in '@app/main'.
 * Thus,
 * the main module remains simplistic and efficient
 * as it receives initialization instructions rather than direct module imports.
 */
initApp({
  renderer: (() => {
    const useDevServer =
      process.env.MODE === "development" && !!process.env.VITE_DEV_SERVER_URL;
    const renderer = useDevServer
      ? new URL(process.env.VITE_DEV_SERVER_URL)
      : {
          // In packaged builds, load renderer via `app://renderer/...` so Electron can
          // serve JS with correct Content-Type (critical for module workers).
          // The `app://` handler maps `renderer` host to @app/renderer/dist.
          //
          // NOTE: This must be a URL (not loadFile) so the origin isn't `file://`.
          // `WindowManager` will call `loadURL(...)` for URL renderers.
          //
          // Keep the old path resolution in place as a fallback for logs/debugging.
          path: require.resolve("@app/renderer/dist/index.html"),
        };
    const resolved =
      renderer instanceof URL ? renderer : new URL("app://renderer/index.html");
    console.log(
      `[entry-point] renderer: ${
        resolved instanceof URL ? resolved.href : resolved.path
      }`,
    );
    return resolved;
  })(),

  preload: {
    path: require.resolve("@app/preload/exposed.mjs"),
  },
});
