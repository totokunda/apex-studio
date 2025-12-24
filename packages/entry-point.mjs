import { initApp } from "@app/main";
import { fileURLToPath } from "node:url";

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
 * We resolve '@app/renderer' and '@app/preload'
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
      : { path: fileURLToPath(import.meta.resolve("@app/renderer")) };
    console.log(
      `[entry-point] renderer: ${
        renderer instanceof URL ? renderer.href : renderer.path
      }`,
    );
    return renderer;
  })(),

  preload: {
    path: fileURLToPath(import.meta.resolve("@app/preload/exposed.mjs")),
  },
});
