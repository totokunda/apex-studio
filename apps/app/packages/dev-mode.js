import { build, createServer } from "vite";
import path from "path";
import { spawn } from "node:child_process";
import electron from "electron";
import fs from "node:fs";

/**
 * This script is designed to run multiple packages of your application in a special development mode.
 * To do this, you need to follow a few steps:
 */

/**
 * 1. We create a few flags to let everyone know that we are in development mode.
 */
const mode = "development";
process.env.NODE_ENV = mode;
process.env.MODE = mode;
// Ensure only this script spawns Electron; signal others to skip spawning
process.env.ELECTRON_SPAWN_MANAGED = "1";

/**
 * 2. We create a development server for the renderer. It is assumed that the renderer exists and is located in the “renderer” package.
 * This server should be started first because other packages depend on its settings.
 */
/**
 * @type {import('vite').ViteDevServer}
 */
const rendererWatchServer = await createServer({
  mode,
  root: path.resolve("packages/renderer"),
});

await rendererWatchServer.listen();

// Derive the renderer dev URL and expose it to Electron via env
const resolvedUrls = rendererWatchServer.resolvedUrls?.local ?? [];
const rendererDevUrl =
  resolvedUrls[0] ??
  `http://localhost:${rendererWatchServer.config.server.port}/`;
process.env.VITE_DEV_SERVER_URL = rendererDevUrl;

/**
 * 3. We are creating a simple provider plugin.
 * Its only purpose is to provide access to the renderer dev-server to all other build processes.
 */
/** @type {import('vite').Plugin} */
const rendererWatchServerProvider = {
  name: "@app/renderer-watch-server-provider",
  api: {
    provideRendererWatchServer() {
      return rendererWatchServer;
    },
  },
};

/**
 * 4. Start building all other packages.
 * For each of them, we add a plugin provider so that each package can implement its own hot update mechanism.
 */

/** @type {string[]} */
const packagesToStart = [
  "packages/preload",
  "packages/main",
  "packages/export-renderer",
];

async function waitForFile(absPath, options) {
  const timeoutMs =
    typeof options?.timeoutMs === "number" && Number.isFinite(options.timeoutMs)
      ? Math.max(0, options.timeoutMs)
      : 20_000;
  const pollMs =
    typeof options?.pollMs === "number" && Number.isFinite(options.pollMs)
      ? Math.max(10, options.pollMs)
      : 50;
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      if (fs.existsSync(absPath)) return;
    } catch {}
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, pollMs));
  }
  throw new Error(`Timed out waiting for build artifact: ${absPath}`);
}

for (const pkg of packagesToStart) {
  await build({
    mode,
    root: path.resolve(pkg),
    plugins: [rendererWatchServerProvider],
  });

  // Ensure the required build outputs exist before launching Electron.
  // Some packages enable Vite/Rollup watch mode in development, and `build()`
  // may return before the initial write completes.
  if (pkg === "packages/preload") {
    await waitForFile(
      path.resolve("packages/preload/dist/exposed.mjs"),
      { timeoutMs: 30_000 },
    );
    await waitForFile(
      path.resolve("packages/preload/dist/_virtual_browser.mjs"),
      { timeoutMs: 30_000 },
    );
  } else if (pkg === "packages/main") {
    await waitForFile(path.resolve("packages/main/dist/index.js"), {
      timeoutMs: 30_000,
    });
  } else if (pkg === "packages/export-renderer") {
    await waitForFile(path.resolve("packages/export-renderer/dist/index.js"), {
      timeoutMs: 30_000,
    });
  }
}

/**
 * 5. Launch Electron pointing at our entry point, with the renderer dev URL in env.
 */
const entryPoint = path.resolve("packages/entry-point.mjs");
const electronProc = spawn(electron, [entryPoint], {
  stdio: "inherit",
  env: {
    ...process.env,
    NODE_ENV: mode,
    MODE: mode,
    ELECTRON_SPAWN_MANAGED: "1",
    VITE_DEV_SERVER_URL: rendererDevUrl,
  },
});

electronProc.on("close", async (code) => {
  try {
    await rendererWatchServer.close();
  } catch {}
  process.exit(code ?? 0);
});

process.on("SIGINT", async () => {
  try {
    electronProc.kill("SIGINT");
  } catch {}
  try {
    await rendererWatchServer.close();
  } catch {}
  process.exit(0);
});