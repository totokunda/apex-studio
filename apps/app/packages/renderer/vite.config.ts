import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { nodePolyfills } from "vite-plugin-node-polyfills";

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    copyDecodeWorker(),
    react(),
    tailwindcss(),
    tsconfigPaths(),
    // Provide path, url for native decoder loading (Electron renderer with nodeIntegration)
    nodePolyfills({
      include: ["path", "crypto"],
      globals: { process: true },
    }),
  ],
  // When packaged, the renderer is loaded via `BrowserWindow.loadFile(...)` (file://...).
  // Vite's default `base: "/"` would emit `/assets/...` URLs which break under file://.
  // Use a relative base for production builds so CSS/JS load correctly from dist/.
  base: mode === "development" ? "/" : "./",
  resolve: {
    alias: [
      // More specific first: @app/preload/src must resolve to browser shim
      {
        find: "@app/preload/src",
        replacement: path.resolve(__dirname, "../preload/dist/_virtual_browser.mjs"),
      },
      // Force preload to the browser shim so we never bundle Node-only code (node:crypto, etc.)
      {
        find: "@app/preload",
        replacement: path.resolve(__dirname, "../preload/dist/_virtual_browser.mjs"),
      },
      {
        find: "@app/export-renderer",
        replacement: path.resolve(__dirname, "../export-renderer/src/index.ts"),
      },
      {
        find: "@app/decoder",
        replacement: path.resolve(__dirname, "../decoder/src/decode.ts"),
      },
    ],
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
    fs: {
      allow: [
        __dirname,
        path.resolve(__dirname, "../export-renderer"),
        path.resolve(__dirname, "..", ".."),
      ],
    },
  },
}));


function copyDecodeWorker() {
  return {
    name: "@app/renderer-copy-decode-worker",
    writeBundle() {
      const root = path.dirname(fileURLToPath(import.meta.url));
      const workerSrc = path.join(root, "..", "decoder", "src", "decode.worker.cjs");
      const workerDest = path.join(root, "dist", "assets", "decode.worker.cjs");
      if (!fs.existsSync(workerSrc)) return;
      fs.mkdirSync(path.dirname(workerDest), { recursive: true });
      fs.copyFileSync(workerSrc, workerDest);
    },
  };
}
