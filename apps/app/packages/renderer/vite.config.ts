import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import path from "node:path";
import { nodePolyfills } from "vite-plugin-node-polyfills";

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    tailwindcss(),
    tsconfigPaths(),
    // Provide path, url for native decoder loading (Electron renderer with nodeIntegration)
    nodePolyfills({
      include: ["path"],
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