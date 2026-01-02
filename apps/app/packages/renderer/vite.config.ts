import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss(), tsconfigPaths()],
  // When packaged, the renderer is loaded via `BrowserWindow.loadFile(...)` (file://...).
  // Vite's default `base: "/"` would emit `/assets/...` URLs which break under file://.
  // Use a relative base for production builds so CSS/JS load correctly from dist/.
  base: mode === "development" ? "/" : "./",
  resolve: {
    alias: {
      "@app/export-renderer": path.resolve(
        __dirname,
        "../export-renderer/src/index.ts",
      ),
    },
  },
  server: {
    fs: {
      allow: [
        __dirname,
        path.resolve(__dirname, "../export-renderer"),
        path.resolve(__dirname, "..", ".."),
      ],
    },
  },
}));