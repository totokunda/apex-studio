import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react(), tailwindcss(), tsconfigPaths()],
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