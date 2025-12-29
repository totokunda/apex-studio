import path from "path";
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default /**
 * @type {import('vite').UserConfig}
 */ ({
  build: {
    sourcemap: false,
    outDir: "dist",
    assetsDir: ".",
    target: "esnext",
    lib: {
      entry: "src/index.ts",
      formats: ["es"],
    },
    rollupOptions: {
      // Do not bundle Electron preload bridge; it's provided at runtime
      external: ["@app/preload"],
      output: {
        entryFileNames: "[name].js",
      },
    },
    emptyOutDir: true,
    reportCompressedSize: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "../renderer/src"),
    },
  },
});
