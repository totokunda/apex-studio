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
      // During this package's own build we can end up traversing into `packages/renderer/src`
      // (via relative imports from this package). Some of those renderer files import
      // `@app/export-renderer`, which would otherwise try to resolve this package via its
      // `package.json` entry (dist output that doesn't exist yet) and fail the build.
      "@app/export-renderer": path.resolve(__dirname, "src/index.ts"),
      "@": path.resolve(__dirname, "../renderer/src"),
    },
  },
});
