import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = fileURLToPath(new URL("../", import.meta.url));
const videoDecodeDir = path.join(root, "src", "lib", "video-decode");

await build({
  entryPoints: [
    path.join(videoDecodeDir, "renderer_2d.ts"),
    path.join(videoDecodeDir, "renderer.ts"),
    path.join(videoDecodeDir, "renderer_webgl.ts"),
  ],
  outdir: videoDecodeDir,
  format: "esm",
  logLevel: "info",
});

await build({
  entryPoints: [path.join(videoDecodeDir, "worker.ts")],
  outfile: path.join(videoDecodeDir, "dist", "worker.js"),
  bundle: true,
  format: "esm",
  platform: "node",
  loader: { ".glsl": "text" },
  mainFields: ["browser", "module", "main"],
  logLevel: "info",
});

await build({
  entryPoints: [path.join(videoDecodeDir, "module.ts")],
  outfile: path.join(videoDecodeDir, "dist", "module.js"),
  bundle: true,
  format: "esm",
  logLevel: "info",
});
