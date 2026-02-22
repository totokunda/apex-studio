import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = fileURLToPath(new URL("../", import.meta.url));

const workers = [
  {
    entryPoint: path.join(root, "src", "lib", "media", "video-decoder.worker.ts"),
    outFile: path.join(root, "src", "lib", "media", "video-decoder.worker.cjs"),
  },
  {
    entryPoint: path.join(root, "src", "lib", "media", "audio-decoder.worker.ts"),
    outFile: path.join(root, "src", "lib", "media", "audio-decoder.worker.cjs"),
  },
];

for (const worker of workers) {
  await build({
    entryPoints: [worker.entryPoint],
    outfile: worker.outFile,
    bundle: true,
    format: "iife",
    platform: "node",
    target: "node20",
    external: ["mediabunny"],
    legalComments: "none",
    logLevel: "info",
  });
}
