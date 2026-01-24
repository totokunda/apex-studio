import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// In production builds we ship `filters/small` + `filters/examples` by default.
// `filters/full` is intentionally excluded to keep install size reasonable.
// To include it (e.g. for internal builds), set: APEX_BUNDLE_FILTERS_FULL=1
const bundleFull = process.env.APEX_BUNDLE_FILTERS_FULL === "1";
if (bundleFull) {
  process.stdout.write(
    "[renderer:postbuild] APEX_BUNDLE_FILTERS_FULL=1; keeping dist/filters/full\n",
  );
  process.exit(0);
}

const distFiltersFull = path.resolve(__dirname, "..", "dist", "filters", "full");
if (!fs.existsSync(distFiltersFull)) {
  process.stdout.write(
    `[renderer:postbuild] No dist/filters/full found at ${distFiltersFull}\n`,
  );
  process.exit(0);
}

process.stdout.write(
  `[renderer:postbuild] Removing dist/filters/full at ${distFiltersFull}\n`,
);
fs.rmSync(distFiltersFull, { recursive: true, force: true });

