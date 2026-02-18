#!/usr/bin/env node
/**
 * Quick test for mlt-backend addon.
 * Run from mlt-backend: npm run test
 * Or: node scripts/test.mjs (from mlt-backend dir)
 *
 * Fixes "Could not find the Qt platform plugin cocoa" on macOS Homebrew MLT.
 */
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// Fix Qt plugin path for Homebrew MLT on macOS (same as melt-transcode.sh)
if (process.platform === "darwin" && !process.env.QT_PLUGIN_PATH) {
  const fs = await import("node:fs");
  for (const p of [
    "/opt/homebrew/opt/qtbase/share/qt/plugins",
    "/opt/homebrew/opt/qt/share/qt/plugins",
  ]) {
    if (fs.existsSync(p)) {
      process.env.QT_PLUGIN_PATH = p;
      break;
    }
  }
}
// Block display-dependent MLT modules to avoid segfault when run headless
// Keep: core, avformat, xml, plus, plusgpl for transcoding
if (!process.env.MLT_REPOSITORY_DENY) {
  process.env.MLT_REPOSITORY_DENY =
    "libmltqt6:libmltqt:libmltgdk:libmltsdl2:libmltdecklink:libmltfrei0r:libmltvidstab:libmltopencv:libmltxine";
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const addonPath = join(root, "build", "Release", "addon.node");

async function main() {
  try {
    const addon = createRequire(import.meta.url)(addonPath);
    console.log("mlt-backend addon loaded");
    console.log("  testLite():", addon.testLite());

    // Full MLT init may segfault when run headless. Use MLT_FULL_TEST=1 to try (e.g. in terminal with display)
    const fullTest = process.env.MLT_FULL_TEST === "1";
    if (fullTest) {
      console.log("  testConnection():", addon.testConnection());
      console.log("  getVersion():", addon.getVersion());
      console.log("  getRepositoryDirectory():", addon.getRepositoryDirectory());
    } else {
      console.log("  (lite only - set MLT_FULL_TEST=1 to run full MLT init)");
    }
  } catch (err) {
    console.error("Failed to load addon:", err.message);
    console.error("\nMake sure you have:");
    console.error("  1. brew install mlt  (macOS) or apt install libmlt-dev (Linux)");
    console.error("  2. npm run build     (in mlt-backend)");
    process.exit(1);
  }
}

main();
