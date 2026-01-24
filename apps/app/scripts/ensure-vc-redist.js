#!/usr/bin/env node
/**
 * Ensure Microsoft Visual C++ Redistributable (x64) installer is available for NSIS bundling.
 *
 * We do NOT commit the installer binary to the repo. Instead, we download it at build time
 * into `apps/app/buildResources/vc_redist.x64.exe`, which the NSIS script embeds and runs
 * if the runtime isn't already installed.
 */

import https from "node:https";
import { createWriteStream, existsSync, mkdirSync, statSync } from "node:fs";
import { pipeline } from "node:stream/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const APP_ROOT = resolve(__dirname, "..");

const VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe";
const DEST = join(APP_ROOT, "buildResources", "vc_redist.x64.exe");

async function download(url, destPath) {
  await new Promise((resolvePromise, rejectPromise) => {
    const fetch = (u, redirectsLeft = 5) => {
      https
        .get(
          u,
          {
            headers: {
              "User-Agent": "apex-studio-build/ensure-vc-redist",
              Accept: "*/*",
            },
          },
          (res) => {
            const status = res.statusCode ?? 0;
            const loc = res.headers.location;
            if ([301, 302, 303, 307, 308].includes(status) && loc && redirectsLeft > 0) {
              res.resume();
              const next = loc.startsWith("http") ? loc : new URL(loc, u).toString();
              return fetch(next, redirectsLeft - 1);
            }
            if (status < 200 || status >= 300) {
              res.resume();
              return rejectPromise(new Error(`Download failed (${status}) for ${u}`));
            }
            const out = createWriteStream(destPath);
            pipeline(res, out).then(resolvePromise, rejectPromise);
          },
        )
        .on("error", rejectPromise);
    };
    fetch(url);
  });
}

export async function ensureVCRedist() {
  if (existsSync(DEST)) {
    try {
      if (statSync(DEST).size > 1024 * 1024) return DEST; // sanity: >1MB
    } catch {
      // fall through to re-download
    }
  }

  mkdirSync(dirname(DEST), { recursive: true });
  console.log(`[ensure-vc-redist] Downloading ${VC_REDIST_URL} -> ${DEST}`);
  await download(VC_REDIST_URL, DEST);

  const size = statSync(DEST).size;
  if (size < 1024 * 1024) {
    throw new Error(`[ensure-vc-redist] Downloaded file looks too small (${size} bytes): ${DEST}`);
  }

  console.log(`[ensure-vc-redist] Ready (${(size / (1024 * 1024)).toFixed(1)} MB)`);
  return DEST;
}

// Run as a script when invoked directly: `node scripts/ensure-vc-redist.js`
try {
  const invoked = process.argv?.[1] ? pathToFileURL(process.argv[1]).href : "";
  if (invoked && import.meta.url === invoked) {
    ensureVCRedist().catch((err) => {
      console.error(String(err?.stack || err));
      process.exit(1);
    });
  }
} catch {
  // If path/url conversion fails for any reason, don't block builds that import this module.
}

