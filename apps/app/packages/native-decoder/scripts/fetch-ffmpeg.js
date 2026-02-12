#!/usr/bin/env node
/**
 * fetch-ffmpeg.js
 *
 * Downloads pre-built FFmpeg static development libraries for the current
 * platform. These include headers and static .a / .lib archives needed to
 * compile the native-decoder addon.
 *
 * Sources:
 *   - Windows / Linux: BtbN/FFmpeg-Builds (GitHub releases)
 *   - macOS: built via Homebrew or fetched from evermeet.cx
 *
 * The script is idempotent — it skips download if the target directory
 * already contains an `include/libavformat` folder.
 */

const fs = require("fs");
const path = require("path");
const https = require("https");
const { execSync } = require("child_process");
const os = require("os");

const DEPS_DIR = path.resolve(__dirname, "..", "deps", "ffmpeg");

// FFmpeg version to fetch
const FFMPEG_VERSION = "7.1";

// BtbN release tag (win/linux)
const BTBN_TAG = `n${FFMPEG_VERSION}-latest`;

function getPlatformKey() {
  const platform = os.platform();
  const arch = os.arch();

  if (platform === "win32") return "win-x64";
  if (platform === "darwin" && arch === "arm64") return "darwin-arm64";
  if (platform === "darwin") return "darwin-x64";
  if (platform === "linux") return "linux-x64";
  throw new Error(`Unsupported platform: ${platform}-${arch}`);
}

function alreadyFetched(destDir) {
  return fs.existsSync(path.join(destDir, "include", "libavformat"));
}

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const follow = (url, redirects = 0) => {
      if (redirects > 5) return reject(new Error("Too many redirects"));
      https
        .get(url, (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            return follow(res.headers.location, redirects + 1);
          }
          if (res.statusCode !== 200) {
            return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
          }
          const stream = fs.createWriteStream(dest);
          res.pipe(stream);
          stream.on("finish", () => {
            stream.close();
            resolve();
          });
        })
        .on("error", reject);
    };
    follow(url);
  });
}

async function fetchWindows(destDir) {
  // BtbN shared GPL builds provide headers, import .lib, and DLLs.
  // The version suffix (e.g. -7.1) is required for release-branch builds.
  const archiveName = `ffmpeg-${BTBN_TAG}-win64-gpl-shared-${FFMPEG_VERSION}`;
  const url = `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${archiveName}.zip`;
  const zipPath = path.join(os.tmpdir(), `${archiveName}.zip`);

  console.log(`Downloading FFmpeg for Windows from ${url}...`);
  await downloadFile(url, zipPath);

  console.log("Extracting...");
  fs.mkdirSync(destDir, { recursive: true });

  // Use PowerShell to extract on Windows
  execSync(
    `powershell -NoProfile -Command "Expand-Archive -Force '${zipPath}' '${os.tmpdir()}'"`
  );

  const extractedDir = path.join(os.tmpdir(), archiveName);
  // Copy include, lib, and bin (DLLs needed at runtime) directories
  copyDirSync(path.join(extractedDir, "include"), path.join(destDir, "include"));
  copyDirSync(path.join(extractedDir, "lib"), path.join(destDir, "lib"));
  copyDirSync(path.join(extractedDir, "bin"), path.join(destDir, "bin"));

  // Cleanup
  try {
    fs.unlinkSync(zipPath);
    fs.rmSync(extractedDir, { recursive: true, force: true });
  } catch {}

  console.log(`FFmpeg dev files installed to ${destDir}`);
}

async function fetchLinux(destDir) {
  // BtbN shared GPL builds provide headers, .so shared libs, and .a import libs.
  // The version suffix (e.g. -7.1) is required for release-branch builds.
  const archiveName = `ffmpeg-${BTBN_TAG}-linux64-gpl-shared-${FFMPEG_VERSION}`;
  const url = `https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${archiveName}.tar.xz`;
  const tarPath = path.join(os.tmpdir(), `${archiveName}.tar.xz`);

  console.log(`Downloading FFmpeg for Linux from ${url}...`);
  await downloadFile(url, tarPath);

  console.log("Extracting...");
  fs.mkdirSync(destDir, { recursive: true });

  execSync(`tar -xf "${tarPath}" -C "${os.tmpdir()}"`);

  const extractedDir = path.join(os.tmpdir(), archiveName);
  copyDirSync(path.join(extractedDir, "include"), path.join(destDir, "include"));
  copyDirSync(path.join(extractedDir, "lib"), path.join(destDir, "lib"));

  try {
    fs.unlinkSync(tarPath);
    fs.rmSync(extractedDir, { recursive: true, force: true });
  } catch {}

  console.log(`FFmpeg dev files installed to ${destDir}`);
}

async function fetchDarwin(destDir, arch) {
  console.log(`Fetching FFmpeg for macOS (${arch}) via Homebrew...`);
  fs.mkdirSync(destDir, { recursive: true });

  // Check if Homebrew FFmpeg is installed
  try {
    const prefix = execSync("brew --prefix ffmpeg", { encoding: "utf8" }).trim();
    if (fs.existsSync(path.join(prefix, "include", "libavformat"))) {
      copyDirSync(path.join(prefix, "include"), path.join(destDir, "include"));
      copyDirSync(path.join(prefix, "lib"), path.join(destDir, "lib"));
      console.log(`FFmpeg dev files copied from Homebrew (${prefix}) to ${destDir}`);
      return;
    }
  } catch {}

  // Fallback: install via Homebrew
  console.log("Installing FFmpeg via Homebrew...");
  execSync("brew install ffmpeg", { stdio: "inherit" });
  const prefix = execSync("brew --prefix ffmpeg", { encoding: "utf8" }).trim();
  copyDirSync(path.join(prefix, "include"), path.join(destDir, "include"));
  copyDirSync(path.join(prefix, "lib"), path.join(destDir, "lib"));
  console.log(`FFmpeg dev files installed to ${destDir}`);
}

function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

async function main() {
  const platformKey = getPlatformKey();
  const destDir = path.join(DEPS_DIR, platformKey);

  if (alreadyFetched(destDir)) {
    console.log(`FFmpeg dev files already present at ${destDir}, skipping download.`);
    return;
  }

  switch (platformKey) {
    case "win-x64":
      await fetchWindows(destDir);
      break;
    case "linux-x64":
      await fetchLinux(destDir);
      break;
    case "darwin-x64":
      await fetchDarwin(destDir, "x64");
      break;
    case "darwin-arm64":
      await fetchDarwin(destDir, "arm64");
      break;
  }
}

main().catch((err) => {
  console.error("Failed to fetch FFmpeg:", err);
  process.exit(1);
});
