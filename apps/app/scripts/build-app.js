#!/usr/bin/env node
/**
 * Apex Studio Full Build Script
 * 
 * This script orchestrates the complete build process:
 * 1. Builds the Python API bundle
 * 2. Builds the Electron app with workspace packages
 * 3. Packages everything together with electron-builder
 * 
 * Usage:
 *   node scripts/build-app.js [options]
 * 
 * Options:
 *   --platform [darwin|linux|win32|all]  Target platform (default: current)
 *   --arch [x64|arm64|universal]         Target architecture (default: current)
 *   --cuda [auto|cuda126|cpu|mps]        GPU support (default: auto)
 *   --skip-python                        Skip Python bundling (use existing)
 *   --skip-sign                          Skip code signing
 *   --publish                            Publish release after build
 *   --draft                              Create draft release
 */

import { spawn, execSync } from "node:child_process";
import { existsSync, mkdirSync, cpSync, rmSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const APP_ROOT = resolve(__dirname, "..");
const API_ROOT = resolve(APP_ROOT, "..", "api");
const PYTHON_BUNDLE_DIR = join(APP_ROOT, "python-api-bundle");

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    platform: process.platform,
    arch: process.arch,
    cuda: "auto",
    skipPython: false,
    skipSign: false,
    publish: false,
    draft: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case "--platform":
        config.platform = args[++i];
        break;
      case "--arch":
        config.arch = args[++i];
        break;
      case "--cuda":
        config.cuda = args[++i];
        break;
      case "--skip-python":
        config.skipPython = true;
        break;
      case "--skip-sign":
        config.skipSign = true;
        break;
      case "--publish":
        config.publish = true;
        break;
      case "--draft":
        config.draft = true;
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
    }
  }

  return config;
}

function printHelp() {
  console.log(`
Apex Studio Build Script

Usage:
  node scripts/build-app.js [options]

Options:
  --platform [darwin|linux|win32|all]  Target platform (default: current)
  --arch [x64|arm64|universal]         Target architecture (default: current)
  --cuda [auto|cuda126|cpu|mps]        GPU support for Python (default: auto)
  --skip-python                        Skip Python bundling (use existing)
  --skip-sign                          Skip code signing
  --publish                            Publish release after build
  --draft                              Create draft release
  --help, -h                           Show this help message

Environment Variables:
  APPLE_ID              Apple ID for notarization
  APPLE_APP_PASSWORD    App-specific password for notarization
  APPLE_TEAM_ID         Apple Developer Team ID
  APPLE_IDENTITY        Code signing identity (certificate name)
  WINDOWS_CERT_FILE     Path to Windows code signing certificate (.pfx)
  WINDOWS_CERT_PASSWORD Password for the certificate
  GITHUB_TOKEN          GitHub token for publishing releases

Examples:
  # Build for current platform
  node scripts/build-app.js

  # Build for macOS with Apple Silicon
  node scripts/build-app.js --platform darwin --arch arm64

  # Build for Windows with CPU-only Python
  node scripts/build-app.js --platform win32 --cuda cpu

  # Build and publish release
  node scripts/build-app.js --publish
`);
}

// Logging utilities
function log(message, type = "info") {
  const prefix = {
    info: "\x1b[34m[INFO]\x1b[0m",
    success: "\x1b[32m[SUCCESS]\x1b[0m",
    warning: "\x1b[33m[WARNING]\x1b[0m",
    error: "\x1b[31m[ERROR]\x1b[0m",
    step: "\x1b[35m[STEP]\x1b[0m",
  };
  console.log(`${prefix[type]} ${message}`);
}

// Run a command and return a promise
function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    log(`Running: ${command} ${args.join(" ")}`, "info");
    
    const proc = spawn(command, args, {
      stdio: "inherit",
      shell: true,
      ...options,
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with code ${code}`));
      }
    });

    proc.on("error", reject);
  });
}

// Step 1: Build Python API bundle
async function buildPythonBundle(config) {
  if (config.skipPython) {
    if (existsSync(PYTHON_BUNDLE_DIR)) {
      log("Skipping Python bundle (--skip-python), using existing bundle", "warning");
      return true;
    } else {
      log("No existing Python bundle found, building...", "warning");
    }
  }

  log("Building Python API bundle...", "step");

  // Clean previous bundle
  if (existsSync(PYTHON_BUNDLE_DIR)) {
    log("Removing previous Python bundle...", "info");
    rmSync(PYTHON_BUNDLE_DIR, { recursive: true, force: true });
  }

  // Build Python bundle using the bundler script
  const pythonCmd = process.platform === "win32" ? "python" : "python3";
  const bundlerScript = join(API_ROOT, "scripts", "bundle_python.py");

  const args = [
    bundlerScript,
    "--platform", config.platform,
    "--output", join(APP_ROOT, "temp-python-build"),
  ];

  if (config.cuda !== "auto") {
    args.push("--cuda", config.cuda);
  }

  if (!config.skipSign) {
    args.push("--sign");
  }

  try {
    await runCommand(pythonCmd, args, { cwd: API_ROOT });
    
    // Copy bundle to expected location
    const builtBundle = join(APP_ROOT, "temp-python-build", "python-api", "apex-engine");
    if (existsSync(builtBundle)) {
      mkdirSync(PYTHON_BUNDLE_DIR, { recursive: true });
      cpSync(builtBundle, PYTHON_BUNDLE_DIR, { recursive: true });
      log("Python bundle created successfully", "success");
    } else {
      throw new Error("Python bundle was not created");
    }

    // Clean up temp directory
    rmSync(join(APP_ROOT, "temp-python-build"), { recursive: true, force: true });
    
    return true;
  } catch (error) {
    log(`Failed to build Python bundle: ${error.message}`, "error");
    log("Continuing without bundled Python (dev mode)...", "warning");
    return false;
  }
}

// Step 2: Build Electron app packages
async function buildElectronPackages() {
  log("Building Electron app packages...", "step");

  try {
    await runCommand("npm", ["run", "build"], { cwd: APP_ROOT });
    log("Electron packages built successfully", "success");
    return true;
  } catch (error) {
    log(`Failed to build Electron packages: ${error.message}`, "error");
    throw error;
  }
}

// Step 3: Package with electron-builder
async function packageApp(config) {
  log("Packaging application with electron-builder...", "step");

  const args = ["electron-builder", "build", "--config", "electron-builder.mjs"];

  // Platform
  if (config.platform !== process.platform || config.platform === "all") {
    if (config.platform === "all") {
      args.push("--mac", "--win", "--linux");
    } else {
      const platformFlags = {
        darwin: "--mac",
        win32: "--win",
        linux: "--linux",
      };
      args.push(platformFlags[config.platform]);
    }
  }

  // Architecture
  if (config.arch && config.arch !== process.arch) {
    if (config.arch === "universal") {
      args.push("--universal");
    } else {
      args.push("--" + config.arch);
    }
  }

  // Publishing
  if (config.publish) {
    if (config.draft) {
      args.push("--publish", "always", "--prerelease");
    } else {
      args.push("--publish", "always");
    }
  }

  try {
    await runCommand("npx", args, { cwd: APP_ROOT });
    log("Application packaged successfully", "success");
    return true;
  } catch (error) {
    log(`Failed to package application: ${error.message}`, "error");
    throw error;
  }
}

// Main build process
async function main() {
  const startTime = Date.now();
  const config = parseArgs();

  console.log(`
╔════════════════════════════════════════════════════════════════╗
║                    Apex Studio Build                           ║
╠════════════════════════════════════════════════════════════════╣
║  Platform: ${config.platform.padEnd(12)} Architecture: ${config.arch.padEnd(12)}  ║
║  CUDA: ${config.cuda.padEnd(16)} Sign: ${(!config.skipSign).toString().padEnd(16)}  ║
╚════════════════════════════════════════════════════════════════╝
`);

  try {
    // Check prerequisites
    log("Checking prerequisites...", "step");
    
    // Check for Node.js
    try {
      execSync("node --version", { stdio: "ignore" });
    } catch {
      throw new Error("Node.js is required but not installed");
    }

    // Check for npm
    try {
      execSync("npm --version", { stdio: "ignore" });
    } catch {
      throw new Error("npm is required but not installed");
    }

    log("Prerequisites check passed", "success");

    // Build Python bundle
    const hasPythonBundle = await buildPythonBundle(config);
    if (!hasPythonBundle) {
      log("Building without bundled Python - app will require external API", "warning");
    }

    // Build Electron packages
    await buildElectronPackages();

    // Package the app
    await packageApp(config);

    // Calculate duration
    const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
    
    console.log(`
╔════════════════════════════════════════════════════════════════╗
║                    Build Complete! 🎉                          ║
╠════════════════════════════════════════════════════════════════╣
║  Duration: ${duration} minutes                                      ║
║  Output: ${join(APP_ROOT, "dist").padEnd(45)}║
╚════════════════════════════════════════════════════════════════╝
`);

    // List output files
    log("Build artifacts:", "info");
    const distDir = join(APP_ROOT, "dist");
    if (existsSync(distDir)) {
      const { readdirSync, statSync } = await import("node:fs");
      const files = readdirSync(distDir)
        .filter((f) => !f.startsWith(".") && statSync(join(distDir, f)).isFile())
        .map((f) => {
          const size = (statSync(join(distDir, f)).size / 1024 / 1024).toFixed(2);
          return `  - ${f} (${size} MB)`;
        });
      console.log(files.join("\n"));
    }

  } catch (error) {
    log(`Build failed: ${error.message}`, "error");
    console.error(error);
    process.exit(1);
  }
}

main();

